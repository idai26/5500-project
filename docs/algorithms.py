"""Best-first shortest-path search on the DC street network.

This module is the Python implementation behind the Methods and Results
sections of ``index.qmd``. It provides Dijkstra's algorithm and the A*
algorithm as two configurations of a single "best-first" search routine that
runs over an OSMnx-derived ``networkx.MultiDiGraph`` of the Washington, DC
drive network.

This paper frames both algorithms in the same vocabulary:

- A **source node** ``s`` and a **target (goal) node** ``t``.
- A **priority queue** that orders candidate nodes by a key.
- An **edge relaxation** step that updates the best known cost ``g(n)`` from
  ``s`` to ``n`` whenever a cheaper route is discovered.
- A **settling** step that locks in the optimal ``g(n)`` for a node ``n`` once
  it is popped from the queue.
- For A*, an **admissible heuristic** ``h(n)`` (a lower bound on the true
  remaining cost to ``t``) and a priority ``f(n) = g(n) + h(n)``.

"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterator

import networkx as nx

# Type alias for a graph node identifier (OSMnx uses integer OSM ids, but we
# only assume the node id is hashable so it can key dictionaries and sets).
Node = Hashable


def _min_edge_weight(
    G: nx.MultiDiGraph, u: Node, v: Node, weight: str
) -> float | None:
    """Return the cheapest cost of any directed edge from ``u`` to ``v``.

    OSM road graphs are stored as ``MultiDiGraph`` instances because two
    intersections can be joined by more than one road segment (for example, a
    service road that runs alongside an avenue, or a divided roadway whose
    two carriageways are modeled as separate edges). For shortest-path
    purposes we treat that bundle of parallel edges as a single logical move
    and pick the minimum-cost option under the active cost model.

    ``weight`` selects which precomputed edge attribute defines "cost":

    - ``"length"`` for the raw distance baseline (meters),
    - ``"travel_time_s"`` for the speed-based travel-time model (seconds),
    - ``"travel_time_grade_s"`` for the absolute-incline-penalized model.

    Returns ``None`` when ``v`` is not a successor of ``u`` or when no edge
    between them carries the requested weight attribute.
    """
    if v not in G[u]:
        return None
    best: float | None = None
    for _, data in G[u][v].items():
        w = data.get(weight)
        if w is None:
            continue
        w = float(w)
        if best is None or w < best:
            best = w
    return best


def _neighbors(G: nx.MultiDiGraph, u: Node, weight: str) -> Iterator[tuple[Node, float]]:
    """Yield successors of ``u`` paired with the edge cost used during relaxation.

    Each pair ``(v, edge_cost)`` is exactly the input the relaxation step
    needs to test whether routing through ``u`` improves the best known
    ``g``-score for ``v``. Successors with no usable weight under the active
    cost model are silently skipped.
    """
    for v in G.successors(u):
        w = _min_edge_weight(G, u, v, weight)
        if w is not None:
            yield v, w


@dataclass
class SearchResult:
    """Output of a best-first search: the path plus diagnostics for the paper.

    The fields here line up with the comparison table in the Results section
    of ``index.qmd``, so that each row of that table is just a few attribute
    lookups on a ``SearchResult``:

    - ``path``: the optimal node sequence from the source to the target,
      reconstructed by backtracking predecessor pointers. Empty if the
      target is unreachable from the source under the active cost model.
    - ``cost``: the total ``g``-score at the target, i.e. the sum of edge
      weights along ``path`` under the active cost model. Reported in
      meters for the ``length`` model and in seconds for the time-based
      models. ``math.inf`` if the target is unreachable.
    - ``nodes_settled``: the count of **settled nodes** -- nodes for which
      the search has proven the shortest path from the source has been
      found. This is the "Nodes settled" column in the comparison table
      and is the headline measure of how much of the graph the algorithm
      had to commit to before terminating.
    - ``pq_pops``: the count of priority-queue pop operations, **including
      stale entries** that were discarded by lazy deletion. This is the
      "PQ pops" column and is always at least ``nodes_settled``; the gap
      reflects how often a better route to a node was discovered after an
      earlier entry for that node had already been queued.
    - ``elapsed_s``:runtime of the search in seconds.
    - ``frames``: ordered snapshots of the search state (settled set,
      frontier, current node, current best predecessor chain) that
      ``routing_viz.build_search_animation`` uses for the Plotly
      animations rendered in the Results section.
    """
    path: list[Node]
    cost: float
    nodes_settled: int
    pq_pops: int
    elapsed_s: float
    frames: list[dict[str, Any]] = field(default_factory=list)


def _euclidean_heuristic(
    xy: dict[Node, tuple[float, float]], goal: Node
) -> Callable[[Node], float]:
    """Build ``h(n)``, the straight-line distance from ``n`` to the goal node.

    This is the default A* heuristic for the ``length`` cost model. It is
    **admissible** because the Euclidean distance between two points is
    always less than or equal to the shortest distance along any road
    network connecting them.

    Coordinates must be supplied in a projected CRS (meters) so that
    ``math.hypot`` is comparable to the ``length`` edge weights. 

    For the time-based cost models (``travel_time_s`` and
    ``travel_time_grade_s``) a different lower-bound heuristic is appropriate
    -- straight-line distance divided by the maximum edge speed in the graph
    -- and is supplied via ``astar``'s explicit ``heuristic_fn`` argument
    rather than constructed here.
    """
    gx, gy = xy[goal]

    def h(n: Node) -> float:
        x, y = xy[n]
        return math.hypot(x - gx, y - gy)

    return h


def _reconstruct_path(predecessor: dict[Node, Node | None], current: Node) -> list[Node]:
    """Backtrack predecessor pointers from the target back to the source.

    ``predecessor`` is the predecessor map maintained by the search: each
    settled node ``v`` stores the predecessor ``u`` that produced its best
    known ``g``-score. Following those pointers from the target node and
    reversing the resulting list yields the optimal path in source-to-target
    order, exactly the "linked-list-style backtracking" described in the
    Methods section of the paper.
    """
    path = [current]
    while predecessor.get(current) is not None:
        current = predecessor[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def _append_frame(
    frames: list[dict[str, Any]],
    settled: set[Node],
    frontier: set[Node],
    current: Node | None,
    predecessor: dict[Node, Node | None],
    max_path_preview: int = 400,
) -> None:
    """Snapshot the current search state for the Plotly animations.

    Each frame captures the two node sets that the animations in the
    Results section color-code, plus the node currently being settled:

    - ``settled`` is the set of **settled** nodes whose final ``g``-score
      is locked in. They render as gray in the animations.
    - ``frontier`` is the set of nodes that have been discovered but not yet
      settled, i.e. the nodes still sitting in the priority queue under
      their best known ``g``-score. They render as orange.
    - ``current`` is the node being settled in this frame (``None`` for the
      initial pre-search frame).
    - ``path_preview`` is the best known predecessor chain from the source
      to ``current``, reconstructed by walking ``predecessor`` backward.
      While the search is running this is the algorithm's current best
      guess at the partial route; on the final frame it is the optimal
      path. It renders as the blue line on intermediate frames and as the
      green line on the last frame.
    """
    path_preview: list[Node] = []
    if current is not None:
        c: Node | None = current
        steps = 0
        while c is not None and steps < max_path_preview:
            path_preview.append(c)
            c = predecessor.get(c)  # type: ignore[assignment]
            steps += 1
        path_preview.reverse()
    frames.append(
        {
            "settled": set(settled),
            "frontier": set(frontier),
            "current": current,
            "path_preview": path_preview,
        }
    )


def _best_first_search(
    G: nx.MultiDiGraph,
    source: Node,
    target: Node,
    *,
    weight: str,
    heuristic: Callable[[Node], float],
    priority_key: Callable[[float, float], float],
    snapshot_every: int,
) -> SearchResult:
    """Unified best-first search underlying both Dijkstra and A*.

    - Dijkstra orders by ``g(n)`` alone (``priority_key = lambda g, h: g``
      and ``heuristic = lambda _: 0.0``).
    - A* orders by ``f(n) = g(n) + h(n)`` (``priority_key = lambda g, h:
      g + h`` and ``heuristic`` is an admissible estimate of the remaining
      cost from ``n`` to the target).

    This function implements the loop. On every iteration it pops
    the lowest-priority node ``u`` from the priority queue, **settles** it
    (locks in ``g(u)`` as optimal), and **relaxes** every outgoing edge
    ``(u, v)`` to update ``g(v)`` whenever the route through ``u`` is
    cheaper than the best route to ``v`` previously discovered.

    Snapshots of the search state are recorded periodically so the Results
    section can animate the exploration. The ``snapshot_every`` knob trades
    off frame density against memory; the very first expansion and the
    final goal expansion are always captured so the animation has clean
    bookends.
    """
    t0 = time.perf_counter()

    # The priority queue stores 4-tuples ``(priority, g, counter, node)``:
    #   - ``priority`` is the ordering key returned by ``priority_key`` and
    #     equals ``g`` for Dijkstra and ``g + h`` for A*.
    #   - ``g`` is the ``g``-score that was current when this entry was
    #     pushed; we re-check it on pop to detect stale entries below.
    #   - ``counter`` is a monotonically increasing tiebreaker that gives
    #     deterministic FIFO ordering whenever two entries share a
    #     priority. It also prevents ``heapq`` from ever attempting to
    #     compare two ``Node`` objects directly, which would fail for any
    #     non-orderable hashable id.
    pq: list[tuple[float, float, int, Node]] = []
    counter = 0
    heapq.heappush(pq, (priority_key(0.0, heuristic(source)), 0.0, counter, source))
    counter += 1

    # ``g[n]`` is the best known cost g(n) from the source to n. It is only
    # finalized (provably optimal) once n is settled; until then it can
    # still be improved by a future relaxation.
    g: dict[Node, float] = {source: 0.0}
    # Predecessor pointers used to backtrack the optimal path at the end.
    predecessor: dict[Node, Node | None] = {source: None}
    # ``settled`` is the paper's "settled set" -- nodes whose final
    # ``g``-score has been locked in.
    settled: set[Node] = set()
    frontier: set[Node] = {source}
    frames: list[dict[str, Any]] = []
    pq_pops = 0
    nodes_settled = 0

    # Initial frame: nothing settled yet, only the source on the frontier.
    _append_frame(frames, settled, frontier, None, predecessor)

    while pq:
        _prio, g_at_pop, _, u = heapq.heappop(pq)
        pq_pops += 1

        # Lazy deletion of stale priority-queue entries. Python's ``heapq``
        # has no decrease-key operation, so when relaxation finds a cheaper
        # route to a node that is already on the queue we simply push a new
        # entry under the lower priority and leave the older one in place.
        # When the older (worse) entry eventually surfaces, its stored ``g``
        # is greater than the current ``g[u]`` and we discard it. This
        # is the source of the gap between ``pq_pops`` and ``nodes_settled``
        # reported in the Results comparison table.
        if g_at_pop > g.get(u, math.inf):
            continue
        # A node is only settled once. If two equally good entries for the
        # same node both reach the front of the queue, the second is a
        # no-op.
        if u in settled:
            continue

        # Settling step: ``u`` is now permanently part of the settled set.
        # For nonnegative edge weights and an admissible heuristic this is
        # the moment when the invariant g(u) = optimal cost from source to u
        # becomes provable, which is precisely the correctness argument the
        # paper makes for both Dijkstra and A*.
        settled.add(u)
        frontier.discard(u)
        nodes_settled += 1

        if nodes_settled == 1 or nodes_settled % snapshot_every == 0 or u == target:
            _append_frame(frames, settled, frontier, u, predecessor)

        # Goal test happens *after* settling. That ordering matters: it is
        # the act of settling the target -- not merely discovering it on the
        # frontier -- that proves g(target) is optimal under the
        # admissibility assumption. There could still be a shorter path via
        # an intermediate node, even if we have discovered the goal node.
        if u == target:
            path = _reconstruct_path(predecessor, target)
            elapsed = time.perf_counter() - t0
            return SearchResult(
                path=path,
                cost=g[target],
                nodes_settled=nodes_settled,
                pq_pops=pq_pops,
                elapsed_s=elapsed,
                frames=frames,
            )

        # Edge relaxation. For each successor ``v`` of the just-settled
        # node ``u``, compute the tentative g-score that goes through ``u``.
        # If that candidate strictly improves the best route to ``v`` known
        # so far, update ``g[v]``, repoint ``v``'s predecessor to ``u``, and
        # push a fresh entry into the priority queue under priority
        # ``priority_key(g(v), h(v))``. ``v`` joins the frontier if it is
        # not already there.
        for v, w in _neighbors(G, u, weight):
            tentative_g = g[u] + w
            if tentative_g < g.get(v, math.inf):
                g[v] = tentative_g
                predecessor[v] = u
                heapq.heappush(
                    pq,
                    (priority_key(tentative_g, heuristic(v)), tentative_g, counter, v),
                )
                counter += 1
                frontier.add(v)

    # Priority queue exhausted without ever settling the target -- the
    # target is unreachable from the source under the active cost model.
    elapsed = time.perf_counter() - t0
    return SearchResult(
        path=[],
        cost=math.inf,
        nodes_settled=nodes_settled,
        pq_pops=pq_pops,
        elapsed_s=elapsed,
        frames=frames,
    )


def dijkstra(
    G: nx.MultiDiGraph,
    source: Node,
    target: Node,
    *,
    weight: str = "length",
    xy: dict[Node, tuple[float, float]] | None = None,
    snapshot_every: int = 80,
) -> SearchResult:
    """Dijkstra's algorithm: best-first search prioritized by ``g(n)`` alone.

Nodes are pulled from the priority queue in order of their
    best known cost ``g(n)`` from the source, settled in that order, and
    their outgoing edges are relaxed. The first time the target is settled
    its ``g``-score is provably the optimal cost from source to target,
    which is how ``cost`` and ``path`` in the returned ``SearchResult`` are
    guaranteed to be optimal.

    Equivalently, this is the special case of A* where the heuristic is
    identically zero (``h(n) == 0`` for every ``n``), so the priority key
    ``f(n) = g(n) + h(n)`` collapses to ``g(n)``.

    ``xy`` is accepted only to keep the public signature symmetric with
    ``astar``; Dijkstra never consults node coordinates.
    """
    return _best_first_search(
        G,
        source,
        target,
        weight=weight,
        heuristic=lambda _n: 0.0,
        priority_key=lambda g, _h: g,
        snapshot_every=snapshot_every,
    )


def astar(
    G: nx.MultiDiGraph,
    source: Node,
    target: Node,
    *,
    weight: str = "length",
    xy: dict[Node, tuple[float, float]] | None = None,
    heuristic_fn: Callable[[Node], float] | None = None,
    snapshot_every: int = 80,
) -> SearchResult:
    """A* search: best-first search prioritized by ``f(n) = g(n) + h(n)``.

    A* extends Dijkstra by ordering the priority queue with the heuristic
    ``f(n) = g(n) + h(n)``, where ``g(n)`` is the best known cost from the
    source to ``n`` and ``h(n)`` is an estimate of the remaining cost from
    ``n`` to the target. Adding the goal-directed term ``h`` biases the
    search toward nodes that look promising relative to the target, which
    typically results in fewer settled nodes than Dijkstra.

    For the returned path to be guaranteed optimal, the heuristic must be
    **admissible**: ``h(n)`` must never overestimate the true remaining
    cost from ``n`` to the target. Combined with nonnegative edge weights,
    admissibility guarantees that the moment the target is settled,
    ``g(target)`` is the optimal cost.

    The heuristic can be supplied in either of two ways:

    - ``heuristic_fn``: any callable mapping a node to a nonnegative float.
      Use this for the time-based cost models, where the appropriate
      admissible lower bound is ``straight_line_distance(n, target) /
      max_edge_speed`` -- the smallest possible travel time, achieved if
      every road were the fastest road in the graph.
    - ``xy``: a mapping from node to projected ``(x, y)`` coordinates in
      meters. When supplied without ``heuristic_fn``, A* automatically
      builds the Euclidean straight-line heuristic, which is the admissible
      choice for the ``length`` cost model.

    Raises ``ValueError`` if neither ``xy`` nor ``heuristic_fn`` is
    supplied, since A* cannot operate without a heuristic.
    """
    if heuristic_fn is not None:
        h_fn = heuristic_fn
    else:
        if xy is None:
            raise ValueError("Either xy or heuristic_fn must be provided to astar.")
        h_fn = _euclidean_heuristic(xy, target)
    return _best_first_search(
        G,
        source,
        target,
        weight=weight,
        heuristic=h_fn,
        priority_key=lambda g, h: g + h,
        snapshot_every=snapshot_every,
    )


def nx_xy_from_graph(G: nx.MultiDiGraph) -> dict[Node, tuple[float, float]]:
    """Pull projected ``(x, y)`` coordinates off every node in an OSMnx graph.

    The graph is expected to have already been reprojected (typically with
    ``osmnx.project_graph``) so that ``x`` and ``y`` are in meters in a local
    Cartesian CRS rather than raw lon/lat in degrees. That projection is
    what makes ``math.hypot(dx, dy)`` a meaningful, units-consistent lower
    bound for the ``length`` cost model used by the Euclidean A* heuristic.
    """
    return {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}
