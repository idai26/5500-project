"""Shortest-path search helpers with trace snapshots for visualization.

This module runs Dijkstra and A* over a ``networkx.MultiDiGraph`` and records
intermediate search state ("frames") so callers can animate how the frontier
evolves over time.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterator

import networkx as nx

# Type alias: node id
Node = Hashable


def _min_edge_weight(
    G: nx.MultiDiGraph, u: Node, v: Node, weight: str
) -> float | None:
    """Return the lightest parallel edge weight from ``u`` to ``v``.

    ``MultiDiGraph`` can hold multiple directed edges between the same nodes;
    this helper treats that bundle as one logical move with the minimum cost.
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
    """Yield reachable neighbors from ``u`` as ``(neighbor, edge_cost)`` pairs."""
    for v in G.successors(u):
        w = _min_edge_weight(G, u, v, weight)
        if w is not None:
            yield v, w


@dataclass
class SearchResult:
    """Container for path output plus search-efficiency diagnostics.

    - ``path``: node sequence from source to target (empty if unreachable)
    - ``cost``: total path weight (``math.inf`` if unreachable)
    - ``nodes_expanded``: number of nodes actually expanded
    - ``pq_pops``: number of priority-queue pops (including stale entries)
    - ``elapsed_s``: wall-clock runtime in seconds
    - ``frames``: sampled snapshots for animation/debugging
    """
    path: list[Node]
    cost: float
    nodes_expanded: int
    pq_pops: int
    elapsed_s: float
    frames: list[dict[str, Any]] = field(default_factory=list)


def _euclidean_heuristic(
    xy: dict[Node, tuple[float, float]], goal: Node
) -> Callable[[Node], float]:
    """Build ``h(n)`` as straight-line distance from ``n`` to ``goal``."""
    gx, gy = xy[goal]

    def h(n: Node) -> float:
        x, y = xy[n]
        return math.hypot(x - gx, y - gy)

    return h


def _reconstruct_path(came_from: dict[Node, Node | None], current: Node) -> list[Node]:
    """Walk predecessor links backward, then reverse to source->target order."""
    path = [current]
    while came_from.get(current) is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def _append_frame(
    frames: list[dict[str, Any]],
    visited: set[Node],
    frontier: set[Node],
    closed: set[Node],
    current: Node | None,
    came_from: dict[Node, Node | None],
    start: Node,
    max_path_preview: int = 400,
) -> None:
    """Store a frame of the current search state for visualization.

    ``path_preview`` is the current best-known predecessor chain from ``start``
    to ``current``. A maximum preview length avoids pathological loops if
    predecessor data is unexpectedly malformed.
    """
    path_preview: list[Node] = []
    if current is not None:
        c: Node | None = current
        steps = 0
        while c is not None and steps < max_path_preview:
            path_preview.append(c)
            c = came_from.get(c)  # type: ignore[assignment]
            steps += 1
        path_preview.reverse()
    frames.append(
        {
            "visited": set(visited),
            "frontier": set(frontier),
            "closed": set(closed),
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
    t0 = time.perf_counter()
    pq: list[tuple[float, float, int, Node]] = []
    counter = 0
    heapq.heappush(pq, (priority_key(0.0, heuristic(source)), 0.0, counter, source))
    counter += 1

    g_score: dict[Node, float] = {source: 0.0}
    came_from: dict[Node, Node | None] = {source: None}
    closed: set[Node] = set()
    frontier: set[Node] = {source}
    visited: set[Node] = set()
    frames: list[dict[str, Any]] = []
    pq_pops = 0
    expansions = 0

    _append_frame(frames, visited, frontier, closed, None, came_from, source)

    while pq:
        _prio, dist, _, u = heapq.heappop(pq)
        pq_pops += 1
        # Ignore stale queue entries that are worse than the current best g-score.
        if dist > g_score.get(u, math.inf):
            continue
        # Once closed, this node has already been permanently expanded.
        if u in closed:
            continue
        closed.add(u)
        visited.add(u)
        # Frontier tracks nodes discovered but not yet expanded.
        frontier.discard(u)
        expansions += 1

        # Sample frames periodically (and at important milestones) for animation.
        if expansions == 1 or expansions % snapshot_every == 0 or u == target:
            _append_frame(frames, visited, frontier, closed, u, came_from, source)

        if u == target:
            path = _reconstruct_path(came_from, target)
            elapsed = time.perf_counter() - t0
            return SearchResult(
                path=path,
                cost=g_score[target],
                nodes_expanded=expansions,
                pq_pops=pq_pops,
                elapsed_s=elapsed,
                frames=frames,
            )

        for v, w in _neighbors(G, u, weight):
            cand = g_score[u] + w
            # Relaxation step: keep only strictly better routes to each neighbor.
            if cand < g_score.get(v, math.inf):
                g_score[v] = cand
                came_from[v] = u
                heapq.heappush(pq, (priority_key(cand, heuristic(v)), cand, counter, v))
                counter += 1
                frontier.add(v)

    elapsed = time.perf_counter() - t0
    return SearchResult(
        path=[],
        cost=math.inf,
        nodes_expanded=expansions,
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
    """Run Dijkstra by prioritizing nodes with the smallest known path cost ``g``.

    ``xy`` is accepted only to keep the public API shape aligned with ``astar``.
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
    """Run A* with priority ``f = g + h`` for faster goal-directed search.

    Provide either:
    - ``heuristic_fn`` directly, or
    - ``xy`` so this function can build a Euclidean-distance heuristic.

    For optimal paths, edge weights must be nonnegative and the heuristic should
    be admissible (never overestimate remaining cost).
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
    """Extract projected ``(x, y)`` coordinates from an OSMnx-style graph."""
    return {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}
