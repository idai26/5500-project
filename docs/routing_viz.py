"""Plotly map animations for pathfinding traces on lat/lon node coordinates."""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np
import plotly.graph_objects as go

Node = Hashable


def _edge_coords(
    node_xy_ll: dict[Node, tuple[float, float]],
    edge_pairs: list[tuple[Node, Node]],
) -> tuple[list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for u, v in edge_pairs:
        if u not in node_xy_ll or v not in node_xy_ll:
            continue
        lon1, lat1 = node_xy_ll[u]
        lon2, lat2 = node_xy_ll[v]
        xs.extend([lon1, lon2, None])
        ys.extend([lat1, lat2, None])
    return xs, ys


def _trace_streets(ex: list[float | None], ey: list[float | None]) -> go.Scattermapbox:
    return go.Scattermapbox(
        lat=ey,
        lon=ex,
        mode="lines",
        line=dict(width=1, color="rgba(100,100,100,0.4)"),
        name="Streets",
        hoverinfo="skip",
        showlegend=False,
    )


def _trace_nodes(
    node_xy_ll: dict[Node, tuple[float, float]],
    nodes: set[Node],
    *,
    color: str,
    size: int,
    name: str,
) -> go.Scattermapbox:
    lats = [node_xy_ll[n][1] for n in nodes if n in node_xy_ll]
    lons = [node_xy_ll[n][0] for n in nodes if n in node_xy_ll]
    return go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=dict(size=size, color=color, opacity=0.55),
        name=name,
    )


def _trace_path(
    node_xy_ll: dict[Node, tuple[float, float]],
    path: list[Node],
    *,
    color: str,
    width: int,
    name: str,
) -> go.Scattermapbox:
    path_nodes = path if len(path) >= 2 else []
    lats = [node_xy_ll[n][1] for n in path_nodes if n in node_xy_ll]
    lons = [node_xy_ll[n][0] for n in path_nodes if n in node_xy_ll]
    return go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=width, color=color),
        name=name,
    )


def _trace_endpoints(
    node_xy_ll: dict[Node, tuple[float, float]],
    start: Node,
    goal: Node,
) -> go.Scattermapbox:
    return go.Scattermapbox(
        lat=[node_xy_ll[start][1], node_xy_ll[goal][1]],
        lon=[node_xy_ll[start][0], node_xy_ll[goal][0]],
        mode="markers",
        marker=dict(size=14, color=["#22c55e", "#ef4444"]),
        name="Start / Goal",
    )


def _trace_current(
    node_xy_ll: dict[Node, tuple[float, float]],
    current: Node | None,
) -> go.Scattermapbox:
    if current is None or current not in node_xy_ll:
        return go.Scattermapbox(
            lat=[],
            lon=[],
            mode="markers",
            marker=dict(size=12, color="#3b82f6"),
            name="Current",
            showlegend=False,
        )
    return go.Scattermapbox(
        lat=[node_xy_ll[current][1]],
        lon=[node_xy_ll[current][0]],
        mode="markers",
        marker=dict(size=12, color="#3b82f6", symbol="circle"),
        name="Current",
    )


def _default_map_layout(
    node_xy_ll: dict[Node, tuple[float, float]],
    start: Node,
    goal: Node,
    map_title: str,
) -> dict[str, Any]:
    return {
        "mapbox": {
            "style": "open-street-map",
            "center": {
                "lat": (node_xy_ll[start][1] + node_xy_ll[goal][1]) / 2,
                "lon": (node_xy_ll[start][0] + node_xy_ll[goal][0]) / 2,
            },
            "zoom": 12,
        },
        "margin": {"l": 0, "r": 0, "t": 48, "b": 0},
        "title": {"text": map_title, "x": 0.5},
        "legend": {"yanchor": "top", "y": 0.98, "xanchor": "left", "x": 0.01},
        "height": 660,
    }


def build_search_animation(
    frames_data: list[dict[str, Any]],
    node_xy_ll: dict[Node, tuple[float, float]],
    edge_pairs: list[tuple[Node, Node]],
    path_final: list[Node],
    start: Node,
    goal: Node,
    *,
    map_title: str = "Search progress",
    max_frames: int = 55,
) -> go.Figure:
    """Six consistent Scattermapbox traces: streets, visited, frontier, path, endpoints, current."""
    if len(frames_data) > max_frames:
        idx = np.linspace(0, len(frames_data) - 1, max_frames, dtype=int)
        frames_data = [frames_data[i] for i in idx]

    ex, ey = _edge_coords(node_xy_ll, edge_pairs)
    streets = _trace_streets(ex, ey)

    def build_traces(fr: dict[str, Any], frame_index: int, last: bool) -> list[Any]:
        visited = fr.get("visited", set())
        frontier = fr.get("frontier", set())
        current = fr.get("current")
        preview = fr.get("path_preview", [])
        path_line = path_final if (last and path_final) else preview
        path_color = "#10b981" if (last and path_final) else "#2563eb"
        path_width = 6 if (last and path_final) else 4
        return [
            streets,
            _trace_nodes(node_xy_ll, visited, color="#64748b", size=5, name="Visited"),
            _trace_nodes(node_xy_ll, frontier, color="#f59e0b", size=7, name="Frontier"),
            _trace_path(
                node_xy_ll,
                path_line,
                color=path_color,
                width=path_width,
                name="Path",
            ),
            _trace_endpoints(node_xy_ll, start, goal),
            _trace_current(node_xy_ll, current),
        ]

    n = len(frames_data)
    initial = build_traces(frames_data[0], 0, n == 1)

    fig = go.Figure(
        data=initial,
        layout=_default_map_layout(node_xy_ll, start, goal, map_title),
    )

    plot_frames: list[go.Frame] = []
    for i, fr in enumerate(frames_data):
        traces = build_traces(fr, i, i == n - 1)
        plot_frames.append(go.Frame(data=traces, name=str(i)))

    fig.frames = tuple(plot_frames)

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.14,
                x=0.08,
                xanchor="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(k)],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True)),
                        ],
                        label=str(k + 1),
                    )
                    for k in range(n)
                ],
                x=0.08,
                len=0.84,
                xanchor="left",
                y=0,
                yanchor="top",
                pad=dict(b=10, t=40),
                currentvalue=dict(prefix="Step: "),
            )
        ],
    )

    return fig


def comparison_table_row(
    label: str,
    cost_m: float,
    nodes_expanded: int,
    pq_pops: int,
    elapsed_ms: float,
) -> dict[str, Any]:
    return {
        "Algorithm": label,
        "Path length (m)": round(cost_m, 2),
        "Nodes expanded": nodes_expanded,
        "PQ pops": pq_pops,
        "Time (ms)": round(elapsed_ms, 3),
    }


def _weight_label(weights: tuple[float, float, float]) -> str:
    wd, wt, wi = weights
    return f"dist {wd:.2f} / time {wt:.2f} / incline {wi:.2f}"


def _blended_annotation(stat: dict[str, Any]) -> str:
    incline_extra = float(
        stat.get("incline_extra_s", stat.get("uphill_extra_s", 0.0))
    )
    return (
        f"length={float(stat.get('length_m', 0.0)):.1f} m<br>"
        f"time={float(stat.get('travel_time_s', 0.0)):.1f} s<br>"
        f"incline_extra={incline_extra:.1f} s<br>"
        f"nodes={int(stat.get('path_nodes', 0))}"
    )


def build_blended_paths_figure(
    node_xy_ll: dict[Node, tuple[float, float]],
    edge_pairs: list[tuple[Node, Node]],
    weights: list[tuple[float, float, float]],
    paths: list[list[Node]],
    stats: list[dict[str, Any]],
    start: Node,
    goal: Node,
    *,
    map_title: str = "Custom cost mix (Dijkstra)",
) -> go.Figure:
    """Render precomputed shortest paths with a dropdown over weight combinations."""
    if not paths:
        raise ValueError("Expected at least one blended path.")
    if not (len(weights) == len(paths) == len(stats)):
        raise ValueError("weights, paths, and stats must have the same length.")

    ex, ey = _edge_coords(node_xy_ll, edge_pairs)
    streets = _trace_streets(ex, ey)
    endpoints = _trace_endpoints(node_xy_ll, start, goal)

    fig = go.Figure(layout=_default_map_layout(node_xy_ll, start, goal, map_title))
    fig.add_trace(streets)
    fig.add_trace(endpoints)

    for i, path in enumerate(paths):
        fig.add_trace(
            _trace_path(
                node_xy_ll,
                path,
                color="#16a34a",
                width=6,
                name="Path",
            )
        )
        fig.data[2 + i].visible = i == 0

    first_label = _weight_label(weights[0])
    fig.update_layout(
        title={"text": f"{map_title} - {first_label}", "x": 0.5},
        annotations=[
            dict(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                align="left",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                text=_blended_annotation(stats[0]),
            )
        ],
    )

    buttons: list[dict[str, Any]] = []
    n = len(paths)
    for i in range(n):
        visible = [True, True] + [k == i for k in range(n)]
        label = _weight_label(weights[i])
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": {"text": f"{map_title} - {label}", "x": 0.5},
                        "annotations": [
                            dict(
                                x=0.01,
                                y=0.99,
                                xref="paper",
                                yref="paper",
                                xanchor="left",
                                yanchor="top",
                                align="left",
                                showarrow=False,
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="rgba(0,0,0,0.2)",
                                borderwidth=1,
                                text=_blended_annotation(stats[i]),
                            )
                        ],
                    },
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                x=0.01,
                y=1.08,
                xanchor="left",
                yanchor="top",
                showactive=True,
                buttons=buttons,
            )
        ]
    )
    return fig
