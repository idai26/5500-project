"""Pre-compute all heavy work (OSM, projection, cost models, algorithm runs).

Writes a single pickle that index.qmd loads quickly, so Quarto rendering
does not re-run OSM/network or algorithm work each time.

Usage:
    python precompute.py            # uses cached graphml if present
    python precompute.py --refresh  # forces fresh OSM download
"""

from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from algorithms import astar, dijkstra, nx_xy_from_graph

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_GRAPH = DATA_DIR / "dc_route_graph.graphml"
RESULTS_PKL = DATA_DIR / "routing_results.pkl"

ORIGIN_QUERY = "1315 Clifton St NW, Washington, DC"
DEST_QUERY = "37th St NW & O St NW, Washington, DC"
ORIGIN_LATLON = (38.9229, -77.0288)
DEST_LATLON = (38.9075, -77.0731)
BUFFER = 0.018

GAMMA_GRADE = 0.02
SNAPSHOT_EVERY = 100
MAX_EDGE_PAIRS = 8_000
WEIGHT_LEVELS = 6

DEFAULT_SPEED_KPH = {
    "motorway": 90.0,
    "trunk": 75.0,
    "primary": 55.0,
    "secondary": 45.0,
    "tertiary": 40.0,
    "unclassified": 35.0,
    "residential": 30.0,
    "service": 20.0,
    "living_street": 12.0,
}


@dataclass(frozen=True)
class Experiment:
    name: str
    weight_attr: str
    heuristic: Callable[[Any], float] | None


def _log(msg: str) -> None:
    print(f"[precompute] {msg}")


def _first_tag_value(v: Any) -> Any:
    if isinstance(v, list):
        return v[0] if v else None
    return v


def parse_speed_mps(raw: Any, highway: Any) -> float:
    v = _first_tag_value(raw)
    if v is not None:
        s = str(v).lower().strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            num = float(m.group(0))
            if "mph" in s:
                return num * 0.44704
            return num / 3.6
    h = str(_first_tag_value(highway) or "residential")
    return DEFAULT_SPEED_KPH.get(h, 30.0) / 3.6


def parse_incline_pct(raw: Any) -> float:
    v = _first_tag_value(raw)
    if v is None:
        return 0.0
    s = str(v).lower().strip()
    if s in {"up", "uphill"}:
        return 5.0
    if s in {"down", "downhill"}:
        return -5.0
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if m:
        return float(m.group(0))
    return 0.0


def load_or_download_graph(refresh: bool) -> nx.MultiDiGraph:
    if CACHE_GRAPH.exists() and not refresh:
        return ox.load_graphml(CACHE_GRAPH)
    lat1, lon1 = ORIGIN_LATLON
    lat2, lon2 = DEST_LATLON
    north = max(lat1, lat2) + BUFFER
    south = min(lat1, lat2) - BUFFER
    east = max(lon1, lon2) + BUFFER
    west = min(lon1, lon2) - BUFFER
    bbox_wsen = (west, south, east, north)
    # OSMnx >=2 accepts tuple bbox in (west, south, east, north) order.
    G_ll = ox.graph_from_bbox(bbox_wsen, network_type="drive", simplify=True)
    ox.save_graphml(G_ll, CACHE_GRAPH)
    return G_ll


def annotate_cost_models(G: nx.MultiDiGraph) -> dict[str, int]:
    edge_total = edge_has_maxspeed = edge_has_incline = 0
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        edge_total += 1
        length_m = float(data.get("length", 0.0))
        speed_mps = parse_speed_mps(data.get("maxspeed"), data.get("highway"))
        if data.get("maxspeed") is not None:
            edge_has_maxspeed += 1
        incline_pct = parse_incline_pct(data.get("incline"))
        if data.get("incline") is not None:
            edge_has_incline += 1
        travel_time_s = length_m / max(speed_mps, 0.1)
        abs_incline_pct = abs(incline_pct)
        travel_time_grade_s = travel_time_s * (1.0 + GAMMA_GRADE * abs_incline_pct)
        data["speed_mps"] = speed_mps
        data["incline_pct"] = incline_pct
        data["travel_time_s"] = travel_time_s
        data["travel_time_grade_s"] = travel_time_grade_s
    return {
        "edge_total": edge_total,
        "edge_has_maxspeed": edge_has_maxspeed,
        "edge_has_incline": edge_has_incline,
    }


def _normalize(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return (value - vmin) / (vmax - vmin)


def normalize_edge_metrics(G: nx.MultiDiGraph) -> dict[str, float]:
    length_vals: list[float] = []
    time_vals: list[float] = []
    incline_vals: list[float] = []
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        length_vals.append(float(data.get("length", 0.0)))
        time_vals.append(float(data.get("travel_time_s", 0.0)))
        incline_vals.append(abs(float(data.get("incline_pct", 0.0))))

    length_min, length_max = min(length_vals), max(length_vals)
    time_min, time_max = min(time_vals), max(time_vals)
    incline_min, incline_max = min(incline_vals), max(incline_vals)

    for _u, _v, _k, data in G.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0))
        travel_time_s = float(data.get("travel_time_s", 0.0))
        abs_incline_pct = abs(float(data.get("incline_pct", 0.0)))
        data["length_norm"] = _normalize(length_m, length_min, length_max)
        data["time_norm"] = _normalize(travel_time_s, time_min, time_max)
        data["incline_norm"] = _normalize(abs_incline_pct, incline_min, incline_max)

    return {
        "length_min": length_min,
        "length_max": length_max,
        "time_min": time_min,
        "time_max": time_max,
        "incline_min": incline_min,
        "incline_max": incline_max,
    }


def simplex_weights(levels: int) -> list[tuple[float, float, float]]:
    den = levels - 1
    combos: list[tuple[float, float, float]] = []
    for i in range(levels):
        for j in range(levels - i):
            k = den - i - j
            combos.append((i / den, j / den, k / den))
    return combos


def _set_composite_costs(G: nx.MultiDiGraph, wd: float, wt: float, wi: float) -> None:
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        data["composite_cost"] = (
            wd * float(data.get("length_norm", 0.0))
            + wt * float(data.get("time_norm", 0.0))
            + wi * float(data.get("incline_norm", 0.0))
        )


def _min_edge_data_by_weight(
    G: nx.MultiDiGraph, u: Any, v: Any, weight: str
) -> dict[str, Any] | None:
    if v not in G[u]:
        return None
    best_data: dict[str, Any] | None = None
    best_weight: float | None = None
    for _key, data in G[u][v].items():
        w = data.get(weight)
        if w is None:
            continue
        wf = float(w)
        if best_weight is None or wf < best_weight:
            best_weight = wf
            best_data = data
    return best_data


def _path_metrics_for_weight(
    G: nx.MultiDiGraph, path: list[Any], *, weight_attr: str
) -> dict[str, float]:
    length_m = 0.0
    travel_time_s = 0.0
    incline_extra_s = 0.0
    if len(path) < 2:
        return {
            "length_m": length_m,
            "travel_time_s": travel_time_s,
            "incline_extra_s": incline_extra_s,
        }
    for u, v in zip(path[:-1], path[1:]):
        data = _min_edge_data_by_weight(G, u, v, weight_attr)
        if data is None:
            continue
        length_m += float(data.get("length", 0.0))
        tt = float(data.get("travel_time_s", 0.0))
        tt_grade = float(data.get("travel_time_grade_s", tt))
        travel_time_s += tt
        incline_extra_s += max(tt_grade - tt, 0.0)
    return {
        "length_m": length_m,
        "travel_time_s": travel_time_s,
        "incline_extra_s": incline_extra_s,
    }


def sample_edge_pairs(G: nx.MultiDiGraph) -> list[tuple[Any, Any]]:
    pairs: set[tuple[Any, Any]] = set()
    for u, v in G.edges():
        a, b = (u, v) if u < v else (v, u)
        pairs.add((a, b))
    rng = np.random.default_rng(42)
    all_pairs = list(pairs)
    if len(all_pairs) > MAX_EDGE_PAIRS:
        pick = rng.choice(len(all_pairs), size=MAX_EDGE_PAIRS, replace=False)
        return [all_pairs[int(i)] for i in np.sort(pick)]
    return all_pairs


def _make_time_heuristic(
    xy_proj: dict[Any, tuple[float, float]],
    goal: Any,
    vmax_mps: float,
) -> Callable[[Any], float]:
    gx, gy = xy_proj[goal]
    safe_vmax = max(vmax_mps, 0.1)

    def time_heuristic(node: Any) -> float:
        x, y = xy_proj[node]
        return ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5 / safe_vmax

    return time_heuristic


def precompute(refresh: bool = False) -> Path:
    _log("Loading graph...")
    G_ll = load_or_download_graph(refresh)
    _log(f"Loaded graph with {G_ll.number_of_nodes()} nodes / {G_ll.number_of_edges()} edges.")

    _log("Projecting graph...")
    G = ox.project_graph(G_ll.copy())
    xy_proj = nx_xy_from_graph(G)
    node_xy_ll = {n: (float(G_ll.nodes[n]["x"]), float(G_ll.nodes[n]["y"])) for n in G.nodes}

    _log("Snapping endpoints...")
    crs = G.graph.get("crs", "EPSG:4326")

    def _nearest(graph, lon, lat):
        pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs)
        return ox.distance.nearest_nodes(graph, float(pt.x.iloc[0]), float(pt.y.iloc[0]))

    orig_node = _nearest(G, ORIGIN_LATLON[1], ORIGIN_LATLON[0])
    dest_node = _nearest(G, DEST_LATLON[1], DEST_LATLON[0])

    _log("Annotating cost models...")
    coverage = annotate_cost_models(G)
    norm_ranges = normalize_edge_metrics(G)

    _log("Sampling edge pairs for visualization...")
    edge_pairs = sample_edge_pairs(G)

    vmax_mps = max(float(d.get("speed_mps", 0.0)) for _, _, _, d in G.edges(keys=True, data=True))
    time_heuristic = _make_time_heuristic(xy_proj, dest_node, vmax_mps)

    experiments = [
        Experiment("distance", "length", None),
        Experiment("time", "travel_time_s", time_heuristic),
        Experiment("time+grade", "travel_time_grade_s", time_heuristic),
    ]

    results: dict[str, Any] = {}
    for exp in experiments:
        _log(f"Running Dijkstra/A* for model={exp.name}...")
        d_res = dijkstra(
            G,
            orig_node,
            dest_node,
            weight=exp.weight_attr,
            snapshot_every=SNAPSHOT_EVERY,
        )
        a_res = astar(
            G,
            orig_node,
            dest_node,
            weight=exp.weight_attr,
            heuristic_fn=exp.heuristic,
            xy=xy_proj if exp.heuristic is None else None,
            snapshot_every=SNAPSHOT_EVERY,
        )
        assert abs(d_res.cost - a_res.cost) < 1e-6, f"Costs differ for {exp.name}"
        results[exp.name] = {
            "weight": exp.weight_attr,
            "dijkstra": d_res,
            "astar": a_res,
        }

    _log("Running blended normalized-cost Dijkstra grid...")
    blended_weights = simplex_weights(WEIGHT_LEVELS)
    blended_paths: list[list[Any]] = []
    blended_stats: list[dict[str, Any]] = []
    for wd, wt, wi in blended_weights:
        _set_composite_costs(G, wd, wt, wi)
        blend_res = dijkstra(
            G,
            orig_node,
            dest_node,
            weight="composite_cost",
            snapshot_every=SNAPSHOT_EVERY,
        )
        blend_res.frames = []
        blended_paths.append(blend_res.path)
        path_metrics = _path_metrics_for_weight(G, blend_res.path, weight_attr="composite_cost")
        blended_stats.append(
            {
                "w_distance": round(wd, 3),
                "w_time": round(wt, 3),
                "w_incline": round(wi, 3),
                "composite_cost": float(blend_res.cost),
                "length_m": round(path_metrics["length_m"], 3),
                "travel_time_s": round(path_metrics["travel_time_s"], 3),
                "incline_extra_s": round(path_metrics["incline_extra_s"], 3),
                "path_nodes": len(blend_res.path),
            }
        )

    bundle = {
        "origin_query": ORIGIN_QUERY,
        "dest_query": DEST_QUERY,
        "origin_latlon": ORIGIN_LATLON,
        "dest_latlon": DEST_LATLON,
        "orig_node": orig_node,
        "dest_node": dest_node,
        "node_xy_ll": node_xy_ll,
        "edge_pairs": edge_pairs,
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "coverage": coverage,
        "norm_ranges": norm_ranges,
        "vmax_mps": vmax_mps,
        "results": results,
        "blended": {
            "weights": blended_weights,
            "paths": blended_paths,
            "stats": blended_stats,
        },
    }

    with RESULTS_PKL.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"Wrote {RESULTS_PKL.relative_to(PROJECT_ROOT)}")
    return RESULTS_PKL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download OSM graph")
    args = parser.parse_args()
    precompute(refresh=args.refresh)


if __name__ == "__main__":
    main()
