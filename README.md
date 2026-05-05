# DC street pathfinding (Quarto manuscript)

Quarto manuscript comparing Dijkstra and A* on an OpenStreetMap drive network.

Cost models included:

- `length` (meters)
- `travel_time_s` (speed-based seconds from OSM `maxspeed` + highway defaults)
- `travel_time_grade_s` (uphill-penalized travel time using OSM `incline` where present)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



## Workflow

Heavy work (OSM download, projection, snapping, cost-model derivation, and Dijkstra/A* runs) lives in [`precompute.py`](precompute.py). \

```bash
python precompute.py        # one-time; writes data/routing_results.pkl
quarto render               # builds the manuscript site
```

After `quarto render`, open `docs/index.html`. Use `quarto preview` for a live manuscript preview while editing. Re-run `python precompute.py` whenever cost model parameters or endpoints change. Use `python precompute.py --refresh` to force a fresh OSM download.

The repository keeps **source** and **`data/`** (cached graph and precomputed results) only. Rendered output (`docs/`), Quarto cache (`_freeze/`, `.quarto/`), and Python bytecode are listed in `.gitignore` and are recreated when you render.

## Files

- [`index.qmd`](index.qmd) — manuscript and executable analysis (uses [`references.bib`](references.bib) for citations)
- [`references.bib`](references.bib) — BibTeX bibliography; copied into the built site under `docs/` for readers
- [`algorithms.py`](algorithms.py) — Dijkstra / A* with exploration traces
- [`routing_viz.py`](routing_viz.py) — Plotly map animations
- [`precompute.py`](precompute.py) — OSM download and routing precomputation
- [`data/`](data/) — `routing_results.pkl` and `dc_route_graph.graphml` (regenerate with `precompute.py` if missing)
- [`_quarto.yml`](_quarto.yml) — manuscript project defaults and output formats
