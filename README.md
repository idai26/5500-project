# DC street pathfinding (Quarto)

Quarto manuscript comparing Dijkstra and A* on an OpenStreetMap drive network between **1315 Clifton St NW** and the **37th St NW & O St NW intersection**.

Cost models included:

- `length` (meters)
- `travel_time_s` (speed-based seconds from OSM `maxspeed` + highway defaults)
- `travel_time_grade_s` (uphill-penalized travel time using OSM `incline` where present)

## Setup

```bash
cd final
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install [Quarto](https://quarto.org/docs/get-started/) if it is not already available (`quarto --version`).

## Workflow

Heavy work (OSM download, projection, snapping, cost-model derivation, and Dijkstra/A* runs) lives in [`precompute.py`](precompute.py). Quarto only loads the resulting pickle and draws figures, so renders are fast and avoid Box.com / Jupyter freezes.

```bash
python precompute.py        # one-time; writes data/routing_results.pkl
quarto render index.qmd     # fast: loads pickle, builds figures
```

Open `_site/index.html`. Re-run `python precompute.py` whenever cost model parameters or endpoints change. Use `python precompute.py --refresh` to force a fresh OSM download.

## Files

- [`index.qmd`](index.qmd) — manuscript and executable analysis
- [`algorithms.py`](algorithms.py) — Dijkstra / A* with exploration traces
- [`routing_viz.py`](routing_viz.py) — Plotly map animations
- [`_quarto.yml`](_quarto.yml) — project defaults
