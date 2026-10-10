# Final 0.25° dataset preparation

The frozen design is `configs/modeling_design_025.json`. It records the user-selected 2022-12-31 cutoff, FRP-sum training Q50/Q75 state boundaries, four dynamic predictors and operational eligibility rules. No model fitting occurs in these commands.

From the repository root, with DuckDB, NumPy, pandas, PyArrow, Rasterio, Shapely and SciPy available:

```bash
python scripts/prepare_modeling_panel_025.py
python scripts/finalize_modeling_design_025.py
python -m unittest discover -s tests -v
```

In the current environment DuckDB is installed separately, so prefix the two preparation commands with `PYTHONPATH=/tmp/sumatra-audit-deps`.

Inputs include the unchanged verified 0.1° panel, canonical base grid, local Sumatra province polygons, raw FIRMS CSVs and monthly ERA5 files. Exact 0.25° geometry requires point reassignment because 0.25° is not nested in 0.1°. Only humidity is recomputed from ERA5; other environmental fields reuse existing preprocessing with land-area overlap weights (EPSG:6933). This is a piecewise-constant approximation of base-cell covariates, not recovery of finer environmental information.

Preparation caches FIRMS daily aggregates and corrected humidity monthly files under `data/processed/modeling_025/`. If source files, geography, dates or those preprocessing rules change, remove those derived caches before rerunning; do not remove or modify the verified source panel. The scripts require local raw assets and are not the standalone Kaggle resolution-audit script.

Outputs:

- `data/processed/modeling_025/grid.parquet`: exact cell rectangles plus separate clipped study geometries.
- `data/processed/modeling_025/grid_day.parquet`: complete daily panel, missingness and coverage flags preserved.
- `data/processed/modeling_025/grid_day_states.parquet`: labeled panel.
- `data/processed/modeling_025/transitions.parquet`: all daily transitions, including availability and eligibility flags.
- `data/processed/modeling_025/eligible_transitions.parquet`: only eligible grids with current-day predictors available; may legitimately be empty.
- `outputs/reports/modeling_design_025/`: audit report, per-grid eligibility, and per-grid/current-state sample counts, destination counts, training feature preprocessing and numerical/separation diagnostics.

Eligibility is decided from training transitions only. The 50-observation, 10-destination-observation and 5-observations-per-free-parameter rules are conservative operational screens, not validated scientific sufficiency thresholds. Passing them is necessary under this design but does not demonstrate stable estimation or forecasting performance. Every current state must pass; single-destination groups are explicitly deterministic. No unsupported state fallback or alternate model is introduced. If no grid passes, preserve that finding and revisit the scientific design with the user before fitting.
