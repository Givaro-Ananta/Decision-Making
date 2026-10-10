# Run the Sumatra spatial-resolution audit on Kaggle

Keep the reusable panel in its own Kaggle Dataset. Keep the standalone Python script in the notebook (or a separate small code Dataset). Updating the script never requires repackaging or uploading the panel again. Use a CPU session; no GPU is needed. These instructions prepare an audit, not model fitting.

## 1. Package the data once, locally

From the repository root:

```bash
mkdir -p /tmp/sumatra-kaggle-upload
tar -czf /tmp/sumatra-kaggle-upload/sumatra_panel.tar.gz \
  data/processed/sumatra_grid_day.parquet
tar -tzf /tmp/sumatra-kaggle-upload/sumatra_panel.tar.gz
sha256sum /tmp/sumatra-kaggle-upload/sumatra_panel.tar.gz
```

The archive must contain only `data/processed/sumatra_grid_day.parquet`. Parquet is already compressed; gzip may not substantially reduce its size. The command reads the verified panel without modifying it. Upload this archive once through **Create Dataset** in Kaggle, and attach that Dataset to the notebook. Alternatively upload the Parquet directly, avoiding extraction and the extra archive disk space.

## 2. Keep the script separate and install DuckDB

Copy the contents of `scripts/spatial_resolution_audit.py` into a notebook cell starting with:

```python
%%writefile /kaggle/working/spatial_resolution_audit.py
# Paste the complete standalone script here, below the cell magic.
```

For later revisions, replace this cell with the updated script and rerun it. Another option is a separate code Dataset containing just the script; update its version independently of the panel Dataset.

In a Python notebook cell:

```python
%pip install "duckdb>=1.4,<2"
```

Install once per runtime. Internet must be enabled if DuckDB is not already installed; for offline sessions, attach a compatible DuckDB wheel in a separate dependency Dataset and install it with pip. No repository imports, pandas, GPU packages, raw rasters, or boundary files are required by the script.

## 3. Locate and extract the data

Run these commands in a `%%bash` notebook cell. Replace `<dataset-name>` with the attached data Dataset's actual folder name.

```bash
find /kaggle/input -type f \( -name 'sumatra_panel.tar.gz' -o -name 'sumatra_grid_day.parquet' \)
mkdir -p /kaggle/working/data
tar -xzf /kaggle/input/<dataset-name>/sumatra_panel.tar.gz \
  -C /kaggle/working/data
find /kaggle/working/data -name 'sumatra_grid_day.parquet' -type f
ls -lh /kaggle/working/data/data/processed/sumatra_grid_day.parquet
df -h /kaggle/working
```

Extract only once per runtime. If a Parquet was uploaded directly, skip extraction and use the path returned by `find` under `/kaggle/input`. Kaggle's mount layout can vary; use the discovered path rather than guessing it.

## 4. Run the audit

In a `%%bash` cell:

```bash
python /kaggle/working/spatial_resolution_audit.py \
  --input /kaggle/working/data/data/processed/sumatra_grid_day.parquet \
  --output /kaggle/working/results
```

For a directly uploaded Parquet:

```bash
python /kaggle/working/spatial_resolution_audit.py \
  --input /kaggle/input/<dataset-name>/sumatra_grid_day.parquet \
  --output /kaggle/working/results
```

Default resolutions are `0.1 0.2 0.25 0.5`; default DuckDB settings are a `4GB` memory limit and two threads. Temporary disk-backed databases process one resolution and one calendar month at a time, and are removed on successful completion or a Python exception. Ensure adequate working disk space for extraction, one temporary panel/database and disk spill. Actual memory and disk use depend on the runtime and data; DuckDB's memory setting is not a hard cap on the entire Python process ([DuckDB memory guidance](https://duckdb.org/docs/current/guides/performance/oom)).

To rerun after editing only the script:

```bash
python /kaggle/working/spatial_resolution_audit.py \
  --input /kaggle/working/data/data/processed/sumatra_grid_day.parquet \
  --output /kaggle/working/results --overwrite
```

`--overwrite` replaces generated files with matching names; it does not clear unrelated or older resolution outputs. Prefer a new output directory when changing the selected resolutions. To audit a subset, use `--resolutions 0.1 0.25`. Optional `--keep-panels` retains the aggregated Parquet files and requires more disk space. `--memory-limit 2GB --threads 1` can reduce resource use.

The recommendation uses an explicitly provisional screen: median ≥50 active days and ≥50% of units with ≥20 active days. It selects the finest passing resolution, or reports that none passes. Change these diagnostic assumptions using `--min-median-active-days` and `--min-pct-units-ge-20`; neither parameter defines hotspot states or establishes statistical model eligibility.

## 5. Check the generated files

```bash
ls -lh /kaggle/working/results
head -n 5 /kaggle/working/results/spatial_resolution_feasibility.csv
cat /kaggle/working/results/spatial_resolution_feasibility.md
python -m json.tool /kaggle/working/results/spatial_resolution_run_metadata.json
```

Outputs:

- `spatial_resolution_feasibility.csv`: one row per resolution, all requested sparsity metrics and thresholds.
- `spatial_resolution_feasibility.md`: methodology, comparison, intensity quantiles, observed sparsity and provisional recommendation.
- `active_day_intensity.csv`: active-grid-day distributions for count, FRP sum and FRP max.
- `per_unit_sparsity_<resolution>.csv`: each spatial unit's days, activity and hotspot/FRP totals.
- `grid_mapping_<resolution>.csv`: source cells and assigned target IDs, preserving traceability.
- `spatial_resolution_run_metadata.json`: source period, DuckDB version, configuration and covariate missingness.
- `panel_<resolution>.parquet`: only when `--keep-panels` is supplied.

The verified panel provides static WorldCover category codes, not within-cell compositions. The audit averages category indicators over known cells and reports missing-member coverage. The 0.25° audit assigns whole base cells by centroid because 0.25° does not align with the 0.1° lattice; this is documented as an approximate grouping. Coarse support consists only of assigned source cells, never newly filled rectangle areas.

## 6. Download results and copy them back

```bash
tar -czf /kaggle/working/spatial_resolution_results.tar.gz \
  -C /kaggle/working/results .
tar -tzf /kaggle/working/spatial_resolution_results.tar.gz
```

Download the archive from Kaggle's output/file browser. Save or commit the notebook run with outputs if you need them retained after the session. Locally, from the repository root, replacing the download path as needed:

```bash
mkdir -p outputs/reports
tar -xzf ~/Downloads/spatial_resolution_results.tar.gz -C outputs/reports
ls -lh outputs/reports/spatial_resolution_feasibility.*
```

This produces the requested local comparison report paths. Until Kaggle executes the audit and results are downloaded, no comparison results for the full panel have been produced locally.
