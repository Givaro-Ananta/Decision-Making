"""Summarize panel coverage, covariate gaps, and per-grid hotspot sparsity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import FEATURE_COLS, GRID_DAY_FILE, GRID_FILE, REPORTS_DIR

QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "std": None, "min": None, "p25": None,
                "p50": None, "p75": None, "p90": None, "p95": None,
                "p99": None, "max": None}
    quant = np.quantile(values, [0.25, *QUANTILES])
    return {
        "n": int(len(values)), "mean": float(np.mean(values)),
        "std": float(np.std(values)), "min": float(np.min(values)),
        "p25": float(quant[0]), "p50": float(quant[1]), "p75": float(quant[2]),
        "p90": float(quant[3]), "p95": float(quant[4]), "p99": float(quant[5]),
        "max": float(np.max(values)),
    }


def audit_panel(source_audit: dict[str, Any], boundary_metadata: dict[str, Any]) -> dict[str, Any]:
    if not GRID_DAY_FILE.exists():
        raise FileNotFoundError(GRID_DAY_FILE)
    grid = pd.read_parquet(GRID_FILE, columns=["grid_id"])
    grid_ids = grid["grid_id"].astype(str).tolist()
    grid_stats = {
        gid: {"grid_id": gid, "total_days": 0, "active_days": 0,
              "total_hotspots": 0, "total_frp": 0.0,
              "era5_missing_days": 0, "chirps_missing_days": 0}
        for gid in grid_ids
    }
    counts_parts: list[np.ndarray] = []
    frp_sum_parts: list[np.ndarray] = []
    active_count_parts: list[np.ndarray] = []
    active_frp_sum_parts: list[np.ndarray] = []
    active_frp_max_parts: list[np.ndarray] = []
    era5_missing_grid_days = chirps_missing_grid_days = 0
    era5_missing_by_date: dict[str, list[int]] = {}
    chirps_missing_by_date: dict[str, list[int]] = {}
    era5_valid_dates: set[str] = set()
    chirps_valid_dates: set[str] = set()
    total_rows = 0
    first_date = last_date = None
    parquet = pq.ParquetFile(GRID_DAY_FILE)
    columns = [
        "grid_id", "date", "hotspot_count", "frp_sum", "frp_max",
        "temperature_t", "precipitation_t", "humidity_t", "wind_speed_t", "rain_t",
    ]
    for batch in parquet.iter_batches(batch_size=500_000, columns=columns):
        frame = batch.to_pandas()
        total_rows += len(frame)
        min_date, max_date = frame["date"].min(), frame["date"].max()
        first_date = min_date if first_date is None else min(first_date, min_date)
        last_date = max_date if last_date is None else max(last_date, max_date)
        count = frame["hotspot_count"].to_numpy(dtype=np.int64)
        frp_sum = frame["frp_sum"].fillna(0).to_numpy(dtype=float)
        active = count > 0
        counts_parts.append(count)
        frp_sum_parts.append(frp_sum)
        active_count_parts.append(count[active])
        active_frp_sum_parts.append(frp_sum[active])
        active_frp_max_parts.append(frame.loc[active, "frp_max"].dropna().to_numpy(dtype=float))

        era_cols = ["temperature_t", "precipitation_t", "humidity_t", "wind_speed_t"]
        era_missing = frame[era_cols].isna().any(axis=1).to_numpy()
        rain_missing = frame["rain_t"].isna().to_numpy()
        era5_missing_grid_days += int(era_missing.sum())
        chirps_missing_grid_days += int(rain_missing.sum())
        dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        era5_valid_dates.update(dates.loc[~frame[era_cols].isna().all(axis=1)].unique())
        chirps_valid_dates.update(dates.loc[~rain_missing].unique())
        for key, indices in pd.Series(np.arange(len(frame))).groupby(dates, sort=False).groups.items():
            name = str(key)
            state = era5_missing_by_date.setdefault(name, [0, 0])
            state[0] += len(indices)
            state[1] += int(era_missing[np.asarray(list(indices), dtype=int)].sum())
            state = chirps_missing_by_date.setdefault(name, [0, 0])
            state[0] += len(indices)
            state[1] += int(rain_missing[np.asarray(list(indices), dtype=int)].sum())

        stats = frame.assign(_era_missing=era_missing, _rain_missing=rain_missing)
        for gid, group in stats.groupby("grid_id", sort=False, observed=True):
            entry = grid_stats[str(gid)]
            entry["total_days"] += len(group)
            entry["active_days"] += int((group["hotspot_count"] > 0).sum())
            entry["total_hotspots"] += int(group["hotspot_count"].sum())
            entry["total_frp"] += float(group["frp_sum"].fillna(0).sum())
            entry["era5_missing_days"] += int(group["_era_missing"].sum())
            entry["chirps_missing_days"] += int(group["_rain_missing"].sum())

    days = int((pd.Timestamp(last_date) - pd.Timestamp(first_date)).days + 1) if total_rows else 0
    expected_rows = len(grid_ids) * days
    assert total_rows == expected_rows, f"panel rows {total_rows} != expected {expected_rows}"
    per_grid = pd.DataFrame(grid_stats.values())
    per_grid["inactive_days"] = per_grid["total_days"] - per_grid["active_days"]
    per_grid["active_fraction"] = per_grid["active_days"] / per_grid["total_days"].replace(0, np.nan)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    per_grid_path = REPORTS_DIR / "sumatra_per_grid_sparsity.csv"
    per_grid.to_csv(per_grid_path, index=False)

    counts = np.concatenate(counts_parts) if counts_parts else np.array([], dtype=float)
    frp_sums = np.concatenate(frp_sum_parts) if frp_sum_parts else np.array([], dtype=float)
    active_counts = np.concatenate(active_count_parts) if active_count_parts else np.array([], dtype=float)
    active_sums = np.concatenate(active_frp_sum_parts) if active_frp_sum_parts else np.array([], dtype=float)
    active_max = np.concatenate(active_frp_max_parts) if active_frp_max_parts else np.array([], dtype=float)
    report = {
        "canonical_grid": boundary_metadata,
        "panel": {
            "grid_count": len(grid_ids), "start_date": pd.Timestamp(first_date).date().isoformat(),
            "end_date": pd.Timestamp(last_date).date().isoformat(), "date_count": days,
            "expected_grid_day_rows": expected_rows, "actual_grid_day_rows": total_rows,
            "missing_grid_day_rows": expected_rows - total_rows,
            "duplicate_key_check": "passed by panel writer per date; row-count invariant passed",
        },
        "sources": source_audit,
        "covariates": {
            "active_feature_configuration": FEATURE_COLS,
            "ERA5_grid_days_with_any_missing_value": era5_missing_grid_days,
            "ERA5_dates_with_any_valid_covariate": len(era5_valid_dates),
            "ERA5_valid_date_start": min(era5_valid_dates) if era5_valid_dates else None,
            "ERA5_valid_date_end": max(era5_valid_dates) if era5_valid_dates else None,
            "ERA5_grids_with_any_missing_value": int((per_grid["era5_missing_days"] > 0).sum()),
            "ERA5_dates_with_all_covariates_missing_for_all_rows": sorted(
                date for date, (n, missing) in era5_missing_by_date.items() if n == missing
            ),
            "CHIRPS_grid_days_missing_rainfall": chirps_missing_grid_days,
            "CHIRPS_dates_with_any_valid_rainfall": len(chirps_valid_dates),
            "CHIRPS_valid_date_start": min(chirps_valid_dates) if chirps_valid_dates else None,
            "CHIRPS_valid_date_end": max(chirps_valid_dates) if chirps_valid_dates else None,
            "CHIRPS_grids_with_any_missing_rainfall": int((per_grid["chirps_missing_days"] > 0).sum()),
            "CHIRPS_missing_grid_days_on_unreadable_dates": int(
                sum(
                    chirps_missing_by_date.get(date, [0, 0])[0]
                    for date, (n, missing) in chirps_missing_by_date.items()
                    if n == missing
                    and any(date.replace("-", ".") in item.get("file", "") for item in source_audit["chirps"].get("unreadable_files", []))
                )
            ),
            "CHIRPS_missing_grid_days_outside_unreadable_dates": int(
                chirps_missing_grid_days - sum(
                    chirps_missing_by_date.get(date, [0, 0])[0]
                    for date, (n, missing) in chirps_missing_by_date.items()
                    if n == missing
                    and any(date.replace("-", ".") in item.get("file", "") for item in source_audit["chirps"].get("unreadable_files", []))
                )
            ),
            "CHIRPS_dates_with_rainfall_missing_for_all_rows": sorted(
                date for date, (n, missing) in chirps_missing_by_date.items() if n == missing
            ),
            "WorldCover_grids_missing": int(source_audit["worldcover"]["missing_grids"]),
        },
        "hotspot_distributions": {
            "all_grid_days_hotspot_count": _distribution(counts),
            "all_grid_days_frp_sum": _distribution(frp_sums),
            "frp_max_active_grid_days_only": _distribution(active_max),
            "active_grid_days": {
                "hotspot_count": _distribution(active_counts),
                "frp_sum": _distribution(active_sums),
                "frp_max": _distribution(active_max),
            },
        },
        "per_grid_sparsity_distributions": {
            key: _distribution(per_grid[key].to_numpy(dtype=float))
            for key in ("active_days", "active_fraction", "total_hotspots")
        },
        "state_and_transition_audit": {
            "state_0": "No Activity assigned from hotspot_count == 0",
            "states_1_to_3": "not assigned; thresholds pending empirical review",
            "active_grid_days": int(len(active_counts)),
            "active_grid_day_fraction": float(len(active_counts) / total_rows) if total_rows else None,
            "grids_with_no_active_days": int((per_grid["active_days"] == 0).sum()),
            "median_active_days_per_grid": float(per_grid["active_days"].median()),
            "per_grid_current_state_transition_counts": "not computed until state thresholds are approved",
        },
        "per_grid_sparsity_file": str(per_grid_path),
    }
    report_path = REPORTS_DIR / "sumatra_grid_day_audit.md"
    _write_markdown_report(report, report_path)
    report["report_path"] = str(report_path)
    return report


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    panel = report["panel"]
    grid = report["canonical_grid"]
    sources = report["sources"]
    cov = report["covariates"]
    dist = report["hotspot_distributions"]
    lines = [
        "# Sumatra Grid-Day Data Audit", "",
        "This report evaluates preprocessing support for per-grid transition construction. It does not fit a predictive model.", "",
        "## Canonical grid and geographic coverage", "",
        f"- Boundary source: {grid['boundary_source']}.",
        f"- Mainland Sumatra geometry: {grid['boundary_components']} component(s), valid={grid['geometry_valid']}, bounds={grid['boundary_bounds']}.",
        f"- Grid: {grid['resolution_deg']}°; centroids are ERA5 coordinate centers; rows north-to-south and columns west-to-east.",
        f"- Cells retained by polygon intersection: {panel['grid_count']:,}; IDs use `SUM_######`; geometry in EPSG:4326 WKB.",
        f"- Canonical grid edges: {grid['grid_edge_bounds']}; polygon area outside the canonical grid extent is {grid['uncovered_polygon_area_deg2']:.6f} square degrees.",
        "- FIRMS points are filtered against the exact derived main-island polygon; the bounding box is only a coarse filter.",
        "- Province scope: Aceh, Sumatera Utara, Sumatera Barat, Riau, Jambi, Bengkulu, Sumatera Selatan, and Lampung; the largest dissolved component is used to retain the main island and exclude detached islands.",
        "",
        "## Daily panel", "",
        f"- Period: {panel['start_date']} through {panel['end_date']} ({panel['date_count']:,} dates).",
        f"- Expected rows: {panel['expected_grid_day_rows']:,}; written: {panel['actual_grid_day_rows']:,}; missing: {panel['missing_grid_day_rows']:,}.",
        f"- Key/row validation: {panel['duplicate_key_check']}.",
        "- No-detection days use `hotspot_count=0`, `frp_sum=0`; `frp_mean` and `frp_max` remain null because they are undefined without detections.",
        "",
        "## Source integrity", "",
        f"- FIRMS: {sources['firms']['retained_observations']:,} retained in-polygon observations; {sources['firms']['active_grids']:,} grids with detections; source dates {sources['firms']['raw_start']} to {sources['firms']['raw_end']}; study-window detections {sources['firms']['study_start']} to {sources['firms']['study_end']}.",
        f"- FIRMS in-polygon observations outside the canonical lattice: {sources['firms']['outside_canonical_lattice']:,}.",
        f"- ERA5: {sources['era5']['files_found']} monthly files; source grid {sources['era5']['grid_shape']}; missing expected dates {len(sources['era5']['missing_days'])}; missing grid-days with one or more missing variables {cov['ERA5_grid_days_with_any_missing_value']:,}.",
        f"- ERA5 panel covariate dates: {cov['ERA5_valid_date_start']} to {cov['ERA5_valid_date_end']}; {cov['ERA5_grids_with_any_missing_value']:,} grids have at least one day with one or more missing ERA5 variables; {cov['ERA5_grid_days_with_any_missing_value']:,} grid-days ({100 * cov['ERA5_grid_days_with_any_missing_value'] / panel['actual_grid_day_rows']:.2f}%) have at least one missing ERA5 variable.",
        f"- CHIRPS: {sources['chirps']['files_found']:,} files; valid rainfall dates {cov['CHIRPS_valid_date_start']} to {cov['CHIRPS_valid_date_end']} ({cov['CHIRPS_dates_with_any_valid_rainfall']:,} dates); unreadable {len(sources['chirps']['unreadable_files'])}; missing grid-days after merge {cov['CHIRPS_grid_days_missing_rainfall']:,}, including {cov['CHIRPS_missing_grid_days_on_unreadable_dates']:,} rows on unreadable dates and {cov['CHIRPS_missing_grid_days_outside_unreadable_dates']:,} on other dates; {cov['CHIRPS_grids_with_any_missing_rainfall']:,} grids have rainfall gaps; unreadable details: `{sources['chirps']['unreadable_files']}`.",
        f"- WorldCover: {sources['worldcover']['tiles_found']} tiles; unreadable {len(sources['worldcover'].get('unreadable_tiles', []))}; grids missing land cover {sources['worldcover']['missing_grids']:,}; unreadable details: `{sources['worldcover'].get('unreadable_tiles', [])}`.",
        "",
        "## Hotspot distributions", "",
        f"- All grid-days: hotspot_count {_fmt_dist(dist['all_grid_days_hotspot_count'])}; frp_sum {_fmt_dist(dist['all_grid_days_frp_sum'])}.",
        f"- Active grid-days only: hotspot_count {_fmt_dist(dist['active_grid_days']['hotspot_count'])}; frp_sum {_fmt_dist(dist['active_grid_days']['frp_sum'])}; frp_max {_fmt_dist(dist['active_grid_days']['frp_max'])}.",
        f"- Across-grid sparsity: active_days {_fmt_dist(report['per_grid_sparsity_distributions']['active_days'])}; active_fraction {_fmt_dist(report['per_grid_sparsity_distributions']['active_fraction'])}; total_hotspots {_fmt_dist(report['per_grid_sparsity_distributions']['total_hotspots'])}.",
        f"- Per-grid audit table: `{report['per_grid_sparsity_file']}`.",
        "- FRP max is null on inactive days; its all-grid-day distribution is therefore reported for active days only.",
        "",
        "## Covariates and feasibility", "",
        f"- Active features are same-grid only: {', '.join(cov['active_feature_configuration'])}.",
        "- No neighbor features, `is_peat`, or `elev_mean` are used. Peat/elevation sources were not found in the local data inventory.",
        f"- Activity is sparse: {report['state_and_transition_audit']['active_grid_days']:,} of {panel['actual_grid_day_rows']:,} grid-days are active ({100 * report['state_and_transition_audit']['active_grid_day_fraction']:.3f}%); {report['state_and_transition_audit']['grids_with_no_active_days']:,} grids have no active days, and the median grid has {report['state_and_transition_audit']['median_active_days_per_grid']:.0f} active days in {panel['date_count']:,} days.",
        "- This sparsity is a substantial warning for separately estimating each grid × current-state multinomial model. Low/Moderate/High state thresholds remain unresolved, so state-specific transition counts and exact per-grid estimation feasibility cannot yet be assessed.",
        "- Transition dataset readiness: **not ready** until active states are defined and the remaining missing covariate grid-days are reviewed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_dist(summary: dict[str, Any]) -> str:
    if summary["n"] == 0:
        return "n=0"
    return (
        f"n={summary['n']:,}, mean={summary['mean']:.4g}, median={summary['p50']:.4g}, "
        f"p75={summary['p75']:.4g}, p90={summary['p90']:.4g}, p95={summary['p95']:.4g}, "
        f"p99={summary['p99']:.4g}, max={summary['max']:.4g}"
    )
