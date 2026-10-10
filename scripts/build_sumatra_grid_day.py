"""Build the verified Sumatra-only grid-day dataset and its feasibility audit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FIRMS_DAILY_FILE,
)
from src.preprocessing.audit_grid_day import audit_panel  # noqa: E402
from src.preprocessing.build_grid_day_panel import build_complete_panel  # noqa: E402
from src.preprocessing.canonical_grid import (  # noqa: E402
    GRID_EDGE_BOUNDS,
    MAINLAND_SUMATRA_PROVINCES,
    grid_indices,
    load_mainland_sumatra_geometry,
)
from src.preprocessing.preprocess_chirps import build_chirps_yearly  # noqa: E402
from src.preprocessing.preprocess_era5 import build_era5_monthly  # noqa: E402
from src.preprocessing.preprocess_firms import (  # noqa: E402
    filter_sumatra,
    load_firms_raw,
    to_grid_daily,
)
from src.preprocessing.preprocess_grid import run as build_grid_and_landcover  # noqa: E402


def main() -> dict[str, object]:
    geometry = load_mainland_sumatra_geometry(PROJECT_ROOT)
    grid, landcover = build_grid_and_landcover(overwrite=True)
    valid_grid_ids = set(grid["grid_id"])

    firms_raw = load_firms_raw()
    raw_start = pd.to_datetime(firms_raw["acq_date"]).min().date().isoformat()
    raw_end = pd.to_datetime(firms_raw["acq_date"]).max().date().isoformat()
    retained = filter_sumatra(firms_raw, geometry)
    row, col = grid_indices(retained["longitude"].to_numpy(), retained["latitude"].to_numpy())
    outside_lattice = int(((row < 0) | (col < 0)).sum())
    retained_in_lattice = retained.loc[(row >= 0) & (col >= 0)]
    firms_daily = to_grid_daily(retained_in_lattice, valid_grid_ids)
    firms_daily.to_parquet(FIRMS_DAILY_FILE, index=False)
    firms_audit = {
        "raw_observations": len(firms_raw),
        "raw_start": raw_start,
        "raw_end": raw_end,
        "retained_observations": len(retained),
        "observations_in_canonical_lattice": len(retained_in_lattice),
        "outside_canonical_lattice": outside_lattice,
        "active_grids": int(firms_daily["grid_id"].nunique()),
        "study_start": firms_daily["date"].min().date().isoformat() if len(firms_daily) else None,
        "study_end": firms_daily["date"].max().date().isoformat() if len(firms_daily) else None,
    }

    era5_audit = build_era5_monthly(overwrite=True)
    chirps_audit = build_chirps_yearly(overwrite=True)
    panel_audit = build_complete_panel(overwrite=True)

    worldcover_audit = {
        "tiles_found": len(list((PROJECT_ROOT / "data" / "worldcover").glob("*.tif"))),
        "missing_grids": len(grid) - int(landcover["grid_id"].nunique()),
        "grids_with_landcover": int(landcover["grid_id"].nunique()),
        "unreadable_tiles": landcover.attrs.get("unreadable_tiles", []),
    }
    source_audit = {
        "firms": firms_audit,
        "era5": era5_audit,
        "chirps": chirps_audit,
        "worldcover": worldcover_audit,
    }
    boundary_metadata = {
        "boundary_source": "local indonesia-geojson GADM v4.0; eight mainland provinces dissolved, largest component retained",
        "provinces_used": list(MAINLAND_SUMATRA_PROVINCES),
        "boundary_components": 1,
        "geometry_valid": bool(geometry.is_valid),
        "boundary_bounds": [round(float(v), 6) for v in geometry.bounds],
        "resolution_deg": 0.1,
        "grid_edge_bounds": [round(float(v), 6) for v in GRID_EDGE_BOUNDS],
        "uncovered_polygon_area_deg2": float(geometry.difference(box(*GRID_EDGE_BOUNDS)).area),
    }
    report = audit_panel(source_audit, boundary_metadata)
    report["panel_build"] = panel_audit
    print(f"Panel written: {panel_audit['panel_path']}")
    print(f"Audit written: {report['report_path']}")
    return report


if __name__ == "__main__":
    main()
