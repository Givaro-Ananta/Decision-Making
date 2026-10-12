"""Aggregate FIRMS VIIRS detections onto the canonical Sumatra grid.

This module writes active grid-days only. ``build_grid_day_panel`` fills
zero-hotspot dates after all covariates have been prepared.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import (
    FIRMS_DIR, FIRMS_DAILY_FILE, START_DATE, END_DATE,
    FIRMS_MIN_CONFIDENCE, FIRMS_DAY_ONLY,
)
from src.preprocessing.canonical_grid import (
    filter_coordinates_to_geometry,
    grid_ids_from_indices,
    grid_indices,
)

_CONF_ORDER = {"l": 0, "n": 1, "h": 2}
FIRMS_FILES = ("fire_archive_SV-C2_809701.csv", "fire_nrt_SV-C2_809701.csv")


def load_firms_raw() -> pd.DataFrame:
    frames = []
    for name in FIRMS_FILES:
        path = FIRMS_DIR / name
        if path.exists():
            frames.append(pd.read_csv(
                path,
                usecols=["latitude", "longitude", "acq_date", "confidence", "frp", "daynight"],
                dtype={"confidence": "string", "daynight": "string"},
            ))
    if not frames:
        raise FileNotFoundError(f"No FIRMS input files found under {FIRMS_DIR}")
    return pd.concat(frames, ignore_index=True)


def filter_sumatra(df: pd.DataFrame, sumatra_geometry) -> pd.DataFrame:
    """Apply a coarse extent filter followed by the exact Sumatra polygon."""
    if sumatra_geometry is None:
        raise ValueError("A Sumatra polygon is required; a bounding box is not a region mask")
    min_lon, min_lat, max_lon, max_lat = sumatra_geometry.bounds
    coarse = df.loc[
        df["longitude"].between(min_lon, max_lon)
        & df["latitude"].between(min_lat, max_lat)
    ].copy()
    inside = filter_coordinates_to_geometry(
        coarse["longitude"].to_numpy(), coarse["latitude"].to_numpy(), sumatra_geometry
    )
    out = coarse.loc[inside].copy()
    if FIRMS_MIN_CONFIDENCE in _CONF_ORDER:
        threshold = _CONF_ORDER[FIRMS_MIN_CONFIDENCE]
        out = out.loc[out["confidence"].map(_CONF_ORDER).fillna(-1) >= threshold]
    if FIRMS_DAY_ONLY:
        out = out.loc[out["daynight"] == "D"]
    out = out.loc[out["acq_date"].between(START_DATE, END_DATE)].copy()
    return out


def to_grid_daily(df: pd.DataFrame, valid_grid_ids: set[str] | None = None) -> pd.DataFrame:
    """Assign detections to canonical centers and aggregate by grid and day."""
    out = df.copy()
    row, col = grid_indices(out["longitude"].to_numpy(), out["latitude"].to_numpy())
    out["grid_id"] = grid_ids_from_indices(row, col)
    out = out.loc[out["grid_id"].notna()].copy()
    if valid_grid_ids is not None:
        out = out.loc[out["grid_id"].isin(valid_grid_ids)]
    out["date"] = pd.to_datetime(out["acq_date"], errors="raise").dt.normalize()
    active = out.groupby(["grid_id", "date"], observed=True).agg(
        hotspot_count=("frp", "size"),
        frp_sum=("frp", "sum"),
        frp_mean=("frp", "mean"),
        frp_max=("frp", "max"),
    ).reset_index()
    if active[["grid_id", "date"]].duplicated().any():
        raise AssertionError("FIRMS aggregation produced duplicate grid-days")
    return active


def build_firms_daily(sumatra_geometry, valid_grid_ids: set[str], verbose: bool = True) -> pd.DataFrame:
    raw = load_firms_raw()
    retained = filter_sumatra(raw, sumatra_geometry)
    daily = to_grid_daily(retained, valid_grid_ids)
    if verbose:
        print(f"FIRMS observations retained in Sumatra: {len(retained):,}")
        print(f"FIRMS active grid-days: {len(daily):,}; grids: {daily['grid_id'].nunique():,}")
        print(f"FIRMS date coverage: {daily['date'].min().date()} to {daily['date'].max().date()}")
    daily.to_parquet(FIRMS_DAILY_FILE, index=False)
    return daily
