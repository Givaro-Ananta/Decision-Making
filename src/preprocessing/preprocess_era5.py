"""Convert ERA5-Land NetCDF bands to daily values on the canonical grid."""

from __future__ import annotations

import glob
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.config import (
    END_DATE, ERA5_DIR, ERA5_MONTHLY_DIR, GRID_FILE, START_DATE,
)
from src.preprocessing.canonical_grid import grid_ids_from_indices

_MONTH_RE = re.compile(r"era5_land_(\d{4})_(\d{2})\.nc$")
_VARIABLES = ("t2m", "tp", "d2m", "u10", "v10")


def _time_by_band(src) -> dict[pd.Timestamp, list[int]]:
    result: dict[pd.Timestamp, list[int]] = defaultdict(list)
    for band in range(1, src.count + 1):
        stamp = src.tags(band).get("NETCDF_DIM_valid_time")
        if stamp is None:
            raise ValueError(f"Missing valid_time tag for band {band}")
        day = pd.Timestamp(datetime.fromtimestamp(int(float(stamp)), tz=timezone.utc).date())
        result[day].append(band)
    return result


def _relative_humidity(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    a, b = 17.625, 243.04
    es = 6.1094 * np.exp((a * temp_c) / (b + temp_c))
    e = 6.1094 * np.exp((a * dewpoint_c) / (b + dewpoint_c))
    return np.clip(100.0 * e / es, 1.0, 100.0)


def _daily_reduce(values: np.ndarray, method: str) -> np.ndarray:
    """Reduce a complete set of 6-hourly values without masking missing pixels."""
    values = np.asarray(values, dtype=float)
    complete = np.isfinite(values).all(axis=0)
    if method == "max":
        reduced = np.max(np.where(np.isfinite(values), values, -np.inf), axis=0)
    elif method == "mean":
        reduced = np.mean(np.where(np.isfinite(values), values, 0.0), axis=0)
    elif method == "sum":
        reduced = np.sum(np.where(np.isfinite(values), values, 0.0), axis=0)
    else:
        raise ValueError(f"Unknown daily reduction: {method}")
    return np.where(complete, reduced, np.nan)


def process_one_month(path: str, valid_grid: pd.DataFrame) -> pd.DataFrame:
    match = _MONTH_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Unexpected ERA5 filename: {path}")
    datasets = {}
    try:
        for var in _VARIABLES:
            datasets[var] = rasterio.open(f"NETCDF:{path}:{var}")
        reference = datasets["t2m"]
        for var, src in datasets.items():
            if (src.width, src.height, src.transform) != (
                reference.width, reference.height, reference.transform
            ):
                raise ValueError(f"ERA5 variable {var} is not aligned in {path}")
        times = _time_by_band(reference)
        for var in _VARIABLES[1:]:
            if _time_by_band(datasets[var]) != times:
                raise ValueError(f"ERA5 variable {var} has a different time axis in {path}")

        row, col = np.indices((reference.height, reference.width))
        grid_ids = grid_ids_from_indices(row.ravel(), col.ravel())
        keep = np.isin(grid_ids, valid_grid["grid_id"].to_numpy())
        if not keep.any():
            raise ValueError("No canonical Sumatra grid IDs align with ERA5 raster cells")
        ids = grid_ids[keep]
        output = []
        for day, bands in sorted(times.items()):
            if day < pd.Timestamp(START_DATE) or day > pd.Timestamp(END_DATE):
                continue
            if len(bands) != 4:
                daily = {
                    name: np.full(reference.width * reference.height, np.nan)
                    for name in ("t2m", "tp", "wind", "humidity")
                }
            else:
                arrays = {
                    var: np.stack([datasets[var].read(b, masked=True).filled(np.nan) for b in bands])
                    for var in _VARIABLES
                }
                daily = {
                    "t2m": _daily_reduce(arrays["t2m"], "max").ravel(),
                    "tp": _daily_reduce(arrays["tp"], "sum").ravel(),
                    "d2m": _daily_reduce(arrays["d2m"], "mean").ravel(),
                    "wind": _daily_reduce(np.hypot(arrays["u10"], arrays["v10"]), "mean").ravel(),
                }
                daily["humidity"] = _relative_humidity(
                    daily["t2m"] - 273.15, daily["d2m"] - 273.15
                )
            daily = {k: v[keep] for k, v in daily.items()}
            frame = pd.DataFrame({
                "grid_id": ids,
                "date": day,
                "temperature_t": daily["t2m"] - 273.15,
                "precipitation_t": daily["tp"] * 1000.0,
                "humidity_t": daily["humidity"],
                "wind_speed_t": daily["wind"],
            })
            output.append(frame)
        result = pd.concat(output, ignore_index=True) if output else pd.DataFrame()
        if not result.empty and result[["grid_id", "date"]].duplicated().any():
            raise AssertionError("ERA5 output has duplicate grid-days")
        return result
    finally:
        for src in datasets.values():
            src.close()


def build_era5_monthly(overwrite: bool = False) -> dict[str, object]:
    grid = pd.read_parquet(GRID_FILE)
    files = sorted(glob.glob(str(ERA5_DIR / "era5_land_*.nc")))
    if not files:
        raise FileNotFoundError(f"No ERA5 NetCDF files found under {ERA5_DIR}")
    missing_days: list[str] = []
    processed_days: set[str] = set()
    for path in files:
        match = _MONTH_RE.search(Path(path).name)
        if not match:
            continue
        month = pd.Period(f"{match.group(1)}-{match.group(2)}", freq="M")
        if month.end_time.normalize() < pd.Timestamp(START_DATE) or month.start_time.normalize() > pd.Timestamp(END_DATE):
            continue
        stem = f"era5_{match.group(1)}_{match.group(2)}"
        out = ERA5_MONTHLY_DIR / f"{stem}.parquet"
        if out.exists() and not overwrite:
            existing = pd.read_parquet(out, columns=["date"])
            processed_days.update(pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d").unique())
            continue
        daily = process_one_month(path, grid)
        daily.to_parquet(out, index=False)
        processed_days.update(pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d").unique())
    expected = pd.date_range(START_DATE, END_DATE, freq="D").strftime("%Y-%m-%d")
    missing_days = sorted(set(expected) - processed_days)
    return {
        "files_found": len(files), "processed_days": len(processed_days),
        "missing_days": missing_days,
        "grid_shape": [121, 111],
    }


if __name__ == "__main__":
    print(build_era5_monthly())
