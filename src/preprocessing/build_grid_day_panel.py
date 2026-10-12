"""Create the complete same-grid daily panel from preprocessed sources."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    CHIRPS_YEARLY_DIR, END_DATE, FIRMS_DAILY_FILE, GRID_DAY_FILE, GRID_FILE,
    LANDCOVER_FILE, START_DATE, ERA5_MONTHLY_DIR,
)
from src.preprocessing.build_states import assign_hotspot_state

FIRMS_COLUMNS = ["grid_id", "date", "hotspot_count", "frp_sum", "frp_mean", "frp_max"]
ERA5_COLUMNS = ["grid_id", "date", "temperature_t", "precipitation_t", "humidity_t", "wind_speed_t"]
CHIRPS_COLUMNS = ["grid_id", "date", "rain_t"]


def _read_slice(path: Path, date_col: str = "date") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if date_col in frame:
        frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    return frame


def _daily_rows(frame: pd.DataFrame, date: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
    if frame.empty or "date" not in frame:
        return pd.DataFrame(columns=columns)
    day = frame.loc[frame["date"].eq(date)]
    return day.reindex(columns=columns)


def build_complete_panel(overwrite: bool = False) -> dict[str, object]:
    """Build all canonical grid × day rows, preserving covariate missingness.

    Zero-hotspot days use ``hotspot_count=0`` and ``frp_sum=0``. ``frp_mean``
    and ``frp_max`` are null because they are undefined without a detection.
    """
    if GRID_DAY_FILE.exists() and not overwrite:
        raise FileExistsError(f"{GRID_DAY_FILE} exists; pass overwrite=True to replace it")
    for path in (GRID_FILE, LANDCOVER_FILE, FIRMS_DAILY_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required preprocessing output is missing: {path}")

    grid = pd.read_parquet(GRID_FILE)
    if grid["grid_id"].duplicated().any():
        raise AssertionError("Canonical grid contains duplicate grid IDs")
    base = grid[["grid_id", "centroid_lat", "centroid_lon", "row_index", "col_index"]].copy()
    valid_ids = set(base["grid_id"])
    landcover = pd.read_parquet(LANDCOVER_FILE)[["grid_id", "landcover", "landcover_code"]]
    if landcover["grid_id"].duplicated().any():
        raise AssertionError("WorldCover has duplicate grid IDs")
    base = base.merge(landcover, on="grid_id", how="left", validate="one_to_one")

    firms = _read_slice(FIRMS_DAILY_FILE)
    firms = firms.reindex(columns=FIRMS_COLUMNS)
    firms = firms.loc[firms["grid_id"].isin(valid_ids)]
    if firms[["grid_id", "date"]].duplicated().any():
        raise AssertionError("FIRMS active grid-day table contains duplicate keys")

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    expected_rows = len(base) * len(dates)
    GRID_DAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="sumatra_grid_day_", suffix=".parquet", dir=GRID_DAY_FILE.parent)
    os.close(fd)
    writer = None
    actual_rows = 0
    try:
        current_month = None
        era5_cache = pd.DataFrame(columns=ERA5_COLUMNS)
        current_year = None
        chirps_cache = pd.DataFrame(columns=CHIRPS_COLUMNS)
        for date in dates:
            month_key = date.strftime("%Y_%m")
            if month_key != current_month:
                era5_cache = _read_slice(ERA5_MONTHLY_DIR / f"era5_{month_key}.parquet")
                if era5_cache.empty:
                    era5_cache = pd.DataFrame(columns=ERA5_COLUMNS)
                else:
                    era5_cache = era5_cache.reindex(columns=ERA5_COLUMNS)
                current_month = month_key
            year_key = date.year
            if year_key != current_year:
                chirps_cache = _read_slice(CHIRPS_YEARLY_DIR / f"chirps_{year_key}.parquet")
                if chirps_cache.empty:
                    chirps_cache = pd.DataFrame(columns=CHIRPS_COLUMNS)
                else:
                    chirps_cache = chirps_cache.reindex(columns=CHIRPS_COLUMNS)
                current_year = year_key

            day = base.copy()
            day["date"] = date
            day = day.merge(_daily_rows(firms, date, FIRMS_COLUMNS), on=["grid_id", "date"], how="left", validate="one_to_one")
            day["hotspot_count"] = day["hotspot_count"].fillna(0).astype("int32")
            day["frp_sum"] = day["frp_sum"].fillna(0.0).astype("float64")
            day = day.merge(_daily_rows(era5_cache, date, ERA5_COLUMNS), on=["grid_id", "date"], how="left", validate="one_to_one")
            day = day.merge(_daily_rows(chirps_cache, date, CHIRPS_COLUMNS), on=["grid_id", "date"], how="left", validate="one_to_one")
            state = assign_hotspot_state(day["hotspot_count"].to_numpy(), day["frp_sum"].to_numpy())
            day["state"] = pd.array(state, dtype="Int8")
            if len(day) != len(base) or day[["grid_id", "date"]].duplicated().any():
                raise AssertionError(f"Panel key/row invariant failed on {date.date()}")
            actual_rows += len(day)
            table = pa.Table.from_pandas(day, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_name, table.schema, compression="zstd")
            writer.write_table(table)
        if writer is None:
            raise ValueError("No dates were generated for the panel")
        writer.close()
        writer = None
        if actual_rows != expected_rows:
            raise AssertionError(f"Expected {expected_rows:,} rows, wrote {actual_rows:,}")
        os.replace(tmp_name, GRID_DAY_FILE)
    except Exception:
        if writer is not None:
            writer.close()
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return {
        "grid_count": len(base), "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(), "date_count": len(dates),
        "expected_rows": expected_rows, "actual_rows": actual_rows,
        "missing_grid_day_rows": expected_rows - actual_rows,
        "panel_path": str(GRID_DAY_FILE),
    }
