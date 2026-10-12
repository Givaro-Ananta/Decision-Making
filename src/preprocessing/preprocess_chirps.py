"""Aggregate CHIRPS rainfall pixels to daily canonical Sumatra grid cells."""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

from src.config import CHIRPS_DIR, CHIRPS_YEARLY_DIR, GRID_FILE, START_DATE, END_DATE
from src.preprocessing.canonical_grid import (
    GRID_EDGE_BOUNDS, coordinates_within_geometry, grid_ids_from_indices,
    grid_indices, load_mainland_sumatra_geometry,
)

_DATE_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\.tif$")


class UnreadableRasterError(RuntimeError):
    pass


_PIXEL_MAP_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _pixel_map(src, valid_grid: pd.DataFrame, sumatra_geometry) -> tuple[np.ndarray, np.ndarray]:
    """Map the raster window to canonical grid IDs; polygon-test once per raster layout."""
    window = from_bounds(*GRID_EDGE_BOUNDS, transform=src.transform).round_offsets().round_lengths()
    window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
    if window.width <= 0 or window.height <= 0:
        return np.empty(0, dtype=object), np.empty(0, dtype=bool)
    transform = src.window_transform(window)
    key = (
        src.width, src.height, tuple(src.transform),
        int(window.row_off), int(window.col_off), int(window.height), int(window.width),
        tuple(transform),
    )
    cached = _PIXEL_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    rows, cols = np.indices((int(window.height), int(window.width)))
    xs = transform.c + (cols.ravel() + 0.5) * transform.a
    ys = transform.f + (rows.ravel() + 0.5) * transform.e
    row_index, col_index = grid_indices(xs, ys)
    ids = grid_ids_from_indices(row_index, col_index)
    valid = np.isin(ids, valid_grid["grid_id"].to_numpy())
    valid &= coordinates_within_geometry(xs, ys, sumatra_geometry)
    _PIXEL_MAP_CACHE[key] = (ids, valid)
    return ids, valid


def process_one_day(path: str, valid_grid: pd.DataFrame, sumatra_geometry) -> pd.DataFrame:
    match = _DATE_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Unexpected CHIRPS filename: {path}")
    date = pd.Timestamp(*map(int, match.groups()))
    if date < pd.Timestamp(START_DATE) or date > pd.Timestamp(END_DATE):
        return pd.DataFrame(columns=["grid_id", "date", "rain_t"])
    try:
        with rasterio.open(path) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise ValueError(f"CHIRPS raster must be EPSG:4326: {path}")
            window = from_bounds(*GRID_EDGE_BOUNDS, transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            if window.width <= 0 or window.height <= 0:
                return pd.DataFrame(columns=["grid_id", "date", "rain_t"])
            values = src.read(1, window=window, masked=True).astype("float64").filled(np.nan)
            ids, spatial_keep = _pixel_map(src, valid_grid, sumatra_geometry)
            rain = values.ravel()
            keep = spatial_keep & pd.notna(rain) & np.isfinite(rain) & (rain != -9999.0)
            if not keep.any():
                return pd.DataFrame(columns=["grid_id", "date", "rain_t"])
            frame = pd.DataFrame({"grid_id": ids[keep], "rain_t": rain[keep]})
            result = frame.groupby("grid_id", as_index=False, observed=True)["rain_t"].mean()
            result["date"] = date
            return result[["grid_id", "date", "rain_t"]]
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise UnreadableRasterError(f"Could not read CHIRPS raster {path}: {exc}") from exc


def build_chirps_yearly(overwrite: bool = False) -> dict[str, object]:
    grid = pd.read_parquet(GRID_FILE)
    geometry = load_mainland_sumatra_geometry(Path(__file__).resolve().parents[2])
    files = sorted(glob.glob(str(CHIRPS_DIR / "chirps-v3.0.rnl.*.tif")))
    if not files:
        raise FileNotFoundError(f"No CHIRPS GeoTIFFs found under {CHIRPS_DIR}")
    unreadable: list[dict[str, str]] = []
    skipped_existing: list[int] = []
    paths_by_year: dict[int, list[str]] = {}
    for path in files:
        match = _DATE_RE.search(Path(path).name)
        if not match:
            raise ValueError(f"Unexpected CHIRPS filename: {path}")
        year = int(match.group(1))
        out = CHIRPS_YEARLY_DIR / f"chirps_{year}.parquet"
        if out.exists() and not overwrite:
            continue
        paths_by_year.setdefault(year, []).append(path)
    for year, year_paths in sorted(paths_by_year.items()):
        frames = []
        for path in year_paths:
            try:
                frames.append(process_one_day(path, grid, geometry))
            except UnreadableRasterError as exc:
                unreadable.append({"file": Path(path).name, "error": str(exc)})
        nonempty = [frame for frame in frames if not frame.empty]
        if nonempty:
            result = pd.concat(nonempty, ignore_index=True)
        else:
            result = pd.DataFrame(columns=["grid_id", "date", "rain_t"])
        if result[["grid_id", "date"]].duplicated().any():
            raise AssertionError(f"CHIRPS {year} output contains duplicate grid-days")
        result.to_parquet(CHIRPS_YEARLY_DIR / f"chirps_{year}.parquet", index=False)
    if not overwrite:
        present_years = {int(p.stem.split("_")[-1]) for p in CHIRPS_YEARLY_DIR.glob("chirps_*.parquet")}
        skipped_existing = sorted(present_years)
    return {
        "files_found": len(files), "unreadable_files": unreadable,
        "years_skipped_existing": sorted(set(skipped_existing)),
    }


if __name__ == "__main__":
    print(build_chirps_yearly())
