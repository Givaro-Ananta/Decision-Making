"""Build canonical mainland-Sumatra cells and same-grid WorldCover summaries."""

from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from src.config import (
    GRID_FILE, LANDCOVER_FILE, PROJECT_ROOT, WORLDCOVER_DIR,
)
from src.preprocessing.canonical_grid import (
    GRID_EDGE_BOUNDS,
    GRID_COLS,
    GRID_ROWS,
    coordinates_within_geometry,
    grid_ids_from_indices,
    grid_indices,
    load_mainland_sumatra_geometry,
    make_canonical_grid,
)

WC_LABEL = {
    0: "nodata", 10: "tree", 20: "shrub", 30: "grass", 40: "cropland",
    50: "builtup", 60: "bare", 70: "snow", 80: "water", 90: "wetland",
    95: "mangrove", 100: "moss", 200: "ocean",
}


def build_grid(sumatra_geometry=None) -> pd.DataFrame:
    if sumatra_geometry is None:
        sumatra_geometry = load_mainland_sumatra_geometry(PROJECT_ROOT)
    return make_canonical_grid(sumatra_geometry)


def sample_landcover(grid: pd.DataFrame, sumatra_geometry) -> pd.DataFrame:
    """Sample each WorldCover tile at ~100 m and compute a pooled grid mode."""
    valid_mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    valid_mask[grid["row_index"].to_numpy(), grid["col_index"].to_numpy()] = True
    tile_files = sorted(glob.glob(str(WORLDCOVER_DIR / "*.tif")))
    if not tile_files:
        raise FileNotFoundError(f"No WorldCover GeoTIFFs found under {WORLDCOVER_DIR}")

    parts = []
    unreadable_tiles = []
    for path in tile_files:
        with rasterio.open(path) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise ValueError(f"WorldCover tile must be EPSG:4326: {path}")
            window = from_bounds(*GRID_EDGE_BOUNDS, transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            if window.width <= 0 or window.height <= 0:
                continue
            out_h = max(1, math.ceil(window.height / 10))
            out_w = max(1, math.ceil(window.width / 10))
            try:
                values = src.read(
                    1, window=window, out_shape=(out_h, out_w),
                    resampling=Resampling.nearest,
                )
            except Exception as exc:
                unreadable_tiles.append({"file": Path(path).name, "error": str(exc)})
                continue
            transform = src.window_transform(window) * rasterio.Affine.scale(
                window.width / out_w, window.height / out_h
            )
            xs = transform.c + (np.arange(out_w) + 0.5) * transform.a
            ys = transform.f + (np.arange(out_h) + 0.5) * transform.e
            xx, yy = np.meshgrid(xs, ys)
            row, col = grid_indices(xx.ravel(), yy.ravel())
            in_grid = (row >= 0) & (col >= 0)
            keep = np.zeros(row.shape, dtype=bool)
            keep[in_grid] = valid_mask[row[in_grid], col[in_grid]]
            keep &= coordinates_within_geometry(xx.ravel(), yy.ravel(), sumatra_geometry)
            keep &= values.ravel() != 0
            if not keep.any():
                continue
            sample = pd.DataFrame({
                "grid_id": grid_ids_from_indices(row[keep], col[keep]),
                "landcover_code": values.ravel()[keep].astype(np.int16)
            })
            counts = sample.groupby(["grid_id", "landcover_code"], observed=True).size()
            parts.append(counts.rename("pixel_samples").reset_index())

    if not parts:
        result = pd.DataFrame(columns=["grid_id", "landcover", "landcover_code"])
        result.attrs["unreadable_tiles"] = unreadable_tiles
        return result
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby(["grid_id", "landcover_code"], as_index=False)["pixel_samples"].sum()
    # Stable tie break: choose the lower WorldCover code if counts are equal.
    combined = combined.sort_values(
        ["grid_id", "pixel_samples", "landcover_code"], ascending=[True, False, True]
    ).drop_duplicates("grid_id")
    combined["landcover"] = combined["landcover_code"].map(WC_LABEL)
    result = combined[["grid_id", "landcover", "landcover_code"]].reset_index(drop=True)
    result.attrs["unreadable_tiles"] = unreadable_tiles
    return result


def run(overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not overwrite and GRID_FILE.exists() and LANDCOVER_FILE.exists():
        raise FileExistsError("Canonical grid outputs already exist; pass overwrite=True to replace")
    geometry = load_mainland_sumatra_geometry(PROJECT_ROOT)
    grid = build_grid(geometry)
    if grid.empty:
        raise ValueError("No canonical grid centroids fall inside the Sumatra polygon")
    lc = sample_landcover(grid, geometry)
    if lc["grid_id"].duplicated().any():
        raise AssertionError("WorldCover output has duplicate grid IDs")
    missing = len(grid) - grid["grid_id"].isin(lc["grid_id"]).sum()
    grid.to_parquet(GRID_FILE, index=False)
    lc.to_parquet(LANDCOVER_FILE, index=False)
    print(f"Mainland Sumatra grids: {len(grid):,}; missing WorldCover: {missing:,}; unreadable tiles: {len(lc.attrs.get('unreadable_tiles', []))}")
    return grid, lc


if __name__ == "__main__":
    run()
