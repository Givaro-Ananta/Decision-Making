"""Canonical 0.1-degree grid aligned to the ERA5-Land coordinate centers.

ERA5 coordinates are cell centers at longitude 95..106 and latitude 6..-6.
Each cell is therefore centered on an ERA5 coordinate, with edges half a cell
around it. Rows run north-to-south and columns west-to-east. FIRMS points and
CHIRPS pixel centers use the same half-open cell convention.

The grid uses EPSG:4326 coordinates. ``geometry`` is stored as OGC WKB bytes
when serialized to Parquet so Arrow does not need a GeoPandas extension type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import contains_xy, covers, points, to_wkb
from shapely.geometry import box, shape
from shapely.ops import unary_union

from src.config import GRID_RES, SUMATRA_BBOX

GRID_ROWS = 121
GRID_COLS = 111
GRID_LAT_NORTH = SUMATRA_BBOX["lat_max"]
GRID_LON_WEST = SUMATRA_BBOX["lon_min"]
GRID_EDGE_BOUNDS = (
    GRID_LON_WEST - GRID_RES / 2,
    GRID_LAT_NORTH - (GRID_ROWS - 0.5) * GRID_RES,
    GRID_LON_WEST + (GRID_COLS - 0.5) * GRID_RES,
    GRID_LAT_NORTH + GRID_RES / 2,
)

# Administrative provinces that intersect the main Sumatra landmass. The
# largest component of their union is the main island; offshore islands and
# the separate Bangka Belitung / Kepulauan Riau provinces are excluded.
MAINLAND_SUMATRA_PROVINCES = (
    "Aceh",
    "Sumatera Utara",
    "Sumatera Barat",
    "Riau",
    "Jambi",
    "Bengkulu",
    "Sumatera Selatan",
    "Lampung",
)


def load_mainland_sumatra_geometry(repository_root: Path) -> Any:
    """Dissolve local GADM province GeoJSONs and return Sumatra's main component.

    Raises a descriptive error when the local boundary source is unavailable;
    no bounding box is accepted as a replacement for the polygon.
    """
    root = Path(repository_root) / "indonesia-geojson"
    geometries = []
    for province in MAINLAND_SUMATRA_PROVINCES:
        path = root / province / "Provinsi" / f"{province}.geojson"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Sumatra boundary component: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        features = data.get("features", [])
        if not features:
            raise ValueError(f"No GeoJSON features found in {path}")
        geometries.extend(shape(feature["geometry"]) for feature in features if feature.get("geometry"))
    if not geometries:
        raise ValueError(f"No province geometries found under {root}")
    dissolved = unary_union(geometries)
    components = list(dissolved.geoms) if hasattr(dissolved, "geoms") else [dissolved]
    mainland = max(components, key=lambda geom: geom.area)
    if mainland.is_empty or not mainland.is_valid:
        raise ValueError("Derived main-island Sumatra boundary is empty or invalid")
    return mainland


def make_canonical_grid(sumatra_geometry: Any | None = None) -> pd.DataFrame:
    """Return canonical cells, optionally retaining those intersecting Sumatra.

    Columns are ``grid_id``, ``geometry`` (WKB bytes), ``centroid_lat``,
    ``centroid_lon``, ``row_index``, and ``col_index``. Row zero is the
    northernmost ERA5 row; column zero is the westernmost ERA5 column.
    """
    rows: list[dict[str, Any]] = []
    for row_index in range(GRID_ROWS):
        lat = GRID_LAT_NORTH - row_index * GRID_RES
        for col_index in range(GRID_COLS):
            lon = GRID_LON_WEST + col_index * GRID_RES
            cell_geometry = box(
                lon - GRID_RES / 2, lat - GRID_RES / 2,
                lon + GRID_RES / 2, lat + GRID_RES / 2,
            )
            if sumatra_geometry is not None and not sumatra_geometry.intersects(cell_geometry):
                continue
            rows.append(
                {
                    "grid_id": f"SUM_{row_index * GRID_COLS + col_index + 1:06d}",
                    "geometry": to_wkb(cell_geometry),
                    "centroid_lat": lat,
                    "centroid_lon": lon,
                    "row_index": row_index,
                    "col_index": col_index,
                }
            )
    return pd.DataFrame(rows)


def grid_indices(longitudes: Any, latitudes: Any) -> tuple[np.ndarray, np.ndarray]:
    """Map point coordinates to canonical ``(row_index, col_index)`` arrays.

    Cell edges are half a resolution from ERA5 centers. The west and north
    edges are included; the east and south outer edges are excluded.
    Out-of-domain coordinates receive -1.
    """
    lon, lat = np.broadcast_arrays(
        np.asarray(longitudes, dtype=float), np.asarray(latitudes, dtype=float)
    )
    row = np.full(lon.shape, -1, dtype=np.int32)
    col = np.full(lon.shape, -1, dtype=np.int32)
    finite = np.isfinite(lon) & np.isfinite(lat)
    row[finite] = np.floor(
        ((GRID_LAT_NORTH + GRID_RES / 2) - lat[finite]) / GRID_RES
    ).astype(np.int32)
    col[finite] = np.floor(
        (lon[finite] - (GRID_LON_WEST - GRID_RES / 2)) / GRID_RES
    ).astype(np.int32)
    invalid = (row < 0) | (row >= GRID_ROWS) | (col < 0) | (col >= GRID_COLS)
    row[invalid] = -1
    col[invalid] = -1
    return row, col


def grid_ids_from_indices(row_indices: Any, col_indices: Any) -> np.ndarray:
    row = np.asarray(row_indices, dtype=np.int32)
    col = np.asarray(col_indices, dtype=np.int32)
    valid = (row >= 0) & (row < GRID_ROWS) & (col >= 0) & (col < GRID_COLS)
    ids = np.full(row.shape, None, dtype=object)
    ids[valid] = np.char.add(
        "SUM_",
        np.char.zfill((row[valid] * GRID_COLS + col[valid] + 1).astype(str), 6),
    )
    return ids


def filter_coordinates_to_geometry(longitudes: Any, latitudes: Any, geometry: Any) -> np.ndarray:
    """Vectorized point-in-polygon mask for source observations."""
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat)
    result = np.zeros(lon.shape, dtype=bool)
    if valid.any():
        result[valid] = covers(geometry, points(lon[valid], lat[valid]))
    return result


def coordinates_within_geometry(longitudes: Any, latitudes: Any, geometry: Any) -> np.ndarray:
    """Fast point-in-polygon mask for raster pixel centers (boundary excluded)."""
    lon, lat = np.broadcast_arrays(
        np.asarray(longitudes, dtype=float), np.asarray(latitudes, dtype=float)
    )
    valid = np.isfinite(lon) & np.isfinite(lat)
    result = np.zeros(lon.shape, dtype=bool)
    if valid.any():
        result[valid] = contains_xy(geometry, lon[valid], lat[valid])
    return result
