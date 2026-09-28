"""
Preprocessing grid & WorldCover -> grid daratan Sumatra + tutupan lahan.

Input : data/worldcover/ESA_WorldCover_10m_2021_v200_*.tif (18 tile, 10m)
Output: data/processed/grid_sumatra.parquet    (grid_id, grid_lat, grid_lon)
        data/processed/landcover_grid.parquet  (grid_id, landcover, landcover_code)
        data/processed/neighbors_8.parquet     (grid_id, neighbor grid_ids)

Keputusan desain:
- Grid 0.1° disamakan dengan ERA5 (121 x 111 = 13.431 sel).
- Grid dianggap DARATAN jika mayoritas pixel WorldCover-nya bukan 0 (nodata).
  WorldCover 10m -> ~11,000 pixel per grid 0.1°: sampling 1/100 untuk speed.
- Kelas WorldCover: 10 Tree, 20 Shrub, 30 Grass, 40 Crop, 50 Built, 60 Bare,
  70 Snow, 80 Water, 90 HerbWet, 95 Mangrove -> modus per grid.
- neighbors_8: 8-tetangga Moore untuk fitur spasial (laporan mingguan tetangga).
"""

import glob
import numpy as np
import pandas as pd
import rioxarray as rio
import xarray as xr

from src.config import (
    WORLDCOVER_DIR, PROCESSED_DIR, SUMATRA_BBOX, GRID_RES,
    GRID_FILE, LANDCOVER_FILE, NEIGHBOR_FILE,
)

WC_LABEL = {
    0: "nodata", 10: "tree", 20: "shrub", 30: "grass", 40: "cropland",
    50: "builtup", 60: "bare", 70: "snow", 80: "water", 90: "wetland",
    95: "mangrove", 100: "moss", 200: "ocean",
}

N_LON, N_LAT = 111, 121


def build_grid() -> pd.DataFrame:
    bb = SUMATRA_BBOX
    rows = []
    for iy in range(N_LAT):
        for ix in range(N_LON):
            gid = f"g{iy:03d}_{ix:03d}"
            lat = bb["lat_min"] + iy * GRID_RES + GRID_RES / 2
            lon = bb["lon_min"] + ix * GRID_RES + GRID_RES / 2
            rows.append({"grid_id": gid, "grid_lat": round(lat, 3), "grid_lon": round(lon, 3)})
    return pd.DataFrame(rows)


def sample_landcover() -> pd.DataFrame:
    """Mosaik semua tile, sampling tiap ~1 km (10 pixel WorldCover), modus per grid.
    Memori-aman: agregasi dilakukan per tile, tidak menumpuk 170 juta record."""
    tiles = sorted(glob.glob(str(WORLDCOVER_DIR / "*.tif")))
    print(f"  Membaca {len(tiles)} tile WorldCover (sampling 1/100)...")
    parts = []
    for t in tiles:
        da = rio.open_rasterio(t).squeeze("band", drop=True)
        sub = da.values[::10, ::10]  # sampling 100m
        xs = da.x.values[::10]
        ys = da.y.values[::10]
        bb = SUMATRA_BBOX
        ix = np.floor((xs - bb["lon_min"]) / GRID_RES).astype(int)
        iy = np.floor((ys - bb["lat_min"]) / GRID_RES).astype(int)
        ix2, iy2 = np.meshgrid(ix, iy)
        ok = (ix2 >= 0) & (ix2 < N_LON) & (iy2 >= 0) & (iy2 < N_LAT)
        # hitung modus per grid di tile ini saja (DataFrame kecil)
        df = pd.DataFrame({"iy": iy2[ok], "ix": ix2[ok], "wc": sub[ok]})
        g = (df.groupby(["iy", "ix", "wc"]).size().reset_index(name="n")
               .sort_values("n", ascending=False)
               .drop_duplicates(["iy", "ix"], keep="first"))
        parts.append(g)
        del da, sub, df
    # gabung antar tile: jumlahkan count kelas yang sama, lalu modus final
    allw = pd.concat(parts, ignore_index=True)
    g = (allw.groupby(["iy", "ix", "wc"])["n"].sum().reset_index(name="n")
           .sort_values("n", ascending=False)
           .drop_duplicates(["iy", "ix"], keep="first"))
    g["grid_id"] = "g" + g["iy"].astype(str).str.zfill(3) + "_" + g["ix"].astype(str).str.zfill(3)
    g["landcover_code"] = g["wc"]
    g["landcover"] = g["wc"].map(WC_LABEL)
    return g[["grid_id", "landcover", "landcover_code"]]


def build_neighbors(grid: pd.DataFrame) -> pd.DataFrame:
    """8-tetangga Moore per grid (untuk fitur spasial lag)."""
    ids = {g: (int(g[1:4]), int(g[5:8])) for g in grid["grid_id"]}
    out = []
    for gid, (iy, ix) in ids.items():
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                n = f"g{iy+dy:03d}_{ix+dx:03d}"
                if n in ids:
                    nbrs.append(n)
        out.append({"grid_id": gid, "neighbors": "|".join(nbrs)})
    return pd.DataFrame(out)


def run(overwrite: bool = False) -> None:
    print("== Preprocessing Grid + WorldCover ==")
    grid = build_grid()
    print(f"  Grid penuh: {len(grid):,} sel ({N_LAT} x {N_LON})")

    if not LANDCOVER_FILE.exists() or overwrite:
        lc = sample_landcover()
        lc.to_parquet(LANDCOVER_FILE, index=False)
        print(f"  Landcover: {len(lc):,} grid terklasifikasi")
    else:
        lc = pd.read_parquet(LANDCOVER_FILE)
        print(f"  Landcover sudah ada: {len(lc):,} grid")

    # grid daratan = grid yang punya klasifikasi landcover (bukan mayoritas nodata)
    land = lc[lc["landcover"] != "nodata"]
    grid_land = grid.merge(land[["grid_id"]], on="grid_id", how="inner")
    grid_land.to_parquet(GRID_FILE, index=False)
    print(f"  Grid daratan: {len(grid_land):,} sel -> {GRID_FILE.name}")

    if not NEIGHBOR_FILE.exists() or overwrite:
        nb = build_neighbors(grid_land)
        nb.to_parquet(NEIGHBOR_FILE, index=False)
        print(f"  Tetangga 8: {len(nb):,} grid -> {NEIGHBOR_FILE.name}")

    print(f"  Distribusi landcover (top 8):")
    print(land["landcover"].value_counts().head(8).to_string())


if __name__ == "__main__":
    run()
