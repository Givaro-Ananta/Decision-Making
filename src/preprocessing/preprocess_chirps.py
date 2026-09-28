"""
Preprocessing CHIRPS v3.0 harian -> panel grid-harian 0.1°.

Input : data/chirps/chirps-v3.0.rnl.YYYY.MM.DD.tif  (GeoTIFF 0.05°, nodata=-9999)
Output: data/processed/chirps_yearly/chirps_YYYY.parquet
        kolom: grid_id, date, rain_t (mm)

Keputusan desain (audit 2026-09-28: 3.631 file, x 95.025..105.975, dx=0.05):
- Nodata -9999 -> NaN sebelum agregasi.
- Resolusi 0.05° -> 0.1° dengan MEAN dari 2x2 blok (bukan sum; curah hujan
  raster adalah intensitas spasial, rata-rata area lebih tepat untuk resolusi
  grid yang lebih kasar).
- Alokasi grid 0.1°: pixel center jatuh di dalam sel grid (floor).
"""

import glob
import re
import numpy as np
import pandas as pd
import rioxarray as rio

from src.config import CHIRPS_DIR, CHIRPS_YEARLY_DIR, SUMATRA_BBOX, GRID_RES

_DATE_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\.tif$")


def process_one_day(path: str) -> pd.DataFrame:
    m = _DATE_RE.search(path.replace("\\", "/"))
    if not m:
        return pd.DataFrame()
    date = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    try:
        da = rio.open_rasterio(path).squeeze("band", drop=True)
    except Exception:
        return pd.DataFrame()  # file korup -> hari ini jadi NaN di batch builder
    da = da.where(da != -9999.0)  # nodata -> NaN

    # floor ke grid 0.1°: pixel 0.05° tepat 4 pixel per grid 0.1°
    bb = SUMATRA_BBOX
    da = da.assign_coords(
        ix=("x", np.floor((da.x.values - bb["lon_min"]) / GRID_RES).astype(int)),
        iy=("y", np.floor((da.y.values - bb["lat_min"]) / GRID_RES).astype(int)),
    )
    # kelompokkan pixel per grid sel, rata-rata
    df = da.to_dataframe(name="rain_t").reset_index()
    df = df.dropna(subset=["rain_t"])
    df = df[(df["ix"] >= 0) & (df["ix"] < 111) & (df["iy"] >= 0) & (df["iy"] < 121)]
    g = df.groupby(["iy", "ix"], observed=True)["rain_t"].mean().reset_index()
    g["grid_id"] = "g" + g["iy"].astype(str).str.zfill(3) + "_" + g["ix"].astype(str).str.zfill(3)
    g["date"] = date
    return g[["grid_id", "date", "rain_t"]]


def build_chirps_yearly(overwrite: bool = False) -> None:
    files = sorted(glob.glob(str(CHIRPS_DIR / "chirps-v3.0.rnl.*.tif")))
    print(f"== Preprocessing CHIRPS: {len(files)} file harian ==")
    by_year: dict[int, list[pd.DataFrame]] = {}
    for i, f in enumerate(files, 1):
        yr = int(_DATE_RE.search(f.replace("\\", "/")).group(1))
        out = CHIRPS_YEARLY_DIR / f"chirps_{yr}.parquet"
        if out.exists() and not overwrite:
            continue
        by_year.setdefault(yr, []).append(process_one_day(f))
        if i % 500 == 0:
            print(f"  ... {i}/{len(files)} file")
    for yr, frames in by_year.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(CHIRPS_YEARLY_DIR / f"chirps_{yr}.parquet", index=False)
        print(f"  {yr}: {len(df):,} baris grid-hari")
    print(f"  Selesai. Output di {CHIRPS_YEARLY_DIR}")


if __name__ == "__main__":
    build_chirps_yearly()
