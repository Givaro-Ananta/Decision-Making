"""
Preprocessing ERA5-Land -> harian per grid 0.1°.

Input : data/era5/era5_land_YYYY_MM.nc  (NetCDF4, 6-hourly, 0.1°, t2m/tp/d2m/u10/v10)
Output: data/processed/era5_monthly/era5_YYYY_MM.parquet
        kolom: grid_id, date, temperature_t, precipitation_t, humidity_t, wind_speed_t

Keputusan desain (audit 2026-09-28: 124 langkah 6-hourly per bulan):
- t2m  -> agregasi MAX harian (suhu puncak siang paling relevan untuk kebakaran)
- d2m  -> agregasi MEAN harian
- u10,v10 -> kecepatan angin mean + max dipilih max; di sini pakai mean magnitudo
  dari kedua komponen per langkah lalu dirata-rata harian (kecepatan, bukan vektor)
- tp   -> SUM harian (akumulasi curah hujan; ERA5 tp dalam meter -> konversi mm)
- Kelembapan relatif dihitung dari t2m & d2m (rumus Magnus), bukan dipakai mentah.
- Grid ERA5 sudah 0.1° dan sejajar bbox -> langsung mapping grid_id seperti FIRMS.
"""

import glob
import numpy as np
import pandas as pd
import xarray as xr

from src.config import ERA5_DIR, ERA5_MONTHLY_DIR, SUMATRA_BBOX, GRID_RES


def _rh_magnus(t2m_c: np.ndarray, d2m_c: np.ndarray) -> np.ndarray:
    """RH (%) dari suhu dan dew point (Magnus). Clip 1-100."""
    a, b = 17.625, 243.04
    es = 6.1094 * np.exp((a * t2m_c) / (b + t2m_c))
    e = 6.1094 * np.exp((a * d2m_c) / (b + d2m_c))
    return np.clip(100.0 * e / es, 1.0, 100.0)


def process_one_month(path: str) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    time_name = "valid_time" if "valid_time" in ds.dims else "time"

    # komponen angin per langkah -> kecepatan skalar
    ds["wind"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)

    # resample 6-hourly -> harian (agregasi berbeda per variabel)
    daily = xr.Dataset(
        {
            "temperature_t": ds["t2m"].resample({time_name: "1D"}).max(),   # K -> C
            "precipitation_t": ds["tp"].resample({time_name: "1D"}).sum(),  # m -> mm
            "dewpoint_mean": ds["d2m"].resample({time_name: "1D"}).mean(),
            "wind_speed_t": ds["wind"].resample({time_name: "1D"}).mean(),
        }
    )
    daily = daily.compute()

    df = daily.to_dataframe().reset_index()
    df = df.rename(columns={time_name: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # konversi satuan
    df["temperature_t"] = df["temperature_t"] - 273.15          # K -> C
    df["precipitation_t"] = df["precipitation_t"] * 1000.0      # m  -> mm
    df["humidity_t"] = _rh_magnus(df["temperature_t"].values,
                                  (df["dewpoint_mean"] - 273.15).values)
    df = df.drop(columns=["dewpoint_mean"])

    # grid_id konsisten dengan FIRMS (floor ke bbox, zfill)
    bb = SUMATRA_BBOX
    df["ix"] = np.floor((df["longitude"] - bb["lon_min"]) / GRID_RES).astype(int)
    df["iy"] = np.floor((df["latitude"] - bb["lat_min"]) / GRID_RES).astype(int)
    df = df[(df["ix"] >= 0) & (df["ix"] < 111) & (df["iy"] >= 0) & (df["iy"] < 121)]
    df["grid_id"] = "g" + df["iy"].astype(str).str.zfill(3) + "_" + df["ix"].astype(str).str.zfill(3)
    df = df[[
        "grid_id", "date", "temperature_t", "precipitation_t",
        "humidity_t", "wind_speed_t",
    ]]
    return df


def build_era5_monthly(overwrite: bool = False) -> None:
    files = sorted(glob.glob(str(ERA5_DIR / "era5_land_*.nc")))
    print(f"== Preprocessing ERA5-Land: {len(files)} file bulanan ==")
    for f in files:
        stem = f.replace("\\", "/").split("/")[-1].replace(".nc", "")
        out = ERA5_MONTHLY_DIR / f"{stem}.parquet"
        if out.exists() and not overwrite:
            continue
        df = process_one_month(f)
        df.to_parquet(out, index=False)
        print(f"  {stem}: {len(df):,} baris grid-hari -> {out.name}")
    print(f"  Selesai. Output di {ERA5_MONTHLY_DIR}")


if __name__ == "__main__":
    build_era5_monthly()
