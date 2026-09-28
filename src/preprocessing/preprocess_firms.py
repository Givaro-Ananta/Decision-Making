"""
Preprocessing NASA FIRMS VIIRS -> panel grid-harian.

Input : data/DL_FIRE_SV-C2_809701/fire_archive_*.csv + fire_nrt_*.csv
        (cakupan seluruh Indonesia; WAJIB difilter ke bbox Sumatra)
Output: data/processed/firms_daily_grid.parquet
        kolom: grid_id, date, hotspot_count, FRP_sum, FRP_mean, FRP_max,
               confidence_high_pct, state

Keputusan desain (dari audit data 2026-09-28):
- Archive berhenti 2026-06-30, NRT mulai 2026-07-01 -> digabung, dedup
  tidak dilakukan karena rentang tidak tumpang tindih (terverifikasi).
- Grid 0.1° sejajar ERA5: lon floor ke 95.0 + k*0.1, lat floor ke -6.0 + k*0.1.
- state = 1 jika grid-hari punya >= 1 hotspot lolos filter confidence.
- -9999 / NaN dibersihkan; baris di luar bbox DIDROP (bukan error).
"""

import pandas as pd
import numpy as np

from src.config import (
    FIRMS_DIR, FIRMS_DAILY_FILE, SUMATRA_BBOX, GRID_RES,
    START_DATE, END_DATE, FIRMS_MIN_CONFIDENCE, FIRMS_DAY_ONLY,
)

# Confidence VIIRS SNPP di CSV: 'l' (low), 'n' (nominal), 'h' (high).
_CONF_ORDER = {"l": 0, "n": 1, "h": 2}


def load_firms_raw() -> pd.DataFrame:
    """Gabung archive + NRT, sekali dibaca, return DataFrame mentah."""
    frames = []
    for name in ("fire_archive_SV-C2_809701.csv", "fire_nrt_SV-C2_809701.csv"):
        path = FIRMS_DIR / name
        if path.exists():
            df = pd.read_csv(
                path,
                usecols=[
                    "latitude", "longitude", "acq_date", "confidence",
                    "frp", "daynight", "satellite",
                ],
                dtype={"confidence": "string", "daynight": "string", "satellite": "string"},
            )
            frames.append(df)
        else:
            print(f"  [WARN] {name} tidak ditemukan, dilewati")
    out = pd.concat(frames, ignore_index=True)
    print(f"  FIRMS mentah gabungan: {len(out):,} titik")
    return out


def filter_sumatra(df: pd.DataFrame) -> pd.DataFrame:
    """Filter ke bbox Sumatra + filter confidence + daynight."""
    bb = SUMATRA_BBOX
    n0 = len(df)
    df = df[
        (df["latitude"] >= bb["lat_min"]) & (df["latitude"] <= bb["lat_max"]) &
        (df["longitude"] >= bb["lon_min"]) & (df["longitude"] <= bb["lon_max"])
    ]
    print(f"  Filter bbox Sumatra: {n0:,} -> {len(df):,} titik ({len(df)/n0:.1%})")

    if FIRMS_MIN_CONFIDENCE in _CONF_ORDER:
        thr = _CONF_ORDER[FIRMS_MIN_CONFIDENCE]
        n1 = len(df)
        df = df[df["confidence"].map(_CONF_ORDER).fillna(-1) >= thr]
        print(f"  Filter confidence >= '{FIRMS_MIN_CONFIDENCE}': {n1:,} -> {len(df):,}")

    if FIRMS_DAY_ONLY:
        df = df[df["daynight"] == "D"]

    # periode analisis (dengan margin rolling di handled oleh builder, bukan di sini)
    df = df[(df["acq_date"] >= START_DATE) & (df["acq_date"] <= END_DATE)]
    print(f"  Periode {START_DATE}..{END_DATE}: sisa {len(df):,} titik")
    return df


def to_grid_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Assign tiap titik ke grid 0.1°, agregasi jadi grid-hari."""
    bb = SUMATRA_BBOX
    # kolom grid: floor ke tepi bbox, lalu index integer biar hemat
    df = df.copy()
    df["ix"] = np.floor((df["longitude"] - bb["lon_min"]) / GRID_RES).astype(int)
    df["iy"] = np.floor((df["latitude"] - bb["lat_min"]) / GRID_RES).astype(int)
    df["grid_id"] = "g" + df["iy"].astype(str).str.zfill(3) + "_" + df["ix"].astype(str).str.zfill(3)

    g = df.groupby(["grid_id", "acq_date"], observed=True).agg(
        hotspot_count=("frp", "size"),
        FRP_sum=("frp", "sum"),
        FRP_mean=("frp", "mean"),
        FRP_max=("frp", "max"),
    ).reset_index().rename(columns={"acq_date": "date"})

    # persentase deteksi high-confidence (proxy kualitas deteksi per grid-hari)
    hi = df[df["confidence"] == "h"].groupby(["grid_id", "acq_date"], observed=True).size()
    g = g.merge(hi.rename("conf_high_n").reset_index().rename(columns={"acq_date": "date"}),
                on=["grid_id", "date"], how="left")
    g["conf_high_n"] = g["conf_high_n"].fillna(0)
    g["confidence_high_pct"] = (g["conf_high_n"] / g["hotspot_count"]).round(4)
    g = g.drop(columns=["conf_high_n"])

    g["state"] = (g["hotspot_count"] > 0).astype(np.int8)
    g["date"] = pd.to_datetime(g["date"])
    return g


def build_firms_daily(verbose: bool = True) -> pd.DataFrame:
    """Pipeline lengkap FIRMS. Return dataframe grid-hari & simpan parquet."""
    if verbose:
        print("== Preprocessing FIRMS VIIRS ==")
    raw = load_firms_raw()
    flt = filter_sumatra(raw)
    daily = to_grid_daily(flt)
    if verbose:
        print(f"  Hasil: {len(daily):,} grid-hari aktif | "
              f"{daily['hotspot_count'].sum():,} hotspot | grid unik: {daily['grid_id'].nunique():,}")
        print(f"  Periode hasil: {daily['date'].min().date()} s.d. {daily['date'].max().date()}")
        print(f"  Simpan -> {FIRMS_DAILY_FILE}")
    daily.to_parquet(FIRMS_DAILY_FILE, index=False)
    return daily


if __name__ == "__main__":
    build_firms_daily()
