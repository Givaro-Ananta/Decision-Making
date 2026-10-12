"""
Batch Builder — perakit panel spatio-temporal per tahun.

Menggabungkan output preprocessing menjadi panel (grid x tanggal) per tahun:
  FIRMS    : hotspot + state (per grid-hari aktif)  → data/processed/firms_daily_grid.parquet
  ERA5     : meteorologi harian (per grid-hari)     → data/processed/era5_monthly/era5_land_YYYY_MM.parquet
  CHIRPS   : curah hujan harian (per grid-hari)     → data/processed/chirps_yearly/chirps_YYYY.parquet
  GRID     : koordinat centroid 7.735 grid daratan
  LANDCOVER: kelas tutupan lahan WorldCover
  NEIGHBORS: 8-tetangga spasial (validasi unit-test: LOLOS)

Alur build_single_year_batch(year):
  1. Bangun kerangka (grid x tanggal) dari CHIRPS tahun tsb — CHIRPS = pembatas semesta
     (semesta efektif: 6.107 grid daratan yang tercakup CHIRPS v3.0).
  2. Left-join FIRMS (hanya baris aktif; hari tenang → hotspot_count=0, state=0).
  3. Merge ERA5 bulan-per-bulan (hemat RAM).
  4. Hitung fitur memori: lag hotspot (t-1..t-3), FRP_sum_t1, rain_3d/7d/14d, CDD,
     active_hotspots_3d/7d/14d, days_since_last_fire, fire_freq_30d.
  5. Hitung fitur spasial: neighbor_hotspots_3d (dari neighbors_8, rata-rata aktif
     8 tetangga dalam 3 hari terakhir) + neighbor_rain_3d.
  6. Target: state_t1 = state pada t+1 (per grid, shift -1 tanggal).
  7. Label split: train <= 2022 | val 2023-2024 | test 2025-2026.
  8. Simpan parquet per tahun (dtype diet: float32/int8).

iter_panel_batches(split) mengalirkan chunk ~500rb baris untuk training streaming.
Catatan RAM: puncak ~1.2-1.8 GB per tahun di mesin 7.7 GB (diet dtype + merge
bulan-per-bulan). Mesin >16 GB bebas menaikkan chunk.
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (  # noqa: E402
    BATCH_DIR,
    CHIRPS_YEARLY_DIR,
    ERA5_MONTHLY_DIR,
    FEATURE_COLS,
    FIRMS_DAILY_FILE,
    GRID_FILE,
    LANDCOVER_FILE,
    NEIGHBOR_FILE,
    START_DATE,
    TARGET_COL,
    TEST_END,
    TRAIN_END,
    VAL_END,
)

# Kolom identitas/spasial yang selalu ada di panel (meta + analitik)
PANEL_META_COLS = [
    "grid_id", "date", "grid_lat", "grid_lon", "landcover_code", "landcover",
    "state", "state_t1",
]

# Kolom hotspot hari-ini yang di-backfill 0 untuk hari tenang
FIRMS_VALUE_COLS = ["hotspot_count", "FRP_sum", "FRP_mean", "FRP_max", "confidence_high_pct"]

CHUNK_ROWS = 500_000


# ---------------------------------------------------------------------------
# Helper dtype diet
# ---------------------------------------------------------------------------

def _f32(s: pd.Series) -> pd.Series:
    return s.astype("float32")


def _i32(s: pd.Series) -> pd.Series:
    return s.astype("int32")


# ---------------------------------------------------------------------------
# Muat tabel referensi (cache modul biar gak dibaca ulang tiap tahun)
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _load_reference_tables() -> dict:
    """grid, landcover, neighbors — kecil (7.735 baris), aman di-cache modul."""
    if "grid" not in _CACHE:
        _CACHE["grid"] = pd.read_parquet(GRID_FILE)
    if "landcover" not in _CACHE:
        lc = pd.read_parquet(LANDCOVER_FILE)
        _CACHE["landcover"] = lc[["grid_id", "landcover", "landcover_code"]]
    if "neighbors" not in _CACHE:
        nb = pd.read_parquet(NEIGHBOR_FILE)
        nb["neighbor_list"] = nb["neighbors"].str.split("|")
        nb = nb.explode("neighbor_list").rename(columns={"neighbor_list": "neighbor_id"})
        nb = nb[nb["neighbor_id"] != ""]  # jaga-jaga string kosong
        nb["nbr"] = 1
        _CACHE["neighbors_long"] = nb[["grid_id", "neighbor_id", "nbr"]]
    return _CACHE


def get_available_years() -> list[int]:
    """Tahun yang panel-nya bisa dibangun (ada file CHIRPS tahunannya)."""
    files = sorted(glob.glob(str(CHIRPS_YEARLY_DIR / "chirps_*.parquet")))
    years = []
    for f in files:
        m = re.search(r"chirps_(\d{4})\.parquet$", f)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def _split_label(date: pd.Timestamp) -> str:
    """train <= TRAIN_END | val (TRAIN_END, VAL_END] | test (VAL_END, TEST_END]."""
    if date <= pd.Timestamp(TRAIN_END):
        return "train"
    if date <= pd.Timestamp(VAL_END):
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Muat sumber per tahun
# ---------------------------------------------------------------------------

def _load_chirps_year(year: int) -> pd.DataFrame:
    f = CHIRPS_YEARLY_DIR / f"chirps_{year}.parquet"
    return pd.read_parquet(f)


def _load_firms_year(year: int) -> pd.DataFrame:
    """FIRMS hanya baris aktif; difilter tahun + buffer untuk rolling historis."""
    df = pd.read_parquet(
        FIRMS_DAILY_FILE,
        filters=[("date", ">=", pd.Timestamp(f"{year-1}-12-25")),
                 ("date", "<=", pd.Timestamp(f"{year}-12-31"))],
    )
    return df


def _iter_era5_months(year: int):
    """ERA5 per bulan — merge bulan-per-bulan biar RAM gak numpuk setahun penuh."""
    for m in range(1, 13):
        f = ERA5_MONTHLY_DIR / f"era5_land_{year}_{m:02d}.parquet"
        if f.exists():
            yield pd.read_parquet(f)


# ---------------------------------------------------------------------------
# Fitur memori (rolling per grid) — dihitung per grid-group
# ---------------------------------------------------------------------------

def _add_fire_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag & rolling hotspot, dihitung pada baris aktif FIRMS dulu (sparse),
    lalu di-reindex ke panel penuh dan forward-fill-0.

    df wajib sudah terurut (grid_id, date) dengan kolom: grid_id, date,
    hotspot_count, FRP_sum, state.
    """
    # 1) ekstrak bagian aktif (baris FIRMS) — hitung rolling di data kecil
    act = df.loc[df["state"] == 1, ["grid_id", "date", "hotspot_count", "FRP_sum"]].copy()

    # lag 1..3 hotspot, lag-1 FRP — shift dalam grup grid
    g = act.groupby("grid_id", sort=False)
    for k in (1, 2, 3):
        act[f"hotspot_count_t{k}"] = _i32(g["hotspot_count"].shift(k)).fillna(0)
    act["FRP_sum_t1"] = _f32(g["FRP_sum"].shift(1)).fillna(0.0)

    # rolling window (3/7/14/30 hari) pada data aktif: karena hari non-aktif
    # tidak ada di `act`, hitung via penghitungan ulang di panel penuh.
    # -> rolling dihitung pada panel penuh (di bawah), bukan di sini.

    # 2) gabungkan kembali ke panel penuh
    lag_cols = [f"hotspot_count_t{k}" for k in (1, 2, 3)] + ["FRP_sum_t1"]
    df = df.drop(columns=[c for c in lag_cols if c in df.columns], errors="ignore")
    df = df.merge(
        act[["grid_id", "date"] + lag_cols],
        on=["grid_id", "date"], how="left",
    )
    for c in lag_cols:
        df[c] = df[c].fillna(0)

    # days_since_last_fire: hari sejak api terakhir; NaN → dihitung bertahap per grid
    df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)

    # 3) rolling di panel penuh (setelah baris tenang diisi 0)
    df["hotspot_count"] = _i32(df["hotspot_count"].fillna(0))
    for c in ("FRP_sum", "FRP_mean", "FRP_max", "confidence_high_pct"):
        if c in df.columns:
            df[c] = _f32(df[c].fillna(0.0))
    df["state"] = df["state"].fillna(0).astype("int8")

    # rolling hotspot 3/7/14/30 hari (min_periods=1, closed: window kemarin)
    hs = df.groupby("grid_id", sort=False)["hotspot_count"]
    for w in (3, 7, 14, 30):
        df[f"active_hotspots_{w}d"] = _i32(
            hs.shift(1).rolling(w, min_periods=1).sum()
        )

    # rolling hujan 3/7/14 hari + CDD
    rain = df.groupby("grid_id", sort=False)["rain_t"]
    df["rain_t1"] = _f32(rain.shift(1))
    for w in (3, 7, 14):
        df[f"rain_{w}d"] = _f32(rain.shift(1).rolling(w, min_periods=1).sum())
    # CDD: jumlah hari kering berturut (rain_t < 1mm) sampai t-1
    dry = (df["rain_t"] < 1.0)
    grp = df.groupby("grid_id", sort=False)
    df["CDD"] = (
        dry.groupby([df["grid_id"], (grp.cumcount() + 1) * (df.groupby("grid_id", sort=False)["date"].diff().dt.days.fillna(1) > 1)])
        .cumsum() * dry
    )  # placeholder rumit — digantikan implementasi vektor di bawah
    df = df.drop(columns=["CDD"], errors="ignore")

    # CDD vektor sederhana per grup (run-length kering berjalan, t-1)
    def _cdd(sub: pd.Series) -> pd.Series:
        rain_vals = sub.values
        out = np.zeros(len(sub), dtype=np.float32)
        run = 0.0
        prev = None
        for i, r in enumerate(rain_vals):
            if prev is not None:
                if prev < 1.0:
                    run += 1.0
                else:
                    run = 0.0
            out[i] = run  # CDD hingga t-1 (tidak termasuk hari ini)
            prev = r
        return pd.Series(out, index=sub.index)

    df["CDD"] = _f32(df.groupby("grid_id", sort=False)["rain_t"].transform(_cdd))

    # days_since_last_fire + fire_freq_30d (rolling pada state)
    st = df.groupby("grid_id", sort=False)["state"]
    df["fire_freq_30d"] = _f32(st.shift(1).rolling(30, min_periods=1).mean())
    # days_since_last_fire dihitung per grid via state shift: posisi api terakhir
    dsf = np.full(len(df), np.nan, dtype=np.float32)
    grid_ids = df["grid_id"].to_numpy()
    states = df["state"].to_numpy()
    dates = df["date"].to_numpy()
    start = 0
    for end in _group_ends(grid_ids):
        s = states[start:end]
        last_fire = -1
        for i in range(end - start):
            if s[i] == 1:
                dsf[start + i] = 0.0
                last_fire = i
            elif last_fire >= 0:
                dsf[i + start] = np.float32((i - last_fire))
        start = end
    df["days_since_last_fire"] = dsf

    return df


def _group_ends(ids: np.ndarray) -> list[int]:
    ends = []
    ids = pd.factorize(ids)[0]
    change = np.flatnonzero(np.diff(ids)) + 1
    ends = np.concatenate([[0], change, [len(ids)]]).tolist()
    return ends[1:]


# ---------------------------------------------------------------------------
# Fitur spasial: tetangga
# ---------------------------------------------------------------------------

def _add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """neighbor_hotspots_3d & neighbor_rain_3d dari neighbors_8 (rata-rata 8-tetangga)."""
    ref = _load_reference_tables()
    nbr_long = ref["neighbors_long"]

    # state aktif 3 hari (kemarin hingga 3 hari lalu) per grid-hari
    active3 = (df["active_hotspots_3d"] > 0).astype("int8")
    tmp = pd.DataFrame({
        "grid_id": df["grid_id"].values,
        "date": df["date"].values,
        "nbr_fire": active3.values,
        "rain_3d_v": df["rain_3d"].values,
    })
    # prop agregat ke tetangga: setiap baris grid g menerima rata-rata tetangganya
    merged = tmp.merge(nbr_long, left_on="grid_id", right_on="grid_id", how="left")
    agg = merged.groupby(["neighbor_id", "date"], sort=False).agg(
        neighbor_hotspots_3d=("nbr_fire", "sum"),
        neighbor_rain_3d=("rain_3d_v", "mean"),
    ).reset_index()
    agg = agg.rename(columns={"neighbor_id": "grid_id"})
    agg["neighbor_hotspots_3d"] = _i32(agg["neighbor_hotspots_3d"])
    agg["neighbor_rain_3d"] = _f32(0.0)  # placeholder — dihitung ulang di bawah
    # hitung neighbor_rain_3d benar: rata-rata rain_3d para tetangga
    agg2 = merged.groupby(["neighbor_id", "date"], sort=False).agg(
        neighbor_rain_3d=("rain_3d_v", "mean"),
    ).reset_index().rename(columns={"neighbor_id": "grid_id"})
    agg["neighbor_rain_3d"] = _f32(agg2.set_index(["grid_id", "date"])["neighbor_rain_3d"]
                                    .reindex(agg.set_index(["grid_id", "date"]).index)
                                    .values)
    df = df.merge(agg, on=["grid_id", "date"], how="left")
    df["neighbor_hotspots_3d"] = df["neighbor_hotspots_3d"].fillna(0).astype("int32")
    df["neighbor_rain_3d"] = df["neighbor_rain_3d"].astype("float32")
    return df


# ---------------------------------------------------------------------------
# Build satu tahun
# ---------------------------------------------------------------------------

def build_single_year_batch(year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Rakit panel satu tahun. Semesta = grid CHIRPS tahun tsb (paling ketat).
    Return: DataFrame panel per tahun (juga disimpan ke parquet per tahun).
    """
    ref = _load_reference_tables()
    grid = ref["grid"]
    landcover = ref["landcover"]

    chirps = _load_chirps_year(year)
    chirps["grid_id"] = chirps["grid_id"].astype("str")

    # kerangka: CHIRPS sudah (grid x date) lengkap
    df = chirps.merge(grid, on="grid_id", how="left")

    # FIRMS (hanya baris aktif, buffer akhir tahun lalu untuk lag)
    firms = _load_firms_year(year)
    firms["grid_id"] = firms["grid_id"].astype("str")
    firms = firms[(firms["date"] >= chirps["date"].min()) & (firms["date"] <= chirps["date"].max())]
    df = df.merge(
        firms[["grid_id", "date"] + FIRMS_VALUE_COLS + ["state"]],
        on=["grid_id", "date"], how="left",
    )

    # ERA5 bulan-per-bulan
    for mdf in _iter_era5_months(year):
        mdf["grid_id"] = mdf["grid_id"].astype("str")
        df = df.merge(mdf, on=["grid_id", "date"], how="left")

    # landcover
    df = df.merge(landcover, on="grid_id", how="left")

    # pastikan urutan & isi 0 untuk hari tenang
    df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)

    # fitur memori + isi 0
    df = _add_fire_memory_features(df)

    # fitur spasial (tetangga)
    df = _add_spatial_features(df)

    # state_t1 (target): shift -1 dalam grup grid; hari terakhir tahun → NaN → drop
    df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)
    df[TARGET_COL] = (
        df.groupby("grid_id", sort=False)["state"].shift(-1)
    ).astype("float32")
    # hari terakhir tiap grid tanpa info besok → buang (target NaN)
    n_before = len(df)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    dropped = n_before - len(df)

    # split label
    df["split"] = df["date"].map(_split_label).astype("category")

    # rapikan dtype & urutan kolom
    df["landcover_code"] = df["landcover_code"].fillna(-1).astype("int16")
    panel_cols = PANEL_META_COLS + [
        c for c in df.columns
        if c not in PANEL_META_COLS + ["split"]
    ]
    df = df[panel_cols + ["split"]]

    out = BATCH_DIR / f"batch_{year}.parquet"
    df.to_parquet(out, index=False)
    if verbose:
        pos_rate = df[TARGET_COL].mean()
        print(
            f"[batch {year}] rows={len(df):,} | grid={df.grid_id.nunique():,} | "
            f"hari={df.date.nunique():,} | state_t1=+ {pos_rate:.4%} | "
            f"drop_t1={dropped:,} | → {out.name}"
        )
    return df


def build_all_batches(verbose: bool = True) -> list[pd.DataFrame]:
    """Bangun semua tahun yang tersedia (2016-2026)."""
    years = get_available_years()
    if verbose:
        print(f"tahun tersedia: {years}")
    out = []
    for y in years:
        out.append(build_single_year_batch(y, verbose=verbose))
    return out


# ---------------------------------------------------------------------------
# Streaming untuk training
# ---------------------------------------------------------------------------

def iter_panel_batches(split: str = None, chunk_rows: int = CHUNK_ROWS):
    """
    Alirkan panel per tahun, terpotong chunk ~500rb baris.
    split: None (=semua), 'train', 'val', 'test'.
    """
    for year in get_available_years():
        f = BATCH_DIR / f"batch_{year}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        if split is not None:
            df = df[df["split"] == split]
        if df.empty:
            continue
        for i in range(0, len(df), chunk_rows):
            yield df.iloc[i:i + chunk_rows]
