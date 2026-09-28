"""
Konfigurasi global proyek — single source of truth.
Semua path, konstanta, dan keputusan desain hidup di sini.

Struktur data mentah (hasil audit 2026-09-28):
  data/DL_FIRE_SV-C2_809701/   FIRMS VIIRS SNPP (archive + NRT), CSV
  data/era5/                   ERA5-Land bulanan, NetCDF4, 6-hourly, 0.1°
  data/chirps/                 CHIRPS v3.0 harian GeoTIFF, 0.05°, nodata=-9999
  data/worldcover/             ESA WorldCover 2021 v200, 10m tiles

Struktur data antara (dihasilkan preprocessing):
  data/processed/firms_daily_grid.parquet
  data/processed/era5_monthly/era5_YYYY_MM.parquet      (harian per grid)
  data/processed/chirps_yearly/chirps_YYYY.parquet      (harian per grid)
  data/processed/grid_sumatra.parquet                   (0.1°, daratan saja)
  data/processed/landcover_grid.parquet
  data/processed/batch/batch_YYYY.parquet               (panel final per tahun)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR
FIRMS_DIR = DATA_DIR / "DL_FIRE_SV-C2_809701"
ERA5_DIR = DATA_DIR / "era5"
CHIRPS_DIR = DATA_DIR / "chirps"
WORLDCOVER_DIR = DATA_DIR / "worldcover"

PROCESSED_DIR = DATA_DIR / "processed"
BATCH_DIR = PROCESSED_DIR / "batch"
ERA5_MONTHLY_DIR = PROCESSED_DIR / "era5_monthly"
CHIRPS_YEARLY_DIR = PROCESSED_DIR / "chirps_yearly"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
EDA_FIG_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, BATCH_DIR, MODELS_DIR, REPORTS_DIR, EDA_FIG_DIR,
           ERA5_MONTHLY_DIR, CHIRPS_YEARLY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# File antara yang dibaca notebook
FIRMS_DAILY_FILE = PROCESSED_DIR / "firms_daily_grid.parquet"
GRID_FILE = PROCESSED_DIR / "grid_sumatra.parquet"
LANDCOVER_FILE = PROCESSED_DIR / "landcover_grid.parquet"
NEIGHBOR_FILE = PROCESSED_DIR / "neighbors_8.parquet"

# ---------------------------------------------------------------------------
# Domain spasial — Sumatra (bbox dari data ERA5 yang sudah terverifikasi)
# ERA5: lat -6..6 (121 titik @0.1°), lon 95..106 (111 titik @0.1°)
# Grid 0.1° = unit keputusan proyek
# ---------------------------------------------------------------------------
SUMATRA_BBOX = {"lon_min": 95.0, "lon_max": 106.0, "lat_min": -6.0, "lat_max": 6.0}
GRID_RES = 0.1  # derajat (~11 km x 11 km di ekuator)

# ---------------------------------------------------------------------------
# Periode analisis
# Dibatasi CHIRPS: 2016-09-22 s.d. 2026-08-31 (audit: nol hari bolong).
# FIRMS lebih panjang di dua ujungnya -> margin untuk rolling features.
# ---------------------------------------------------------------------------
START_DATE = "2016-09-22"
END_DATE = "2026-08-31"

# Split temporal strict (out-of-time) — dihitung ulang dari data asli,
# bukan dari proyeksi notebook.
TRAIN_END = "2022-12-31"    # train: 2016-09-22 s.d. 2022-12-31
VAL_END = "2024-12-31"      # val  : 2023-01-01 s.d. 2024-12-31
TEST_END = "2026-08-31"     # test : 2025-01-01 s.d. 2026-08-31

# ---------------------------------------------------------------------------
# Preprocessing FIRMS
# ---------------------------------------------------------------------------
FIRMS_MIN_CONFIDENCE = "n"   # 'l' = low saja; 'n' = normal ke atas. Bisa dinaikkan.
FIRMS_DAY_ONLY = False       # VIIRS daynight: D = siang, N = malam. Keduanya dipakai.

# ---------------------------------------------------------------------------
# Fitur panel (harus konsisten dengan batch_builder)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # meteorologi (ERA5-Land, harian per grid)
    "temperature_t", "precipitation_t", "humidity_t", "wind_speed_t",
    # curah hujan CHIRPS + rolling
    "rain_t", "rain_3d", "rain_7d", "rain_14d",
    # histori hotspot
    "active_hotspots_3d", "active_hotspots_7d", "active_hotspots_14d",
    "days_since_last_fire", "fire_freq_30d",
    # spasial-static
    "landcover_code", "is_peat", "elev_mean",
    # tetangga spasial
    "neighbor_hotspots_3d", "neighbor_rain_3d",
    # state hari ini
    "state",
]
TARGET_COL = "state_t1"
META_COLS = ["grid_id", "date"]
