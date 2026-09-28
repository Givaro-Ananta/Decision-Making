"""Smoke test preprocessing: jalankan tiap modul pada data asli dan verifikasi output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.preprocess_grid import run as run_grid
from src.preprocessing.preprocess_firms import build_firms_daily
from src.preprocessing.preprocess_era5 import build_era5_monthly
from src.preprocessing.preprocess_chirps import build_chirps_yearly

if __name__ == "__main__":
    run_grid()
    print()
    build_firms_daily()
    print()
    build_era5_monthly()
    print()
    build_chirps_yearly()
    print("\nSMOKE TEST SELESAI")
