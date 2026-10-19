"""Download + clean the dataset.  Usage:  python scripts/download_data.py"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import ensure_clean_data  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = ensure_clean_data(refresh=True)
    print(f"Done. Cleaned dataset shape: {df.shape}")
