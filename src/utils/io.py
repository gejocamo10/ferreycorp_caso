"""Data I/O helpers — single entrypoint for loading the raw dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "compras_data.csv"

BRANDS = [1, 2, 3, 4, 5]


def load_raw() -> pd.DataFrame:
    """Load raw transactional data, sorted by customer and visit day."""
    df = pd.read_csv(RAW_CSV)
    df = df.sort_values(["id", "dia_visita"]).reset_index(drop=True)
    if "tamanio_ciudad" in df.columns:
        df = df.drop(columns=["tamanio_ciudad"])
    return df


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
