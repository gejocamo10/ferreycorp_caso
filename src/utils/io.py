"""Utilidades de entrada y salida de datos.

Concentro aqui la carga del CSV crudo para que todo el resto del codigo
(EDA, features, scoring) use la misma fuente y el mismo orden.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "compras_data.csv"

# Las 5 marcas que aparecen en el dataset, en el orden del diccionario de datos.
BRANDS = [1, 2, 3, 4, 5]


def load_raw() -> pd.DataFrame:
    """Carga el CSV crudo, lo ordena por cliente y dia de visita y descarta
    la columna `tamanio_ciudad` que el diccionario indica no considerar.
    """
    df = pd.read_csv(RAW_CSV)
    df = df.sort_values(["id", "dia_visita"]).reset_index(drop=True)
    if "tamanio_ciudad" in df.columns:
        df = df.drop(columns=["tamanio_ciudad"])
    return df


def ensure_dir(path: str | Path) -> Path:
    """Crea el directorio si no existe y devuelve el Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
