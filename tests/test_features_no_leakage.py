"""Tests críticos: validar que las features no tienen data leakage.

Si estos tests fallan, el modelo está aprendiendo del futuro y todas las métricas
de evaluación están infladas. Es el bug más peligroso en proyectos de ML.

Run:
    pytest tests/ -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.features import build_features
from src.utils.io import BRANDS, load_raw


@pytest.fixture(scope="module")
def df_features() -> pd.DataFrame:
    df = load_raw()
    return build_features(df)


def test_first_visit_priors_are_zero(df_features: pd.DataFrame) -> None:
    """En la primera visita de cada cliente, todas las features 'prior_*' deben ser 0."""
    first = df_features.groupby("id").head(1)
    for col in ["prior_visits", "prior_purchases", "prior_buy_rate", "prior_qty"]:
        assert (first[col] == 0).all(), f"{col} no es 0 en primera visita de algún cliente"


def test_first_visit_recency_sentinel(df_features: pd.DataFrame) -> None:
    """recency_days debe ser -1 (sentinel 'sin compra previa') en primera visita."""
    first = df_features.groupby("id").head(1)
    assert (first["recency_days"] == -1).all()


def test_loyalty_sums_to_one_when_history_exists(df_features: pd.DataFrame) -> None:
    """Las 5 lealtades deben sumar exactamente 1.0 cuando hay compras previas."""
    loy_cols = [f"loyalty_b{b}" for b in BRANDS]
    df = df_features.copy()
    df["loy_sum"] = df[loy_cols].sum(axis=1)
    with_history = df[df.prior_purchases > 0]
    assert with_history["loy_sum"].between(0.999, 1.001).all()


def test_loyalty_zero_when_no_history(df_features: pd.DataFrame) -> None:
    """Si el cliente no tiene compras previas, todas las lealtades son 0."""
    loy_cols = [f"loyalty_b{b}" for b in BRANDS]
    no_history = df_features[df_features.prior_purchases == 0]
    assert (no_history[loy_cols].sum(axis=1) == 0).all()


def test_prior_purchases_is_lag(df_features: pd.DataFrame) -> None:
    """prior_purchases en la fila k debe ser sum(incidencia) en filas 0..k-1 del cliente."""
    sample = df_features[df_features.id == df_features.id.iloc[0]].head(20)
    expected_lag = sample["incidencia_compra"].cumsum().shift(1, fill_value=0)
    assert (sample["prior_purchases"].values == expected_lag.values).all()


def test_prior_visits_monotonic(df_features: pd.DataFrame) -> None:
    """prior_visits debe crecer monotónicamente para cada cliente."""
    for cust_id in df_features.id.unique()[:5]:
        cust = df_features[df_features.id == cust_id].sort_values("dia_visita")
        diffs = cust["prior_visits"].diff().dropna()
        assert (diffs == 1).all(), f"prior_visits no incrementa en 1 para cliente {cust_id}"


def test_promo_uplift_bounded(df_features: pd.DataFrame) -> None:
    """promo_uplift = buy_rate_with_promo - buy_rate_no_promo, ambos en [0,1]."""
    assert df_features["promo_uplift"].between(-1.0, 1.0).all()
