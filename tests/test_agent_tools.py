"""Pruebas del modulo de tools del agente.

Verifica que la whitelist de columnas y operadores funciona
y que las queries devuelven resultados consistentes.
"""
from __future__ import annotations

import pytest

from src.agent.tools import (
    ALLOWED_COLUMNS,
    aggregate_predictions,
    query_predictions,
    schema_info,
)


def test_schema_info_returns_columns() -> None:
    info = schema_info()
    assert "columns" in info
    assert len(info["columns"]) >= 15
    for col in ["score_compra", "cluster_label", "decile"]:
        assert col in info["columns"]


def test_query_basic_filter() -> None:
    out = query_predictions(
        filters=[{"column": "decile", "operator": "=", "value": "D1"}],
        limit=10,
    )
    assert out["row_count"] <= 10
    assert all(r["decile"] == "D1" for r in out["rows"])


def test_query_rejects_unknown_column() -> None:
    with pytest.raises(ValueError, match="Columna no permitida"):
        query_predictions(filters=[{"column": "drop_table", "operator": "=", "value": 1}])


def test_query_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="Operador no permitido"):
        query_predictions(filters=[{"column": "edad", "operator": "DROP", "value": 1}])


def test_query_in_operator() -> None:
    out = query_predictions(
        filters=[{"column": "decile", "operator": "IN", "value": ["D1", "D2"]}],
        limit=20,
    )
    assert all(r["decile"] in {"D1", "D2"} for r in out["rows"])


def test_query_between_operator() -> None:
    out = query_predictions(
        filters=[{"column": "edad", "operator": "BETWEEN", "value": [30, 40]}],
        limit=20,
    )
    assert all(30 <= r["edad"] <= 40 for r in out["rows"])


def test_aggregate_by_segment() -> None:
    out = aggregate_predictions(
        group_by=["cluster_label"],
        aggregations=[
            {"function": "count", "column": "id", "alias": "n"},
            {"function": "avg", "column": "score_compra", "alias": "avg_score"},
        ],
    )
    assert out["row_count"] == 4  # 4 segmentos
    labels = {r["cluster_label"] for r in out["rows"]}
    assert labels == {
        "Leales premium",
        "Cazadores de oferta",
        "Compradores moderados",
        "Visitantes ocasionales",
    }


def test_top_decile_has_higher_score() -> None:
    """Sanity: el score promedio debe ser mayor en D1 que en D10."""
    out = aggregate_predictions(
        group_by=["decile"],
        aggregations=[{"function": "avg", "column": "score_compra", "alias": "avg_score"}],
    )
    rows = {r["decile"]: r["avg_score"] for r in out["rows"]}
    assert rows["D1"] > rows["D10"], "D1 debería tener score mayor que D10"


def test_aggregate_rejects_unknown_function() -> None:
    with pytest.raises(ValueError, match="Funcion de agregacion no permitida"):
        aggregate_predictions(
            aggregations=[{"function": "DROP", "column": "id"}],
        )


def test_limit_capped() -> None:
    """El parámetro limit debe estar acotado (defensa contra DoS)."""
    out = query_predictions(limit=99999)
    assert out["row_count"] <= 500  # cap hardcoded


def test_sql_injection_blocked() -> None:
    """Intentar inyectar SQL en valores debe ser inocuo (params escapan)."""
    out = query_predictions(
        filters=[{"column": "cluster_label", "operator": "=", "value": "Leales premium'; DROP TABLE x; --"}],
        limit=5,
    )
    assert out["row_count"] == 0  # no hay match, pero NO crashea ni ejecuta SQL malicioso
