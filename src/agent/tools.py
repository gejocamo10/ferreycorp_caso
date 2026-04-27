"""DuckDB-backed query tools the agent can call.

Architecture decision: instead of text-to-SQL (risky, can produce invalid/dangerous
SQL), the LLM emits a structured filter object that we translate to parameterized
SQL. Safer, predictable, easier to test.

The Parquet file is read by DuckDB on every query — at this volume (~60k rows)
queries return in <100ms even from cloud object storage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd

from src.config import PROJECT_ROOT, config
from src.storage.object_storage import storage

# Allowed columns and operators — strict whitelist prevents SQL injection
ALLOWED_COLUMNS: dict[str, str] = {
    "id": "INTEGER",
    "dia_visita": "INTEGER",
    "score_compra": "DOUBLE",
    "decile": "VARCHAR",
    "score_band": "VARCHAR",
    "cluster_id": "INTEGER",
    "cluster_label": "VARCHAR",
    "edad": "INTEGER",
    "ingreso_anual": "INTEGER",
    "genero": "INTEGER",
    "estado_civil": "INTEGER",
    "nivel_educacion": "INTEGER",
    "ocupacion": "INTEGER",
    "loyal_brand_id": "INTEGER",
    "prior_purchases": "INTEGER",
    "prior_buy_rate": "DOUBLE",
    "recency_days": "DOUBLE",
    "any_promo_today": "INTEGER",
    "incidencia_compra": "INTEGER",
}

ALLOWED_OPERATORS = {">", ">=", "<", "<=", "=", "!=", "IN", "BETWEEN"}
ALLOWED_AGG = {"count", "avg", "sum", "min", "max"}
ALLOWED_SORT_DIR = {"asc", "desc"}


# ============================================================
# Connection / data loading
# ============================================================
def _predictions_uri() -> str:
    if config.cloud_provider == "local":
        return str(config.local_predictions_path)
    return storage.get_duckdb_uri("predictions.parquet")


def get_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    storage.configure_duckdb(conn)
    return conn


# ============================================================
# Filter language → SQL
# ============================================================
def _build_where(filters: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    for f in filters:
        col = f.get("column")
        op = f.get("operator", "=").upper()
        val = f.get("value")
        if col not in ALLOWED_COLUMNS:
            raise ValueError(f"Column not allowed: {col}")
        if op not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator not allowed: {op}")
        if op == "IN":
            if not isinstance(val, list) or not val:
                raise ValueError("IN requires a non-empty list value")
            placeholders = ",".join(["?"] * len(val))
            clauses.append(f"{col} IN ({placeholders})")
            params.extend(val)
        elif op == "BETWEEN":
            if not isinstance(val, list) or len(val) != 2:
                raise ValueError("BETWEEN requires a [low, high] list")
            clauses.append(f"{col} BETWEEN ? AND ?")
            params.extend(val)
        else:
            clauses.append(f"{col} {op} ?")
            params.append(val)
    return " WHERE " + " AND ".join(clauses), params


def _build_order(order_by: list[dict[str, str]] | None) -> str:
    if not order_by:
        return ""
    parts = []
    for ob in order_by:
        col = ob.get("column")
        direction = ob.get("direction", "desc").lower()
        if col not in ALLOWED_COLUMNS:
            raise ValueError(f"Order column not allowed: {col}")
        if direction not in ALLOWED_SORT_DIR:
            raise ValueError(f"Order direction not allowed: {direction}")
        parts.append(f"{col} {direction}")
    return " ORDER BY " + ", ".join(parts)


# ============================================================
# Public tool functions (called by the agent via tool use)
# ============================================================
def query_predictions(
    filters: list[dict[str, Any]] | None = None,
    order_by: list[dict[str, str]] | None = None,
    limit: int = 50,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """List individual customer-day predictions matching the filters."""
    cols = columns or list(ALLOWED_COLUMNS.keys())
    for c in cols:
        if c not in ALLOWED_COLUMNS:
            raise ValueError(f"Column not allowed: {c}")
    where_sql, params = _build_where(filters or [])
    order_sql = _build_order(order_by)
    limit = max(1, min(int(limit), 500))

    sql = f"SELECT {', '.join(cols)} FROM read_parquet('{_predictions_uri()}'){where_sql}{order_sql} LIMIT {limit}"
    conn = get_conn()
    df = conn.execute(sql, params).fetchdf()
    conn.close()
    return {
        "row_count": len(df),
        "rows": df.to_dict(orient="records"),
        "sql_executed": sql,
    }


def aggregate_predictions(
    group_by: list[str] | None = None,
    aggregations: list[dict[str, str]] | None = None,
    filters: list[dict[str, Any]] | None = None,
    order_by: list[dict[str, str]] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Group + aggregate the predictions table (e.g., avg score by cluster)."""
    if aggregations is None:
        aggregations = [{"function": "count", "column": "id", "alias": "n"},
                        {"function": "avg", "column": "score_compra", "alias": "avg_score"}]
    select_parts: list[str] = []
    if group_by:
        for g in group_by:
            if g not in ALLOWED_COLUMNS:
                raise ValueError(f"Group column not allowed: {g}")
            select_parts.append(g)
    for agg in aggregations:
        fn = agg.get("function", "count").lower()
        col = agg.get("column", "id")
        alias = agg.get("alias", f"{fn}_{col}")
        if fn not in ALLOWED_AGG:
            raise ValueError(f"Agg function not allowed: {fn}")
        if col not in ALLOWED_COLUMNS:
            raise ValueError(f"Agg column not allowed: {col}")
        if fn == "count":
            select_parts.append(f"COUNT({col}) AS {alias}")
        else:
            select_parts.append(f"{fn.upper()}({col}) AS {alias}")

    where_sql, params = _build_where(filters or [])
    group_sql = f" GROUP BY {', '.join(group_by)}" if group_by else ""
    order_sql = _build_order(order_by)
    limit = max(1, min(int(limit), 500))

    sql = f"SELECT {', '.join(select_parts)} FROM read_parquet('{_predictions_uri()}'){where_sql}{group_sql}{order_sql} LIMIT {limit}"
    conn = get_conn()
    df = conn.execute(sql, params).fetchdf()
    conn.close()
    return {"row_count": len(df), "rows": df.to_dict(orient="records"), "sql_executed": sql}


def schema_info() -> dict[str, Any]:
    """Return the available columns + their meaning so the LLM picks valid ones."""
    descriptions = {
        "id": "ID del cliente (500 únicos).",
        "dia_visita": "Día de la visita (1-730, ~2 años de datos).",
        "score_compra": "Probabilidad estimada de compra (0-1).",
        "decile": "Decil de score: D1=top 10% más probable, D10=bottom 10%.",
        "score_band": "Banda de score: 'bajo','medio_bajo','medio_alto','alto'.",
        "cluster_id": "ID del segmento de cliente (0-3).",
        "cluster_label": "Etiqueta del segmento: 'Leales premium', 'Cazadores de oferta', 'Compradores moderados', 'Visitantes ocasionales'.",
        "edad": "Edad del cliente (18-75).",
        "ingreso_anual": "Ingreso anual (38k-309k).",
        "genero": "Género codificado (0/1).",
        "estado_civil": "Estado civil codificado.",
        "nivel_educacion": "Nivel educativo (0-3).",
        "ocupacion": "Ocupación (0-2).",
        "loyal_brand_id": "Marca a la que el cliente es más leal históricamente (1-5, 0 si sin historia).",
        "prior_purchases": "# compras previas del cliente antes de esa visita.",
        "prior_buy_rate": "Tasa de compra histórica del cliente (0-1).",
        "recency_days": "Días desde la última compra del cliente (-1 si nunca).",
        "any_promo_today": "1 si al menos una marca está en promo ese día, 0 si no.",
        "incidencia_compra": "Variable real (0/1) — útil para validar el modelo, no para predecir.",
    }
    return {"columns": descriptions}
