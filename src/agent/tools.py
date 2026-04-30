"""Herramientas que el agente puede llamar, ejecutadas con DuckDB sobre el Parquet.

Decision de diseño: en vez de exponer text-to-SQL (riesgoso, puede generar SQL
invalido o peligroso), el LLM emite un objeto de filtros estructurado que el
backend traduce a SQL parametrizado. Es mas seguro, predecible y facil de testear.

DuckDB lee el Parquet en cada consulta. Para este volumen (alrededor de 60 mil filas)
las consultas devuelven en menos de 100 ms incluso leyendo desde object storage en la nube.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd

from src.config import PROJECT_ROOT, config
from src.storage.object_storage import storage

# Whitelist estricta de columnas y operadores. Cualquier valor que no este
# aqui se rechaza antes de tocar SQL, lo que previene inyeccion.
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
# Conexion y carga de datos
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
# Traduccion del lenguaje de filtros estructurados a SQL parametrizado
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
            raise ValueError(f"Columna no permitida: {col}")
        if op not in ALLOWED_OPERATORS:
            raise ValueError(f"Operador no permitido: {op}")
        if op == "IN":
            if not isinstance(val, list) or not val:
                raise ValueError("El operador IN requiere una lista no vacia como valor")
            placeholders = ",".join(["?"] * len(val))
            clauses.append(f"{col} IN ({placeholders})")
            params.extend(val)
        elif op == "BETWEEN":
            if not isinstance(val, list) or len(val) != 2:
                raise ValueError("El operador BETWEEN requiere una lista [min, max]")
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
            raise ValueError(f"Columna de ordenamiento no permitida: {col}")
        if direction not in ALLOWED_SORT_DIR:
            raise ValueError(f"Direccion de ordenamiento no permitida: {direction}")
        parts.append(f"{col} {direction}")
    return " ORDER BY " + ", ".join(parts)


# ============================================================
# Funciones publicas que el agente puede invocar via tool use
# ============================================================
def query_predictions(
    filters: list[dict[str, Any]] | None = None,
    order_by: list[dict[str, str]] | None = None,
    limit: int = 50,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Lista filas individuales de predicciones (cliente y dia) que cumplen
    los filtros indicados. Util para preguntas tipo "dame los 50 clientes
    con mayor score que sean de Leales premium".
    """
    cols = columns or list(ALLOWED_COLUMNS.keys())
    for c in cols:
        if c not in ALLOWED_COLUMNS:
            raise ValueError(f"Columna no permitida: {c}")
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
    """Agrupa y agrega la tabla de predicciones para responder preguntas tipo
    "score promedio por segmento" o "cuantos clientes hay en el decil top
    con promo activa".
    """
    if aggregations is None:
        aggregations = [{"function": "count", "column": "id", "alias": "n"},
                        {"function": "avg", "column": "score_compra", "alias": "avg_score"}]
    select_parts: list[str] = []
    if group_by:
        for g in group_by:
            if g not in ALLOWED_COLUMNS:
                raise ValueError(f"Columna de agrupacion no permitida: {g}")
            select_parts.append(g)
    for agg in aggregations:
        fn = agg.get("function", "count").lower()
        col = agg.get("column", "id")
        alias = agg.get("alias", f"{fn}_{col}")
        if fn not in ALLOWED_AGG:
            raise ValueError(f"Funcion de agregacion no permitida: {fn}")
        if col not in ALLOWED_COLUMNS:
            raise ValueError(f"Columna de agregacion no permitida: {col}")
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
    """Devuelve la lista de columnas disponibles con su descripcion. El agente
    puede llamar esta funcion cuando duda de que columna usar para un filtro.
    """
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
        "incidencia_compra": "Variable real (0 o 1). Util para validar el modelo, no para predecir.",
    }
    return {"columns": descriptions}
