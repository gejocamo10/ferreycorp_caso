"""Pipeline de scoring batch.

Carga las features ya construidas, predice la probabilidad de compra para
cada par (cliente, dia de visita) con el LightGBM entrenado, enriquece la
salida con demografia y segmento, y escribe el resultado como Parquet usando
la capa de storage agnostica (local, DO Spaces, AWS S3 o GCS).

El Parquet resultante es lo que el agente conversacional consulta en linea.
Cuando llegan datos nuevos, basta volver a correr este script para refrescar
las predicciones.

Para ejecutar:
    python -m src.model.score
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from src.config import PROJECT_ROOT
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.storage.object_storage import storage
from src.utils.io import ensure_dir

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / "data" / "predictions")

# Columnas que conservamos junto a las predicciones para que el agente pueda filtrar por ellas
ENRICHMENT_COLS = [
    "id", "dia_visita",
    "edad", "ingreso_anual", "genero", "estado_civil", "nivel_educacion", "ocupacion",
    "loyal_brand_id", "prior_purchases", "prior_buy_rate", "recency_days",
    "any_promo_today", "incidencia_compra",
]


def main() -> None:
    print("Cargando modelo y features...")
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_propensity.txt"))
    features = pd.read_parquet(PROCESSED / "features.parquet")
    segments = pd.read_parquet(PROCESSED / "customer_segments.parquet")
    print(f"  filas={len(features):,}  clientes={features['id'].nunique()}")

    X = features[FEATURE_COLUMNS].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")

    print("Scoring de todas las filas...")
    proba = booster.predict(X)

    print("Construyendo tabla de predicciones...")
    preds = features[ENRICHMENT_COLS].copy()
    preds["score_compra"] = proba
    preds = preds.merge(segments[["id", "cluster_id", "cluster_label"]], on="id", how="left")

    preds["decile"] = pd.qcut(preds["score_compra"].rank(method="first"), 10,
                              labels=[f"D{i}" for i in range(10, 0, -1)]).astype(str)
    preds["score_band"] = pd.cut(
        preds["score_compra"],
        bins=[-0.01, 0.20, 0.40, 0.60, 1.01],
        labels=["bajo", "medio_bajo", "medio_alto", "alto"],
    ).astype(str)

    print("\nResumen de predicciones por decil:")
    print(preds.groupby("decile", observed=True).agg(
        n=("id", "count"),
        avg_score=("score_compra", "mean"),
        actual_rate=("incidencia_compra", "mean"),
    ).round(3))

    print("\nResumen de predicciones por segmento:")
    print(preds.groupby("cluster_label").agg(
        n=("id", "count"),
        avg_score=("score_compra", "mean"),
        actual_rate=("incidencia_compra", "mean"),
    ).round(3))

    print("\nGuardando predicciones...")
    local_path = OUTPUT_DIR / "predictions.parquet"
    preds.to_parquet(local_path, index=False)
    print(f"  local: {local_path}")

    if storage.provider != "local":
        uri = storage.write_parquet(preds, "predictions.parquet")
        print(f"  nube ({storage.provider}): {uri}")
    else:
        print("  upload a la nube omitido (CLOUD_PROVIDER=local)")

    print(f"\nListo. {len(preds):,} predicciones escritas.")


if __name__ == "__main__":
    main()
