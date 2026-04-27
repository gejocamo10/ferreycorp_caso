"""Batch scoring pipeline.

Loads features → predicts probability of purchase per (cliente, dia_visita)
→ enriches with customer demographics + segment → writes Parquet via the
cloud-agnostic storage layer (local / DO Spaces / AWS S3 / GCS).

Run:
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

# Columns we keep alongside predictions for the agent to filter on
ENRICHMENT_COLS = [
    "id", "dia_visita",
    "edad", "ingreso_anual", "genero", "estado_civil", "nivel_educacion", "ocupacion",
    "loyal_brand_id", "prior_purchases", "prior_buy_rate", "recency_days",
    "any_promo_today", "incidencia_compra",
]


def main() -> None:
    print("Loading model + features...")
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_propensity.txt"))
    features = pd.read_parquet(PROCESSED / "features.parquet")
    segments = pd.read_parquet(PROCESSED / "customer_segments.parquet")
    print(f"  rows={len(features):,}  customers={features['id'].nunique()}")

    X = features[FEATURE_COLUMNS].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")

    print("Scoring all rows...")
    proba = booster.predict(X)

    print("Building predictions table...")
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

    print("\nPredictions summary by decile:")
    print(preds.groupby("decile", observed=True).agg(
        n=("id", "count"),
        avg_score=("score_compra", "mean"),
        actual_rate=("incidencia_compra", "mean"),
    ).round(3))

    print("\nPredictions summary by segment:")
    print(preds.groupby("cluster_label").agg(
        n=("id", "count"),
        avg_score=("score_compra", "mean"),
        actual_rate=("incidencia_compra", "mean"),
    ).round(3))

    print("\nSaving predictions...")
    local_path = OUTPUT_DIR / "predictions.parquet"
    preds.to_parquet(local_path, index=False)
    print(f"  local: {local_path}")

    if storage.provider != "local":
        uri = storage.write_parquet(preds, "predictions.parquet")
        print(f"  cloud ({storage.provider}): {uri}")
    else:
        print(f"  cloud upload skipped (CLOUD_PROVIDER=local)")

    print(f"\nDone. {len(preds):,} predictions written.")


if __name__ == "__main__":
    main()
