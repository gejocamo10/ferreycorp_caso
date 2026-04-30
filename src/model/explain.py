"""Explicabilidad y graficos de negocio sobre el modelo LightGBM entrenado.

Genera:
  - Importancia global de features con SHAP (bar plot y beeswarm)
  - Curva de ganancia acumulada (lift curve)
  - Grafico de calibracion (probabilidad predicha vs. tasa real)
  - Matriz de confusion a un umbral elegido
  - Tabla de deciles para targeting (CSV)

Para ejecutar:
    python -m src.model.explain
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from src.config import PROJECT_ROOT
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.model.train import TARGET, VAL_DAYS_END, temporal_split
from src.utils.io import ensure_dir

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIG_DIR = ensure_dir(PROJECT_ROOT / "docs" / "model" / "figures")
ensure_dir(PROJECT_ROOT / "docs" / "model")


def fig_shap(booster: lgb.Booster, X_sample: pd.DataFrame) -> None:
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()


def fig_lift_curve(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p", ascending=False).reset_index(drop=True)
    df["rank_pct"] = (df.index + 1) / len(df) * 100
    df["cum_positives"] = df["y"].cumsum()
    df["pct_positives_captured"] = df["cum_positives"] / df["y"].sum() * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["rank_pct"], df["pct_positives_captured"], color="#2a9d8f", lw=2.5, label="Modelo LightGBM")
    ax.plot([0, 100], [0, 100], color="grey", ls="--", label="Random (baseline)")
    ax.fill_between(df["rank_pct"], df["pct_positives_captured"], df["rank_pct"], alpha=0.15, color="#2a9d8f")
    ax.set_xlabel("% de clientes contactados (orden por score)")
    ax.set_ylabel("% de compras capturadas")
    ax.set_title("Curva de Ganancia Acumulada (Lift Curve)")
    for k in [10, 20, 30]:
        idx = int(np.ceil(len(df) * k / 100)) - 1
        captured = df.loc[idx, "pct_positives_captured"]
        ax.scatter([k], [captured], color="#e76f51", zorder=5)
        ax.annotate(f"  Top {k}% captura {captured:.0f}% de compras", (k, captured), fontsize=9)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lift_curve.png", dpi=120)
    plt.close(fig)

    deciles = pd.qcut(-df["p"], q=10, labels=[f"D{i}" for i in range(1, 11)])
    decile_table = (
        df.assign(decile=deciles)
        .groupby("decile", observed=True)
        .agg(n=("y", "count"), positives=("y", "sum"), avg_score=("p", "mean"))
        .assign(rate=lambda d: d["positives"] / d["n"])
    )
    decile_table["lift_vs_base"] = decile_table["rate"] / df["y"].mean()
    return decile_table


def fig_calibration(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "o-", color="#2a9d8f", lw=2, label="LightGBM")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Calibración perfecta")
    ax.set_xlabel("Probabilidad predicha (media por bin)")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Calibración del modelo")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration.png", dpi=120)
    plt.close(fig)


def fig_confusion(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["Real 0", "Real 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_title(f"Matriz de confusión (umbral={threshold})")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion.png", dpi=120)
    plt.close(fig)
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1))}


def main() -> None:
    print("Cargando modelo y datos...")
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_propensity.txt"))
    df = pd.read_parquet(PROCESSED / "features.parquet")
    _, _, test = temporal_split(df)
    X_test, y_test = test[FEATURE_COLUMNS].copy(), test[TARGET].values
    for c in CATEGORICAL_FEATURES:
        X_test[c] = X_test[c].astype("category")

    proba = booster.predict(X_test)

    print("Generando graficos SHAP (muestra de 3000 filas)...")
    sample = X_test.sample(min(3000, len(X_test)), random_state=42)
    fig_shap(booster, sample)

    print("Generando curva de ganancia (lift)...")
    decile_table = fig_lift_curve(y_test, proba)
    decile_table.to_csv(PROJECT_ROOT / "docs" / "model" / "deciles.csv")
    print(decile_table.round(3))

    print("Generando grafico de calibracion...")
    fig_calibration(y_test, proba)

    print("Generando matriz de confusion...")
    cm_metrics = fig_confusion(y_test, proba, threshold=0.5)
    print(f"  precision={cm_metrics['precision']:.3f}  recall={cm_metrics['recall']:.3f}")

    summary = {
        "test_set_size": int(len(y_test)),
        "test_base_rate": float(y_test.mean()),
        "decile_lift_top10pct": float(decile_table.iloc[0]["lift_vs_base"]),
        "decile_lift_top20pct": float(decile_table.iloc[:2]["lift_vs_base"].mean()),
        "confusion_at_0.5": cm_metrics,
    }
    (PROJECT_ROOT / "docs" / "model" / "explainability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nListo. Figuras generadas en docs/model/figures/")


if __name__ == "__main__":
    main()
