"""Train propensity models — Logistic Regression baseline + LightGBM tuned with Optuna.

Temporal split (no random shuffling): train < day 510, val 510-619, test 620-730.
Target: incidencia_compra.

Run:
    python -m src.model.train
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.utils.io import ensure_dir

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = ensure_dir(PROJECT_ROOT / "models")
METRICS_DIR = ensure_dir(PROJECT_ROOT / "docs" / "metrics")

TARGET = "incidencia_compra"
TRAIN_DAYS_END = 510   # ~70%
VAL_DAYS_END = 620     # ~85%


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df.dia_visita < TRAIN_DAYS_END].copy()
    val = df[(df.dia_visita >= TRAIN_DAYS_END) & (df.dia_visita < VAL_DAYS_END)].copy()
    test = df[df.dia_visita >= VAL_DAYS_END].copy()
    return train, val, test


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> float:
    """Lift @ top-K: among the top-K% scored, what's the purchase rate vs. baseline."""
    n = len(y_true)
    cutoff = int(np.ceil(n * k))
    order = np.argsort(-y_score)
    top_rate = y_true[order[:cutoff]].mean()
    base_rate = y_true.mean()
    return float(top_rate / base_rate) if base_rate > 0 else float("nan")


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, label: str) -> dict:
    metrics = {
        "set": label,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "lift@10%": lift_at_k(y_true, y_proba, 0.10),
        "lift@20%": lift_at_k(y_true, y_proba, 0.20),
        "lift@30%": lift_at_k(y_true, y_proba, 0.30),
    }
    return metrics


def train_logreg(X_train, y_train, X_val, y_val) -> tuple[Pipeline, dict]:
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(X_train, y_train)
    proba_val = pipe.predict_proba(X_val)[:, 1]
    return pipe, evaluate(y_val.values, proba_val, "val")


def tune_lgbm(X_train, y_train, X_val, y_val, cat_idx: list[int], n_trials: int = 30) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
        }
        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_idx)
        val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_idx, reference=train_set)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        proba = booster.predict(X_val, num_iteration=booster.best_iteration)
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_lgbm_final(X_trval, y_trval, X_val, y_val, cat_idx, best_params) -> lgb.Booster:
    """Refit on train+val combined for final model (use val for early-stop signal still)."""
    params = {"objective": "binary", "metric": "auc", "verbosity": -1, **best_params}
    train_set = lgb.Dataset(X_trval, label=y_trval, categorical_feature=cat_idx)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_idx, reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return booster


def main() -> None:
    print("Loading features...")
    df = pd.read_parquet(PROCESSED / "features.parquet")
    print(f"  shape={df.shape}")

    train, val, test = temporal_split(df)
    print(f"  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    print(f"  train base rate: {train[TARGET].mean():.3f}  val: {val[TARGET].mean():.3f}  test: {test[TARGET].mean():.3f}")

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_val, y_val = val[FEATURE_COLUMNS], val[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    for c in CATEGORICAL_FEATURES:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = X_val[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    cat_idx = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES]

    # ============================================================
    # Baseline: Logistic Regression
    # ============================================================
    print("\n[1/3] Training Logistic Regression baseline...")
    X_train_lr = X_train.copy()
    X_val_lr = X_val.copy()
    X_test_lr = X_test.copy()
    for c in CATEGORICAL_FEATURES:
        X_train_lr[c] = X_train_lr[c].astype(int)
        X_val_lr[c] = X_val_lr[c].astype(int)
        X_test_lr[c] = X_test_lr[c].astype(int)
    lr_model, lr_val = train_logreg(X_train_lr, y_train, X_val_lr, y_val)
    lr_test = evaluate(y_test.values, lr_model.predict_proba(X_test_lr)[:, 1], "test")
    print(f"  LR val AUC={lr_val['auc_roc']:.4f}  test AUC={lr_test['auc_roc']:.4f}  lift@10={lr_test['lift@10%']:.2f}")

    # ============================================================
    # LightGBM with Optuna tuning
    # ============================================================
    print("\n[2/3] Tuning LightGBM with Optuna (30 trials)...")
    best_params = tune_lgbm(X_train, y_train, X_val, y_val, cat_idx, n_trials=30)
    print(f"  best params: {best_params}")

    booster = train_lgbm_final(X_train, y_train, X_val, y_val, cat_idx, best_params)
    val_proba = booster.predict(X_val, num_iteration=booster.best_iteration)
    test_proba = booster.predict(X_test, num_iteration=booster.best_iteration)
    lgb_val = evaluate(y_val.values, val_proba, "val")
    lgb_test = evaluate(y_test.values, test_proba, "test")
    print(f"  LightGBM val AUC={lgb_val['auc_roc']:.4f}  test AUC={lgb_test['auc_roc']:.4f}  lift@10={lgb_test['lift@10%']:.2f}")

    # ============================================================
    # Save
    # ============================================================
    print("\n[3/3] Saving models and metrics...")
    booster.save_model(str(MODELS_DIR / "lgbm_propensity.txt"))
    joblib.dump(lr_model, MODELS_DIR / "logreg_propensity.joblib")
    joblib.dump({"feature_columns": FEATURE_COLUMNS, "categorical": CATEGORICAL_FEATURES}, MODELS_DIR / "feature_schema.joblib")

    metrics = {
        "logreg": {"val": lr_val, "test": lr_test},
        "lightgbm": {"val": lgb_val, "test": lgb_test, "best_params": best_params, "best_iteration": int(booster.best_iteration)},
        "split": {"train_days_end": TRAIN_DAYS_END, "val_days_end": VAL_DAYS_END, "n_train": len(train), "n_val": len(val), "n_test": len(test)},
        "feature_count": len(FEATURE_COLUMNS),
    }
    (METRICS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== FINAL TEST METRICS ===")
    print(f"{'Metric':<15} {'LogReg':>10} {'LightGBM':>10}")
    for k in ["auc_roc", "pr_auc", "log_loss", "brier", "lift@10%", "lift@20%", "lift@30%"]:
        print(f"{k:<15} {lr_test[k]:>10.4f} {lgb_test[k]:>10.4f}")


if __name__ == "__main__":
    main()
