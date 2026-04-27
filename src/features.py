"""Feature engineering — strictly causal features per (customer, visit_day).

All historical aggregates use only data with `dia < dia_visita_actual` (no leakage).
Computed efficiently with groupby cumulative ops + shift, no Python loops.

Run:
    python -m src.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT
from src.utils.io import BRANDS, ensure_dir, load_raw

PROCESSED_DIR = ensure_dir(PROJECT_ROOT / "data" / "processed")


# ============================================================
# Customer-level causal aggregates (RFM, loyalty, promo sens.)
# ============================================================
def add_prior_visit_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, add aggregates computed over PRIOR visits of same customer."""
    df = df.sort_values(["id", "dia_visita"]).reset_index(drop=True).copy()
    grp = df.groupby("id", sort=False)

    df["prior_visits"] = grp.cumcount()
    df["prior_purchases"] = grp["incidencia_compra"].cumsum() - df["incidencia_compra"]
    df["prior_buy_rate"] = np.where(df["prior_visits"] > 0, df["prior_purchases"] / df["prior_visits"], 0.0)

    df["_qty"] = df["cantidad"]
    df["prior_qty"] = grp["_qty"].cumsum() - df["_qty"]
    df["prior_avg_qty_per_purchase"] = np.where(df["prior_purchases"] > 0, df["prior_qty"] / df["prior_purchases"], 0.0)

    purchase_day = df["dia_visita"].where(df["incidencia_compra"] == 1)
    df["_last_purchase_day"] = grp[purchase_day.name if hasattr(purchase_day, "name") else None].transform(
        lambda s: s
    )
    last_purchase_day = (
        df.assign(_p=df["dia_visita"].where(df["incidencia_compra"] == 1))
        .groupby("id")["_p"]
        .ffill()
        .shift(1)
    )
    last_purchase_day = last_purchase_day.where(df.groupby("id").cumcount() > 0)
    df["recency_days"] = (df["dia_visita"] - last_purchase_day).fillna(-1).astype(float)
    df = df.drop(columns=["_qty", "_last_purchase_day"])

    return df


def add_brand_loyalty(df: pd.DataFrame) -> pd.DataFrame:
    """Per (customer, brand): prior fraction of customer's purchases that went to brand b."""
    grp = df.groupby("id", sort=False)
    for b in BRANDS:
        bought_b = ((df["id_marca"] == b) & (df["incidencia_compra"] == 1)).astype(int)
        cum_b = bought_b.groupby(df["id"]).cumsum() - bought_b
        df[f"prior_buys_b{b}"] = cum_b
        df[f"loyalty_b{b}"] = np.where(df["prior_purchases"] > 0, cum_b / df["prior_purchases"], 0.0)
    return df


def add_promo_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """How much does this customer respond to promotions historically?"""
    promo_cols = [f"promo_marca_{b}" for b in BRANDS]
    df["any_promo_today"] = (df[promo_cols].sum(axis=1) > 0).astype(int)

    grp_id = df.groupby("id", sort=False)
    df["prior_visits_with_promo"] = grp_id["any_promo_today"].cumsum() - df["any_promo_today"]

    bought_on_promo = (df["incidencia_compra"] * df["any_promo_today"]).astype(int)
    df["prior_purchases_on_promo"] = bought_on_promo.groupby(df["id"]).cumsum() - bought_on_promo

    df["prior_visits_no_promo"] = df["prior_visits"] - df["prior_visits_with_promo"]
    df["prior_purchases_no_promo"] = df["prior_purchases"] - df["prior_purchases_on_promo"]

    df["buy_rate_with_promo"] = np.where(
        df["prior_visits_with_promo"] > 0, df["prior_purchases_on_promo"] / df["prior_visits_with_promo"], 0.0
    )
    df["buy_rate_no_promo"] = np.where(
        df["prior_visits_no_promo"] > 0, df["prior_purchases_no_promo"] / df["prior_visits_no_promo"], 0.0
    )
    df["promo_uplift"] = df["buy_rate_with_promo"] - df["buy_rate_no_promo"]
    return df


def add_relative_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Today's price for each brand vs. historical average (causal).

    Same prices apply to all customers on a given day, so we compute at day level.
    """
    price_cols = [f"precio_marca_{b}" for b in BRANDS]
    daily_prices = df.groupby("dia_visita")[price_cols].first().sort_index()
    historical_avg = daily_prices.expanding().mean().shift(1)
    df = df.merge(
        historical_avg.add_suffix("_hist_avg").reset_index(),
        on="dia_visita",
        how="left",
    )
    for b in BRANDS:
        col = f"precio_marca_{b}"
        hist = f"{col}_hist_avg"
        df[f"rel_price_b{b}"] = np.where(df[hist].notna(), df[col] / df[hist] - 1, 0.0)
    df = df.drop(columns=[f"{c}_hist_avg" for c in price_cols])

    df["min_rel_price_today"] = df[[f"rel_price_b{b}" for b in BRANDS]].min(axis=1)
    df["pct_brands_on_promo"] = df[[f"promo_marca_{b}" for b in BRANDS]].sum(axis=1) / len(BRANDS)
    return df


def add_loyal_brand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per row: which brand is the customer most loyal to historically? Is it on promo today?
    Is its price below historical average?
    """
    loyalty_cols = [f"loyalty_b{b}" for b in BRANDS]
    df["loyal_brand_id"] = df[loyalty_cols].values.argmax(axis=1) + 1
    no_history = df["prior_purchases"] == 0
    df.loc[no_history, "loyal_brand_id"] = 0

    promo_for_loyal = np.zeros(len(df))
    relprice_for_loyal = np.zeros(len(df))
    for b in BRANDS:
        mask = df["loyal_brand_id"] == b
        promo_for_loyal[mask] = df.loc[mask, f"promo_marca_{b}"].values
        relprice_for_loyal[mask] = df.loc[mask, f"rel_price_b{b}"].values
    df["loyal_brand_on_promo"] = promo_for_loyal.astype(int)
    df["loyal_brand_rel_price"] = relprice_for_loyal
    return df


# ============================================================
# Customer-level segmentation (for the AGENT to filter on)
# Computed at end of period — used for filtering, NOT model input.
# ============================================================
def compute_customer_segments(df: pd.DataFrame, n_clusters: int = 4, seed: int = 42) -> pd.DataFrame:
    """K-Means on customer-level RFM + demographics. Returns DataFrame with id + cluster + label."""
    cust = (
        df.groupby("id")
        .agg(
            total_visits=("dia_visita", "count"),
            total_purchases=("incidencia_compra", "sum"),
            total_qty=("cantidad", "sum"),
            edad=("edad", "first"),
            ingreso_anual=("ingreso_anual", "first"),
            genero=("genero", "first"),
            estado_civil=("estado_civil", "first"),
            nivel_educacion=("nivel_educacion", "first"),
        )
        .reset_index()
    )
    cust["buy_rate"] = cust["total_purchases"] / cust["total_visits"]
    cust["avg_qty_per_purchase"] = np.where(cust["total_purchases"] > 0, cust["total_qty"] / cust["total_purchases"], 0)

    feats = ["buy_rate", "total_purchases", "avg_qty_per_purchase", "edad", "ingreso_anual"]
    X = StandardScaler().fit_transform(cust[feats])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    cust["cluster_id"] = km.fit_predict(X)

    cluster_stats = cust.groupby("cluster_id").agg(
        avg_buy_rate=("buy_rate", "mean"),
        avg_purchases=("total_purchases", "mean"),
        avg_income=("ingreso_anual", "mean"),
        avg_age=("edad", "mean"),
    )
    label_map = label_clusters(cluster_stats)
    cust["cluster_label"] = cust["cluster_id"].map(label_map)
    return cust[["id", "cluster_id", "cluster_label", "buy_rate", "total_purchases", "avg_qty_per_purchase"]]


def label_clusters(stats: pd.DataFrame) -> dict[int, str]:
    """Heuristic labels based on buy_rate + income."""
    labels = {}
    sorted_by_rate = stats.sort_values("avg_buy_rate", ascending=False).index.tolist()
    sorted_by_income = stats.sort_values("avg_income", ascending=False).index.tolist()
    rate_rank = {c: i for i, c in enumerate(sorted_by_rate)}
    income_rank = {c: i for i, c in enumerate(sorted_by_income)}
    for c in stats.index:
        if rate_rank[c] == 0:
            labels[c] = "Leales premium" if income_rank[c] <= 1 else "Leales recurrentes"
        elif rate_rank[c] == len(stats) - 1:
            labels[c] = "Visitantes ocasionales"
        else:
            labels[c] = "Cazadores de oferta" if income_rank[c] >= len(stats) - 2 else "Compradores moderados"
    return labels


# ============================================================
# Main entrypoint
# ============================================================
FEATURE_COLUMNS = (
    [
        "prior_visits",
        "prior_purchases",
        "prior_buy_rate",
        "prior_qty",
        "prior_avg_qty_per_purchase",
        "recency_days",
        "buy_rate_with_promo",
        "buy_rate_no_promo",
        "promo_uplift",
        "any_promo_today",
        "pct_brands_on_promo",
        "min_rel_price_today",
        "loyal_brand_id",
        "loyal_brand_on_promo",
        "loyal_brand_rel_price",
        "ultima_marca_comprada",
        "ultima_cantidad_comprada",
        "edad",
        "ingreso_anual",
        "genero",
        "estado_civil",
        "nivel_educacion",
        "ocupacion",
    ]
    + [f"loyalty_b{b}" for b in BRANDS]
    + [f"rel_price_b{b}" for b in BRANDS]
    + [f"precio_marca_{b}" for b in BRANDS]
    + [f"promo_marca_{b}" for b in BRANDS]
)

CATEGORICAL_FEATURES = [
    "genero",
    "estado_civil",
    "nivel_educacion",
    "ocupacion",
    "ultima_marca_comprada",
    "loyal_brand_id",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_prior_visit_aggregates(df)
    df = add_brand_loyalty(df)
    df = add_promo_sensitivity(df)
    df = add_relative_prices(df)
    df = add_loyal_brand_features(df)
    return df


def main() -> None:
    print("Loading raw data...")
    df = load_raw()
    print(f"  shape={df.shape}")

    print("Building features (causal)...")
    df_feat = build_features(df)
    print(f"  shape after features={df_feat.shape}")
    print(f"  feature columns: {len(FEATURE_COLUMNS)}")

    print("Computing customer segments (KMeans, k=4)...")
    segments = compute_customer_segments(df)
    print(segments.groupby("cluster_label").agg(n=("id", "count"), buy_rate=("buy_rate", "mean")))

    df_feat = df_feat.merge(segments[["id", "cluster_id", "cluster_label"]], on="id", how="left")

    out_path = PROCESSED_DIR / "features.parquet"
    df_feat.to_parquet(out_path, index=False)
    print(f"\nSaved features to {out_path}")

    seg_path = PROCESSED_DIR / "customer_segments.parquet"
    segments.to_parquet(seg_path, index=False)
    print(f"Saved segments to {seg_path}")

    print("\nSample feature row:")
    print(df_feat.iloc[1000][FEATURE_COLUMNS[:15]].to_string())


if __name__ == "__main__":
    main()
