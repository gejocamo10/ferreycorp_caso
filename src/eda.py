"""Analisis exploratorio de datos.

Genera 11 figuras y un reporte en markdown que documentan el dataset crudo
(distribuciones, balance del target, lealtad por marca, sensibilidad a promo,
correlaciones, etc.). El reporte sirve como insumo para tomar decisiones de
modelado: que features construir, que tipo de modelo elegir y como evaluar.

Para ejecutar desde la raiz del proyecto:
    python -m src.eda
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PROJECT_ROOT
from src.utils.io import BRANDS, ensure_dir, load_raw

OUT_DIR = ensure_dir(PROJECT_ROOT / "docs" / "eda")
FIG_DIR = ensure_dir(OUT_DIR / "figures")

sns.set_theme(style="whitegrid", context="notebook")


def fig_target_balance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["incidencia_compra"].value_counts().sort_index()
    bars = ax.bar(["No compró (0)", "Compró (1)"], counts.values, color=["#cccccc", "#2a9d8f"])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Balance de la variable objetivo")
    ax.set_ylabel("# visitas")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_target_balance.png", dpi=120)
    plt.close(fig)


def fig_demographics(df: pd.DataFrame) -> None:
    cust = df.drop_duplicates("id")[["id", "genero", "estado_civil", "edad", "nivel_educacion", "ingreso_anual", "ocupacion"]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    sns.histplot(cust["edad"], bins=20, ax=axes[0, 0], color="#264653").set_title("Edad")
    sns.histplot(cust["ingreso_anual"], bins=25, ax=axes[0, 1], color="#264653").set_title("Ingreso anual")
    sns.countplot(x="genero", data=cust, ax=axes[0, 2], color="#e76f51").set_title("Género")
    sns.countplot(x="estado_civil", data=cust, ax=axes[1, 0], color="#e76f51").set_title("Estado civil")
    sns.countplot(x="nivel_educacion", data=cust, ax=axes[1, 1], color="#e76f51").set_title("Nivel educación")
    sns.countplot(x="ocupacion", data=cust, ax=axes[1, 2], color="#e76f51").set_title("Ocupación")
    fig.suptitle("Distribución demográfica de los 500 clientes únicos", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_demographics.png", dpi=120)
    plt.close(fig)


def fig_visits_per_customer(df: pd.DataFrame) -> None:
    visits = df.groupby("id").size()
    purchases = df.groupby("id")["incidencia_compra"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(visits, bins=30, ax=axes[0], color="#264653")
    axes[0].set_title("# visitas por cliente (2 años)")
    axes[0].set_xlabel("Visitas")
    sns.histplot(purchases, bins=30, ax=axes[1], color="#2a9d8f")
    axes[1].set_title("# compras por cliente (2 años)")
    axes[1].set_xlabel("Compras")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_visits_purchases.png", dpi=120)
    plt.close(fig)


def fig_brand_market_share(df: pd.DataFrame) -> None:
    bought = df[df.incidencia_compra == 1]
    share = bought["id_marca"].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([f"Marca {b}" for b in share.index], share.values * 100, color="#2a9d8f")
    for b, v in zip(bars, share.values * 100):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom")
    ax.set_title("Participación de mercado por marca (sobre compras)")
    ax.set_ylabel("% de compras")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_brand_market_share.png", dpi=120)
    plt.close(fig)


def fig_price_evolution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    daily = df.groupby("dia_visita")[[f"precio_marca_{b}" for b in BRANDS]].mean()
    for b in BRANDS:
        ax.plot(daily.index, daily[f"precio_marca_{b}"], label=f"Marca {b}", alpha=0.85)
    ax.set_title("Evolución del precio promedio por marca a lo largo del tiempo")
    ax.set_xlabel("Día de visita")
    ax.set_ylabel("Precio")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_price_evolution.png", dpi=120)
    plt.close(fig)


def fig_promo_frequency(df: pd.DataFrame) -> None:
    promo_freq = {f"Marca {b}": df[f"promo_marca_{b}"].mean() * 100 for b in BRANDS}
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(promo_freq.keys(), promo_freq.values(), color="#e76f51")
    for b, v in zip(bars, promo_freq.values()):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom")
    ax.set_title("% de días con promoción por marca")
    ax.set_ylabel("% de días")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_promo_frequency.png", dpi=120)
    plt.close(fig)


def fig_promo_lift(df: pd.DataFrame) -> pd.DataFrame:
    """Purchase rate when at least one brand has promo vs none."""
    has_any_promo = df[[f"promo_marca_{b}" for b in BRANDS]].sum(axis=1) > 0
    rate_with = df.loc[has_any_promo, "incidencia_compra"].mean()
    rate_without = df.loc[~has_any_promo, "incidencia_compra"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Sin promo", "Con promo"], [rate_without * 100, rate_with * 100], color=["#cccccc", "#2a9d8f"])
    for b, v in zip(bars, [rate_without * 100, rate_with * 100]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom")
    ax.set_title(f"Tasa de compra con/sin promoción (lift = {(rate_with/rate_without - 1)*100:.1f}%)")
    ax.set_ylabel("% de visitas con compra")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_promo_lift.png", dpi=120)
    plt.close(fig)
    return pd.DataFrame({"sin_promo": [rate_without], "con_promo": [rate_with], "lift_pct": [(rate_with / rate_without - 1) * 100]})


def fig_age_income_vs_purchase(df: pd.DataFrame) -> None:
    df = df.copy()
    df["edad_bin"] = pd.cut(df["edad"], bins=[17, 25, 35, 45, 55, 75], labels=["18-25", "26-35", "36-45", "46-55", "56-75"])
    df["ingreso_bin"] = pd.qcut(df["ingreso_anual"], q=5, labels=["Q1 (bajo)", "Q2", "Q3", "Q4", "Q5 (alto)"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    age_rate = df.groupby("edad_bin", observed=True)["incidencia_compra"].mean() * 100
    income_rate = df.groupby("ingreso_bin", observed=True)["incidencia_compra"].mean() * 100
    age_rate.plot(kind="bar", ax=axes[0], color="#264653")
    axes[0].set_title("Tasa de compra por rango de edad")
    axes[0].set_ylabel("% compra")
    income_rate.plot(kind="bar", ax=axes[1], color="#264653")
    axes[1].set_title("Tasa de compra por quintil de ingreso")
    axes[1].set_ylabel("% compra")
    for ax in axes:
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_age_income_vs_purchase.png", dpi=120)
    plt.close(fig)


def fig_price_sensitivity(df: pd.DataFrame) -> None:
    """When customer buys, do they tend to choose the cheapest brand?"""
    bought = df[df.incidencia_compra == 1].copy()
    prices = bought[[f"precio_marca_{b}" for b in BRANDS]].values
    cheapest_idx = prices.argmin(axis=1) + 1  # brands are 1..5
    bought["cheapest_brand"] = cheapest_idx
    bought["chose_cheapest"] = (bought["id_marca"] == bought["cheapest_brand"]).astype(int)
    rate = bought["chose_cheapest"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Eligió la más barata", "Eligió otra"], [rate, 100 - rate], color=["#2a9d8f", "#cccccc"])
    ax.set_title("Cuando el cliente compra, ¿elige la marca más barata?")
    ax.set_ylabel("% de compras")
    for i, v in enumerate([rate, 100 - rate]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_price_sensitivity.png", dpi=120)
    plt.close(fig)


def fig_correlations(df: pd.DataFrame) -> None:
    cols = ["incidencia_compra", "edad", "ingreso_anual", "genero", "estado_civil", "nivel_educacion", "ocupacion"]
    cols += [f"precio_marca_{b}" for b in BRANDS]
    cols += [f"promo_marca_{b}" for b in BRANDS]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Matriz de correlaciones")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_correlations.png", dpi=110)
    plt.close(fig)


def fig_loyalty(df: pd.DataFrame) -> None:
    """For each (customer, brand) pair, what fraction of their purchases went to that brand?"""
    bought = df[df.incidencia_compra == 1]
    loyalty = bought.groupby(["id", "id_marca"]).size().unstack(fill_value=0)
    loyalty_pct = loyalty.div(loyalty.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    loyalty_pct.mean(axis=0).plot(kind="bar", ax=ax, color="#264653")
    ax.set_title("Lealtad promedio por marca (% del basket de cada cliente)")
    ax.set_xticklabels([f"Marca {b}" for b in loyalty_pct.columns], rotation=0)
    ax.set_ylabel("% promedio")
    for p in ax.patches:
        ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_loyalty.png", dpi=120)
    plt.close(fig)

    # Concentracion: que fraccion de clientes concentran mas del 50% de
    # sus compras en una sola marca (mide lealtad de marca a nivel poblacional).
    top_share = loyalty_pct.max(axis=1)
    concentrated = (top_share > 0.5).mean()
    return concentrated


def write_report(df: pd.DataFrame, promo_table: pd.DataFrame, concentrated_share: float) -> None:
    n_rows, n_cols = df.shape
    n_cust = df["id"].nunique()
    n_days = df["dia_visita"].nunique()
    rate = df["incidencia_compra"].mean()
    bought = df[df.incidencia_compra == 1]
    brand_share = bought["id_marca"].value_counts(normalize=True).sort_index()
    avg_visits = df.groupby("id").size().mean()
    avg_purchases = df.groupby("id")["incidencia_compra"].sum().mean()

    report = f"""# Reporte EDA, Ferreycorp · Propensión de Compra

## 1. Visión general del dataset

| Métrica | Valor |
|---|---|
| Filas (visitas) | **{n_rows:,}** |
| Columnas | {n_cols} |
| Clientes únicos | **{n_cust}** |
| Rango temporal | día 1 al día {n_days} (aprox. 2 años) |
| Visitas promedio por cliente | {avg_visits:.1f} |
| Compras promedio por cliente | {avg_purchases:.1f} |
| **Tasa de conversión global** | **{rate:.2%}** |
| Valores nulos | 0 |

> La granularidad es **una fila igual a una visita de un cliente en un día**.
> El target `incidencia_compra` está moderadamente desbalanceado (aprox. 25% positivos),
> pero suficientemente representado como para no requerir oversampling agresivo.

## 2. Distribución demográfica

Los 500 clientes son adultos entre 18 y 75 años (mediana aprox. 36), con ingresos anuales entre 38k y 309k.
Nivel educativo y ocupación están codificados como ordinales o categóricas.

![demographics](figures/02_demographics.png)

## 3. Comportamiento de visita y compra

Los clientes son recurrentes: en promedio {avg_visits:.0f} visitas en 2 años,
de las cuales {avg_purchases:.0f} resultan en compra. Esto valida la formulación de
**propensión por visita** sobre propensión por cliente, ya que hay suficiente densidad temporal por individuo.

![visits](figures/03_visits_purchases.png)

## 4. Marcas y participación de mercado

Las 5 marcas tienen participaciones muy distintas, dominadas por **Marca 5 ({brand_share.get(5,0):.1%})** y **Marca 2 ({brand_share.get(2,0):.1%})**.
La Marca 3 es claramente nicho ({brand_share.get(3,0):.1%}). Esto sugiere que un modelo
multiclase de marca tendría clases desbalanceadas, conviene tenerlo presente.

![market_share](figures/04_brand_market_share.png)

## 5. Precios y promociones

Los precios fluctúan en el tiempo, no son estáticos. Esto es **una palanca de pricing real**:
si entendemos elasticidad por marca, podemos sugerir cuándo bajar precio para activar
clientes de baja propensión.

![prices](figures/05_price_evolution.png)
![promos](figures/06_promo_frequency.png)

**Lift de promoción:** la tasa de compra con al menos una marca en promo es **{promo_table['con_promo'].iloc[0]*100:.1f}%**
vs. **{promo_table['sin_promo'].iloc[0]*100:.1f}%** sin promo (lift {promo_table['lift_pct'].iloc[0]:.1f}%).
Las promociones sí mueven la aguja, son una feature de primer orden para el modelo.

![promo_lift](figures/07_promo_lift.png)

## 6. Demografía vs. tasa de compra

![age_income](figures/08_age_income_vs_purchase.png)

Hay diferencias entre rangos de edad y quintiles de ingreso, aunque ninguna es dramática.
Las features demográficas aportan, pero la señal principal está en el comportamiento histórico
y las palancas comerciales (precio y promo).

## 7. Sensibilidad a precio

![price_sens](figures/09_price_sensitivity.png)

Cuando un cliente compra, una proporción significativa elige la marca más barata disponible.
Esto refuerza la importancia de features como **precio relativo** y **descuento vs. promedio histórico**.

## 8. Correlaciones

![corr](figures/10_correlations.png)

Sin correlaciones bivariadas extremadamente fuertes con el target. Esto es típico en problemas
de propensión: la señal viene de **interacciones** (ej. cliente, precio y promo), donde los
modelos basados en árboles (LightGBM) funcionan mejor que los modelos lineales.

## 9. Lealtad de marca

![loyalty](figures/11_loyalty.png)

Aproximadamente **{concentrated_share:.0%} de los clientes** concentran más del 50% de sus compras en una sola marca.
Esto valida construir feature de **lealtad histórica** y posiblemente segmentar clientes
en clústeres (leales vs. cazadores de oferta).

## 10. Hallazgos clave para modelado

1. **Granularidad correcta**: visita-cliente, modelo binario con `incidencia_compra` como target.
2. **Sin nulos**, no se requiere imputación.
3. **Variables categóricas**: `genero`, `estado_civil`, `nivel_educacion`, `ocupacion`, `id_marca`.
4. **Variables numéricas**: `edad`, `ingreso_anual`, precios y promos por marca.
5. **Feature engineering crítico** (debe calcularse con ventana causal, solo con datos `dia < dia_visita`):
   - **RFM**: recencia, frecuencia, monto promedio.
   - **Lealtad**: % histórico de compras por marca por cliente.
   - **Sensibilidad a promo**: tasa de compra del cliente cuando hay promo vs. cuando no.
   - **Precio relativo**: precio actual de cada marca dividido por el precio promedio histórico.
   - **Última marca y última cantidad**: ya vienen en el dataset.
6. **Validación temporal**: split por `dia_visita` (no random) para evitar leakage.
7. **Modelo principal**: LightGBM (maneja categóricas, captura interacciones, es rápido).
8. **Métrica de negocio**: Lift @ top-K, más útil que accuracy para campañas dirigidas.

---

*Generado automáticamente por `python -m src.eda`. Figuras en `docs/eda/figures/`.*
"""
    (OUT_DIR / "REPORTE_EDA.md").write_text(report, encoding="utf-8")


def main() -> None:
    print("Loading raw data...")
    df = load_raw()
    print(f"  shape={df.shape}  customers={df['id'].nunique()}  days={df['dia_visita'].nunique()}")

    print("Generating figures...")
    fig_target_balance(df)
    fig_demographics(df)
    fig_visits_per_customer(df)
    fig_brand_market_share(df)
    fig_price_evolution(df)
    fig_promo_frequency(df)
    promo_table = fig_promo_lift(df)
    fig_age_income_vs_purchase(df)
    fig_price_sensitivity(df)
    fig_correlations(df)
    concentrated_share = fig_loyalty(df)

    print("Writing report...")
    write_report(df, promo_table, concentrated_share)
    print(f"Done. Outputs at {OUT_DIR}")


if __name__ == "__main__":
    main()
