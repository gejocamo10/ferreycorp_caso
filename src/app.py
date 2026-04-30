"""Aplicacion Streamlit del producto.

Tres vistas:
  - Chat con el agente conversacional (preguntas en lenguaje natural).
  - Dashboard de predicciones (KPIs y graficos por decil y segmento).
  - Modelo (metricas de performance, hiperparametros, SHAP, calibracion).

Para ejecutar:
    streamlit run src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit ejecuta este archivo como script y no como modulo, por lo que la
# raiz del proyecto no esta en sys.path y los imports `from src.X` fallarian.
# Inserto la raiz manualmente para que la app funcione sin importar desde
# donde se invoque streamlit.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from src.agent.agent import PropensityAgent
from src.agent.tools import _predictions_uri
from src.config import config

st.set_page_config(page_title="Ferreycorp · Propensión de Compra", layout="wide", page_icon="🎯")


# ============================================================
# Data
# ============================================================
@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    return duckdb.sql(f"SELECT * FROM read_parquet('{_predictions_uri()}')").df()


@st.cache_resource
def get_agent() -> PropensityAgent | None:
    if not config.anthropic_api_key:
        return None
    return PropensityAgent()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("🎯 Ferreycorp")
    st.caption("Propensión de Compra · Modelo + Agente IA")
    st.divider()
    st.markdown(f"**Cloud provider**: `{config.cloud_provider}`")
    st.markdown(f"**Modelo LLM**: `{config.anthropic_model}`")
    if not config.anthropic_api_key:
        st.error("⚠️ Agrega `ANTHROPIC_API_KEY` en `.env` para activar el chat.")
    else:
        st.success("✅ Agente conectado")
    st.divider()
    page = st.radio("Navegación", ["💬 Chat", "📊 Dashboard", "🧠 Modelo"])

# ============================================================
# Pages
# ============================================================
preds = load_predictions()


def page_chat() -> None:
    st.header("💬 Chat con el agente comercial")
    st.caption(
        "Pregúntame sobre las predicciones de compra: filtros por segmento, "
        "edad, ingreso, decil, promo, etc. Ejemplos abajo."
    )

    examples = [
        "Dame los 20 clientes con mayor score que sean Leales premium",
        "¿Cuál es el score promedio por segmento?",
        "¿Cuántos clientes en el decil top tienen ingreso > 150k?",
        "Lista clientes con score > 0.6 que su marca leal es 5",
    ]
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        if c.button(ex, use_container_width=True):
            st.session_state["pending_input"] = ex

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "trace" in msg and msg["trace"]:
                with st.expander(f"🔍 {sum(1 for s in msg['trace'] if s['kind']=='tool_call')} tool call(s)"):
                    for step in msg["trace"]:
                        if step["kind"] == "tool_call":
                            st.markdown(f"**llamada a {step['name']}**")
                            st.json(step["payload"], expanded=False)
                        elif step["kind"] == "tool_result":
                            st.markdown(f"**← resultado de {step['name']}**")
                            payload = step["payload"]
                            if isinstance(payload, dict) and "rows" in payload:
                                st.caption(f"{payload.get('row_count','?')} filas")
                                if payload["rows"]:
                                    st.dataframe(pd.DataFrame(payload["rows"]), use_container_width=True)
                            else:
                                st.json(payload, expanded=False)

    user_input = st.chat_input("Escribe tu pregunta...")
    if not user_input and "pending_input" in st.session_state:
        user_input = st.session_state.pop("pending_input")

    if user_input:
        agent = get_agent()
        if agent is None:
            st.error("Agente no inicializado: falta ANTHROPIC_API_KEY en el .env.")
            return
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Consultando..."):
                try:
                    out = agent.chat(user_input, history=st.session_state.history)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
            st.markdown(out["reply"])
            st.session_state.messages.append({
                "role": "assistant", "content": out["reply"], "trace": out["trace"]
            })
            st.session_state.history = out["messages"]
            st.rerun()


def page_dashboard() -> None:
    st.header("📊 Dashboard de predicciones")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total predicciones", f"{len(preds):,}")
    c2.metric("Clientes únicos", f"{preds['id'].nunique():,}")
    c3.metric("Score promedio", f"{preds['score_compra'].mean():.3f}")
    c4.metric("Tasa real (validación)", f"{preds['incidencia_compra'].mean():.1%}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribución de scores")
        fig = px.histogram(preds, x="score_compra", nbins=40,
                           color_discrete_sequence=["#2a9d8f"])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Score predicho vs. tasa real por decil")
        decile_stats = (preds.groupby("decile", observed=True)
                        .agg(avg_score=("score_compra", "mean"),
                             actual_rate=("incidencia_compra", "mean"),
                             n=("id", "count"))
                        .reset_index()
                        .sort_values("decile"))
        fig = px.bar(decile_stats, x="decile", y=["avg_score", "actual_rate"], barmode="group",
                     labels={"value": "Tasa", "variable": ""},
                     color_discrete_sequence=["#264653", "#2a9d8f"])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Distribución por segmento de cliente")
        seg_stats = (preds.groupby("cluster_label")
                     .agg(n=("id", "count"),
                          avg_score=("score_compra", "mean"),
                          actual_rate=("incidencia_compra", "mean"))
                     .reset_index()
                     .sort_values("avg_score", ascending=False))
        fig = px.bar(seg_stats, x="cluster_label", y="avg_score",
                     color="actual_rate", color_continuous_scale="Teal",
                     hover_data=["n", "actual_rate"])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Score por marca leal histórica")
        brand_stats = (preds[preds.loyal_brand_id > 0]
                       .groupby("loyal_brand_id")
                       .agg(avg_score=("score_compra", "mean"),
                            n=("id", "count"))
                       .reset_index())
        brand_stats["loyal_brand_id"] = brand_stats["loyal_brand_id"].astype(str).map(lambda b: f"Marca {b}")
        fig = px.bar(brand_stats, x="loyal_brand_id", y="avg_score",
                     color_discrete_sequence=["#e76f51"])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Explorador interactivo")
    f1, f2, f3 = st.columns(3)
    seg_filter = f1.multiselect("Segmento", sorted(preds["cluster_label"].unique()), default=[])
    decile_filter = f2.multiselect("Decil", sorted(preds["decile"].unique()), default=[])
    score_min = f3.slider("Score mínimo", 0.0, 1.0, 0.0, 0.05)

    df_view = preds.copy()
    if seg_filter:
        df_view = df_view[df_view["cluster_label"].isin(seg_filter)]
    if decile_filter:
        df_view = df_view[df_view["decile"].isin(decile_filter)]
    df_view = df_view[df_view["score_compra"] >= score_min]
    st.caption(f"{len(df_view):,} filas tras filtros")
    st.dataframe(df_view.head(500), use_container_width=True, height=300)


def page_modelo() -> None:
    st.header("🧠 Performance del modelo")
    metrics_path = Path("docs/metrics/model_metrics.json")
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        lgb_test = m["lightgbm"]["test"]

        # KPIs principales arriba
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC-ROC (test)", f"{lgb_test['auc_roc']:.3f}")
        c2.metric("PR-AUC (test)", f"{lgb_test['pr_auc']:.3f}")
        c3.metric("Lift @ top 10%", f"{lgb_test['lift@10%']:.2f}x")
        c4.metric("Brier score", f"{lgb_test['brier']:.3f}")

        st.divider()
        st.subheader("Comparativa de modelos")

        # Tabla comparativa val + test, LogReg vs LightGBM
        rows = []
        labels = {
            "auc_roc": "AUC-ROC",
            "pr_auc": "PR-AUC",
            "log_loss": "Log loss",
            "brier": "Brier score",
            "lift@10%": "Lift @ top 10%",
            "lift@20%": "Lift @ top 20%",
            "lift@30%": "Lift @ top 30%",
        }
        for key, label in labels.items():
            rows.append({
                "Métrica": label,
                "LogReg (val)": f"{m['logreg']['val'][key]:.3f}",
                "LogReg (test)": f"{m['logreg']['test'][key]:.3f}",
                "LightGBM (val)": f"{m['lightgbm']['val'][key]:.3f}",
                "LightGBM (test)": f"{m['lightgbm']['test'][key]:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Lift @ top-K mide cuántas veces más compradores hay en el top-K% por score vs. la población general. Es la métrica que más importa para campañas dirigidas.")

        # Configuración del split
        st.divider()
        st.subheader("Configuración del experimento")
        split = m.get("split", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas train", f"{split.get('n_train', 0):,}")
        c2.metric("Filas validación", f"{split.get('n_val', 0):,}")
        c3.metric("Filas test", f"{split.get('n_test', 0):,}")
        st.caption(
            f"Split temporal: train hasta día {split.get('train_days_end')}, "
            f"validación hasta día {split.get('val_days_end')}, test el resto. "
            f"Total de features: {m.get('feature_count', 'N/D')}."
        )

        # Hiperparametros encontrados por Optuna
        best_params = m["lightgbm"].get("best_params", {})
        if best_params:
            st.divider()
            st.subheader("Hiperparámetros optimizados (Optuna, 30 trials)")
            param_rows = []
            descriptions = {
                "num_leaves": "Cantidad máxima de hojas por árbol",
                "max_depth": "Profundidad máxima de cada árbol",
                "learning_rate": "Tasa de aprendizaje (paso del boosting)",
                "feature_fraction": "Fracción de features muestreadas por árbol",
                "bagging_fraction": "Fracción de filas muestreadas por árbol",
                "bagging_freq": "Cada cuántas iteraciones se hace bagging",
                "min_child_samples": "Mínimo de muestras en una hoja",
                "lambda_l1": "Regularización L1 (sparsity)",
                "lambda_l2": "Regularización L2 (suavidad)",
            }
            for k, v in best_params.items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                param_rows.append({
                    "Hiperparámetro": k,
                    "Valor óptimo": val,
                    "Qué controla": descriptions.get(k, ""),
                })
            param_rows.append({
                "Hiperparámetro": "best_iteration",
                "Valor óptimo": str(m["lightgbm"].get("best_iteration", "N/D")),
                "Qué controla": "Cantidad de árboles construidos antes de early stopping",
            })
            st.dataframe(pd.DataFrame(param_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Curva de ganancia (Lift Curve)")
    img = Path("docs/model/figures/lift_curve.png")
    if img.exists():
        st.image(str(img), use_column_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Importancia de features (SHAP)")
        img = Path("docs/model/figures/shap_importance.png")
        if img.exists():
            st.image(str(img), use_column_width=True)
    with col_b:
        st.subheader("Calibración")
        img = Path("docs/model/figures/calibration.png")
        if img.exists():
            st.image(str(img), use_column_width=True)


# Router
if page == "💬 Chat":
    page_chat()
elif page == "📊 Dashboard":
    page_dashboard()
elif page == "🧠 Modelo":
    page_modelo()
