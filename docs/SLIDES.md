# Slides Ejecutivos — Caso Ferreycorp (≤ 7 slides)

> Convertir a PDF: abrir en VS Code con extensión "Markdown PDF" o usar pandoc:
> `pandoc docs/SLIDES.md -o docs/SLIDES.pdf -t beamer`

---

## Slide 1 — Objetivos

# Modelo de Propensión de Compra + Agente IA

**Objetivo de negocio**
Predecir, en cada visita, qué clientes comprarán — y permitir al equipo comercial **filtrar y explorar** las predicciones en lenguaje natural.

**Objetivos técnicos**
1. AUC-ROC ≥ 0.65 · Lift @ top-10% ≥ 2.0
2. Modelo explicable (SHAP) global y por cliente
3. Agente conversacional sobre predicciones (acceso < 100 ms)
4. Arquitectura **portable cloud** (DigitalOcean / AWS / GCP)

---

## Slide 2 — Cronograma

| Semana | Entregable |
|---|---|
| **S1** | EDA + diseño de features causales |
| **S2** | Modelo (baseline + LightGBM + tuning) |
| **S3** | Explicabilidad + agente IA |
| **S4** | UI Streamlit + despliegue cloud + documentación |

**Stack**: Python · LightGBM · Optuna · SHAP · DuckDB · Anthropic API · Streamlit · Docker

---

## Slide 3 — EDA · Hallazgos clave

**Dataset**: 58,693 visitas · 500 clientes · 730 días · **25%** de conversión.

| Hallazgo | Implicación |
|---|---|
| **89%** de clientes concentran >50% de compras en una marca | Lealtad histórica = feature top |
| **+23%** de lift cuando hay promo (26.9% vs 21.8%) | Promo es palanca de primer orden |
| Marcas 5 y 2 dominan (**65%** combinado), Marca 3 nicho (5.7%) | Multiclase de marca tendría desbalance |
| Sin nulos · Precios fluctúan en el tiempo | Feature de precio relativo viable |
| Cliente promedio: 117 visitas / 29 compras en 2 años | Densidad temporal suficiente para features causales |

---

## Slide 4 — Metodología

**Pipeline**

```
EDA → Feature Engineering Causal → Train Temporal Split → Tuning → Score → Agent
```

**Features (43)** — todos calculados con info de días anteriores, sin leakage:
- **RFM**: recencia, frecuencia, ticket promedio
- **Lealtad**: % histórico de compras por marca (5 cols)
- **Sensibilidad a promo**: tasa de compra con/sin promo, uplift personal
- **Precio relativo**: precio de hoy vs. promedio histórico por marca
- **Demografía + cluster**: K-Means k=4 → 4 segmentos de cliente

**Modelos**: Logistic Regression (baseline) → **LightGBM + Optuna (30 trials)**

**Validación**: split temporal · métrica objetivo: AUC-ROC + Lift @ top-K

---

## Slide 5 — Resultados

**Métricas test (LightGBM)**

| AUC-ROC | PR-AUC | Brier | Lift @ top 10% | Lift @ top 20% |
|---|---|---|---|---|
| **0.684** | 0.457 | 0.170 | **2.40×** | 1.90× |

**Análisis por decil — la cifra que cuenta al negocio**

| Decil | n | Score predicho | Tasa real |
|---|---|---|---|
| **D1 (top 10%)** | 5.9k | 0.50 | **69.6%** |
| D2 | 5.9k | 0.33 | 42.7% |
| D5 | 5.9k | 0.23 | 20.1% |
| D10 (bottom 10%) | 5.9k | 0.16 | 7.0% |

→ Llamando solo al top 20%, capturas **~38% de las compras totales**.

**Top features (SHAP)**: prior_buy_rate · prior_purchases · recency_days · min_rel_price_today

---

## Slide 6 — Arquitectura

```
   ┌────────────────────────────────────────┐
   │  Streamlit UI (chat + dashboard)        │
   └──────────────────┬─────────────────────┘
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
   ┌─────────────┐        ┌────────────────┐
   │  Claude     │ ◄────► │  predictions   │
   │  + tools    │        │  Parquet on    │
   │  (DuckDB    │        │  S3-compatible │
   │   queries)  │        │  storage       │
   └─────────────┘        └────────────────┘
```

**Cloud-agnóstico** — el mismo Docker corre en DO / AWS / GCP, sólo cambia el `.env`.

| Componente | DigitalOcean | AWS | GCP |
|---|---|---|---|
| Storage | Spaces | S3 | GCS |
| Compute | App Platform | ECS Fargate | Cloud Run |
| LLM | Anthropic API | Anthropic API | Anthropic API |

**Seguridad**: agente NO escribe SQL libre — emite filtros estructurados que se validan contra whitelist y se traducen a SQL parametrizado.

---

## Slide 7 — Captura de valor

**Áreas de impacto**

| Área | Palanca | KPI clave |
|---|---|---|
| 📣 Marketing dirigido | Cortar contactos a bottom-50% | Costo por compra: **-50%** |
| 🏷️ Pricing & promo | Personalizar promos por segmento | Lift incremental por promo |
| ♻️ Reactivación | Detectar baja-propensión activa | Tasa de reactivación: +10pp |
| ⚡ Operación comercial | Self-service vía chat | Time-to-insight: horas → segundos |

**Modelo económico simplificado**
Campaña al top-20% por score:
- ⬇ **80%** menos costo de contacto
- ✅ Mantiene **38%** de la efectividad de la campaña masiva
- → Rentabilidad por contacto **~5×** vs. baseline

**Roadmap próximas fases**: modelo de marca · A/B testing · monitoreo drift · pricing optimizer
