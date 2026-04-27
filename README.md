# Ferreycorp · Modelo de Propensión de Compra + Agente IA

Solución end-to-end al caso técnico de Ferreycorp: modelo de propensión de compra (LightGBM con SHAP) + agente conversacional (Claude) que permite filtrar predicciones en lenguaje natural, sobre una arquitectura **cloud-agnóstica** lista para desplegar en DigitalOcean, AWS o GCP.

---

## 🏗️ Arquitectura

```
                                    ┌────────────────────────────┐
                                    │  Streamlit UI               │
                                    │  ├─ Chat con agente         │
                                    │  ├─ Dashboard predicciones  │
                                    │  └─ Performance del modelo  │
                                    └────────────┬───────────────┘
                                                 │
              ┌──────────────────────────────────┼─────────────────────────┐
              ▼                                  ▼                         ▼
   ┌────────────────────┐         ┌──────────────────────┐    ┌────────────────────┐
   │  PropensityAgent   │         │  Tabla predictions   │    │  LightGBM model    │
   │  (Claude + tools)  │ ◄────►  │  (Parquet)           │ ◄  │  (offline, batch)  │
   │                    │         │                      │    │                    │
   │  - query_predictions│         │  S3-compatible:     │    │  Features causales │
   │  - aggregate_predictions│     │  DO Spaces / AWS S3 │    │  (no leakage)      │
   │  - schema_info     │         │  GCS                 │    │                    │
   └────────────────────┘         └──────────────────────┘    └────────────────────┘
              │                              ▲
              │                              │
              ▼                              │
   ┌────────────────────┐                    │
   │  DuckDB (embedded) │  ── SQL parametrizado, lectura directa de Parquet
   └────────────────────┘
```

## 📁 Estructura del repo

```
ferreycorp_caso/
├── data/
│   ├── raw/                 # CSV original + diccionario
│   ├── processed/           # Features + segmentos (Parquet)
│   └── predictions/         # Tabla final que consume el agente
├── docker/Dockerfile        # Container portable (DO/AWS/GCP)
├── deploy/
│   ├── digitalocean/        # App Platform spec
│   ├── aws/                 # ECS / App Runner specs
│   └── gcp/                 # Cloud Run spec
├── docs/
│   ├── eda/                 # Reporte EDA + 11 figuras
│   ├── model/               # SHAP, lift curve, calibration
│   └── metrics/             # JSON con métricas finales
├── models/                  # LightGBM + LogReg baseline
├── src/
│   ├── config.py            # Config por env vars (no hardcoded)
│   ├── eda.py               # Pipeline EDA reproducible
│   ├── features.py          # Feature engineering causal
│   ├── model/
│   │   ├── train.py         # Entrenamiento con Optuna
│   │   ├── explain.py       # SHAP + lift + calibration
│   │   └── score.py         # Scoring batch → Parquet
│   ├── storage/
│   │   └── object_storage.py # Capa S3/GCS unificada
│   ├── agent/
│   │   ├── tools.py         # Herramientas DuckDB (whitelist)
│   │   ├── agent.py         # Claude + tool use loop
│   │   └── smoke_test.py    # Test del agente end-to-end
│   └── app.py               # Streamlit UI
├── requirements.txt
├── .env.example             # Plantilla de variables
└── README.md                # Este archivo
```

## 🚀 Quickstart local

### Una sola línea

```bash
bash scripts/run_all.sh
```

Ejecuta: setup venv → install deps → EDA → features → train → explain → score → tests → genera todos los PDFs.

### Paso a paso (si prefieres ir manualmente)

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edita .env y agrega tu ANTHROPIC_API_KEY

# 2. Pipeline ML (EDA → features → train → score)
python -m src.eda           # 11 figuras + reporte en docs/eda/
python -m src.features      # 43 features + segmentos K-Means
python -m src.model.train   # LogReg + LightGBM (Optuna 30 trials)
python -m src.model.explain # SHAP + lift curve + calibration
python -m src.model.score   # Genera predictions.parquet

# 3. Tests
python -m pytest tests/ -v   # 18 tests (no leakage + agente)

# 4. App
streamlit run src/app.py     # http://localhost:8501

# 5. (Opcional) test del agente desde CLI
python -m src.agent.smoke_test

# 6. Regenerar entregables PDF/TXT
python scripts/build_pdfs.py         # documento técnico, guía, EDA
python scripts/build_slides_pdf.py   # SLIDES.pdf (7 páginas A4 landscape)
python scripts/build_codigo_txt.py   # CODIGO_SOLUCION.txt
```

## 🌐 Despliegue cloud-agnóstico

El mismo `Dockerfile` y código corre en las tres nubes — solo cambia la configuración por env vars.

### DigitalOcean (recomendado para POC)

```bash
# Instalar doctl
brew install doctl
doctl auth init

# Crear Spaces para storage (equivalente S3)
doctl spaces create ferreycorp-predictions --region nyc3

# Subir el Parquet de predicciones
doctl spaces cp data/predictions/predictions.parquet s3://ferreycorp-predictions/

# Desplegar la app desde GitHub
doctl apps create --spec deploy/digitalocean/app.yaml
```

Costo estimado: **~$25/mes** (App Platform basic + Spaces + Managed Redis opcional).

### AWS

```bash
# Subir Parquet
aws s3 cp data/predictions/predictions.parquet s3://ferreycorp-predictions/

# Desplegar en ECS Fargate
aws ecs register-task-definition --cli-input-json file://deploy/aws/ecs-task.json
aws ecs create-service --cli-input-json file://deploy/aws/ecs-service.json

# Alternativa más simple: AWS App Runner
aws apprunner create-service --cli-input-json file://deploy/aws/apprunner.json
```

### GCP

```bash
# Subir Parquet
gsutil cp data/predictions/predictions.parquet gs://ferreycorp-predictions/

# Build + deploy en Cloud Run
gcloud builds submit --tag gcr.io/PROJECT/ferreycorp-app
gcloud run deploy ferreycorp-app --image gcr.io/PROJECT/ferreycorp-app \
    --env-vars-file deploy/gcp/cloudrun-env.yaml \
    --region us-central1 --allow-unauthenticated
```

## 🎯 Resultados clave

| Métrica | Baseline (LogReg) | **LightGBM** |
|---|---|---|
| AUC-ROC test | 0.673 | **0.684** |
| PR-AUC test | 0.458 | 0.457 |
| **Lift @ top 10%** | 2.36 | **2.40** |
| Lift @ top 20% | 1.87 | 1.90 |
| Brier score | 0.190 | **0.170** |

**Decil top (D1)**: 60.4% de tasa real vs. 25.2% baseline → **2.4× lift** para campañas dirigidas.

**Segmentación de clientes (K-Means k=4)**:
- Leales premium (47): 60% buy rate
- Cazadores de oferta (45): 29%
- Compradores moderados (139): 22%
- Visitantes ocasionales (269): 18%

## 📚 Entregables y documentación

| Archivo | Formato | Descripción |
|---|---|---|
| [docs/CODIGO_SOLUCION.txt](docs/CODIGO_SOLUCION.txt) | TXT | Código fuente consolidado (entregable del brief) |
| [docs/DOCUMENTO_TECNICO.pdf](docs/DOCUMENTO_TECNICO.pdf) | PDF | Documento técnico completo (objetivos, arquitectura, metodología, resultados) |
| [docs/SLIDES.pdf](docs/SLIDES.pdf) | PDF | Presentación de 7 slides (A4 landscape) |
| [docs/GUIA_APRENDIZAJE.md](docs/GUIA_APRENDIZAJE.md) / [.pdf](docs/GUIA_APRENDIZAJE.pdf) | MD/PDF | Guía explicativa paso a paso (uso interno) |
| [docs/eda/REPORTE_EDA.md](docs/eda/REPORTE_EDA.md) / [.pdf](docs/eda/REPORTE_EDA.pdf) | MD/PDF | Reporte de EDA con 11 figuras |
| [docs/metrics/model_metrics.json](docs/metrics/model_metrics.json) | JSON | Métricas finales del modelo |
| [docs/model/deciles.csv](docs/model/deciles.csv) | CSV | Tabla de deciles (lift por bucket) |

## 🔒 Seguridad de la solución

- El agente **no escribe SQL libre**: solo emite filtros estructurados, validados contra una whitelist de columnas y operadores antes de generar SQL parametrizado.
- Las credenciales de cloud storage y de la API de Claude **nunca se hardcodean** — siempre vienen de env vars.
- El Parquet es read-only desde la app; no hay path para que el agente modifique datos.
