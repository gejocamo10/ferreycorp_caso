#!/usr/bin/env bash
# Pipeline completo end-to-end: EDA → features → train → explain → score → docs
# Uso: bash scripts/run_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d ".venv" ]]; then
    echo "==> Creando venv..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Instalando dependencias..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if [[ ! -f ".env" ]]; then
    echo "==> Copiando .env.example -> .env (recordá agregar tu ANTHROPIC_API_KEY)"
    cp .env.example .env
fi

echo "==> [1/7] EDA..."
python -m src.eda

echo "==> [2/7] Feature engineering..."
python -m src.features

echo "==> [3/7] Entrenamiento (LogReg + LightGBM + Optuna)..."
python -m src.model.train

echo "==> [4/7] Explicabilidad (SHAP + lift curve + calibración)..."
python -m src.model.explain

echo "==> [5/7] Scoring batch → predictions.parquet..."
python -m src.model.score

echo "==> [6/7] Tests (pytest)..."
python -m pytest tests/ -q

echo "==> [7/7] Generando documentos PDF..."
python scripts/build_pdfs.py
python scripts/build_slides_pdf.py
python scripts/build_codigo_txt.py

echo ""
echo "✅ Pipeline completo. Para arrancar la UI:"
echo "    source .venv/bin/activate"
echo "    streamlit run src/app.py"
