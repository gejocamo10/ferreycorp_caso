"""Configuracion centralizada del proyecto, leida desde variables de entorno.

Toda la configuracion vive aqui (proveedor cloud, credenciales, paths,
modelo de Claude). El resto del codigo solo importa la instancia `config`.

Cambiar de DigitalOcean a AWS o GCP no requiere tocar codigo, solo se
ajustan variables en el archivo .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cargo el .env con path absoluto para que funcione sin importar desde donde
# se invoque el codigo (Streamlit, scripts, tests o contenedores).
# Uso override=True para que las variables del .env del proyecto tengan
# prioridad sobre variables vacias heredadas del shell del usuario.
load_dotenv(PROJECT_ROOT / ".env", override=True)


@dataclass
class Config:
    cloud_provider: str = os.getenv("CLOUD_PROVIDER", "local")

    storage_bucket: str = os.getenv("STORAGE_BUCKET", "ferreycorp-predictions")
    storage_endpoint: str | None = os.getenv("STORAGE_ENDPOINT") or None
    storage_region: str = os.getenv("STORAGE_REGION", "us-east-1")
    storage_key: str | None = os.getenv("STORAGE_KEY") or None
    storage_secret: str | None = os.getenv("STORAGE_SECRET") or None

    local_data_dir: Path = Path(os.getenv("LOCAL_DATA_DIR", str(PROJECT_ROOT / "data")))
    local_predictions_path: Path = Path(
        os.getenv("LOCAL_PREDICTIONS_PATH", str(PROJECT_ROOT / "data" / "predictions" / "predictions.parquet"))
    )

    database_url: str = os.getenv("DATABASE_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "")

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
