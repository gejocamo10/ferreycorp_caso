"""Centralized configuration loaded from environment variables.

Switching cloud providers (DigitalOcean, AWS, GCP) only requires changing
env vars, never application code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Carga explicita del .env desde la raiz del proyecto para que funcione sin
# importar el cwd (Streamlit, scripts, tests, contenedores).
# override=True para que el .env del proyecto gane sobre variables vacias
# heredadas del shell.
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
