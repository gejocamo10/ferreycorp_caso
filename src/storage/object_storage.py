"""Capa de almacenamiento agnostica del proveedor cloud.

Esta clase resuelve un problema concreto: si el codigo se escribe asumiendo un
proveedor especifico (digamos AWS S3), migrar a otra nube luego implica reescribir
mucho. Aqui abstraigo las diferencias detras de cuatro metodos publicos
(`write_parquet`, `read_parquet`, `get_duckdb_uri`, `configure_duckdb`) y el resto
del codigo no necesita saber en que nube esta corriendo.

Proveedores soportados:
  - local        : sistema de archivos local (modo desarrollo)
  - digitalocean : DO Spaces (API compatible con S3)
  - aws          : AWS S3
  - gcp          : Google Cloud Storage

El mismo archivo Parquet escrito aqui es consultable por DuckDB en cualquier
proveedor sin cambios en el codigo del agente.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import config

Provider = Literal["local", "digitalocean", "aws", "gcp"]


class ObjectStorage:
    """Interfaz unica para leer y escribir Parquet en distintos proveedores.

    Convencion de URIs:
      local        : path relativo o absoluto (./data/predictions/foo.parquet)
      digitalocean : s3://<bucket>/<key>  (usa el endpoint de DO)
      aws          : s3://<bucket>/<key>
      gcp          : gs://<bucket>/<key>
    """

    def __init__(self, provider: Provider | None = None):
        self.provider: Provider = provider or config.cloud_provider  # type: ignore

    # ------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------
    def write_parquet(self, df: pd.DataFrame, key: str) -> str:
        """Escribe el DataFrame como Parquet en el backend configurado y
        devuelve la URI final donde quedo el archivo.
        """
        uri = self._uri(key)
        if self.provider == "local":
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(uri, index=False)
        else:
            storage_options = self._fsspec_options()
            df.to_parquet(uri, index=False, storage_options=storage_options)
        return uri

    def read_parquet(self, key: str) -> pd.DataFrame:
        """Lee un Parquet desde el backend configurado."""
        uri = self._uri(key)
        if self.provider == "local":
            return pd.read_parquet(uri)
        return pd.read_parquet(uri, storage_options=self._fsspec_options())

    def get_duckdb_uri(self, key: str) -> str:
        """Devuelve una URI que DuckDB puede consultar via httpfs o fsspec."""
        return self._uri(key)

    def configure_duckdb(self, conn) -> None:
        """Inyecta credenciales de S3 o GCS en una conexion de DuckDB para que
        pueda leer Parquet remotos sin descargarlos primero.
        """
        if self.provider in ("digitalocean", "aws"):
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            if config.storage_endpoint:
                endpoint = config.storage_endpoint.replace("https://", "").replace("http://", "")
                conn.execute(f"SET s3_endpoint='{endpoint}';")
                conn.execute("SET s3_url_style='path';")
            if config.storage_region:
                conn.execute(f"SET s3_region='{config.storage_region}';")
            if config.storage_key:
                conn.execute(f"SET s3_access_key_id='{config.storage_key}';")
            if config.storage_secret:
                conn.execute(f"SET s3_secret_access_key='{config.storage_secret}';")
        elif self.provider == "gcp":
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            # GCS vía HMAC keys, equivalente al protocolo S3.
            if config.storage_key and config.storage_secret:
                conn.execute("SET s3_endpoint='storage.googleapis.com';")
                conn.execute(f"SET s3_access_key_id='{config.storage_key}';")
                conn.execute(f"SET s3_secret_access_key='{config.storage_secret}';")

    # ------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------
    def _uri(self, key: str) -> str:
        if self.provider == "local":
            base = config.local_predictions_path.parent
            return str(base / key)
        elif self.provider in ("digitalocean", "aws"):
            return f"s3://{config.storage_bucket}/{key}"
        elif self.provider == "gcp":
            return f"gs://{config.storage_bucket}/{key}"
        raise ValueError(f"Proveedor no soportado: {self.provider}")

    def _fsspec_options(self) -> dict:
        if self.provider == "digitalocean":
            return {
                "key": config.storage_key,
                "secret": config.storage_secret,
                "client_kwargs": {"endpoint_url": config.storage_endpoint},
            }
        elif self.provider == "aws":
            return {"key": config.storage_key, "secret": config.storage_secret}
        elif self.provider == "gcp":
            # gcsfs toma GOOGLE_APPLICATION_CREDENTIALS automaticamente.
            return {}
        return {}


storage = ObjectStorage()
