"""Cloud-agnostic object storage layer.

Supports:
  - local      : write/read on local filesystem (dev mode)
  - digitalocean : DO Spaces (S3-compatible API)
  - aws        : AWS S3
  - gcp        : Google Cloud Storage

Switching providers requires only changing env vars — no code change.
The same Parquet file produced here is queryable by DuckDB across all providers.
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
    """Unified read/write interface for Parquet files across cloud providers.

    URI conventions:
      local       → relative or absolute path (./data/predictions/foo.parquet)
      digitalocean→ s3://<bucket>/<key>          (uses DO endpoint)
      aws         → s3://<bucket>/<key>
      gcp         → gs://<bucket>/<key>
    """

    def __init__(self, provider: Provider | None = None):
        self.provider: Provider = provider or config.cloud_provider  # type: ignore

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def write_parquet(self, df: pd.DataFrame, key: str) -> str:
        """Write DataFrame as Parquet to the configured backend. Returns final URI."""
        uri = self._uri(key)
        if self.provider == "local":
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(uri, index=False)
        else:
            storage_options = self._fsspec_options()
            df.to_parquet(uri, index=False, storage_options=storage_options)
        return uri

    def read_parquet(self, key: str) -> pd.DataFrame:
        uri = self._uri(key)
        if self.provider == "local":
            return pd.read_parquet(uri)
        return pd.read_parquet(uri, storage_options=self._fsspec_options())

    def get_duckdb_uri(self, key: str) -> str:
        """Returns a URI that DuckDB can query via httpfs / fsspec."""
        return self._uri(key)

    def configure_duckdb(self, conn) -> None:
        """Inject S3/GCS credentials into a DuckDB connection so it can read remote Parquet."""
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
            # GCS via HMAC keys — equivalent to S3 protocol
            if config.storage_key and config.storage_secret:
                conn.execute("SET s3_endpoint='storage.googleapis.com';")
                conn.execute(f"SET s3_access_key_id='{config.storage_key}';")
                conn.execute(f"SET s3_secret_access_key='{config.storage_secret}';")

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------
    def _uri(self, key: str) -> str:
        if self.provider == "local":
            base = config.local_predictions_path.parent
            return str(base / key)
        elif self.provider in ("digitalocean", "aws"):
            return f"s3://{config.storage_bucket}/{key}"
        elif self.provider == "gcp":
            return f"gs://{config.storage_bucket}/{key}"
        raise ValueError(f"Unknown provider: {self.provider}")

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
            return {}  # gcsfs picks up GOOGLE_APPLICATION_CREDENTIALS automatically
        return {}


storage = ObjectStorage()
