"""Concatena todo el código fuente en un único archivo TXT entregable.

El brief de Ferreycorp pide: "Código de la solución – Formato word o txt".
Este script genera CODIGO_SOLUCION.txt con todos los archivos relevantes,
con separadores claros y cabecera de cada archivo.

Run:
    python scripts/build_codigo_txt.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "CODIGO_SOLUCION.txt"

# Orden de archivos pensado para lectura: config → datos → modelo → storage → agente → UI
FILES = [
    ("Configuración y dependencias", [
        "requirements.txt",
        ".env.example",
        "docker/Dockerfile",
        "src/config.py",
    ]),
    ("Carga de datos y EDA", [
        "src/utils/io.py",
        "src/eda.py",
    ]),
    ("Feature engineering", [
        "src/features.py",
    ]),
    ("Modelo: entrenamiento, explicabilidad, scoring", [
        "src/model/train.py",
        "src/model/explain.py",
        "src/model/score.py",
    ]),
    ("Capa de storage cloud-agnóstica (DO / AWS / GCP)", [
        "src/storage/object_storage.py",
    ]),
    ("Agente conversacional (Claude + tool use)", [
        "src/agent/tools.py",
        "src/agent/agent.py",
        "src/agent/smoke_test.py",
    ]),
    ("UI Streamlit", [
        "src/app.py",
    ]),
    ("Specs de despliegue cloud", [
        "deploy/digitalocean/app.yaml",
        "deploy/aws/ecs-task.json",
        "deploy/gcp/cloudrun.yaml",
    ]),
]


HEADER = """================================================================================
 CASO FERREYCORP — MODELO DE PROPENSIÓN DE COMPRA + AGENTE IA
 Código de la solución (consolidado)
================================================================================

Este documento contiene todo el código fuente de la solución, organizado
por componentes para facilitar la lectura.

Estructura del repositorio:
  - data/raw/         → CSV original
  - data/processed/   → Features y segmentos (Parquet)
  - data/predictions/ → Predicciones del modelo (consumidas por el agente)
  - docs/             → EDA, documento técnico, slides
  - models/           → Modelos serializados (LightGBM, LogReg)
  - src/              → Código fuente (este documento)
  - deploy/           → Specs DigitalOcean / AWS / GCP

Pipeline para reproducir desde cero:
  1. python -m src.eda
  2. python -m src.features
  3. python -m src.model.train
  4. python -m src.model.explain
  5. python -m src.model.score
  6. streamlit run src/app.py

================================================================================

"""


def section_separator(title: str) -> str:
    return f"\n{'=' * 80}\n  SECCIÓN: {title.upper()}\n{'=' * 80}\n\n"


def file_separator(path: str) -> str:
    return f"\n{'-' * 80}\n# Archivo: {path}\n{'-' * 80}\n"


def main() -> None:
    parts: list[str] = [HEADER]
    total_lines = 0
    total_files = 0

    for section_title, files in FILES:
        parts.append(section_separator(section_title))
        for rel_path in files:
            full = ROOT / rel_path
            if not full.exists():
                parts.append(file_separator(rel_path) + "(archivo no encontrado)\n")
                continue
            content = full.read_text(encoding="utf-8")
            parts.append(file_separator(rel_path))
            parts.append(content.rstrip() + "\n")
            total_lines += content.count("\n")
            total_files += 1

    OUT.write_text("".join(parts), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"OK: {OUT}")
    print(f"   {total_files} archivos · ~{total_lines:,} líneas · {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
