"""Convierte los entregables en markdown a PDF usando xhtml2pdf (puro Python).

Genera DOCUMENTO_TECNICO.pdf, GUIA_APRENDIZAJE.pdf y REPORTE_EDA.pdf.
SLIDES.pdf se genera con build_slides_pdf.py por usar formato landscape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #222; line-height: 1.45; }
h1 { color: #264653; font-size: 22pt; border-bottom: 2px solid #2a9d8f; padding-bottom: 4px; }
h2 { color: #264653; font-size: 14pt; margin-top: 18pt; border-bottom: 1px solid #ccc; padding-bottom: 2px; }
h3 { color: #2a9d8f; font-size: 12pt; margin-top: 14pt; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 2px; font-size: 9pt; }
pre { background: #f4f4f4; padding: 8px; border-left: 3px solid #2a9d8f; font-size: 8.5pt; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.5pt; }
th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; }
th { background: #264653; color: white; }
tr:nth-child(even) td { background: #f8f8f8; }
blockquote { border-left: 3px solid #2a9d8f; padding-left: 10px; color: #555; margin: 8pt 0; }
img { max-width: 100%; }
"""


def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    md_path_str = str(md_path.parent.resolve())
    md_text = md_text.replace("](figures/", f"]({md_path_str}/figures/")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html = f"""<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"""
    with open(pdf_path, "wb") as f:
        result = pisa.CreatePDF(html, dest=f)
    if result.err:
        print(f"  WARN: {result.err} errors generating {pdf_path}")
    else:
        print(f"  OK: {pdf_path}  ({pdf_path.stat().st_size/1024:.1f} KB)")


def main() -> None:
    # SLIDES.pdf se genera con build_slides_pdf.py (formato presentación landscape)
    targets = [
        (ROOT / "docs" / "DOCUMENTO_TECNICO.md", ROOT / "docs" / "DOCUMENTO_TECNICO.pdf"),
        (ROOT / "docs" / "GUIA_APRENDIZAJE.md", ROOT / "docs" / "GUIA_APRENDIZAJE.pdf"),
        (ROOT / "docs" / "eda" / "REPORTE_EDA.md", ROOT / "docs" / "eda" / "REPORTE_EDA.pdf"),
    ]
    for md, pdf in targets:
        if not md.exists():
            print(f"  SKIP: {md} not found")
            continue
        md_to_pdf(md, pdf)


if __name__ == "__main__":
    main()
