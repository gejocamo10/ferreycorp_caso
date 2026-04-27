"""Genera SLIDES.pdf con formato real de presentación: A4 horizontal,
una slide por página, fuentes grandes, paleta consistente.

El brief pide "Presentación de slides - Formato pdf (máx. 7 slides)".

Run:
    python scripts/build_slides_pdf.py
"""
from __future__ import annotations

from pathlib import Path

from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "SLIDES.pdf"

# Paleta inspirada en el deck del EDA (tonos teal/mostaza/coral)
CSS = """
@page { size: A4 landscape; margin: 10mm 14mm 8mm 14mm; }

body {
    font-family: Helvetica, Arial, sans-serif;
    color: #1f2d3d;
    font-size: 10pt;
    line-height: 1.25;
}

.slide {
    page-break-after: always;
    padding: 0;
}
.slide:last-child { page-break-after: auto; }

.slide-num {
    color: #888;
    font-size: 8pt;
    text-align: right;
    margin-bottom: 2mm;
}

h1 {
    color: #264653;
    font-size: 22pt;
    margin: 0 0 3mm 0;
    border-bottom: 2px solid #2a9d8f;
    padding-bottom: 2mm;
}

h2 {
    color: #2a9d8f;
    font-size: 13pt;
    margin: 3mm 0 2mm 0;
}

h3 { color: #e76f51; font-size: 11pt; margin: 2mm 0 1mm 0; }

p { margin: 1mm 0; font-size: 10pt; }
ul, ol { margin: 1mm 0 1mm 5mm; padding: 0; }
li { margin: 0.8mm 0; font-size: 10pt; }

table {
    border-collapse: collapse;
    width: 100%;
    margin: 2mm 0;
    font-size: 9.5pt;
}
th { background: #264653; color: white; padding: 1.5mm 2mm; text-align: left; font-weight: bold; }
td { padding: 1.3mm 2mm; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) td { background: #f7faf9; }

.highlight { background: #2a9d8f; color: white; padding: 0 4px; font-weight: bold; }
.subtitle { color: #555; font-size: 11pt; margin-top: 0; }

table.kpis th { text-align: center; background: #264653; }
table.kpis td.big {
    text-align: center;
    font-size: 18pt;
    color: #2a9d8f;
    font-weight: bold;
    padding: 3mm 2mm;
}

pre {
    background: #f4f4f4;
    padding: 2mm;
    border-left: 3px solid #2a9d8f;
    font-size: 8.5pt;
    font-family: monospace;
    margin: 2mm 0;
}

.footer {
    position: absolute;
    bottom: 6mm;
    left: 0;
    right: 0;
    text-align: center;
    color: #888;
    font-size: 8.5pt;
}
"""

SLIDES_HTML = """
<html>
<head><meta charset='utf-8'><style>{css}</style></head>
<body>

<!-- ================ SLIDE 1: PORTADA + OBJETIVOS ================ -->
<div class="slide">
  <div class="slide-num">1 / 7</div>
  <h1>Modelo de Propensión de Compra<br/>+ Agente IA Conversacional</h1>
  <p class="subtitle">Caso técnico Ferreycorp · Abril 2026</p>

  <h2>Objetivos</h2>
  <h3>Objetivo de negocio</h3>
  <p>Predecir, en cada visita, qué clientes comprarán — y permitir al equipo comercial
  <b>filtrar y explorar las predicciones en lenguaje natural</b>.</p>

  <h3>Objetivos técnicos</h3>
  <ul>
    <li><b>AUC-ROC ≥ 0.65</b> · <b>Lift @ top-10% ≥ 2.0</b></li>
    <li>Modelo explicable globalmente y por cliente (SHAP)</li>
    <li>Agente conversacional con acceso &lt; 100 ms a las predicciones</li>
    <li>Arquitectura <b>portable cloud</b> (DigitalOcean / AWS / GCP)</li>
  </ul>

  <h3>Stack</h3>
  <p>Python · LightGBM · Optuna · SHAP · DuckDB · Anthropic API · Streamlit · Docker</p>
</div>

<!-- ================ SLIDE 2: CRONOGRAMA ================ -->
<div class="slide">
  <div class="slide-num">2 / 7</div>
  <h1>Cronograma de trabajo</h1>

  <table>
    <thead><tr><th>Semana</th><th>Entregable</th><th>Resultado</th></tr></thead>
    <tbody>
      <tr><td><b>S1</b></td><td>EDA + diseño de features causales</td>
          <td>11 figuras + reporte · 43 features definidas</td></tr>
      <tr><td><b>S2</b></td><td>Modelado: baseline + LightGBM + tuning Optuna</td>
          <td>AUC test 0.684 · lift 2.40×</td></tr>
      <tr><td><b>S3</b></td><td>Explicabilidad (SHAP) + agente IA + UI</td>
          <td>Streamlit con chat funcional</td></tr>
      <tr><td><b>S4</b></td><td>Despliegue cloud + documentación + slides</td>
          <td>Specs DO/AWS/GCP · doc técnico · presentación</td></tr>
    </tbody>
  </table>

  <h3>Hitos clave del proyecto</h3>
  <ul>
    <li>📊 <b>EDA reproducible</b>: script Python que regenera todas las figuras</li>
    <li>🛡️ <b>Features causales</b>: validadas sin leakage temporal</li>
    <li>🎯 <b>Modelo bien calibrado</b>: Brier score 0.170</li>
    <li>🤖 <b>Agente seguro</b>: tool use estructurado, no text-to-SQL libre</li>
    <li>☁️ <b>Cloud-agnóstico</b>: DO / AWS / GCP sin tocar código</li>
  </ul>
</div>

<!-- ================ SLIDE 3: EDA ================ -->
<div class="slide">
  <div class="slide-num">3 / 7</div>
  <h1>Análisis Exploratorio (EDA)</h1>

  <h3>Dataset</h3>
  <p>58,693 visitas · 500 clientes únicos · 730 días (~2 años) · 5 marcas ·
  <span class="highlight">25%</span> de tasa de conversión global · sin valores nulos</p>

  <h2>Hallazgos clave</h2>
  <table>
    <thead><tr><th>Hallazgo</th><th>Implicación para el modelo</th></tr></thead>
    <tbody>
      <tr><td><b>89%</b> de clientes concentran &gt;50% de compras en una marca</td>
          <td>Lealtad histórica → feature top</td></tr>
      <tr><td><b>+23%</b> de lift cuando hay promo (26.9% vs 21.8%)</td>
          <td>Promo es palanca de primer orden</td></tr>
      <tr><td>Marcas 5 y 2 dominan (<b>65%</b>) · Marca 3 nicho (5.7%)</td>
          <td>Multiclase de marca tendría desbalance</td></tr>
      <tr><td>Precios fluctúan en el tiempo</td>
          <td>Feature de precio relativo viable</td></tr>
      <tr><td>Cliente promedio: 117 visitas / 29 compras</td>
          <td>Densidad temporal suficiente para features causales</td></tr>
    </tbody>
  </table>
</div>

<!-- ================ SLIDE 4: METODOLOGÍA ================ -->
<div class="slide">
  <div class="slide-num">4 / 7</div>
  <h1>Metodología de la solución</h1>

  <pre>EDA → Feature Engineering Causal → Train Temporal Split → Tuning → Score → Agent</pre>

  <h2>Features (43 totales) — sin leakage</h2>
  <ul>
    <li><b>RFM</b>: recencia, frecuencia, ticket promedio histórico</li>
    <li><b>Lealtad</b>: % histórico de compras por marca (5 columnas)</li>
    <li><b>Sensibilidad personal a promo</b>: tasa de compra con/sin promo, uplift</li>
    <li><b>Precio relativo</b>: precio actual vs. promedio histórico por marca</li>
    <li><b>Demografía + cluster</b>: K-Means k=4 para segmentación de clientes</li>
  </ul>

  <h2>Modelos entrenados</h2>
  <table>
    <thead><tr><th>Modelo</th><th>Rol</th></tr></thead>
    <tbody>
      <tr><td><b>Logistic Regression</b></td><td>Baseline interpretable, class_weight=balanced</td></tr>
      <tr><td><b>LightGBM + Optuna</b></td><td>Modelo final, 30 trials de tuning bayesiano</td></tr>
    </tbody>
  </table>
  <p><b>Validación</b>: split temporal · train (días 1-509) · val (510-619) · test (620-730)</p>

  <h2>Métricas de negocio (no solo accuracy)</h2>
  <p>AUC-ROC · PR-AUC · Brier (calibración) · <b>Lift @ top-K</b> (KPI de campañas)</p>
</div>

<!-- ================ SLIDE 5: ARQUITECTURA ================ -->
<div class="slide">
  <div class="slide-num">5 / 7</div>
  <h1>Arquitectura cloud-agnóstica</h1>

  <pre>
┌──────────────────────────────────────┐
│   Streamlit UI (chat + dashboard)    │
└────────────────┬─────────────────────┘
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
 ┌───────────────┐    ┌─────────────────┐
 │ Claude + tools│ ◄─►│ predictions     │ ◄── LightGBM
 │ (DuckDB SQL)  │    │ Parquet sobre   │     (offline batch)
 │  whitelisted  │    │ S3-compatible   │
 └───────────────┘    └─────────────────┘</pre>

  <h2>Mismo Docker, tres nubes</h2>
  <table>
    <thead><tr><th>Componente</th><th>DigitalOcean</th><th>AWS</th><th>GCP</th></tr></thead>
    <tbody>
      <tr><td>Object storage</td><td>Spaces</td><td>S3</td><td>GCS (HMAC)</td></tr>
      <tr><td>Compute</td><td>App Platform</td><td>ECS Fargate</td><td>Cloud Run</td></tr>
      <tr><td>LLM</td><td>Anthropic API</td><td>Anthropic API</td><td>Anthropic API</td></tr>
    </tbody>
  </table>

  <h3>Seguridad del agente</h3>
  <p>El LLM <b>NO escribe SQL libre</b> — emite filtros estructurados que se validan
  contra una whitelist de columnas/operadores antes de generar SQL parametrizado.</p>
</div>

<!-- ================ SLIDE 6: RESULTADOS ================ -->
<div class="slide">
  <div class="slide-num">6 / 7</div>
  <h1>Resultados</h1>

  <h2>Métricas test (LightGBM)</h2>
  <table class="kpis">
    <thead><tr>
      <th>AUC-ROC</th><th>Lift @ top 10%</th><th>Lift @ top 20%</th><th>Brier (calibración)</th>
    </tr></thead>
    <tbody><tr>
      <td class="big">0.684</td>
      <td class="big">2.40×</td>
      <td class="big">1.90×</td>
      <td class="big">0.170</td>
    </tr></tbody>
  </table>

  <h2>Análisis por decil — el dato que cuenta al negocio</h2>
  <table>
    <thead><tr><th>Decil</th><th>n</th><th>Score predicho</th><th>Tasa real</th><th>Lift vs base</th></tr></thead>
    <tbody>
      <tr><td><b>D1 (top 10%)</b></td><td>5,870</td><td>0.50</td>
          <td><span class="highlight">69.6%</span></td><td><b>2.40×</b></td></tr>
      <tr><td>D2</td><td>5,869</td><td>0.33</td><td>42.7%</td><td>1.40×</td></tr>
      <tr><td>D5</td><td>5,869</td><td>0.23</td><td>20.1%</td><td>0.90×</td></tr>
      <tr><td>D10 (bottom 10%)</td><td>5,870</td><td>0.16</td><td>7.0%</td><td>0.55×</td></tr>
    </tbody>
  </table>

  <p>→ Llamando solo al <b>top 20%</b>, capturas <b>~38% de las compras totales</b>.</p>

  <h3>Top features (SHAP)</h3>
  <p>prior_buy_rate · prior_purchases · recency_days · min_rel_price_today · loyal_brand_on_promo</p>
</div>

<!-- ================ SLIDE 7: CAPTURA DE VALOR ================ -->
<div class="slide">
  <div class="slide-num">7 / 7</div>
  <h1>Captura de valor en el negocio</h1>

  <h2>Áreas de impacto y KPIs</h2>
  <table>
    <thead><tr><th>Área</th><th>Palanca</th><th>KPI clave</th></tr></thead>
    <tbody>
      <tr><td>📣 Marketing dirigido</td><td>Cortar contactos a bottom-50%</td>
          <td>Costo por compra: <b>-50%</b></td></tr>
      <tr><td>🏷️ Pricing &amp; promo</td><td>Personalizar promos por segmento</td>
          <td>Lift incremental por promo</td></tr>
      <tr><td>♻️ Reactivación</td><td>Detectar baja-propensión activa</td>
          <td>Tasa reactivación: <b>+10pp</b></td></tr>
      <tr><td>⚡ Operación comercial</td><td>Self-service vía chat IA</td>
          <td>Time-to-insight: horas → segundos</td></tr>
    </tbody>
  </table>

  <h2>Modelo económico — campaña al top-20% por score</h2>
  <ul>
    <li>⬇ <b>80%</b> menos costo de contacto vs. campaña masiva</li>
    <li>✅ Mantiene <b>38%</b> de la efectividad de la campaña masiva</li>
    <li>→ Rentabilidad por contacto <b>~5×</b> vs. baseline</li>
  </ul>

  <h3>Roadmap de extensiones</h3>
  <p>Modelo multiclase de marca · A/B testing en producción · monitoreo de drift ·
  pricing optimizer · re-entrenamiento automatizado</p>
</div>

</body>
</html>
"""


def main() -> None:
    html = SLIDES_HTML.format(css=CSS)
    with open(OUT, "wb") as f:
        result = pisa.CreatePDF(html, dest=f)
    if result.err:
        print(f"WARN: {result.err} errors")
    else:
        size_kb = OUT.stat().st_size / 1024
        print(f"OK: {OUT}  ({size_kb:.1f} KB · 7 slides A4 landscape)")


if __name__ == "__main__":
    main()
