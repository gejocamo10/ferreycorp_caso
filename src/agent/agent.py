"""Agente conversacional, Claude (Anthropic) con tool use sobre la tabla de predicciones.

El LLM no escribe SQL libre. Llama a herramientas estructuradas cuyos argumentos
se validan contra una whitelist de columnas y operadores antes de traducirse a
SQL parametrizado en DuckDB. Eso elimina el riesgo de inyeccion y hace la
solucion testeable y predecible.

API publica:
    PropensityAgent().chat(user_message, history) -> {"reply": str, "trace": [...]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic

from src.agent.tools import aggregate_predictions, query_predictions, schema_info
from src.config import config

SYSTEM_PROMPT = """Eres un asistente comercial especializado en propensión de compra para Ferreycorp.

Tienes acceso a una tabla de predicciones generada por un modelo LightGBM, con una fila
por (cliente, día de visita). El modelo predice la probabilidad de que el cliente compre
durante esa visita.

Tu trabajo:
1. Entender la pregunta de negocio del usuario en lenguaje natural.
2. Llamar a las herramientas (query_predictions / aggregate_predictions) con filtros estructurados.
3. Interpretar los resultados y dar respuestas accionables, claras y breves en español.
4. Cuando sea relevante, sugerir KPIs o acciones (ej: campañas dirigidas al decil top, ofertas a cazadores de oferta).

Reglas importantes:
- NO inventes numeros, SIEMPRE consulta primero antes de afirmar.
- Si el usuario pide un listado, usa query_predictions con un limit razonable (ej: 20-50).
- Si el usuario pide totales o promedios por grupo, usa aggregate_predictions con group_by.
- Si no estás seguro de qué columna usar, llama a schema_info primero.
- El campo `incidencia_compra` es la verdad real (post-evento), úsalo solo para validar el modelo o reportes históricos. Para campañas a futuro, usa `score_compra`.
- Los segmentos disponibles son: "Leales premium", "Cazadores de oferta", "Compradores moderados", "Visitantes ocasionales".
- Los deciles van de D1 (top 10% mayor probabilidad) a D10 (bottom 10%).

Sé conciso. Si el usuario pide datos, dale los datos. No envuelvas todo en disclaimers.
"""


# ============================================================
# Tool schemas exposed to Claude
# ============================================================
TOOLS = [
    {
        "name": "query_predictions",
        "description": (
            "Lista filas individuales (cliente-día) de la tabla de predicciones. "
            "Útil para: 'dame los 50 clientes con mayor score', 'lista clientes mayores de 40 con score>0.5'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "array",
                    "description": "Lista de filtros AND. Ej: [{'column':'edad','operator':'>','value':40}]",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "=", "!=", "IN", "BETWEEN"]},
                            "value": {},
                        },
                        "required": ["column", "operator", "value"],
                    },
                },
                "order_by": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "direction": {"type": "string", "enum": ["asc", "desc"]},
                        },
                        "required": ["column"],
                    },
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columnas a devolver. Si se omite, devuelve todas.",
                },
            },
        },
    },
    {
        "name": "aggregate_predictions",
        "description": (
            "Agrega la tabla de predicciones (group by + aggregations). "
            "Útil para: 'score promedio por segmento', '# clientes en decil top con promo'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columnas por las que agrupar.",
                },
                "aggregations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "function": {"type": "string", "enum": ["count", "avg", "sum", "min", "max"]},
                            "column": {"type": "string"},
                            "alias": {"type": "string"},
                        },
                        "required": ["function", "column"],
                    },
                },
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "=", "!=", "IN", "BETWEEN"]},
                            "value": {},
                        },
                        "required": ["column", "operator", "value"],
                    },
                },
                "order_by": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "direction": {"type": "string", "enum": ["asc", "desc"]},
                        },
                        "required": ["column"],
                    },
                },
                "limit": {"type": "integer", "default": 100},
            },
        },
    },
    {
        "name": "schema_info",
        "description": "Devuelve la lista de columnas disponibles y su descripción. Llámalo si dudas de qué columna existe.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

TOOL_FUNCTIONS = {
    "query_predictions": query_predictions,
    "aggregate_predictions": aggregate_predictions,
    "schema_info": schema_info,
}


@dataclass
class TraceStep:
    kind: str  # "tool_call" | "tool_result" | "text"
    name: str = ""
    payload: Any = None


@dataclass
class PropensityAgent:
    model: str = field(default_factory=lambda: config.anthropic_model)
    max_tokens: int = 1500
    max_tool_iters: int = 6

    def __post_init__(self) -> None:
        if not config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY no esta configurada. Agregala en el archivo .env.")
        self.client = Anthropic(api_key=config.anthropic_api_key)

    def chat(self, user_message: str, history: list[dict] | None = None) -> dict[str, Any]:
        messages = list(history or [])
        messages.append({"role": "user", "content": user_message})

        trace: list[TraceStep] = []
        final_text = ""

        for _ in range(self.max_tool_iters):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                texts = [b.text for b in response.content if b.type == "text"]
                final_text = "\n".join(texts).strip()
                trace.append(TraceStep("text", payload=final_text))
                messages.append({"role": "assistant", "content": response.content})
                break

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results: list[dict] = []
                for block in response.content:
                    if block.type == "tool_use":
                        trace.append(TraceStep("tool_call", name=block.name, payload=block.input))
                        try:
                            result = TOOL_FUNCTIONS[block.name](**block.input)
                            content = json.dumps(result, default=str, ensure_ascii=False)
                            is_error = False
                        except Exception as e:
                            content = json.dumps({"error": str(e)}, ensure_ascii=False)
                            is_error = True
                        trace.append(TraceStep("tool_result", name=block.name,
                                               payload=json.loads(content)))
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                            "is_error": is_error,
                        })

                messages.append({"role": "user", "content": tool_results})
                continue

            break

        return {
            "reply": final_text,
            "trace": [{"kind": s.kind, "name": s.name, "payload": s.payload} for s in trace],
            "messages": messages,
        }
