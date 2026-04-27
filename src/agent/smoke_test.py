"""Smoke test for the conversational agent.

Run after setting ANTHROPIC_API_KEY in .env:
    python -m src.agent.smoke_test
"""
from __future__ import annotations

import json

from src.agent.agent import PropensityAgent
from src.config import config


SAMPLE_QUESTIONS = [
    "Dame los 10 clientes con mayor probabilidad de compra que sean del segmento Leales premium.",
    "¿Cuál es el score promedio por segmento de cliente y cuántas observaciones tiene cada uno?",
    "¿Cuántos clientes únicos hay en el decil top (D1) y cuál es su tasa real de compra histórica?",
    "Si lanzo una campaña al top 20% por score, ¿qué % de las compras del último mes (días 700-730) capturaría?",
]


def main() -> None:
    if not config.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        return

    agent = PropensityAgent()
    print(f"Using model: {agent.model}\n")
    history: list[dict] = []
    for q in SAMPLE_QUESTIONS:
        print(f"USER  > {q}")
        out = agent.chat(q, history=history)
        print(f"AGENT > {out['reply']}\n")
        print(f"  (tool calls: {sum(1 for s in out['trace'] if s['kind']=='tool_call')})")
        print("-" * 70)
        history = out["messages"]


if __name__ == "__main__":
    main()
