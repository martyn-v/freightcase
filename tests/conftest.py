"""Shared scripted-model helpers for graph- and server-level tests."""

import json

from langchain_core.language_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver

from freightcase.execution import StubToolExecutor
from freightcase.graph import build_graph, checkpoint_serde


def extraction_json(*, exclude: tuple[str, ...] = ()) -> str:
    """A complete road-freight extraction (the Rotterdam lathe); `exclude`
    removes dotted paths to create the specific gap a test is about."""
    data: dict = {
        "mode": "road",
        "origin": {"name": "Rotterdam", "locode": "NLRTM"},
        "destination": {"name": "Houston", "locode": "USHOU"},
        "incoterm": {"rule": "EXW", "named_place": "Rotterdam"},
        "cargo": [
            {
                "description": "Crated lathe",
                "pieces": 1,
                "weight": {"value": 950, "unit": "kg"},
                "dimensions": {
                    "length": 200,
                    "width": 120,
                    "height": 150,
                    "unit": "cm",
                },
            }
        ],
    }
    for path in exclude:
        target = data
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        del target[parts[-1]]
    return json.dumps(data)


def fake_graph(
    *model_responses: str,
    classification: str = '{"classification": "quote_request"}',
):
    """A graph with a scripted model and its own stub executor: fully
    deterministic, no LLM, no shared state between tests. The classify node
    consumes the model's first response, so a classification is prepended;
    override it to script the unknown/dead-letter route."""
    executor = StubToolExecutor()
    model = GenericFakeChatModel(messages=iter([classification, *model_responses]))
    graph = build_graph(
        executor=executor,
        checkpointer=InMemorySaver(serde=checkpoint_serde()),
        model=model,
    )
    return graph, executor
