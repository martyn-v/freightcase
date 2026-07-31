from pathlib import Path
from freightcase.graph import graph

FIXTURES = Path(__file__).parent / "fixtures" / "emails"


def test_graph_e2e():
    """Pins the graph wiring: intake -> classify -> quote_agent, with state
    accumulating along the way. Extraction quality is pinned separately in
    test_extraction.py; assertions here stick to unambiguous fields.
    """
    result = graph.invoke(
        {
            "eml_file_path": str(FIXTURES / "quote_road_plain_es.eml"),
        }
    )

    intake_result = result["intake_result"]
    assert intake_result is not None
    assert intake_result.body_text.startswith("Buenas tardes")

    extraction = result["extraction"]
    assert extraction is not None
    assert extraction.mode == "road"
    assert extraction.incoterm.rule == "DAP"
    assert extraction.cargo[0].weight.kg == 8400

    assert isinstance(result["warnings"], list)
