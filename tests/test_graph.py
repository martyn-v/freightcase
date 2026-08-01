from pathlib import Path
from freightcase.graph import graph
from pprint import pprint

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

    pprint(result)

    intake = result["intake"]
    assert intake is not None
    assert intake.body_text.startswith("Buenas tardes")

    results = result["results"]

    assert len(results) == 1
    assert results[0].function == "quote_request"
    assert results[0].status == "extracted"
    assert results[0].output is not None

    # Email states an incoterm but no dimensions: road mode requires dims.
    assert results[0].missing == ["cargo.0.dimensions"]

    extraction = results[0].output

    assert extraction.mode == "road"
    assert extraction.incoterm is not None
    assert extraction.incoterm.rule == "DAP"
    assert extraction.cargo[0].weight.kg == 8400
