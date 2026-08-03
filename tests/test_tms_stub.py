"""The stub TMS's tool logic, called directly (transport is covered by the
MCPToolExecutor integration test)."""

from freightcase.tms_stub import create_quote


def test_create_quote_returns_created_with_unique_references():
    first = create_quote({"mode": "road", "cargo": [{"description": "x"}]})
    second = create_quote({"mode": "air", "cargo": [{"description": "y"}]})

    assert first.status == "created"
    assert second.status == "created"
    assert first.reference.startswith("Q-")
    assert first.reference != second.reference
