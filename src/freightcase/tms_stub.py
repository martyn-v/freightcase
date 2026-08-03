"""A stand-in TMS: the MCP server an adopter would replace with their own.

Run over stdio with `uv run python -m freightcase.tms_stub`. Quotes live in
memory for the lifetime of the process; references increment per quote.
This is demo/test infrastructure, not part of the pipeline — it exists so
the real MCPToolExecutor has a real server to talk to.
"""

from itertools import count

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("freightcase-stub-tms")

_quotes: dict[str, dict] = {}
_references = count(1)


@mcp.tool()
def create_quote(quote: dict) -> dict:
    """Record a quote request and return its TMS reference."""
    reference = f"Q-{next(_references):04d}"
    _quotes[reference] = quote
    return {"reference": reference, "status": "created"}


if __name__ == "__main__":
    mcp.run()
