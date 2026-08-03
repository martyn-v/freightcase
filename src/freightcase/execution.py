import asyncio
import sys
from typing import Literal, Protocol, TextIO

import mcp
from mcp.client.client import Client
from pydantic import BaseModel


class ToolExecutorResult(BaseModel):
    reference: str
    status: Literal["created"]


class ToolExecutorError(Exception):
    """Raised when a tool execution fails."""


class ToolExecutor(Protocol):
    def execute(self, tool: str, payload: dict) -> ToolExecutorResult: ...


class StubToolExecutor:
    """A stub tool executor for testing and demo purposes. It records the calls made to it and returns a fixed reference and status."""

    def __init__(self, reference: str = "Q-STUB-1"):
        self.reference = reference
        self.calls: list[tuple[str, dict]] = []  # (tool, payload) tuples

    def execute(self, tool: str, payload: dict) -> ToolExecutorResult:
        self.calls.append((tool, payload))
        return ToolExecutorResult(reference=self.reference, status="created")


class MCPToolExecutor:
    """A tool executor that communicates with an MCP server to execute tools.

    `errlog` receives the server process's stderr (default: our stderr, so a
    real TMS server's logs stay visible); tests pass a StringIO to keep
    output clean and assert on captured errors."""

    def __init__(
        self, server_command: mcp.StdioServerParameters, errlog: TextIO = sys.stderr
    ):
        self.server_command = server_command
        self.errlog = errlog

    async def _execute(self, tool: str, payload: dict) -> ToolExecutorResult:
        async with Client(
            mcp.stdio_client(self.server_command, errlog=self.errlog)
        ) as client:
            result = await client.call_tool(tool, {"quote": payload})

            if result.is_error:
                raise ToolExecutorError(f"Tool returned error: {result.content}")

            return ToolExecutorResult.model_validate(result.structured_content)

    def execute(self, tool: str, payload: dict) -> ToolExecutorResult:
        try:
            return asyncio.run(self._execute(tool, payload))
        except ToolExecutorError:
            raise
        except Exception as e:
            raise ToolExecutorError("Tool execution failed") from e
