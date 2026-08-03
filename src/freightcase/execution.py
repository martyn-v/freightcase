from typing import Literal, Protocol

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
