import io

import pytest
from mcp import StdioServerParameters

from freightcase.execution import (
    MCPToolExecutor,
    StubToolExecutor,
    ToolExecutorError,
)

# -m keeps the path CWD-independent.
STUB_TMS = StdioServerParameters(
    command="uv", args=["run", "python", "-m", "freightcase.tms_stub"]
)


class TestStubToolExecutor:
    def test_execute_records_call_and_returns_canned_result(self):
        executor = StubToolExecutor(reference="Q-STUB-TEST")

        result = executor.execute("test_tool", {"key": "value"})

        assert result.reference == "Q-STUB-TEST"
        assert result.status == "created"
        assert executor.calls == [("test_tool", {"key": "value"})]


class TestMCPToolExecutor:
    def test_create_quote_over_real_stdio_transport(self):
        executor = MCPToolExecutor(server_command=STUB_TMS)

        # Flat payload, same shape the execute node sends: the executor owns
        # the MCP argument nesting, keeping both executors interchangeable.
        result = executor.execute("create_quote", {})

        assert result.reference.startswith("Q-")
        assert result.status == "created"

    def test_unknown_tool_wraps_to_tool_executor_error(self):
        executor = MCPToolExecutor(server_command=STUB_TMS)

        with pytest.raises(ToolExecutorError):
            executor.execute("invalid_tool", {})

    def test_unspawnable_server_wraps_to_tool_executor_error(self, capfd):
        """Transport-level failure: the server can't even start. The error
        wraps like any other, and the child's stderr goes to our errlog
        instead of polluting test output. (errlog content itself isn't
        asserted: on an instant crash the stderr drain races the session
        teardown.)"""
        executor = MCPToolExecutor(
            server_command=StdioServerParameters(
                command="uv", args=["run", "python", "-m", "freightcase.no_such_module"]
            ),
            errlog=io.StringIO(),
        )

        with pytest.raises(ToolExecutorError):
            executor.execute("create_quote", {})

        assert "no_such_module" not in capfd.readouterr().err  # nothing leaked
