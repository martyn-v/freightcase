import pytest

from freightcase.execution import StubToolExecutor


class TestStubToolExecutor:
    def test_execute(self):
        executor = StubToolExecutor(reference="Q-STUB-TEST")
        tool_name = "test_tool"
        payload = {"key": "value"}

        result = executor.execute(tool_name, payload)

        assert result.reference == "Q-STUB-TEST"
        assert result.status == "created"
        assert executor.calls == [(tool_name, payload)]
