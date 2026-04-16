"""Tests for the agent module. Mocks the OpenAI client."""

import pytest
from unittest.mock import patch, MagicMock

openai = pytest.importorskip("openai", reason="openai not installed")

from src.agent import run_agent, _parse_pseudo_tool_calls  # noqa: E402
from src.client import CompletionResult  # noqa: E402


def _mock_completion_result(content="final answer", tool_calls=None):
    return CompletionResult(
        tool_calls=tool_calls or [],
        content=content,
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
        latency_ms=100.0,
    )


class TestRunAgentFinalMessage:
    """run_agent should append the final assistant message."""

    @patch("src.agent.get_client")
    @patch("src.agent.chat_completion")
    def test_appends_final_assistant_message(self, mock_completion, mock_client):
        mock_client.return_value = MagicMock()
        mock_completion.return_value = _mock_completion_result(content="final answer")

        result = run_agent(
            system_prompt="You are helpful.",
            user_message="Hello",
            model="test-model",
        )

        assert result.messages[-1] == {
            "role": "assistant",
            "content": "final answer",
        }


class TestParsePseudoToolCalls:
    """_parse_pseudo_tool_calls should handle various formats."""

    def test_read_file_call(self):
        content = 'read_file({"path": "test.txt"})'
        calls = _parse_pseudo_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"

    def test_search_web_call(self):
        content = 'search_web({"query": "hello"})'
        calls = _parse_pseudo_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "search_web"

    def test_check_status_call(self):
        content = 'check_status({"target": "server"})'
        calls = _parse_pseudo_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "check_status"

    def test_plain_text_no_calls(self):
        content = "This is just normal text with no tool calls at all."
        calls = _parse_pseudo_tool_calls(content)
        assert len(calls) == 0

    def test_hydrate_format(self):
        content = 'hydrate({"name": "read_file", "arguments": {"path": "x"}})'
        calls = _parse_pseudo_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
