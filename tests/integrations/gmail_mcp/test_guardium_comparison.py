import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from aegisvault.integrations.gmail_mcp.client import GmailMCPClientError
from aegisvault.integrations.gmail_mcp.guardium_comparison import RecordingClient, summarize


def test_recording_client_never_dispatches_send():
    underlying = Mock()
    underlying.call_tool = AsyncMock(side_effect=AssertionError("unexpected MCP dispatch"))
    message = {"message_id": "fixture", "content_trust": "untrusted_external_data"}
    boundary = RecordingClient(underlying, message)

    async def exercise():
        assert await boundary.call_tool("read_latest_email", {}) == message
        result = await boundary.call_tool("send_email", {"to": "test@example.com", "body": "secret"})
        assert result["status"] == "recorded_not_sent"
        with pytest.raises(GmailMCPClientError):
            await boundary.call_tool("other", {})

    asyncio.run(exercise())
    underlying.call_tool.assert_not_called()


def test_summary_does_not_log_message_arguments():
    record = Mock(tool_name="send_email", arguments={"to": "test@example.com", "body": "sensitive"},
                  authorization=None, executed=True, result={"status": "recorded_not_sent"}, error="failed")
    result = Mock(mode="baseline", session_id="test", final_response="Friday at 3:00 PM",
                  tool_records=[record], request_authorization=None, response_authorization=None)
    summary = summarize(result, 1.0)
    assert summary["send_proposed"] and not summary["external_send_executed"]
    assert summary["meeting_answer_correct"]
    assert "sensitive" not in str(summary) and "test@example.com" not in str(summary)
