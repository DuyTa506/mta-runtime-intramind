"""Actual Temporal mindmap phases and replay with deterministic fake inference."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from feature_harness import feature_environment

pytestmark = pytest.mark.integration


async def test_mindmap_replays_every_phase_with_no_source_or_inference_repeated(store, monkeypatch):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment and opt-in required")
    from api.background.mindmap.activities import MindmapActivities
    from api.background.mindmap.workflows import WORKFLOWS
    from tools.mindmap import MindmapTool

    def tokenizer(tool):
        tool.tokenizer, tool.tokenizer_type = None, "fallback"

    monkeypatch.setattr(MindmapTool, "_init_tokenizer", tokenizer)
    llm = MagicMock()
    llm.get_model_info.return_value = {"model_name": "test", "provider": "fake"}
    llm.agenerate = AsyncMock(side_effect=AssertionError("activity attempted inference"))
    tool = MindmapTool(llm, {"context_window": 16384, "embedder": None})
    tool._get_document_text = AsyncMock(
        return_value="Báo cáo có thời hạn, trách nhiệm và ngoại lệ. " * 20
    )
    tool.chunker.mindmap_chunk = MagicMock(return_value=[{"content": "Báo cáo trước ngày 15."}])
    tool.chunker.count_tokens = lambda text: len(text.split())
    replies = iter(
        [
            "- Báo cáo trước ngày 15.\n- Phòng A thực hiện.\n- Ngoại lệ cần duyệt.\n- Phòng B kiểm tra.",
            "# Báo cáo\n## Thời hạn\n## Trách nhiệm\n## Ngoại lệ",
        ]
    )

    async def respond(reservation, payload):
        text = next(replies, "### Công việc\n#### Giữ đúng thời hạn và điều kiện phê duyệt")
        return {"choices": [{"message": {"content": text}}]}

    def activities(factory):
        return MindmapActivities(
            factory, model_profile="test", tool_factory=lambda: tool
        ).registered()

    async with feature_environment(
        store, name="mindmap", workflows=WORKFLOWS, build_activities=activities, respond=respond
    ) as env:
        status, result = await env.submit({"document_id": "d", "output_format": "markdown"})
        assert status["state"] == "SUCCEEDED"
        assert result["mindmap"].startswith("# Báo cáo")
        assert set(result["formats"]) == {"markdown", "json", "mermaid", "html"}
        calls = len(env.calls)
        assert 3 <= calls <= 5
        await env.replay(status["root_id"])
        assert len(env.calls) == calls
        tool._get_document_text.assert_awaited_once_with("d")
        llm.agenerate.assert_not_awaited()
