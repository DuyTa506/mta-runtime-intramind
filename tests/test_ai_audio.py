"""Real Temporal episode recovery with fake inference, voice and publication."""

import json
import os
import wave
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from feature_harness import feature_environment

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("compress", [False, True])
async def test_audio_retries_encoding_and_publication_without_repeating_completed_work(
    store, monkeypatch, compress
):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment and opt-in required")
    from api.background.audio.activities import AudioActivities
    from api.background.audio.workflows import WORKFLOWS as AUDIO_WORKFLOWS
    from api.background.common.models import ModelActivities
    from api.background.summary.activities import SummaryActivities
    from api.background.summary.workflows import WORKFLOWS as SUMMARY_WORKFLOWS
    from tools.audio_overview_utils.durable import LEAVES
    from tools.summary import SummaryTool

    config = SimpleNamespace(
        audio_overview_target_minutes=1,
        audio_overview_max_minutes=1,
        audio_overview_wpm=20,
        audio_overview_context_window=4096,
        audio_overview_context_margin=1024,
        audio_overview_max_documents=5,
        audio_overview_high_quality_voice=False,
        audio_overview_tts_batch_size=1,
        audio_overview_script_timeout_seconds=27.5,
    )
    monkeypatch.setattr(
        SummaryTool, "_init_tokenizer", lambda self: setattr(self, "tokenizer", None)
    )
    llm = MagicMock()
    llm.get_model_info.return_value = {"model_name": "test", "provider": "fake"}
    llm.agenerate = AsyncMock(side_effect=AssertionError("activity bypassed broker"))
    summary = SummaryTool(llm, {"context_window": 16384})
    summary._count_tokens = lambda text: len(text.split())
    summary.chunker.mindmap_chunk = lambda text, _: [{"content": text, "title": "", "level": 0}]
    summary.chunker.last_outline_source = "test"
    summary._fetch_documents = AsyncMock(
        return_value=[
            {"id": str(i), "name": f"Tài liệu {i}", "text": "Nội dung có căn cứ. " * 1500}
            for i in range(2)
        ]
    )
    turns = [
        {"speaker": speaker, "text": "Một hai ba bốn năm sáu bảy tám chín mười."}
        for speaker in ("host", "guest")
    ]
    voice_calls, encodings, uploads, stored = [], [], [], {}
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 2400)

    class Voice:
        async def asynth(self, text, **options):
            voice_calls.append((text, options))
            return buffer.getvalue(), 100

        async def aclose(self):
            pass

    class Episodes:
        def open(self, key):
            if key not in stored:
                raise FileNotFoundError(key)
            data = stored[key]
            return BytesIO(data), "audio/wav", len(data)

        def upload(self, data, key, audio_format):
            uploads.append(key)
            stored[key] = data
            raise OSError("Upload acknowledgement lost after storing bytes")

    def encode(data):
        encodings.append(data)
        if len(encodings) == 1:
            raise OSError("Encoder failed after every clip had completed")
        return data, "wav"

    async def respond(reservation, payload):
        if reservation.operation.max_output_tokens == 12000:
            assert reservation.operation.attempt_timeout_seconds == 27.5
            config.audio_overview_script_timeout_seconds = 1
            reply = json.dumps(turns, ensure_ascii=False)
        else:
            reply = "Nội dung nén giữ đúng căn cứ và ngoại lệ của nguồn. " * 3
        return {"choices": [{"message": {"content": reply}}]}

    def activities(factory):
        audio = AudioActivities(
            factory,
            model_profile="test",
            config_factory=lambda: config,
            summary_factory=lambda: summary,
            voice_factory=Voice,
            storage_factory=Episodes,
            encoder=encode,
        )
        compression = SummaryActivities(
            factory, model_profile="test", variants={"audio": lambda: summary}
        )
        return [
            *audio.registered(),
            *compression.registered(),
            ModelActivities(factory, leaves=LEAVES).plan,
        ]

    async with feature_environment(
        store,
        name="audio-overview",
        workflows=[*AUDIO_WORKFLOWS, *SUMMARY_WORKFLOWS],
        build_activities=activities,
        respond=respond,
    ) as env:
        source = (
            {"document_ids": ["0", "1"]}
            if compress
            else {"text": "Nguồn đầy đủ và có căn cứ cần đọc chính xác. " * 30}
        )
        status, result = await env.submit(
            {
                **source,
                "mode": "podcast",
                "target_minutes": 1,
                "conversation_id": "42",
            }
        )
        assert status["state"] == "PARTIAL"  # WAV fallback is explicit.
        assert result["transcript"] == turns
        assert len(voice_calls) == 2
        assert len(encodings) == 2 and encodings[0] == encodings[1]
        assert uploads == [result["object_key"]]
        assert result["object_key"].startswith("audio-overviews/42/")
        with wave.open(BytesIO(stored[result["object_key"]]), "rb") as wav:
            assert wav.getnframes() > 0
        assert len(result["metadata"]["sources"]["compacted"]) == (2 if compress else 0)
        if compress:
            summary._fetch_documents.assert_awaited_once()
        else:
            summary._fetch_documents.assert_not_awaited()
        llm.agenerate.assert_not_awaited()
        inference_count = len(env.calls)
        assert inference_count == (3 if compress else 1)
        await env.replay(status["root_id"])
        assert len(env.calls) == inference_count
        assert len(voice_calls) == 2
        assert len(uploads) == 1
