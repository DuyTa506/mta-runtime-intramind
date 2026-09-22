"""Real Temporal episode recovery with fake inference, voice and publication."""

import asyncio
import json
import os
import wave
from contextlib import suppress
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from feature_harness import feature_environment
from temporalio.service import RPCError

from intramind_runtime.speech import SpeechProfile
from intramind_runtime.store import rows

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("high_quality", [False, True])
async def test_audio_retries_encoding_and_publication_without_repeating_completed_work(
    store, monkeypatch, compress, high_quality
):
    await audio_case(store, monkeypatch, compress=compress, high_quality=high_quality)


async def test_audio_unknown_voice_does_not_fall_back_or_refund_on_cancel(store, monkeypatch):
    await audio_case(store, monkeypatch, compress=False, high_quality=True, drop_response=True)


async def test_large_wav_fallback_persists_through_runtime_artifact_api(store, monkeypatch):
    await audio_case(store, monkeypatch, compress=False, high_quality=False,
                     wave_frames=5 * 1024 * 1024)


async def audio_case(store, monkeypatch, *, compress, high_quality, drop_response=False,
                     wave_frames=2400):
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
        audio_overview_high_quality_voice=high_quality,
        audio_overview_tts_batch_size=1,
        audio_overview_script_timeout_seconds=27.5,
        tts_timeout_seconds=23.5,
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
        wav.writeframes(b"\x00\x00" * wave_frames)

    async def voice_response(request):
        body = json.loads(request.content)
        voice_calls.append(body)
        if drop_response:
            raise httpx.ReadTimeout("backend termination was not observed")
        headers = {"X-Intramind-Attempt-ID": request.headers["X-Intramind-Attempt-ID"],
                   "X-Intramind-TTS-Contract": "termination-v1"}
        if body["voice"].get("voice_id"):
            return httpx.Response(503, headers=headers | {"X-Intramind-Compute-State": "not_started"})
        return httpx.Response(200, content=buffer.getvalue(), headers=headers | {
            "X-Intramind-Compute-State": "terminated", "Content-Type": "audio/wav",
        })

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
            config.tts_timeout_seconds = 1
            reply = json.dumps(turns, ensure_ascii=False)
        else:
            reply = "Nội dung nén giữ đúng căn cứ và ngoại lệ của nguồn. " * 3
        return {"choices": [{"message": {"content": reply}}]}

    def activities(factory):
        audio = AudioActivities(
            factory,
            model_profile="test",
            speech_profile="test-voice",
            config_factory=lambda: config,
            summary_factory=lambda: summary,
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
        speech_profile=SpeechProfile(
            model_profile="test-voice", capacity_profile_id="voice-v1", character_limit=2000,
            sample_rate=24000, max_audio_bytes=max(100_000, len(buffer.getvalue())),
            voices={"vi_female", "vi_male", "vi_female_hq", "vi_male_hq"},
        ),
        respond_speech=voice_response,
    ) as env:
        source = (
            {"document_ids": ["0", "1"]}
            if compress
            else {"text": "Nguồn đầy đủ và có căn cứ cần đọc chính xác. " * 30}
        )
        source |= {"mode": "podcast", "target_minutes": 1, "conversation_id": "42"}
        if drop_response:
            api = env.runtime_client("user:test")
            accepted = await api.submit({"task_type": "audio-overview/v1",
                "submission_key": uuid4().hex, "input": await api.put_json(source)})
            handle = env.temporal.get_workflow_handle(accepted["run_id"])
            try:
                async with asyncio.timeout(10):
                    while True:
                        status = await api.get_run(accepted["run_id"])
                        if status["operations"].get("RECONCILING"):
                            break
                        await asyncio.sleep(0.02)
                await asyncio.sleep(0.1)
                assert len(voice_calls) == 1
                assert status["state"] == "RUNNING" and status["cleanup_pending"]
                assert status["resource_budgets"]["speech_characters"]["reserved"] > 0
                assert not encodings and not uploads
                await api.cancel(accepted["run_id"])
                cancelled = await api.get_run(accepted["run_id"])
                assert cancelled["state"] == "CANCELLED" and cancelled["cleanup_pending"]
                assert cancelled["resource_budgets"] == status["resource_budgets"]
                await store.confirm_epoch_stopped("voice", "voice-e1", "fake transport joined; no live backend work")
                reconciled = await api.get_run(accepted["run_id"])
                assert not reconciled["cleanup_pending"]
                assert reconciled["resource_budgets"]["speech_characters"]["reserved"] == 0
                assert len(voice_calls) == 1
            finally:
                with suppress(RPCError):
                    if (await handle.describe()).close_time is None:
                        await handle.terminate("disposable UNKNOWN speech test cleanup")
                await api.close()
            return
        status, result = await env.submit(source)
        assert status["state"] == "PARTIAL"  # WAV fallback is explicit.
        assert result["transcript"] == turns
        assert len(voice_calls) == 2 + int(high_quality)
        assert voice_calls[-1]["voice"] == {"gender": "female"}
        assert status["resource_budgets"]["speech_characters"]["spent"] == sum(
            len(request["text"]) for request in voice_calls if "voice_id" not in request["voice"]
        )
        async with store.engine.connect() as connection:
            voice_operations = await rows(connection,
                "SELECT spec FROM runtime_operations WHERE spec->>'kind'='speech'")
        assert len(voice_operations) == len(voice_calls)
        assert all(operation["spec"]["attempt_timeout_seconds"] == 23.5
                   for operation in voice_operations)
        assert len(encodings) == 2 and encodings[0] == encodings[1]
        assert uploads == [result["object_key"]]
        if wave_frames > 2400:
            assert len(stored[result["object_key"]]) > 16 * 1024 * 1024
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
        assert len(voice_calls) == 2 + int(high_quality)
        assert len(uploads) == 1
