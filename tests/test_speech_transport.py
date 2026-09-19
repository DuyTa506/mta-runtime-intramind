import asyncio
import io
import json
import wave

import httpx
import pytest
from conftest import root
from fakes import MemoryArtifacts
from test_speech_ledger import speech, speech_pool

from intramind_runtime.contracts import Artifact, CancelOutcome
from intramind_runtime.drivers import DriverFailure
from intramind_runtime.executor import Executor
from intramind_runtime.store import rows


def waveform():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 2400)
    return buffer.getvalue()


def profile():
    from intramind_runtime.speech import SpeechProfile
    return SpeechProfile(
        model_profile="speech-test", capacity_profile_id="test-v1", character_limit=2000,
        sample_rate=24000, max_audio_bytes=100_000,
        voices={"vi_female", "vi_male", "vi_female_hq", "vi_male_hq"},
    )


def payload():
    return {"text": "Xin chào bạn", "language": "vi", "voice": {"gender": "male"},
            "speed": 1.1, "pause_scale": 0.0}


def proof(request, state="terminated"):
    return {"X-Intramind-Attempt-ID": request.headers["X-Intramind-Attempt-ID"],
            "X-Intramind-TTS-Contract": "termination-v1", "X-Intramind-Compute-State": state}


async def driver_case(store, handler, *, timeout=5):
    from intramind_runtime.speech import ServingSpeechDriver
    await store.create_root(root(resource_budgets={"speech_characters": 1000}))
    await store.configure_pool(speech_pool(target=1), 1)
    blobs = MemoryArtifacts()
    data = json.dumps(payload()).encode()
    ref = await blobs.put("t", data)
    spec = speech(characters=len(payload()["text"]), attempt_timeout_seconds=timeout).model_copy(
        update={"payload": ref, "capacity_profile_id": "test-v1"}
    )
    await store.submit_operation(spec)
    client = httpx.AsyncClient(base_url="http://serving/", transport=httpx.MockTransport(handler))
    driver = ServingSpeechDriver("http://serving/", profile(), client=client)
    return driver, blobs, Executor(store, blobs, driver, "voice", "worker")


async def test_speech_preparation_pins_profile_and_preserves_voice_speed_and_pause():
    from intramind_runtime.speech import SpeechPreparer, SpeechPrepareRequest
    preparer, blobs = SpeechPreparer(profile()), MemoryArtifacts()
    request = SpeechPrepareRequest(
        model_profile="speech-test", capacity_profile_id="test-v1", payload=payload(),
        attempt_timeout_seconds=17.5,
    )
    prepared = await preparer.prepare(request, blobs, "t")
    assert prepared["characters_bound"] == len(payload()["text"])
    assert prepared["attempt_timeout_seconds"] == 17.5
    assert prepared["capacity_profile_id"] == "test-v1"
    assert json.loads(await blobs.get(Artifact.model_validate(prepared["payload"]))) == payload()
    with pytest.raises(ValueError, match="profile"):
        await preparer.prepare(request.model_copy(update={"capacity_profile_id": "old"}), blobs, "t")


@pytest.mark.parametrize("change", [
    {"text": "x" * 2001}, {"text": " "}, {"voice": {"voice_id": "unapproved"}},
    {"speed": 0.1}, {"pause_scale": -1}, {"language": "en"}, {"endpoint": "http://other"},
])
async def test_speech_preparation_rejects_invalid_or_unqualified_payload(change):
    from intramind_runtime.speech import SpeechPreparer, SpeechPrepareRequest
    with pytest.raises(ValueError):
        request = SpeechPrepareRequest(
            model_profile="speech-test", capacity_profile_id="test-v1",
            payload=payload() | change, attempt_timeout_seconds=17.5,
        )
        await SpeechPreparer(profile()).prepare(request, MemoryArtifacts(), "t")


@pytest.mark.integration
async def test_binary_result_storage_failure_never_repeats_synthesis(store, monkeypatch):
    requests = []

    async def backend(request):
        requests.append(request)
        assert json.loads(request.content) == payload()
        return httpx.Response(200, content=waveform(), headers=proof(request) | {
            "Content-Type": "audio/wav", "X-Audio-Duration-Ms": "100"})

    driver, blobs, executor = await driver_case(store, backend)
    original, failures = blobs.put, []

    async def fail_once(tenant, data, content_type="application/json"):
        if content_type == "audio/wav" and not failures:
            failures.append(True)
            assert (await store.drain_status())["compute_held"] == 0
            assert (await store.drain_status())["pending_settlement"] == 1
            raise OSError("object store unavailable after inference")
        return await original(tenant, data, content_type)

    monkeypatch.setattr(blobs, "put", fail_once)
    try:
        await asyncio.wait_for(executor.tick(), 6)
        operation = await store.operation("speech", "t")
        result = json.loads(await blobs.get(Artifact.model_validate(operation["result"])))
        audio = Artifact.model_validate(result["body"]["audio"])
        assert await blobs.get(audio) == waveform()
        assert result["body"]["duration_ms"] == 100
        state = await store.run("r", "t")
        assert state["resource_budgets"]["speech_characters"]["spent"] == len(payload()["text"])
        assert (state["reserved"], state["spent"]) == (0, 0)
        assert len(requests) == 1
        async with store.engine.connect() as connection:
            manifests = await rows(connection, "SELECT manifest FROM runtime_artifacts")
        assert {item["manifest"]["content_type"] for item in manifests} == {
            "application/json", "audio/wav",
        }
    finally:
        await driver.close()


@pytest.mark.integration
@pytest.mark.parametrize("evidence,state", [
    ("absent", "RECONCILING"), ("wrong_attempt", "RECONCILING"),
    ("not_started", "FAILED"), ("terminated", "FAILED"),
])
async def test_503_releases_resources_only_with_matching_termination_proof(store, evidence, state):
    calls = []

    async def backend(request):
        calls.append(request)
        headers = proof(request, evidence)
        if evidence == "absent":
            headers = {}
        if evidence == "wrong_attempt":
            headers = proof(request) | {"X-Intramind-Attempt-ID": "other"}
        return httpx.Response(503, json={"detail": "unavailable"}, headers=headers)

    driver, _, executor = await driver_case(store, backend)
    try:
        await executor.tick()
        assert (await store.operation("speech", "t"))["state"] == state
        status = await store.run("r", "t")
        budget = status["resource_budgets"]["speech_characters"]
        assert budget["reserved"] == (len(payload()["text"]) if state == "RECONCILING" else 0)
        assert budget["spent"] == (len(payload()["text"]) if evidence == "terminated" else 0)
        assert len(calls) == 1
        assert await driver.cancel(None) == CancelOutcome.UNSUPPORTED
    finally:
        await driver.close()


@pytest.mark.integration
async def test_timeout_retains_character_and_compute_reservations_without_hidden_retry(store):
    calls = []

    async def backend(request):
        calls.append(request)
        raise httpx.ReadTimeout("execution may continue")

    driver, _, executor = await driver_case(store, backend)
    try:
        await executor.tick()
        assert (await store.operation("speech", "t"))["state"] == "RECONCILING"
        assert (await store.drain_status())["compute_held"] == 1
        assert len(calls) == 1
    finally:
        await driver.close()


@pytest.mark.integration
async def test_success_without_matching_proof_does_not_publish_audio(store):
    async def backend(request):
        return httpx.Response(200, content=waveform(), headers={"Content-Type": "audio/wav"})

    driver, _, executor = await driver_case(store, backend)
    try:
        await executor.tick()
        state = await store.operation("speech", "t")
        assert state["state"] == "RECONCILING" and state["result"] is None
    finally:
        await driver.close()


@pytest.mark.integration
async def test_speech_driver_rejects_payload_larger_than_reserved_before_http(store):
    calls = []

    async def backend(request):
        calls.append(request)
        raise AssertionError("unreserved inference")

    driver, _, _ = await driver_case(store, backend)
    try:
        reservation = await store.reserve_next("voice", "worker")
        with pytest.raises(DriverFailure) as failure:
            await driver.execute(reservation, payload() | {"text": payload()["text"] + "!"})
        assert failure.value.not_sent and not calls
    finally:
        await driver.close()
