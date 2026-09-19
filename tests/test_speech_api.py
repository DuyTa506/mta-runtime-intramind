import httpx
import pytest
from fakes import MemoryArtifacts
from test_speech_transport import payload, profile

from intramind_runtime.api import create_app
from intramind_runtime.speech import SpeechPreparer


@pytest.mark.integration
async def test_speech_profile_prepare_and_root_budget_are_owned_by_trusted_catalog(store):
    blobs, token = MemoryArtifacts(), "isolated-speech-api-test-token-12345678"
    definitions = {"audio/v1": {"deadline_seconds": 60, "budget_limit": 1000,
        "resource_budgets": {"speech_characters": 300}, "task_queue": "audio"}}
    app = create_app(store, blobs, token, definitions,
                     speech_preparers={"speech-test": SpeechPreparer(profile())})
    async with httpx.AsyncClient(base_url="http://runtime", transport=httpx.ASGITransport(app)) as client:
        assert (await client.get("/v1/speech/profiles/speech-test")).status_code == 401
        client.headers.update({"Authorization": f"Bearer {token}", "X-Tenant-ID": "t"})
        snapshot = await client.get("/v1/speech/profiles/speech-test")
        assert snapshot.json() == profile().model_dump(mode="json")
        request = {"model_profile": "speech-test", "capacity_profile_id": "test-v1",
                   "payload": payload(), "attempt_timeout_seconds": 25}
        prepared = await client.post("/v1/speech/prepare", json=request)
        assert prepared.status_code == 200
        assert prepared.json()["characters_bound"] == len(payload()["text"])
        assert (await client.post("/v1/speech/prepare", json=request | {
            "capacity_profile_id": "changed"})).status_code == 422
        ref = await blobs.put("t", b"{}")
        submission = {"task_type": "audio/v1", "submission_key": "episode", "input": ref.model_dump()}
        accepted = await client.post("/v1/runs", json=submission)
        assert accepted.status_code == 202
        definitions["audio/v1"]["resource_budgets"]["speech_characters"] = 999
        assert (await client.post("/v1/runs", json=submission)).status_code == 202
        status = await client.get(f"/v1/runs/{accepted.json()['run_id']}")
        assert status.json()["resource_budgets"]["speech_characters"]["limit"] == 300
        assert (await client.post("/v1/runs", json=submission | {
            "resource_budgets": {"speech_characters": 1000000}})).status_code == 422
