import asyncio

import httpx
import pytest
from conftest import root
from fakes import MemoryArtifacts

from intramind_runtime.api import create_app
from intramind_runtime.store import rows

pytestmark = pytest.mark.integration
TOKEN = "isolated-test-credential-only-12345678"


async def test_lifecycle_snapshot_is_bounded_tenant_scoped_and_content_free(store):
    await store.create_root(root("visible", "owner"))
    await store.create_root(root("hidden", "other"))
    app = create_app(store, MemoryArtifacts(), TOKEN, {})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        assert (await client.post("/v1/runs/snapshot", json={"run_ids": ["visible"]})).status_code == 401
        client.headers.update({"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "owner"})
        response = await client.post("/v1/runs/snapshot", json={"run_ids": ["visible", "hidden"]})
        assert response.json() == {"items": [{"root_id": "visible", "state": "RUNNING",
            "cleanup_pending": False, "cancel_requested": False}]}
        assert (await client.post("/v1/runs/snapshot", json={"run_ids": ["visible"] * 101})).status_code == 422
        await store.cancel("visible", "owner")
        response = await client.post("/v1/runs/snapshot", json={"run_ids": ["visible"]})
        assert response.json()["items"][0]["state"] == "CANCELLED"


async def test_submit_is_idempotent_and_server_owns_policy(store):
    blobs = MemoryArtifacts()
    ref = await blobs.put("t", b"{}")
    app = create_app(
        store,
        blobs,
        TOKEN,
        {"summary/v1": {"deadline_seconds": 86400, "budget_limit": 300, "task_queue": "ai"}},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "t"},
    ) as client:
        body = {"task_type": "summary/v1", "submission_key": "request-1", "input": ref.model_dump()}
        first = await client.post("/v1/runs", json=body)
        again = await client.post("/v1/runs", json=body)
        assert first.status_code == again.status_code == 202
        assert first.json()["run_id"] == again.json()["run_id"]
        status = await client.get(f"/v1/runs/{first.json()['run_id']}")
        assert status.json()["input_digest"] == ref.sha256
        assert (
            await client.post("/v1/runs", json=body | {"priority": "interactive"})
        ).status_code == 422
        different = await blobs.put("t", b'{"changed":true}')
        assert (
            await client.post("/v1/runs", json=body | {"input": different.model_dump()})
        ).status_code == 409


async def test_cross_tenant_status_cancel_artifact_and_auth(store):
    blobs = MemoryArtifacts()
    ref = await blobs.put("owner", b"secret")
    await store.create_root(root("private", "owner"))
    app = create_app(store, blobs, TOKEN, {})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        assert (await client.get("/v1/runs/private")).status_code == 401
        client.headers.update({"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "other"})
        assert (await client.get("/v1/runs/private")).status_code == 404
        assert (await client.post("/v1/runs/private/cancel")).status_code == 404
        assert (await client.post("/v1/artifacts/read", json=ref.model_dump())).status_code == 404


async def test_concurrent_submissions_keep_first_configuration_and_one_start_intent(store):
    blobs = MemoryArtifacts()
    source = await blobs.put("t", b'{"document_ids":["doc"]}')
    configs = [await blobs.put("t", f'{{"max_modules":{n}}}'.encode()) for n in (3, 9)]
    app = create_app(
        store,
        blobs,
        TOKEN,
        {"teaching/v1": {"deadline_seconds": 86400, "budget_limit": 300, "task_queue": "ai"}},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "t"},
    ) as client:
        body = {"task_type": "teaching/v1", "submission_key": "same", "input": source.model_dump()}
        results = await asyncio.gather(
            *[
                client.post("/v1/runs", json=body | {"configuration": config.model_dump()})
                for config in configs
            ]
        )
        assert [result.status_code for result in results] == [202, 202]
        assert results[0].json()["run_id"] == results[1].json()["run_id"]
        async with store.engine.connect() as connection:
            submitted = await rows(connection, "SELECT spec FROM runtime_submissions")
            events = await rows(
                connection, "SELECT payload FROM runtime_outbox WHERE kind='start_workflow'"
            )
        assert len(submitted) == len(events) == 1
        first = submitted[0]["spec"]["input"]["configuration"]
        assert first in [ref.model_dump() for ref in configs]
        assert events[0]["payload"]["input"]["configuration"] == first
        retry = await client.post(
            "/v1/runs", json=body | {"configuration": configs[1].model_dump()}
        )
        assert retry.status_code == 202
        other_input = await blobs.put("t", b'{"document_ids":["other"]}')
        assert (
            await client.post("/v1/runs", json=body | {"input": other_input.model_dump()})
        ).status_code == 409


async def test_submission_configuration_requires_same_tenant_and_verified_bytes(store):
    blobs = MemoryArtifacts()
    source = await blobs.put("t", b"{}")
    foreign = await blobs.put("another", b'{"policy":"private"}')
    app = create_app(
        store,
        blobs,
        TOKEN,
        {"exam/v1": {"deadline_seconds": 600, "budget_limit": 300, "task_queue": "ai"}},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "t"},
    ) as client:
        body = {"task_type": "exam/v1", "submission_key": "same", "input": source.model_dump()}
        response = await client.post(
            "/v1/runs", json=body | {"configuration": foreign.model_dump()}
        )
        assert response.status_code == 404
        async with store.engine.connect() as connection:
            assert not await rows(connection, "SELECT root_id FROM runtime_roots")
