import httpx
import pytest
from conftest import root
from fakes import MemoryArtifacts

from intramind_runtime.api import create_app

pytestmark = pytest.mark.integration
TOKEN = "isolated-test-credential-only-12345678"


async def test_submit_is_idempotent_and_server_owns_policy(store):
    blobs = MemoryArtifacts()
    ref = await blobs.put("t", b'{}')
    app = create_app(store, blobs, TOKEN, {"summary/v1": {
        "deadline_seconds": 86400, "budget_limit": 300, "task_queue": "ai"}})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test",
                                headers={"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "t"}) as client:
        body = {"task_type": "summary/v1", "submission_key": "request-1", "input": ref.model_dump()}
        first = await client.post("/v1/runs", json=body)
        again = await client.post("/v1/runs", json=body)
        assert first.status_code == again.status_code == 202
        assert first.json()["run_id"] == again.json()["run_id"]
        assert (await client.post("/v1/runs", json=body | {"priority": "interactive"})).status_code == 422
        different = await blobs.put("t", b'{"changed":true}')
        assert (await client.post("/v1/runs", json=body | {"input": different.model_dump()})).status_code == 409


async def test_cross_tenant_status_cancel_artifact_and_auth(store):
    blobs = MemoryArtifacts()
    ref = await blobs.put("owner", b'secret')
    await store.create_root(root("private", "owner"))
    app = create_app(store, blobs, TOKEN, {})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        assert (await client.get("/v1/runs/private")).status_code == 401
        client.headers.update({"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "other"})
        assert (await client.get("/v1/runs/private")).status_code == 404
        assert (await client.post("/v1/runs/private/cancel")).status_code == 404
        assert (await client.post("/v1/artifacts/read", json=ref.model_dump())).status_code == 404
