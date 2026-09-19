import asyncio
import json

import httpx
import pytest
from fakes import MemoryArtifacts

from intramind_runtime.api import create_app
from intramind_runtime.buffering import BufferedSubmission, BufferedSubmissions
from intramind_runtime.contracts import AdmissionDenied, Artifact, NotFound, RuntimeConflict
from intramind_runtime.store import execute, rows

pytestmark = pytest.mark.integration
TOKEN = "isolated-buffer-test-credential-12345678"
DEFINITION = {"deadline_seconds": 3600, "budget_limit": 10000, "task_queue": "ai",
              "buffering": {"max_batch_size": 50, "max_delay_seconds": 7200,
                            "max_pending_per_tenant": 10, "max_pending_per_partition": 4}}


async def setup(store, *, key="turn-1", tenant="t", partition="user", configuration=1, delay=0):
    blobs = MemoryArtifacts()
    inputs = await blobs.put(tenant, json.dumps({"query": key}).encode())
    policy = await blobs.put(tenant, json.dumps({"version": configuration}).encode())
    request = BufferedSubmission(task_type="memory/v1", submission_key=key,
        partition_key=partition, input=inputs, configuration=policy,
        delay_seconds=delay, batch_size=3)
    return BufferedSubmissions(store, blobs), request


async def add(broker, request, key, **changes):
    ref = await broker.artifacts.put("t", json.dumps({"query": key}).encode())
    request = request.model_copy(update={"submission_key": key, "input": ref, **changes})
    return await broker.append("t", request, DEFINITION)


async def test_accepted_turn_retries_preserve_policy_and_detect_input_conflict(store):
    broker, request = await setup(store)
    ids = await asyncio.gather(*[broker.append("t", request, DEFINITION) for _ in range(8)])
    assert len(set(ids)) == 1
    newer = await broker.artifacts.put("t", b'{"version":2}')
    assert await broker.append("t", request.model_copy(update={"configuration": newer}), {}) == ids[0]
    other = await broker.artifacts.put("t", b'{"query":"different"}')
    with pytest.raises(RuntimeConflict):
        await broker.append("t", request.model_copy(update={"input": other}), DEFINITION)
    batch = (await broker.seal())[0]
    assert batch["configuration"] == request.configuration.model_dump(mode="json")
    with pytest.raises(NotFound):
        await broker.status("other-tenant", ids[0])


async def test_concurrent_sealers_one_root_lost_submit_ack_and_ordered_next_batch(store, monkeypatch):
    broker, request = await setup(store)
    first = await broker.append("t", request, DEFINITION)
    await add(broker, request, "turn-2")
    batches = await asyncio.gather(*[broker.seal() for _ in range(4)])
    assert len({batch["batch_id"] for found in batches for batch in found}) == 1
    batch = batches[0][0]
    original = store.submit_run

    async def lost_ack(*args):
        await original(*args)
        raise ConnectionError("submit committed, response lost")

    monkeypatch.setattr(store, "submit_run", lost_ack)
    with pytest.raises(ConnectionError):
        await broker.dispatch(batch)
    later = await add(broker, request, "turn-3")
    monkeypatch.setattr(store, "submit_run", original)
    # Process restart: recover only the sealed snapshot, not the new pending turn.
    restarted = BufferedSubmissions(store, broker.artifacts, control_queue="new-deployment-queue")
    await asyncio.gather(*[restarted.dispatch(batch) for _ in range(4)])
    assert await restarted.seal() == []
    assert (await restarted.status("t", first))["state"] == "RUNNING"
    assert (await restarted.status("t", later))["state"] == "PENDING"
    async with store.engine.connect() as c:
        submits = await rows(c, "SELECT spec FROM runtime_submissions")
        starts = await rows(c, "SELECT * FROM runtime_outbox WHERE kind='start_workflow'")
    assert len(submits) == len(starts) == 1
    assert submits[0]["spec"]["input"]["control_queue"] == "intramind-control"
    payload = json.loads(await broker.artifacts.get(Artifact.model_validate(submits[0]["spec"]["input"]["input"])))
    assert len(payload["items"]) == 2
    await store.finish_run(batch["batch_id"], "t", "FAILED", reason="test-finished")
    next_batch = (await restarted.seal())[0]
    assert next_batch["batch_id"] != batch["batch_id"]


async def test_policy_change_seals_contiguous_batches_in_input_order(store):
    broker, request = await setup(store)
    await broker.append("t", request, DEFINITION)
    newer = await broker.artifacts.put("t", b'{"version":2}')
    second = await add(broker, request, "turn-2", configuration=newer)
    third = await add(broker, request, "turn-3")
    batch = (await broker.seal())[0]
    await broker.dispatch(batch)
    assert (await broker.status("t", second))["state"] == "PENDING"
    assert (await broker.status("t", third))["state"] == "PENDING"
    await store.finish_run(batch["batch_id"], "t", "FAILED", reason="test-finished")
    batch = (await broker.seal())[0]
    assert batch["configuration"] == newer.model_dump(mode="json")
    await broker.dispatch(batch)
    assert (await broker.status("t", third))["state"] == "PENDING"


async def test_delay_and_backpressure_do_not_trim_or_expire_accepted_items(store):
    broker, request = await setup(store, delay=7200)
    first = await broker.append("t", request, DEFINITION)
    assert await broker.seal() == []
    await asyncio.gather(*[add(broker, request, f"turn-{n}") for n in range(2, 5)])
    with pytest.raises(AdmissionDenied, match="backlog full"):
        await add(broker, request, "turn-5")
    assert await broker.append("t", request, DEFINITION) == first
    async with store.transaction() as c:
        await execute(c, "UPDATE runtime_buffer_items SET due_at=now()-interval '2 days'")
    batch = (await broker.seal())[0]
    await broker.dispatch(batch)
    async with store.engine.connect() as c:
        retained = await rows(c, "SELECT * FROM runtime_buffer_items")
    assert len(retained) == 4
    assert len([item for item in retained if item["batch_id"]]) == 3


async def test_unavailable_namespace_does_not_block_others_and_deadline_is_explicit(store, monkeypatch):
    broker, request = await setup(store)
    first = await broker.append("t", request, DEFINITION)
    second = await add(broker, request, "turn-2", partition_key="different")
    original = broker.dispatch

    async def unavailable(batch):
        if batch["partition_key"] == "user":
            raise ConnectionError("artifact unavailable")
        await original(batch)

    monkeypatch.setattr(broker, "dispatch", unavailable)
    assert await broker.tick() == 2
    assert (await broker.status("t", first))["state"] == "PREPARING"
    assert (await broker.status("t", second))["state"] == "RUNNING"
    async with store.transaction() as c:
        await execute(c, """UPDATE runtime_buffer_batches SET deadline=now()-interval '1 second',
            retry_at=now()-interval '1 second' WHERE state='PREPARING'""")
    assert await broker.tick() == 1
    status = await broker.status("t", first)
    assert status["state"] == "FAILED"
    assert status["terminal_reason"] == "buffer_dispatch_deadline_exceeded"


async def test_api_auth_artifact_scope_and_buffer_status(store):
    broker, request = await setup(store)
    app = create_app(store, broker.artifacts, TOKEN, {"memory/v1": DEFINITION})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        body = request.model_dump(mode="json")
        assert (await client.post("/v1/buffers", json=body)).status_code == 401
        client.headers.update({"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "other"})
        assert (await client.post("/v1/buffers", json=body)).status_code == 404
        client.headers["X-Tenant-ID"] = "t"
        response = await client.post("/v1/buffers", json=body)
        assert response.status_code == 202
        item_id = response.json()["item_id"]
        assert (await client.get(f"/v1/buffers/{item_id}")).json()["state"] == "PENDING"
        client.headers["X-Tenant-ID"] = "other"
        assert (await client.get(f"/v1/buffers/{item_id}")).status_code == 404


async def test_size_bound_prevents_an_oversized_batch_before_acceptance(store):
    broker, request = await setup(store)
    definition = DEFINITION | {"buffering": DEFINITION["buffering"] | {"max_item_bytes": 1}}
    with pytest.raises(AdmissionDenied, match="size bound"):
        await broker.append("t", request, definition)
    assert await broker.seal() == []
