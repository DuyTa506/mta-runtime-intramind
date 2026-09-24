import asyncio
from datetime import UTC, datetime, timedelta
from time import monotonic

import pytest
from conftest import operation, pool, root
from fakes import IndependentEngine, MemoryArtifacts, TestPermits
from test_embedding import embedding, embedding_pool

from intramind_runtime.contracts import EmbeddingResult
from intramind_runtime.direct import DirectRequest
from intramind_runtime.executor import Executor
from intramind_runtime.memory_scheduler import MemoryScheduler
from intramind_runtime.store import row

pytestmark = pytest.mark.integration


async def prepare(store):
    await store.create_root(root())
    await store.configure_pool(pool(target=1), 1)
    blobs = MemoryArtifacts()
    spec = operation()
    blobs.data[spec.payload.key] = b'{"messages":[{"role":"user","content":"test"}]}'
    await store.submit_operation(spec)
    engine = IndependentEngine()
    return engine, Executor(store, blobs, engine, "p", "executor", TestPermits())


async def test_embedding_qa_capacity_wait_does_not_spend_durable_attempt_or_budget(store):
    await store.create_root(root(max_attempts=1,
                                 resource_budgets={"embedding_characters": 4}))
    await store.configure_pool(embedding_pool(target=1, transport_limit=1,
                                              background_transport_limit=1), 1)
    blobs = MemoryArtifacts()
    payload = await blobs.put("t", b'{"texts":["ab","cd"],"input_type":"document"}')
    spec = embedding(max_attempts=1, attempt_timeout_seconds=0.3).model_copy(
        update={"payload": payload})
    await store.submit_operation(spec)
    scheduler = MemoryScheduler(store)
    await scheduler.start()

    class Permits:
        async def acquire(self, reservation):
            return await scheduler.durable(
                reservation.attempt_id, reservation.pool_id, reservation.owner_id,
                reservation.engine_epoch, reservation.workload_class,
                reservation.attempt_deadline)

        async def heartbeat(self, reservation):
            scheduler.durable_heartbeat(reservation.attempt_id, reservation.owner_id)

        async def release(self, reservation):
            await scheduler.durable_release(reservation.attempt_id, reservation.owner_id)

    class EmbeddingEngine:
        calls = 0

        async def execute(self, reservation, payload):
            self.calls += 1
            return EmbeddingResult(body={"embeddings": [[0.1], [0.2]]}, characters=4)

        async def cancel(self, reservation):
            raise AssertionError("QA capacity wait must not send to the engine")

    engine = EmbeddingEngine()
    executor = Executor(store, blobs, engine, "embedding", "worker", Permits())
    qa = DirectRequest(request_id="qa-occupies-embedding", tenant_id="t",
        payload_digest="a" * 64, model_profile="embedding-test",
        capacity_profile_id="embed-v1", kind="embedding", request_bound=4,
        batch_size=2, workload_class="qa",
        deadline=datetime.now(UTC) + timedelta(minutes=1))
    await scheduler.enqueue(qa, "embedding", "qa-owner")
    held = await scheduler.reserve(qa, "embedding", "qa-owner")
    assert held is not None
    await scheduler.mark_send(held)
    try:
        started = monotonic()
        assert await executor.tick()
        assert monotonic() - started >= 0.3
        assert engine.calls == 0
        assert (await store.operation(spec.operation_id, "t"))["attempts"] == 0
        state = await store.run("r", "t")
        assert state["reserved"] == 0
        assert state["resource_budgets"]["embedding_characters"]["reserved"] == 0
        async with store.engine.connect() as c:
            root_attempts = await row(c, "SELECT attempts FROM runtime_roots WHERE root_id='r'")
            first = await row(c, """SELECT state,compute_held,budget_held FROM runtime_attempts
                WHERE operation_id=:id""", id=spec.operation_id)
        assert root_attempts["attempts"] == 0
        assert dict(first) == {"state": "FAILED_NOT_SENT", "compute_held": False,
                               "budget_held": False}
        await scheduler.finish(held)
        assert await executor.tick(), "background work must run as soon as QA releases the slot"
        assert engine.calls == 1
        assert (await store.operation(spec.operation_id, "t"))["state"] == "SUCCEEDED"
        assert (await store.operation(spec.operation_id, "t"))["attempts"] == 1
        async with store.engine.connect() as c:
            assert (await row(c, "SELECT attempts FROM runtime_roots WHERE root_id='r'"))["attempts"] == 1
        assert (await store.run("r", "t"))["resource_budgets"]["embedding_characters"] == {
            "limit": 4, "reserved": 0, "spent": 4}
        assert not scheduler.permits
    finally:
        await scheduler.close()


async def test_transport_cancel_does_not_stop_backend_or_release_quota(store):
    engine, executor = await prepare(store)
    transport = asyncio.create_task(executor.tick())
    await engine.started.wait()
    transport.cancel()
    with pytest.raises(asyncio.CancelledError):
        await transport
    assert not engine.finished
    assert (await store.run("r", "t"))["reserved"] == 30
    assert await store.reserve_next("p", "other") is None
    engine.gate.set()
    await asyncio.gather(*engine.jobs)
    assert len(engine.finished) == 1
    assert (await store.operation("o", "t"))["state"] == "RECONCILING"


async def test_result_persistence_failure_does_not_rerun_inference(store, monkeypatch):
    engine, executor = await prepare(store)
    original = executor.artifacts.put
    failures = 0

    async def flaky(*args, **kwargs):
        nonlocal failures
        failures += 1
        if failures == 1:
            raise OSError("injected storage outage")
        return await original(*args, **kwargs)

    monkeypatch.setattr(executor.artifacts, "put", flaky)
    engine.gate.set()
    await asyncio.wait_for(executor.tick(), 10)
    assert len(engine.calls) == 1 and failures == 2
    assert (await store.operation("o", "t"))["state"] == "SUCCEEDED"


async def test_db_failure_after_compute_preserves_in_memory_result(store, monkeypatch):
    engine, executor = await prepare(store)
    original = store.compute_finished
    failures = 0

    async def flaky(reservation):
        nonlocal failures
        failures += 1
        if failures == 1:
            raise OSError("injected database outage")
        return await original(reservation)

    monkeypatch.setattr(store, "compute_finished", flaky)
    engine.gate.set()
    await asyncio.wait_for(executor.tick(), 10)
    assert len(engine.calls) == 1 and failures == 2
    assert (await store.operation("o", "t"))["state"] == "SUCCEEDED"


@pytest.mark.parametrize("phase", ["artifact", "commit"])
async def test_drain_waits_for_result_persistence_after_compute_ends(store, monkeypatch, phase):
    engine, executor = await prepare(store)
    pending = asyncio.Event()
    resume = asyncio.Event()
    target, method = (executor.artifacts, "put") if phase == "artifact" else (store, "commit_result")
    original = getattr(target, method)

    async def blocked(*args, **kwargs):
        pending.set()
        await resume.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(target, method, blocked)
    engine.gate.set()
    attempt = asyncio.create_task(executor.tick())
    try:
        await asyncio.wait_for(pending.wait(), 5)
        assert await store.drain_status() == {
            "compute_held": 0,
            "pending_settlement": 1,
            "unknown_attempts": 0,
            "unsettled_attempts": 1,
        }
        assert (await store.run("r", "t"))["reserved"] == 30
        assert not attempt.done()
    finally:
        resume.set()
        await asyncio.wait_for(attempt, 5)
    assert not any((await store.drain_status()).values())
    assert (await store.run("r", "t"))["spent"] == 20
    assert (await store.operation("o", "t"))["state"] == "SUCCEEDED"
    assert len(engine.calls) == 1


async def test_unknown_with_terminated_compute_blocks_drain_until_reconciled(store):
    await prepare(store)
    reservation = await store.reserve_next("p", "executor")
    await store.mark_send(reservation)
    await store.compute_finished(reservation)
    await store.unknown(reservation, "result_lost_after_compute")
    assert await store.drain_status() == {
        "compute_held": 0,
        "pending_settlement": 1,
        "unknown_attempts": 1,
        "unsettled_attempts": 1,
    }
    await store.confirm_epoch_stopped("p", "e1", "independent test epoch termination evidence")
    assert not any((await store.drain_status()).values())
