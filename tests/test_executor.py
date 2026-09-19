import asyncio

import pytest
from conftest import operation, pool, root
from fakes import IndependentEngine, MemoryArtifacts

from intramind_runtime.executor import Executor

pytestmark = pytest.mark.integration


async def prepare(store):
    await store.create_root(root())
    await store.configure_pool(pool(target=1), 1)
    blobs = MemoryArtifacts()
    spec = operation()
    blobs.data[spec.payload.key] = b'{"messages":[{"role":"user","content":"test"}]}'
    await store.submit_operation(spec)
    engine = IndependentEngine()
    return engine, Executor(store, blobs, engine, "p", "executor")


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
