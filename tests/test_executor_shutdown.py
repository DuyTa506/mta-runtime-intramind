"""Graceful process shutdown retains completed output through a persistence outage."""

import asyncio
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from conftest import operation, pool, root
from fakes import IndependentEngine, MemoryArtifacts
from pydantic import SecretStr

from intramind_runtime import cli
from intramind_runtime.contracts import Reservation


@pytest.mark.parametrize("phase", ["artifact", "commit"])
async def test_sigterm_waits_for_pending_output_and_does_not_dispatch_again(monkeypatch, phase):
    spec = operation()
    reservation = Reservation(
        attempt_id="a", operation=spec, pool_id="p", engine_epoch="e1",
        model_revision="model-1", owner_id="executor", lease_epoch=1,
        attempt_deadline=root().deadline,
    )
    store = SimpleNamespace(
        lease_seconds=60,
        reserve_next=AsyncMock(return_value=reservation),
        mark_send=AsyncMock(),
        compute_finished=AsyncMock(),
        commit_result=AsyncMock(),
        unknown=AsyncMock(),
        fail=AsyncMock(),
        close=AsyncMock(),
    )
    blobs = MemoryArtifacts()
    blobs.data[spec.payload.key] = b'{"messages":[{"role":"user","content":"test"}]}'
    blobs.ready = AsyncMock()
    engine = IndependentEngine()
    engine.gate.set()
    engine.close = AsyncMock()
    pending = asyncio.Event()
    restored = asyncio.Event()
    target, method = (blobs, "put") if phase == "artifact" else (store, "commit_result")
    original = getattr(target, method)
    writes = 0

    async def outage(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 1:
            pending.set()
            await restored.wait()
            raise OSError("injected persistence outage")
        return await original(*args, **kwargs)

    monkeypatch.setattr(target, method, outage)
    monkeypatch.setattr(cli, "Store", lambda *args, **kwargs: store)
    monkeypatch.setattr(cli, "artifacts", lambda _settings: blobs)
    monkeypatch.setattr(cli, "OpenAICompletionDriver", lambda *args: engine)
    wakeup = SimpleNamespace(generation=0, start=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(cli, "Wakeup", lambda _url: wakeup)
    callbacks = {}
    monkeypatch.setattr(
        asyncio.get_running_loop(), "add_signal_handler",
        lambda sig, callback: callbacks.__setitem__(sig, callback),
    )
    monkeypatch.setenv("SHUTDOWN_TEST_KEY", "local-fake-key")
    settings = SimpleNamespace(
        database_url=SecretStr("unused"), lease_seconds=60, executor_count=1,
    )
    config = {"pools": [{
        "base_url": "http://unused.invalid", "api_key_env": "SHUTDOWN_TEST_KEY",
        "model": "test", "admission": pool().model_dump(mode="json"),
    }]}
    service = asyncio.create_task(cli.services("executor", settings, config))
    try:
        await asyncio.wait_for(pending.wait(), 5)
        store.compute_finished.assert_awaited_once()
        callbacks[signal.SIGTERM]()
        # Let the signal path reach task cleanup while persistence is still blocked.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not service.done()
        store.close.assert_not_awaited()
        engine.close.assert_not_awaited()
        store.unknown.assert_not_awaited()
        restored.set()
        await asyncio.wait_for(service, 5)
    finally:
        restored.set()
        if not service.done():
            service.cancel()
        await asyncio.gather(service, return_exceptions=True)
    assert len(engine.calls) == 1 and writes == 2
    store.reserve_next.assert_awaited_once()
    if phase == "commit":
        original.assert_awaited_once()
    else:
        store.commit_result.assert_awaited_once()
    store.unknown.assert_not_awaited()
    store.fail.assert_not_awaited()
    store.close.assert_awaited_once()
    engine.close.assert_awaited_once()
