"""Permit waiting does not consume send attempts or misclassify budget changes."""

import logging

import pytest
from conftest import operation, pool, root
from fakes import MemoryArtifacts, TestPermits

from intramind_runtime.drivers import DriverFailure
from intramind_runtime.executor import Executor

pytestmark = pytest.mark.integration


class NeverRun:
    def __init__(self):
        self.calls = 0

    async def execute(self, reservation, payload):
        self.calls += 1
        raise AssertionError("no budget means no engine send")


async def test_budget_changed_before_send_retries_without_fenced_warning(store, monkeypatch, caplog):
    await store.configure_pool(pool(target=2), 2)
    await store.create_root(root(budget_limit=30))
    await store.submit_operation(operation("first"))
    await store.submit_operation(operation("second"))
    first = await store.reserve_next("p", "worker-1")
    second = await store.reserve_next("p", "worker-2")
    await store.mark_send(first)
    artifacts = MemoryArtifacts()
    artifacts.data[second.operation.payload.key] = (
        b'{"messages":[{"role":"user","content":"test"}]}')
    driver = NeverRun()

    async def selected(_pool, _owner):
        return second

    monkeypatch.setattr(store, "reserve_next", selected)
    with caplog.at_level(logging.INFO):
        assert await Executor(store, artifacts, driver, "p", "worker-2", TestPermits()).tick()
    state = await store.operation("second", "t")
    assert state["state"] == "RETRY_WAIT"
    assert state["wait_reason"] == "send_budget_unavailable"
    assert state["attempts"] == 0 and driver.calls == 0
    assert (await store.run("r", "t"))["reserved"] == 30
    assert "attempt could not send under current budget" in caplog.text
    assert "attempt fenced" not in caplog.text


async def test_backoff_uses_sent_attempts_not_ledger_reservations(store, monkeypatch):
    await store.configure_pool(pool(target=1), 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    artifacts = MemoryArtifacts()
    artifacts.data[operation().payload.key] = (
        b'{"messages":[{"role":"user","content":"test"}]}')

    class BusyThenReady(TestPermits):
        def __init__(self):
            self.calls = 0

        async def acquire(self, reservation):
            self.calls += 1
            if self.calls <= 3:
                raise TimeoutError("permit wait expired")
            return await super().acquire(reservation)

    class ConnectFailure:
        async def execute(self, reservation, payload):
            raise DriverFailure("connect_failed", not_sent=True, retry=True)

    bounds = []

    def zero_delay(low, high):
        bounds.append(high)
        return 0

    monkeypatch.setattr("intramind_runtime.executor.random.uniform", zero_delay)
    executor = Executor(store, artifacts, ConnectFailure(), "p", "worker", BusyThenReady())
    for _ in range(4):
        assert await executor.tick()
    assert bounds == [2]
    state = await store.operation("o", "t")
    assert state["attempts"] == 1
    assert state["state"] == "RETRY_WAIT"
