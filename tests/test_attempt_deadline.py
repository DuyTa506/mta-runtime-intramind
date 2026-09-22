"""Attempt expiry never proves that independently running inference has stopped."""

import asyncio
from datetime import timedelta

import httpx
import pytest
from conftest import operation, pool, root
from fakes import IndependentEngine, MemoryArtifacts
from pydantic import ValidationError

from intramind_runtime.contracts import RuntimeConflict
from intramind_runtime.executor import Executor
from intramind_runtime.preparation import LlamaCppPromptSizer, PrepareRequest
from intramind_runtime.store import execute, row


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True, "600"])
def test_attempt_timeout_requires_finite_positive_numeric_policy(seconds):
    with pytest.raises(ValidationError):
        operation(attempt_timeout_seconds=seconds)


async def test_sizing_preserves_timeout_as_policy_outside_model_payload():
    async def backend(request):
        return httpx.Response(200, json={"prompt": "text", "tokens": [1, 2]})

    async with httpx.AsyncClient(
        base_url="http://fake/", transport=httpx.MockTransport(backend)
    ) as client:
        request = PrepareRequest(
            model_profile="test", payload={"messages": [{"role": "user", "content": "text"}]},
            max_output_tokens=20, attempt_timeout_seconds=17.5,
        )
        artifacts = MemoryArtifacts()
        result = await LlamaCppPromptSizer(
            client, model="test", capacity_profile_id="test-v1", context_limit=1024,
            token_margin=2, expected_output_tokens=10,
        ).prepare(request, artifacts, "t")
        assert result["attempt_timeout_seconds"] == 17.5
        assert b"attempt_timeout" not in next(iter(artifacts.data.values()))


@pytest.mark.integration
@pytest.mark.parametrize("seconds", [10, 7200])
async def test_reservation_deadline_uses_ledger_time_and_never_extends_root(store, seconds):
    accepted = root()
    await store.create_root(accepted)
    await store.configure_pool(pool(), 1)
    await store.submit_operation(operation(attempt_timeout_seconds=seconds))
    reserved = await store.reserve_next("p", "worker")
    async with store.engine.connect() as connection:
        attempt = await row(connection, "SELECT created_at FROM runtime_attempts")
    assert reserved.attempt_deadline == min(
        accepted.deadline, attempt["created_at"] + timedelta(seconds=seconds)
    )


@pytest.mark.integration
async def test_old_operation_without_timeout_remains_idempotent_but_policy_change_conflicts(store):
    await store.create_root(root())
    await store.submit_operation(operation())
    async with store.engine.begin() as connection:
        await execute(connection, "UPDATE runtime_operations SET spec=spec-'attempt_timeout_seconds'")
    assert await store.submit_operation(operation()) == "o"
    with pytest.raises(RuntimeConflict):
        await store.submit_operation(operation(attempt_timeout_seconds=300))


@pytest.mark.integration
async def test_send_rechecks_attempt_expiry_using_authoritative_time(store):
    await store.create_root(root())
    await store.configure_pool(pool(), 1)
    await store.submit_operation(operation(attempt_timeout_seconds=10))
    reserved = await store.reserve_next("p", "worker")
    async with store.engine.begin() as connection:
        await execute(connection, "UPDATE runtime_attempts SET created_at=now()-interval '11 seconds'")
    with pytest.raises(RuntimeConflict):
        await store.mark_send(reserved)
    async with store.engine.connect() as connection:
        assert (await row(connection, "SELECT state FROM runtime_attempts"))["state"] == "RESERVED"


async def prepared_executor(store, *, timeout=0.5):
    await store.create_root(root())
    await store.configure_pool(pool(target=1), 1)
    artifacts = MemoryArtifacts()
    spec = operation(attempt_timeout_seconds=timeout)
    artifacts.data[spec.payload.key] = b'{"messages":[{"role":"user","content":"test"}]}'
    await store.submit_operation(spec)
    engine = IndependentEngine()
    return engine, Executor(store, artifacts, engine, "p", "worker")


@pytest.mark.integration
@pytest.mark.parametrize("sent", [False, True])
async def test_total_timeout_releases_only_when_transport_has_not_started(store, monkeypatch, sent):
    engine, executor = await prepared_executor(store)
    if not sent:
        async def blocked_payload(_ref):
            await asyncio.Event().wait()

        monkeypatch.setattr(executor.artifacts, "get", blocked_payload)
    try:
        await asyncio.wait_for(executor.tick(), 5)
        state = await store.operation("o", "t")
        assert state["state"] == ("RECONCILING" if sent else "RETRY_WAIT")
        assert state["attempts"] == 1
        assert len(engine.calls) == int(sent)
        assert engine.finished == []
        assert (await store.run("r", "t"))["reserved"] == (30 if sent else 0)
        if sent:
            assert await store.reserve_next("p", "other") is None
        async with store.engine.connect() as connection:
            attempt = await row(connection, "SELECT * FROM runtime_attempts")
        assert attempt["error_class"] == "attempt_deadline_exceeded"
        assert attempt["compute_held"] == sent
    finally:
        engine.gate.set()
        await asyncio.gather(*engine.jobs)


@pytest.mark.integration
async def test_attempt_timeout_does_not_discard_completed_output_during_persistence(store, monkeypatch):
    engine, executor = await prepared_executor(store)
    original = executor.artifacts.put
    output_pending, resume = asyncio.Event(), asyncio.Event()

    async def blocked_put(*args, **kwargs):
        output_pending.set()
        await resume.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(executor.artifacts, "put", blocked_put)
    engine.gate.set()
    task = asyncio.create_task(executor.tick())
    try:
        await asyncio.wait_for(output_pending.wait(), 5)
        await asyncio.sleep(0.6)
        assert not task.done()
        assert (await store.drain_status())["compute_held"] == 0
        assert (await store.drain_status())["pending_settlement"] == 1
    finally:
        resume.set()
        await asyncio.wait_for(task, 5)
    assert (await store.operation("o", "t"))["state"] == "SUCCEEDED"
    assert len(engine.calls) == 1
