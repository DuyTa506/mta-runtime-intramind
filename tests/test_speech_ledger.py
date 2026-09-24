import asyncio

import pytest
from conftest import operation, pool, root
from pydantic import ValidationError

from intramind_runtime import contracts
from intramind_runtime.store import execute, row


def speech(operation_id="speech", *, characters=12, **kwargs):
    return contracts.SpeechOperationSpec(
        operation_id=operation_id, root_id="r", tenant_id="t", payload=operation().payload,
        model_profile="speech-test", characters_bound=characters, expected_cost=characters,
        **kwargs,
    )


def speech_pool(**kwargs):
    original = pool().model_dump(exclude={"context_limit", "model_profile", "pool_id"})
    return contracts.SpeechPoolSpec(
        **(original | {"pool_id": "voice", "model_profile": "speech-test",
                       "character_limit": 2000} | kwargs),
    )


def test_speech_contract_has_its_own_units_and_no_token_fields():
    spec = speech()
    assert spec.budget_unit == "speech_characters"
    assert spec.budget_bound == 12
    assert "max_output_tokens" not in spec.model_dump()
    assert "input_tokens_bound" not in spec.model_dump()
    assert contracts.parse_operation(spec.model_dump()) == spec
    with pytest.raises(ValidationError):
        speech(max_output_tokens=1)


@pytest.mark.parametrize("budgets", [{"tokens": 10}, {"speech_characters": 0},
                                    {"speech_characters": True}])
def test_extra_root_budget_requires_a_supported_unit_and_strict_positive_limit(budgets):
    with pytest.raises(ValidationError):
        root(resource_budgets=budgets)


@pytest.mark.integration
async def test_speech_without_accepted_root_budget_is_rejected_before_materializing(store):
    await store.create_root(root())
    with pytest.raises(contracts.AdmissionDenied, match="speech_characters"):
        await store.submit_operation(speech())
    assert (await store.run("r", "t"))["operations"] == {}


@pytest.mark.integration
async def test_character_budget_is_reserved_atomically_and_does_not_charge_llm_tokens(store):
    await store.create_root(root(resource_budgets={"speech_characters": 12}))
    await store.configure_pool(speech_pool(), 4)
    for index in range(3):
        await store.submit_operation(speech(str(index)))
    results = await asyncio.gather(*(store.reserve_next("voice", str(i)) for i in range(8)))
    reserved = [result for result in results if result]
    assert len(reserved) == 3
    state = await store.run("r", "t")
    assert (state["reserved"], state["spent"]) == (0, 0)
    assert state["resource_budgets"]["speech_characters"] == {
        "limit": 12, "reserved": 0, "spent": 0,
    }
    attempt = reserved[0]
    await store.mark_send(attempt)
    for waiting in reserved[1:]:
        with pytest.raises(contracts.RuntimeConflict, match="budget unavailable"):
            await store.mark_send(waiting)
        await store.fail(waiting, "budget_wait", not_sent=True)
    assert (await store.run("r", "t"))["resource_budgets"]["speech_characters"]["reserved"] == 12
    await store.compute_finished(attempt)
    await store.commit_result(attempt, attempt.operation.payload, 12)
    await store.commit_result(attempt, attempt.operation.payload, 12)
    assert (await store.run("r", "t"))["resource_budgets"]["speech_characters"] == {
        "limit": 12, "reserved": 0, "spent": 12,
    }
    assert await store.reserve_next("voice", "next") is None
    async with store.engine.connect() as connection:
        assert (await row(connection, "SELECT attempts FROM runtime_roots"))["attempts"] == 1
        assert (await row(connection, "SELECT budget_unit FROM runtime_attempts"))["budget_unit"] == "speech_characters"


@pytest.mark.integration
async def test_speech_and_completion_share_group_capacity_and_root_attempt_limit(store):
    await store.create_root(root(max_attempts=1, resource_budgets={"speech_characters": 100}))
    await store.configure_pool(pool(), 1)
    await store.configure_pool(speech_pool(), 1)
    await store.submit_operation(speech())
    await store.submit_operation(operation())
    attempt = await store.reserve_next("voice", "speech-worker")
    waiting = await store.reserve_next("p", "llm-worker")
    assert waiting is not None
    await store.mark_send(attempt)
    with pytest.raises(contracts.RuntimeConflict, match="attempt budget"):
        await store.mark_send(waiting)
    await store.fail(waiting, "root_attempt_budget", not_sent=True, retry=True)
    await store.fail(attempt, "connect_failed", not_sent=True)
    assert await store.reserve_next("p", "llm-worker") is None
    assert (await store.operation("o", "t"))["wait_reason"] == "root_attempt_budget"


@pytest.mark.integration
async def test_speech_unknown_survives_expiry_and_cancel_until_epoch_proof(store):
    await store.create_root(root(resource_budgets={"speech_characters": 100}))
    await store.configure_pool(speech_pool(target=1), 1)
    await store.submit_operation(speech())
    attempt = await store.reserve_next("voice", "worker")
    await store.mark_send(attempt)
    async with store.engine.begin() as connection:
        await execute(connection, "UPDATE runtime_attempts SET lease_expires_at=now()-interval '1 second'")
    await store.reconcile_expired()
    await store.cancel("r", "t")
    assert await store.reserve_next("voice", "replacement") is None
    state = await store.run("r", "t")
    assert state["cleanup_pending"]
    assert state["resource_budgets"]["speech_characters"]["reserved"] == 12
    await store.confirm_epoch_stopped("voice", "e1", "test engine joined and exited")
    await store.confirm_epoch_stopped("voice", "e1", "duplicate proof")
    state = await store.run("r", "t")
    assert state["state"] == "CANCELLED" and not state["cleanup_pending"]
    assert state["resource_budgets"]["speech_characters"] == {
        "limit": 100, "reserved": 0, "spent": 12,
    }


@pytest.mark.integration
async def test_speech_rejects_oversized_request_without_blocking_smaller_work(store):
    await store.create_root(root(resource_budgets={"speech_characters": 10000}))
    await store.configure_pool(speech_pool(character_limit=10), 1)
    await store.submit_operation(speech("large"))
    await store.submit_operation(speech("small", characters=10))
    assert (await store.reserve_next("voice", "worker")).operation.operation_id == "small"
    await store.reconcile_expired()
    assert (await store.operation("large", "t"))["wait_reason"] == "request_exceeds_all_pools"


@pytest.mark.integration
async def test_legacy_root_without_extra_budgets_attaches_without_changing_policy(store):
    accepted = root()
    await store.create_root(accepted)
    async with store.engine.begin() as connection:
        await execute(connection, "UPDATE runtime_roots SET spec=spec-'resource_budgets'")
    await store.create_root(accepted)
    with pytest.raises(contracts.RuntimeConflict):
        await store.create_root(accepted.model_copy(update={"resource_budgets": {"speech_characters": 12}}))


@pytest.mark.integration
async def test_unsent_speech_lease_refunds_once_and_fences_old_worker(store):
    await store.create_root(root(resource_budgets={"speech_characters": 12}))
    await store.configure_pool(speech_pool(), 1)
    await store.submit_operation(speech())
    attempt = await store.reserve_next("voice", "old")
    async with store.engine.begin() as connection:
        await execute(connection, "UPDATE runtime_attempts SET lease_expires_at=now()-interval '1 second'")
    await store.reconcile_expired()
    await store.reconcile_expired()
    assert (await store.run("r", "t"))["resource_budgets"]["speech_characters"]["reserved"] == 0
    with pytest.raises(contracts.RuntimeConflict):
        await store.mark_send(attempt)
    assert await store.reserve_next("voice", "new")


@pytest.mark.integration
@pytest.mark.parametrize("cancelled", [False, True])
async def test_speech_usage_overrun_stops_pool_and_preserves_existing_cancellation(store, cancelled):
    await store.create_root(root(resource_budgets={"speech_characters": 12}))
    await store.configure_pool(speech_pool(), 1)
    await store.submit_operation(speech())
    attempt = await store.reserve_next("voice", "worker")
    await store.mark_send(attempt)
    if cancelled:
        await store.cancel("r", "t")
    await store.compute_finished(attempt)
    await store.commit_result(attempt, attempt.operation.payload, 13)
    state = await store.run("r", "t")
    assert state["state"] == ("CANCELLED" if cancelled else "FAILED")
    assert state["spent"] == 0
    assert state["resource_budgets"]["speech_characters"]["spent"] == 13
    async with store.engine.connect() as connection:
        capacity = await row(connection, "SELECT health,target FROM runtime_pools WHERE pool_id='voice'")
    assert dict(capacity) == {"health": "DEGRADED", "target": 0}
