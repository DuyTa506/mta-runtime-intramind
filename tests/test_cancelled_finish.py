"""SDK cancellation finalization must complete without losing cleanup accounting."""

import pytest
from conftest import operation, pool, root
from sqlalchemy import text

from intramind_runtime.temporal_adapter import BrokerActivities

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("already_cancelled", [False, True])
async def test_finish_cancelled_is_idempotent_and_keeps_sent_work_accounted(store, already_cancelled):
    await store.create_root(root())
    await store.configure_pool(pool(target=1), 1)
    await store.submit_operation(operation())
    attempt = await store.reserve_next("p", "worker")
    await store.mark_send(attempt)
    if already_cancelled:
        await store.cancel("r", "t")
    activity = BrokerActivities(store)
    payload = {"root_id": "r", "tenant_id": "t", "state": "CANCELLED", "reason": "user_cancelled"}
    await activity.finish(payload)
    await activity.finish(payload)
    status = await store.run("r", "t")
    assert status["state"] == "CANCELLED"
    assert status["cleanup_pending"] and status["reserved"] == 30
    assert (await store.operation("o", "t"))["state"] == "CANCELLED"
    async with store.engine.connect() as connection:
        assert (await connection.execute(text(
            "SELECT count(*) FROM runtime_outbox WHERE kind='cancel_workflow'"
        ))).scalar_one() == 1
    await store.compute_finished(attempt)
    await store.commit_result(attempt, attempt.operation.payload, 12)
    status = await store.run("r", "t")
    assert status["state"] == "CANCELLED" and not status["cleanup_pending"]
    assert status["spent"] == 12 and status["reserved"] == 0
    assert (await store.operation("o", "t"))["result"] is None


async def test_finish_cancelled_does_not_change_a_completed_run(store):
    await store.create_root(root())
    result = operation().payload
    await store.finish_run("r", "t", "SUCCEEDED", result)
    await BrokerActivities(store).finish({"root_id": "r", "tenant_id": "t", "state": "CANCELLED"})
    status = await store.run("r", "t")
    assert status["state"] == "SUCCEEDED" and status["result"] == result.model_dump(mode="json")


async def test_finish_cancelled_rejects_wrong_tenant_without_mutation(store):
    from temporalio.exceptions import ApplicationError

    await store.create_root(root())
    with pytest.raises(ApplicationError) as error:
        await BrokerActivities(store).finish(
            {"root_id": "r", "tenant_id": "another", "state": "CANCELLED"}
        )
    assert error.value.non_retryable
    assert (await store.run("r", "t"))["state"] == "RUNNING"


async def test_finish_cancelled_rejects_a_result(store):
    await store.create_root(root())
    with pytest.raises(ValueError, match="cannot publish a result"):
        await store.finish_run("r", "t", "CANCELLED", operation().payload)
    assert (await store.run("r", "t"))["state"] == "RUNNING"
