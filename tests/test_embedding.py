import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from conftest import operation, pool, root
from fakes import MemoryArtifacts, TestPermits

from intramind_runtime.api import create_app
from intramind_runtime.client import RuntimeClient
from intramind_runtime.contracts import (
    AdmissionDenied,
    Artifact,
    EmbeddingOperationSpec,
    EmbeddingPoolSpec,
    Reservation,
    parse_operation,
    parse_pool,
)
from intramind_runtime.drivers import DriverFailure
from intramind_runtime.embedding import (
    EmbeddingPreparer,
    EmbeddingPrepareRequest,
    EmbeddingProfile,
    ServingEmbeddingDriver,
)
from intramind_runtime.executor import Executor


def profile(**changes):
    return EmbeddingProfile(**({
        "model_profile": "embedding-test", "capacity_profile_id": "embed-v1", "model": "native",
        "model_revision": "model-v1", "dimension": 2, "max_batch_size": 2,
        "character_limit": 40, "max_text_characters": 24, "max_response_bytes": 4096,
    } | changes))


def embedding(op="embedding", **changes):
    return EmbeddingOperationSpec(**({
        "operation_id": op, "root_id": "r", "tenant_id": "t", "payload": operation().payload,
        "model_profile": "embedding-test", "capacity_profile_id": "embed-v1", "model_revision": "model-v1",
        "characters_bound": 4, "texts_count": 2, "expected_cost": 4,
    } | changes))


def embedding_pool(**changes):
    data = pool().model_dump(exclude={"context_limit"}) | {
        "pool_id": "embedding", "model_profile": "embedding-test", "profile_id": "embed-v1",
        "model_revision": "model-v1", "character_limit": 40, "max_batch_size": 2,
    }
    return EmbeddingPoolSpec(**(data | changes))


def reservation(**changes):
    return Reservation(operation=embedding(**changes), attempt_id="attempt-1", pool_id="embedding",
        engine_epoch="epoch-1", model_revision="model-v1", owner_id="executor", lease_epoch=1,
        attempt_deadline=datetime.now(UTC)+timedelta(seconds=30))


def proof(request, state="terminated", **changes):
    return {"X-Intramind-Attempt-ID": request.headers["X-Intramind-Attempt-ID"],
            "X-Intramind-Embedding-Contract": "termination-v1",
            "X-Intramind-Compute-State": state, "X-Intramind-Model-Revision": "model-v1"} | changes


def vectors(**changes):
    return {"embeddings": [[0.1, 0.2], [0.3, 0.4]], "dimension": 2, "model": "native"} | changes


def test_embedding_roundtrip_keeps_budget_and_batch_units_separate():
    spec = embedding()
    assert parse_operation(spec.model_dump()) == spec
    capacity = embedding_pool()
    assert parse_pool(capacity.model_dump()) == capacity
    assert spec.budget_unit == "embedding_characters" and spec.budget_bound == 4
    assert "max_output_tokens" not in spec.model_dump()
    assert embedding(characters_bound=0).budget_bound == 1


@pytest.mark.parametrize("payload", [
    {"texts": []}, {"texts": ["x"] * 3}, {"texts": ["x" * 25]},
    {"texts": ["x" * 21, "y" * 21]}, {"texts": [True]},
    {"texts": ["x"], "input_type": "passage"}, {"texts": ["x"], "endpoint": "http://other"},
])
async def test_preparation_rejects_unqualified_batch_without_truncation(payload):
    with pytest.raises(ValueError):
        await EmbeddingPreparer(profile()).prepare(EmbeddingPrepareRequest(
            model_profile="embedding-test", capacity_profile_id="embed-v1", payload=payload,
            attempt_timeout_seconds=30), MemoryArtifacts(), "t")


@pytest.mark.parametrize("input_type", ["document", "query"])
async def test_driver_preserves_input_order_empty_text_and_query_semantics(input_type):
    calls = []
    payload = {"texts": ["", "xin chào"], "input_type": input_type}

    async def backend(request):
        calls.append(request)
        assert json.loads(request.content) == payload
        assert request.headers["X-Intramind-Expected-Revision"] == "model-v1"
        return httpx.Response(200, json=vectors(), headers=proof(request))

    async with httpx.AsyncClient(base_url="http://serving/", transport=httpx.MockTransport(backend)) as client:
        result = await ServingEmbeddingDriver("http://serving", profile(), client=client).execute(
            reservation(characters_bound=8), payload)
    assert result.body == vectors() and result.characters == 8 and len(calls) == 1


@pytest.mark.parametrize("changed", [
    {"dimension": 3}, {"model": "different"}, {"embeddings": [[1., 2.]]},
    {"embeddings": [[1.], [2.]]}, {"embeddings": [[True, 1.], [2., 3.]]},
])
async def test_completed_but_invalid_vectors_are_not_published_or_retried(changed):
    async def backend(request):
        return httpx.Response(200, json=vectors(**changed), headers=proof(request))

    async with httpx.AsyncClient(base_url="http://serving/", transport=httpx.MockTransport(backend)) as client:
        with pytest.raises(DriverFailure, match="embedding_invalid_vectors") as failed:
            await ServingEmbeddingDriver("http://serving", profile(), client=client).execute(
                reservation(), {"texts": ["ab", "cd"]})
    assert failed.value.finished and not failed.value.retry


@pytest.mark.parametrize("evidence,expected", [
    ({}, "embedding_response_unconfirmed"),
    ({"X-Intramind-Attempt-ID": "stale"}, "embedding_response_unconfirmed"),
    ({"X-Intramind-Compute-State": "unknown"}, "embedding_response_unconfirmed"),
    ({"X-Intramind-Model-Revision": "different"}, "embedding_model_revision_changed"),
])
async def test_termination_and_model_proofs_have_distinct_failure_accounting(evidence, expected):
    async def backend(request):
        headers = proof(request) | evidence if evidence else {}
        return httpx.Response(200, json=vectors(), headers=headers)

    async with httpx.AsyncClient(base_url="http://serving/", transport=httpx.MockTransport(backend)) as client:
        with pytest.raises(DriverFailure, match=expected) as failed:
            await ServingEmbeddingDriver("http://serving", profile(), client=client).execute(
                reservation(), {"texts": ["ab", "cd"]})
    assert failed.value.finished == (expected == "embedding_model_revision_changed")
    assert not failed.value.retry and not failed.value.not_sent


@pytest.mark.parametrize("content", [b'{"embeddings":[[NaN,0],[1,2]],"dimension":2,"model":"native"}',
                                     b"x" * 4097])
async def test_nonfinite_or_oversized_completed_output_cannot_enter_the_index(content):
    async def backend(request):
        return httpx.Response(200, content=content,
            headers=proof(request) | {"Content-Type": "application/json"})

    async with httpx.AsyncClient(base_url="http://serving/", transport=httpx.MockTransport(backend)) as client:
        with pytest.raises(DriverFailure) as failed:
            await ServingEmbeddingDriver("http://serving", profile(), client=client).execute(
                reservation(), {"texts": ["ab", "cd"]})
    assert failed.value.finished and not failed.value.retry


@pytest.mark.parametrize("phase", ["connect", "before_headers", "after_termination"])
async def test_transport_retry_requires_not_sent_or_confirmed_termination(phase):
    calls = []

    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"embeddings":'
            raise httpx.ReadError("output transfer failed")

    async def backend(request):
        calls.append(request)
        if phase == "connect":
            raise httpx.ConnectError("connection refused")
        if phase == "before_headers":
            raise httpx.ReadTimeout("compute may continue")
        return httpx.Response(200, stream=Broken(), headers=proof(request))

    async with httpx.AsyncClient(base_url="http://serving/", transport=httpx.MockTransport(backend)) as client:
        with pytest.raises(DriverFailure) as failed:
            await ServingEmbeddingDriver("http://serving", profile(), client=client).execute(
                reservation(), {"texts": ["ab", "cd"]})
    assert len(calls) == 1
    assert failed.value.not_sent == (phase == "connect")
    assert failed.value.finished == (phase == "after_termination")
    assert failed.value.retry == (phase != "before_headers")


@pytest.mark.integration
async def test_embedding_budget_reservation_is_atomic_while_llm_transport_is_independent(store):
    await store.create_root(root(resource_budgets={"embedding_characters": 4}))
    await store.configure_pool(pool(), 1)
    await store.configure_pool(embedding_pool(), 1)
    for i in range(3):
        await store.submit_operation(embedding(str(i)))
    await store.submit_operation(operation())
    attempts = await asyncio.gather(*(store.reserve_next("embedding", str(i)) for i in range(8)))
    winners = [a for a in attempts if a]
    assert len(winners) == 1
    assert await store.reserve_next("p", "llm") is not None
    attempt = winners[0]
    await store.mark_send(attempt)
    await store.compute_finished(attempt)
    for _ in range(2):
        await store.commit_result(attempt, attempt.operation.payload, 4)
    state = await store.run("r", "t")
    assert state["reserved"] == 30 and state["spent"] == 0
    assert state["resource_budgets"]["embedding_characters"] == {"limit": 4, "reserved": 0, "spent": 4}
    assert await store.reserve_next("embedding", "next") is None
    assert await store.reserve_next("p", "llm") is None


@pytest.mark.integration
async def test_missing_budget_and_oversized_batch_are_explicit_not_infinite_queue(store):
    await store.create_root(root())
    with pytest.raises(AdmissionDenied, match="embedding_characters"):
        await store.submit_operation(embedding())
    await store.create_root(root("with-budget", resource_budgets={"embedding_characters": 100}))
    await store.configure_pool(embedding_pool(), 1)
    await store.submit_operation(embedding("too-many", root_id="with-budget", texts_count=3))
    await store.submit_operation(embedding("fits", root_id="with-budget"))
    assert (await store.reserve_next("embedding", "worker")).operation.operation_id == "fits"
    await store.reconcile_expired()
    assert (await store.operation("too-many", "t"))["wait_reason"] == "request_exceeds_all_pools"


@pytest.mark.integration
async def test_unknown_embedding_keeps_budget_through_cancel_until_epoch_proof(store):
    await store.create_root(root(resource_budgets={"embedding_characters": 4}))
    await store.configure_pool(embedding_pool(), 1)
    await store.submit_operation(embedding())
    attempt = await store.reserve_next("embedding", "worker")
    await store.mark_send(attempt)
    await store.unknown(attempt, "read_timeout")
    await store.cancel("r", "t")
    assert (await store.run("r", "t"))["resource_budgets"]["embedding_characters"]["reserved"] == 4
    assert (await store.drain_status())["compute_held"] == 1
    await store.confirm_epoch_stopped("embedding", "e1", "isolated backend exited")
    await store.confirm_epoch_stopped("embedding", "e1", "duplicate stop proof")
    state = await store.run("r", "t")
    assert state["state"] == "CANCELLED" and not state["cleanup_pending"]
    assert state["resource_budgets"]["embedding_characters"] == {"limit": 4, "spent": 4, "reserved": 0}


@pytest.mark.integration
async def test_reusing_profile_id_cannot_route_accepted_embedding_to_another_model_revision(store):
    await store.create_root(root(resource_budgets={"embedding_characters": 100}))
    await store.configure_pool(embedding_pool(model_revision="different-model"), 1)
    await store.submit_operation(embedding())
    assert await store.reserve_next("embedding", "worker") is None
    await store.reconcile_expired()
    assert (await store.operation("embedding", "t"))["wait_reason"] == "model_revision_changed"
    assert (await store.run("r", "t"))["resource_budgets"]["embedding_characters"]["spent"] == 0


@pytest.mark.integration
async def test_embedding_result_persistence_failure_does_not_repeat_compute(store, monkeypatch):
    blobs, calls, failures = MemoryArtifacts(), [], []
    payload = await blobs.put("t", b'{"texts":["ab","cd"],"input_type":"query"}')
    await store.create_root(root(resource_budgets={"embedding_characters": 4}))
    await store.configure_pool(embedding_pool(), 1)
    await store.submit_operation(embedding(payload=payload))

    async def backend(request):
        calls.append(request)
        return httpx.Response(200, json=vectors(), headers=proof(request))

    put = blobs.put

    async def flaky(*args, **kwargs):
        if not failures:
            failures.append(True)
            assert (await store.drain_status())["compute_held"] == 0
            raise OSError("lost object write")
        return await put(*args, **kwargs)

    monkeypatch.setattr(blobs, "put", flaky)
    async with httpx.AsyncClient(base_url="http://serving/", transport=httpx.MockTransport(backend)) as client:
        driver = ServingEmbeddingDriver("http://serving", profile(), client=client)
        await asyncio.wait_for(Executor(store, blobs, driver, "embedding", "worker", TestPermits()).tick(), 6)
    op = await store.operation("embedding", "t")
    body = json.loads(await blobs.get(Artifact.model_validate(op["result"])))
    assert op["state"] == "SUCCEEDED" and body["body"] == vectors() and len(calls) == 1
    assert (await store.run("r", "t"))["resource_budgets"]["embedding_characters"]["spent"] == 4


@pytest.mark.integration
async def test_trusted_embedding_catalog_freezes_budget_and_preserves_source_order(store):
    blobs, token = MemoryArtifacts(), "embedding-api-token-used-only-for-tests"
    definitions = {"ingestion/v1": {"deadline_seconds": 60, "budget_limit": 100,
        "resource_budgets": {"embedding_characters": 30}, "task_queue": "ingestion"}}
    app = create_app(store, blobs, token, definitions,
                     embedding_preparers={"embedding-test": EmbeddingPreparer(profile())})
    async with httpx.AsyncClient(base_url="http://runtime", transport=httpx.ASGITransport(app)) as http:
        assert (await http.get("/v1/embedding/profiles/embedding-test")).status_code == 401
        http.headers.update({"Authorization": f"Bearer {token}", "X-Tenant-ID": "t"})
        client = RuntimeClient("http://runtime", token, "t", client=http)
        assert (await client.embedding_profile("embedding-test"))["dimension"] == 2
        prepared = await client.prepare_embedding(model_profile="embedding-test",
            capacity_profile_id="embed-v1", payload={"texts": ["xin chào", ""], "input_type": "query"},
            attempt_timeout_seconds=30)
        assert prepared["characters_bound"] == 8 and prepared["texts_count"] == 2
        assert await client.read_json(prepared["payload"]) == {"texts": ["xin chào", ""], "input_type": "query"}
        submission = {"task_type": "ingestion/v1", "submission_key": "source", "input": prepared["payload"]}
        accepted = await client.submit(submission)
        definitions["ingestion/v1"]["resource_budgets"]["embedding_characters"] = 999
        assert (await client.submit(submission))["run_id"] == accepted["run_id"]
        assert (await client.get_run(accepted["run_id"]))["resource_budgets"]["embedding_characters"]["limit"] == 30
        assert (await http.post("/v1/runs", json=submission | {
            "resource_budgets": {"embedding_characters": 999}})).status_code == 422
