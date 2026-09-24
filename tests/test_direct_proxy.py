"""Direct streaming must retain inference ownership after caller disconnect."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
import pytest
import uvicorn
from conftest import operation, pool, root
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from intramind_runtime.contracts import EmbeddingPoolSpec
from intramind_runtime.direct import DirectRequest
from intramind_runtime.direct_proxy import DirectProxy

pytestmark = pytest.mark.integration


PAYLOAD = {
    "model": "model-1", "messages": [{"role": "user", "content": "hello"}],
    "stream": True, "max_tokens": 20,
}


@asynccontextmanager
async def serve(app):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    address = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("fixture HTTP server did not start")
                await asyncio.sleep(0.01)
        yield address
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except TimeoutError:
            server.force_exit = True
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        listener.close()


async def attempt(store, attempt_id):
    async with store.engine.connect() as connection:
        return (await connection.execute(text("""SELECT state,compute_held FROM runtime_direct_attempts
            WHERE attempt_id=:id"""), {"id": attempt_id})).mappings().one()


async def consume(response):
    return b"".join([chunk async for chunk in response.body_iterator])


async def test_stream_is_delivered_before_completion_and_disconnect_drains_backend(store):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    finish = asyncio.Event()
    upstream_closed = asyncio.Event()
    downstream_closed = asyncio.Event()
    backend_calls = []
    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def complete(request: Request):
        backend_calls.append(await request.json())
        assert request.headers.get("X-Intramind-Attempt-ID")
        assert request.headers["X-Intramind-Expected-Revision"] == "model-1"

        async def chunks():
            try:
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                await finish.wait()
                yield b"data: [DO"
                yield b"NE]\n\n"
            finally:
                upstream_closed.set()

        return StreamingResponse(chunks(), media_type="text/event-stream")

    async with serve(backend) as backend_url:
        upstream = httpx.AsyncClient(base_url=backend_url + "/v1/", trust_env=False)
        proxy = DirectProxy(store, client=upstream, pool=spec, timeout_seconds=30)
        gateway = FastAPI()

        @gateway.post("/chat/completions")
        async def foreground():
            return await proxy.open("tenant", PAYLOAD, request_bound=30)

        async def observed_gateway(scope, receive, send):
            try:
                await gateway(scope, receive, send)
            finally:
                if scope["type"] == "http":
                    downstream_closed.set()

        try:
            async with serve(observed_gateway) as gateway_url:
                async with httpx.AsyncClient(base_url=gateway_url, trust_env=False, timeout=5) as caller:
                    async with caller.stream("POST", "/chat/completions") as response:
                        assert response.status_code == 200
                        async with asyncio.timeout(5):
                            async for chunk in response.aiter_bytes():
                                if b"hello" in chunk:
                                    break
                            else:
                                pytest.fail("engine token was not streamed")
                        assert not upstream_closed.is_set()
                    await asyncio.wait_for(downstream_closed.wait(), timeout=5)
                    assert not upstream_closed.is_set()
                    assert (await store.drain_status())["compute_held"] == 1
                    assert await store.reserve_next("p", "background-owner") is None
                    finish.set()
                    await asyncio.wait_for(upstream_closed.wait(), timeout=5)
                    async with asyncio.timeout(5):
                        while (await store.drain_status())["compute_held"]:
                            await asyncio.sleep(0.01)
                    assert await store.reserve_next("p", "background-owner") is not None
                    assert backend_calls == [PAYLOAD]
            async with store.engine.connect() as connection:
                assert (await connection.execute(text("SELECT count(*) FROM runtime_outbox"))).scalar_one() == 0
        finally:
            finish.set()
            await proxy.close()


async def test_refused_connection_releases_only_the_unsent_attempt(store):
    spec = pool(target=1)
    await store.configure_pool(spec, 1)
    # A bound, non-listening socket reserves the port while producing ECONNREFUSED.
    with socket.socket() as refused:
        refused.bind(("127.0.0.1", 0))
        upstream = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{refused.getsockname()[1]}/v1/", trust_env=False, timeout=1,
        )
        proxy = DirectProxy(store, client=upstream, pool=spec, timeout_seconds=5)
        try:
            response = await proxy.open("tenant", PAYLOAD | {"stream": False}, request_bound=30)
            assert response.status_code == 502
            assert json.loads(await consume(response))["error"]
            state = await attempt(store, response.headers["X-Intramind-Attempt-ID"])
            assert state == {"state": "FAILED_NOT_SENT", "compute_held": False}
            assert (await store.drain_status())["unsettled_attempts"] == 0
        finally:
            await proxy.close()


@pytest.mark.parametrize("detach_phase", ["reservation", "resumed_control"])
async def test_detach_during_reservation_cannot_leak_a_permit(store, detach_phase):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    backend_calls = []

    async def backend(request):
        backend_calls.append(request)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    upstream = httpx.AsyncClient(base_url="http://engine/v1/", transport=httpx.MockTransport(backend))
    proxy = DirectProxy(store, client=upstream, pool=spec)
    original = proxy.admission.reserve
    reserved = asyncio.Event()
    release = asyncio.Event()
    resuming = asyncio.Event()
    release_control = asyncio.Event()

    async def pause_after_commit(*args):
        granted = await original(*args)
        if granted is not None:
            reserved.set()
            await release.wait()
        return granted

    proxy.admission.reserve = pause_after_commit
    try:
        response = await proxy.open("tenant", PAYLOAD, request_bound=30)
        await asyncio.wait_for(reserved.wait(), 5)
        original_put = response.channel.put

        async def pause_resumed_control(chunk):
            if b'"type":"resumed"' in chunk:
                resuming.set()
                await release_control.wait()
            await original_put(chunk)

        response.channel.put = pause_resumed_control
        if detach_phase == "resumed_control":
            release.set()
            await asyncio.wait_for(resuming.wait(), 5)
        await response.channel.detach()
        release.set()
        release_control.set()
        async with asyncio.timeout(5):
            while (await store.drain_status())["compute_held"]:
                await asyncio.sleep(0.01)
        async with store.engine.connect() as connection:
            state = (await connection.execute(text(
                "SELECT state,compute_held FROM runtime_direct_attempts"))).mappings().one()
            waiters = (await connection.execute(text(
                "SELECT count(*) FROM runtime_direct_waiters"))).scalar_one()
        assert state == {"state": "FAILED_NOT_SENT", "compute_held": False}
        assert waiters == 0 and backend_calls == []
    finally:
        release.set()
        release_control.set()
        await proxy.close()


async def test_transient_orphan_settlement_failure_does_not_stop_next_request(store):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    backend_calls = []

    async def backend(request):
        backend_calls.append(request)
        return httpx.Response(200, content=b"data: [DONE]\n\n",
                              headers={"Content-Type": "text/event-stream"})

    upstream = httpx.AsyncClient(base_url="http://engine/v1/", transport=httpx.MockTransport(backend))
    proxy = DirectProxy(store, client=upstream, pool=spec)
    try:
        await proxy.start()
        request = DirectRequest(request_id="orphan", tenant_id="tenant",
            payload_digest="a" * 64, model_profile="test", capacity_profile_id="test-v1",
            request_bound=30, deadline=datetime(3000, 1, 1, tzinfo=UTC),
            workload_class="background")
        await proxy.admission.enqueue(request, "p", proxy.owner_id)
        orphan = await proxy.admission.reserve(request, "p", proxy.owner_id)
        assert orphan is not None
        original_finish = proxy.admission.finish
        calls = 0

        async def transient_failure(reservation, *, evidence):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("transient ledger outage")
            return await original_finish(reservation, evidence=evidence)

        proxy.admission.finish = transient_failure
        proxy._settlements[orphan.attempt_id] = (orphan, "not_sent")
        proxy._pending_signal.set()
        response = await proxy.open("tenant", PAYLOAD, request_bound=30)
        streamed = await asyncio.wait_for(consume(response), 8)
        assert b"[DONE]" in streamed and len(backend_calls) == 1
        assert calls >= 2 and not proxy._settlements
        assert not proxy._dispatch_task.done()
        assert (await store.drain_status())["compute_held"] == 0
    finally:
        await proxy.close()


async def test_dispatcher_recovers_after_wakeup_exception(store, monkeypatch):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)

    class FlakyWakeup:
        generation = 0
        calls = 0

        async def wait(self, generation, timeout):
            self.calls += 1
            if self.calls == 1:
                raise OSError("wakeup connection reset")
            await asyncio.sleep(0.05)

    wakeup = FlakyWakeup()

    async def direct_wakeup():
        return wakeup

    monkeypatch.setattr(store, "direct_wakeup", direct_wakeup)
    upstream = httpx.AsyncClient(base_url="http://engine/v1/", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"data: [DONE]\n\n",
            headers={"Content-Type": "text/event-stream"})))
    proxy = DirectProxy(store, client=upstream, pool=spec)
    try:
        await proxy.start()
        async with asyncio.timeout(5):
            while not wakeup.calls:
                await asyncio.sleep(0.01)
        response = await proxy.open("tenant", PAYLOAD, request_bound=30)
        assert b"[DONE]" in await asyncio.wait_for(consume(response), 8)
        assert wakeup.calls >= 1 and not proxy._dispatch_task.done()
    finally:
        await proxy.close()


async def test_dispatcher_uses_committed_waiter_order_when_enqueue_returns_out_of_order(store):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    calls = []

    async def backend(request):
        calls.append(request)
        return httpx.Response(200, content=b"data: [DONE]\n\n",
                              headers={"Content-Type": "text/event-stream"})

    proxy = DirectProxy(store, pool=spec, client=httpx.AsyncClient(
        base_url="http://engine/v1/", transport=httpx.MockTransport(backend)))
    original_enqueue = proxy.admission.enqueue
    first_committed = asyncio.Event()
    release_first = asyncio.Event()
    enqueues = 0

    async def reordered_enqueue(*args, **kwargs):
        nonlocal enqueues
        enqueues += 1
        order = enqueues
        created_at = await original_enqueue(*args, **kwargs)
        if order == 1:
            first_committed.set()
            await release_first.wait()
        return created_at

    proxy.admission.enqueue = reordered_enqueue
    first = asyncio.create_task(proxy.open("tenant", PAYLOAD, request_bound=30,
        workload_class="qa"))
    try:
        await asyncio.wait_for(first_committed.wait(), 5)
        second_response = await asyncio.wait_for(proxy.open("tenant", PAYLOAD,
            request_bound=30, workload_class="qa"), 5)
        release_first.set()
        first_response = await asyncio.wait_for(first, 5)
        completed = await asyncio.wait_for(asyncio.gather(
            consume(first_response), consume(second_response)), 8)
        assert all(b"[DONE]" in body for body in completed)
        assert len(calls) == 2
        assert (await store.drain_status())["compute_held"] == 0
    finally:
        release_first.set()
        if not first.done():
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
        await proxy.close()


async def test_stale_proxy_epoch_does_not_send_to_the_reconfigured_pool(store):
    spec = pool(target=1)
    await store.configure_pool(spec, 1)
    calls = []

    async def backend(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "wrong epoch"}}]})

    upstream = httpx.AsyncClient(base_url="http://engine/v1/", transport=httpx.MockTransport(backend))
    proxy = DirectProxy(store, client=upstream, pool=spec, timeout_seconds=30)
    try:
        await store.configure_pool(spec.model_copy(update={"engine_epoch": "replacement"}), 1)
        response = await proxy.open("tenant", PAYLOAD | {"stream": False}, request_bound=30)
        assert response.status_code == 503
        assert json.loads(await consume(response))["error"]
        assert calls == []
        async with store.engine.connect() as connection:
            assert (await connection.execute(text(
                "SELECT count(*) FROM runtime_direct_attempts"))).scalar_one() == 0
    finally:
        await proxy.close()


@pytest.mark.parametrize("failure", ["read_timeout", "missing_done"])
async def test_incomplete_stream_waits_for_proof_then_resets_generation(store, failure):
    spec = pool(target=1)
    await store.configure_pool(spec, 1)
    backend_calls = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            if failure == "read_timeout":
                raise httpx.ReadTimeout("fixture lost backend response")

    async def backend(request):
        backend_calls.append(request)
        if len(backend_calls) == 1:
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=Stream())
        return httpx.Response(200, content=(
            b'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
            b'data: [DONE]\n\n'), headers={"Content-Type": "text/event-stream"})

    upstream = httpx.AsyncClient(base_url="http://engine/v1/", transport=httpx.MockTransport(backend))
    proxy = DirectProxy(store, client=upstream, pool=spec, timeout_seconds=30)
    try:
        response = await proxy.open("tenant", PAYLOAD, request_bound=30)
        consuming = asyncio.create_task(consume(response))
        async with asyncio.timeout(5):
            while (await store.drain_status())["unknown_attempts"] == 0:
                await asyncio.sleep(0.01)
        assert backend_calls and len(backend_calls) == 1
        assert (await store.drain_status())["compute_held"] == 1
        await asyncio.sleep(0.1)
        assert not consuming.done(), "timeout or EOF alone must never retry inference"
        token = "watchdog-test-owner"
        await store.quiesce_engine("p", "e1", token)
        proof = json.dumps({
            "pool_id": "p", "engine_epoch": "e1", "verification": "container_and_child_absent",
            "container_id": "old-container", "model_pid": 1234,
            "process_started_at": "2026-09-24T01:00:00Z",
            "verified_stopped_at": "2026-09-24T01:01:00Z",
        })
        assert await store.confirm_epoch_stopped("p", "e1", proof, recover=True) == 1
        await store.configure_pool(spec.model_copy(update={"engine_epoch": "e2"}), 1)
        streamed = await asyncio.wait_for(consuming, timeout=10)
        assert b"hello" in streamed and b"recovered" in streamed
        assert streamed.index(b"generation_reset") < streamed.index(b"recovered")
        assert len(backend_calls) == 2
        assert (await store.drain_status())["compute_held"] == 0
    finally:
        await proxy.close()


@pytest.mark.parametrize("response_kind", ["valid", "different_revision", "missing_proof", "foreign_proof"])
async def test_embedding_proof_headers_are_preserved_and_invalid_proof_blocks_vectors(store, response_kind):
    spec = EmbeddingPoolSpec(**(pool(target=1).model_dump(exclude={"context_limit"}) | {
        "character_limit": 40, "max_batch_size": 2,
    }))
    await store.configure_pool(spec, 1)
    payload = {"texts": ["hello"], "input_type": "query"}
    vectors = {"embeddings": [[1., 2.]], "dimension": 2, "model": "model-1"}

    async def backend(request):
        assert request.url.path == "/embed"
        assert json.loads(request.content) == payload
        assert request.headers["X-Intramind-Expected-Revision"] == "model-1"
        headers = {
            "X-Intramind-Attempt-ID": request.headers["X-Intramind-Attempt-ID"],
            "X-Intramind-Embedding-Contract": "termination-v1",
            "X-Intramind-Compute-State": "terminated",
            "X-Intramind-Model-Revision": "different" if response_kind == "different_revision" else "model-1",
            "Retry-After": "2",
            "Set-Cookie": "private-engine-cookie=must-not-forward",
        }
        if response_kind == "missing_proof":
            headers.pop("X-Intramind-Embedding-Contract")
        if response_kind == "foreign_proof":
            headers["X-Intramind-Attempt-ID"] = "another-attempt"
        return httpx.Response(200, json=vectors, headers=headers)

    upstream = httpx.AsyncClient(base_url="http://engine/", transport=httpx.MockTransport(backend))
    proxy = DirectProxy(store, client=upstream, pool=spec, timeout_seconds=30)
    try:
        response = await asyncio.wait_for(proxy.open("tenant", payload, request_bound=5,
                                                     path="embed"), timeout=5)
        if response_kind == "valid":
            assert response.status_code == 200
            assert response.headers["X-Intramind-Embedding-Contract"] == "termination-v1"
            assert response.headers["X-Intramind-Compute-State"] == "terminated"
            assert response.headers["X-Intramind-Model-Revision"] == "model-1"
            assert response.headers["Retry-After"] == "2"
            assert "Set-Cookie" not in response.headers
            assert json.loads(await consume(response)) == vectors
        else:
            assert response.status_code == 502
            assert "embeddings" not in json.loads(await consume(response))
        assert (await store.drain_status())["compute_held"] == (
            1 if response_kind in {"missing_proof", "foreign_proof"} else 0
        )
    finally:
        await proxy.close()
