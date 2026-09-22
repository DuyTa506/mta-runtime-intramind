"""Direct streaming must retain inference ownership after caller disconnect."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from conftest import operation, pool, root
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from intramind_runtime.contracts import EmbeddingPoolSpec
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
    spec = pool(target=1)
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
                        assert b"hello" in await anext(response.aiter_bytes())
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
            response = await proxy.open("tenant", PAYLOAD, request_bound=30)
            assert response.status_code == 502
            assert json.loads(await consume(response))["error"]
            state = await attempt(store, response.headers["X-Intramind-Attempt-ID"])
            assert state == {"state": "FAILED_NOT_SENT", "compute_held": False}
            assert (await store.drain_status())["unsettled_attempts"] == 0
        finally:
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
        assert response.status_code == 502
        assert json.loads(await consume(response))["error"]
        assert calls == []
        assert (await attempt(store, response.headers["X-Intramind-Attempt-ID"])) == {
            "state": "FAILED_NOT_SENT", "compute_held": False,
        }
    finally:
        await proxy.close()


@pytest.mark.parametrize("failure", ["read_timeout", "missing_done", "shutdown"])
async def test_incomplete_stream_keeps_unknown_compute_after_reconciliation(store, failure):
    spec = pool(target=1)
    await store.configure_pool(spec, 1)
    gate = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            await gate.wait()
            if failure == "read_timeout":
                raise httpx.ReadTimeout("fixture lost backend response")

    async def backend(request):
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=Stream())

    upstream = httpx.AsyncClient(base_url="http://engine/v1/", transport=httpx.MockTransport(backend))
    proxy = DirectProxy(store, client=upstream, pool=spec, timeout_seconds=30)
    try:
        response = await proxy.open("tenant", PAYLOAD, request_bound=30)
        iterator = response.body_iterator
        assert b"hello" in await asyncio.wait_for(anext(iterator), timeout=1)
        assert (await store.drain_status())["compute_held"] == 1
        if failure == "shutdown":
            await proxy.close()
        else:
            gate.set()
        with pytest.raises(RuntimeError, match="interrupted"):
            await asyncio.wait_for(anext(iterator), timeout=5)
        state = await attempt(store, response.headers["X-Intramind-Attempt-ID"])
        assert state == {"state": "UNKNOWN", "compute_held": True}
        await store.reconcile_expired()
        assert (await store.drain_status())["unknown_attempts"] == 1
    finally:
        gate.set()
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
        response = await proxy.open("tenant", payload, request_bound=5, path="embed")
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
