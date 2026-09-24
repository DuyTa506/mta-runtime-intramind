"""A direct request has one bounded logical deadline across every phase."""

import asyncio
import json

import httpx
import pytest
from conftest import pool

from intramind_runtime.direct_proxy import DirectProxy

PAYLOAD = {"model": "model-1", "messages": [{"role": "user", "content": "hello"}],
           "stream": True, "max_tokens": 20}


def held(proxy):
    return [item for item in proxy.admission.permits.values() if item.kind == "direct"]


def unknown(proxy):
    return [item for item in held(proxy) if item.state == "UNKNOWN"]


def test_workload_deadlines_are_configurable_and_client_only_shortens():
    spec = pool()
    proxy = DirectProxy(None, client=None, pool=spec, timeout_seconds=45)
    assert [proxy._deadline_seconds(kind, None) for kind in
            ("qa", "user_task", "background", "maintenance")] == [600, 1800, 45, 45]
    assert proxy._deadline_seconds("qa", 900) == 600
    assert proxy._deadline_seconds("qa", 7) == 7
    configured = DirectProxy(None, client=None, pool=spec, timeout_seconds=45,
        workload_deadline_seconds={"qa": 90, "user_task": 150, "maintenance": 20})
    assert [configured._deadline_seconds(kind, None) for kind in
            ("qa", "user_task", "background", "maintenance")] == [90, 150, 45, 20]
    assert configured._deadline_seconds("qa", 600) == 90
    with pytest.raises(ValueError, match="deadline"):
        DirectProxy(None, client=None, pool=spec, workload_deadline_seconds={"qa": 0})


@pytest.mark.integration
@pytest.mark.parametrize("streaming", [True, False])
async def test_waiting_capacity_expires_without_sending_or_leaving_waiter(store, streaming):
    spec = pool(target=0, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    calls = []

    async def backend(request):
        calls.append(request)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    proxy = DirectProxy(store, pool=spec, client=httpx.AsyncClient(
        base_url="http://engine/v1/", transport=httpx.MockTransport(backend)))
    try:
        response = await asyncio.wait_for(proxy.open("tenant", PAYLOAD | {"stream": streaming},
            request_bound=30, workload_class="qa", deadline_seconds=1), 3)
        body = b"".join([chunk async for chunk in response.body_iterator])
        assert response.status_code == (200 if streaming else 504)
        assert b'deadline_exceeded' in body
        assert b'admission_timeout' in body
        assert calls == []
        assert held(proxy) == []
        assert proxy.admission.waiters == {}
        assert proxy.admission.direct_attempts == {}
    finally:
        await proxy.close()


@pytest.mark.integration
async def test_timeout_after_send_keeps_unknown_compute_held_even_after_reconcile(store):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    calls = []

    async def backend(request):
        calls.append(request)
        await asyncio.sleep(3)
        return httpx.Response(200, json={"choices": []})

    proxy = DirectProxy(store, pool=spec, client=httpx.AsyncClient(
        base_url="http://engine/v1/", transport=httpx.MockTransport(backend)),
        timeout_seconds=5)
    try:
        response = await asyncio.wait_for(proxy.open("tenant", PAYLOAD | {"stream": False},
            request_bound=30, workload_class="qa", deadline_seconds=1), 3)
        assert response.status_code == 504
        assert json.loads(b"".join([chunk async for chunk in response.body_iterator])) == {
            "error": {"message": "inference request deadline exceeded",
                      "type": "upstream_error", "reason": "deadline_exceeded"}}
        assert len(calls) == 1
        assert len(unknown(proxy)) == 1
        await store.reconcile_expired()
        assert len(unknown(proxy)) == 1
    finally:
        await proxy.close()


@pytest.mark.integration
async def test_recovery_wait_expires_without_replaying_unknown_inference(store):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    calls = []

    async def backend(request):
        calls.append(request)
        return httpx.Response(200,
            content=b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            headers={"Content-Type": "text/event-stream"})

    proxy = DirectProxy(store, pool=spec, client=httpx.AsyncClient(
        base_url="http://engine/v1/", transport=httpx.MockTransport(backend)))
    try:
        response = await proxy.open("tenant", PAYLOAD, request_bound=30,
                                    workload_class="qa", deadline_seconds=1)
        body = await asyncio.wait_for(_consume(response), 3)
        assert b"partial" in body and b"engine_termination_unconfirmed" in body
        assert b'deadline_exceeded' in body
        assert len(calls) == 1
        assert len(unknown(proxy)) == 1
        await store.reconcile_expired()
        assert len(held(proxy)) == 1
    finally:
        await proxy.close()


@pytest.mark.integration
async def test_retry_capacity_wait_expires_without_requeue_or_second_send(store):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    calls = []

    async def backend(request):
        calls.append(request)
        return httpx.Response(503, content=b"Loading model")

    proxy = DirectProxy(store, pool=spec, client=httpx.AsyncClient(
        base_url="http://engine/v1/", transport=httpx.MockTransport(backend)))
    retried = []

    async def no_capacity(*args):
        retried.append(args)
        return None

    proxy.admission.retry_confirmed = no_capacity
    try:
        response = await proxy.open("tenant", PAYLOAD, request_bound=30,
                                    workload_class="qa", deadline_seconds=1)
        body = await asyncio.wait_for(_consume(response), 3)
        assert b"deadline_exceeded" in body
        assert len(calls) == 1 and retried
        assert proxy.admission.waiters == {}
        assert held(proxy) == []
    finally:
        await proxy.close()


@pytest.mark.integration
async def test_streamed_token_then_stall_expires_without_replay_or_releasing_unknown(store):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    calls = []

    class StalledStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            await asyncio.sleep(3)
            yield b"data: [DONE]\n\n"

    async def backend(request):
        calls.append(request)
        return httpx.Response(200, stream=StalledStream(),
            headers={"Content-Type": "text/event-stream"})

    proxy = DirectProxy(store, pool=spec, client=httpx.AsyncClient(
        base_url="http://engine/v1/", transport=httpx.MockTransport(backend)),
        timeout_seconds=5)
    try:
        response = await proxy.open("tenant", PAYLOAD, request_bound=30,
                                    workload_class="qa", deadline_seconds=1)
        body = await asyncio.wait_for(_consume(response), 3)
        assert b"first" in body and b"deadline_exceeded" in body
        assert len(calls) == 1
        await store.reconcile_expired()
        assert len(unknown(proxy)) == 1
    finally:
        await proxy.close()


async def test_proxy_start_is_inside_the_logical_deadline():
    client = httpx.AsyncClient(base_url="http://engine/v1/")
    proxy = DirectProxy(None, pool=pool(), client=client)

    async def slow_start():
        await asyncio.sleep(3)

    proxy.start = slow_start
    try:
        response = await asyncio.wait_for(proxy.open("tenant", PAYLOAD,
            request_bound=30, workload_class="qa", deadline_seconds=1), 3)
        assert b"deadline_exceeded" in await _consume(response)
    finally:
        await proxy.close()


async def _consume(response):
    return b"".join([chunk async for chunk in response.body_iterator])
