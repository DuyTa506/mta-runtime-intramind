"""One-shot warehouse admission never waits, writes SQL, or refunds unknown compute."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from conftest import pool

from intramind_runtime.admission import AdmissionDeferred, admission_scope
from intramind_runtime.direct import DirectRequest
from intramind_runtime.direct_client import DirectBinding, inference_scope
from intramind_runtime.direct_proxy import DirectProxy
from intramind_runtime.memory_scheduler import MemoryScheduler, _Pool

PAYLOAD = {"model": "model-1", "messages": [{"role": "user", "content": "hello"}],
           "stream": True, "max_tokens": 20}


def scheduler(limit=1):
    spec = pool(transport_limit=limit, background_transport_limit=limit)
    manager = MemoryScheduler(object())  # Any SQL access is an immediate test failure.
    manager._started = True
    manager.pools["p"] = _Pool(spec, "e1", "HEALTHY", 1, "gpu", "HEALTHY", 4)
    manager._ready_by_pool["p"] = 0
    return manager, spec


def request(identity, **kwargs):
    return DirectRequest(request_id=identity, tenant_id="tenant", payload_digest="a" * 64,
        model_profile="test", model_revision="model-1", capacity_profile_id="test-v1",
        request_bound=30, deadline=datetime.now(UTC) + timedelta(seconds=60),
        admission_mode="try", workload_class=kwargs.pop("workload_class", "background"), **kwargs)


async def body(response):
    if hasattr(response, "body"):
        return response.body
    return b"".join([chunk async for chunk in response.body_iterator])


async def test_try_does_not_jump_waiters_and_never_leaves_refused_state():
    manager, _ = scheduler()
    first = request("older")
    await manager.enqueue(first, "p", "owner")
    for index in range(100):
        assert manager.try_reserve(request(str(index)), "p", "owner") == (
            None, "inference_capacity")
    assert list(manager.waiters) == ["older"]
    assert list(manager.request_identity) == ["older"]
    assert len(manager._queues["p"]) == 1
    assert not manager.permits and not manager.direct_attempts
    qa = request("qa", workload_class="qa")
    granted, _ = manager.try_reserve(qa, "p", "owner")
    assert granted.request == qa
    await manager.finish(granted, "not_sent")
    assert await manager.reserve(first, "p", "owner")


async def test_eight_background_requests_use_all_qualified_slots():
    manager, _ = scheduler(8)
    grants = [manager.try_reserve(request(str(i)), "p", "owner")[0] for i in range(8)]
    assert all(grants)
    assert manager.try_reserve(request("full"), "p", "owner") == (None, "inference_capacity")
    for granted in grants:
        await manager.finish(granted, "not_sent")
    assert not manager.permits


@pytest.mark.parametrize("reason", ["capacity", "health", "cutoff"])
async def test_try_returns_409_without_engine_or_request_sql(reason):
    manager, spec = scheduler()
    if reason == "capacity":
        manager.try_reserve(request("occupy"), "p", "owner")
    elif reason == "health":
        manager.pools["p"].health = "UNHEALTHY"
    calls = []
    client = httpx.AsyncClient(base_url="http://engine/", transport=httpx.MockTransport(
        lambda req: calls.append(req)))
    proxy = DirectProxy(object(), client=client, pool=spec, scheduler=manager)
    await proxy.start()
    try:
        response = await proxy.open("tenant", PAYLOAD, request_bound=30, admission_mode="try",
            dispatch_before=datetime.now(UTC) - timedelta(seconds=1) if reason == "cutoff" else None)
        assert response.status_code == 409
        error = json.loads(await body(response))["error"]
        assert error["type"] == "admission_deferred" and error["compute_state"] == "not_sent"
        assert not calls and not manager.waiters
    finally:
        await proxy.close()


async def test_cutoff_race_after_grant_before_engine_releases_unsent_permit():
    manager, spec = scheduler()
    calls = []
    proxy = DirectProxy(object(), pool=spec, scheduler=manager,
        client=httpx.AsyncClient(base_url="http://engine/", transport=httpx.MockTransport(
            lambda req: calls.append(req))))
    original = manager.mark_send

    async def slow_mark(reservation):
        await original(reservation)
        await asyncio.sleep(.03)

    manager.mark_send = slow_mark
    await proxy.start()
    try:
        response = await proxy.open("tenant", PAYLOAD, request_bound=30, admission_mode="try",
            dispatch_before=datetime.now(UTC) + timedelta(seconds=.01))
        assert b"dispatch_cutoff" in await body(response)
        assert not calls and not manager.permits and not manager.waiters
    finally:
        await proxy.close()


async def test_timeout_is_unknown_and_keeps_capacity_without_hidden_retry():
    manager, spec = scheduler()
    calls = []

    async def hang(req):
        calls.append(req)
        await asyncio.Future()

    proxy = DirectProxy(object(), pool=spec, scheduler=manager,
        client=httpx.AsyncClient(base_url="http://engine/", transport=httpx.MockTransport(hang)))
    await proxy.start()
    try:
        response = await proxy.open("tenant", PAYLOAD, request_bound=30, admission_mode="try",
            execution_timeout_seconds=.02, logical_request_id="work-1")
        result = await asyncio.wait_for(body(response), 1)
        assert b'"compute_state":"unknown"' in result
        assert b'"logical_request_id":"work-1"' in result
        assert len(calls) == 1 and len(manager.permits) == 1
        assert next(iter(manager.permits.values())).state == "UNKNOWN"
    finally:
        await proxy.close()
    assert len(manager.permits) == 1


async def test_disconnected_try_grant_is_released_before_engine():
    manager, spec = scheduler()
    calls = []
    proxy = DirectProxy(object(), pool=spec, scheduler=manager,
        client=httpx.AsyncClient(base_url="http://engine/", transport=httpx.MockTransport(
            lambda req: calls.append(req))))
    await proxy.start()
    try:
        response = await proxy.open("tenant", PAYLOAD, request_bound=30, admission_mode="try")
        await response.channel.detach()
        await asyncio.gather(*list(proxy.tasks))
        assert not calls and not manager.permits and not manager.waiters
    finally:
        await proxy.close()


async def test_sdk_forwards_scope_and_preserves_deferral_evidence():
    def deferred(req):
        assert req.headers["X-Intramind-Admission-Mode"] == "try"
        assert req.headers["X-Intramind-Step-Timeout-Seconds"] == "30"
        assert req.headers["X-Intramind-Dispatch-Before"].endswith("+00:00")
        return httpx.Response(409, json={"error": {"type": "admission_deferred",
            "reason": "inference_capacity", "compute_state": "not_sent"}})

    binding = DirectBinding("http://runtime", "a" * 32, "tool")
    with inference_scope("tenant"), admission_scope(mode="try",
        execution_timeout_seconds=30, dispatch_before=datetime.now(UTC)) as scope:
        async with httpx.AsyncClient(**binding.client_kwargs(observe_admission=True),
            transport=httpx.MockTransport(deferred)) as client:
            with pytest.raises(AdmissionDeferred, match="inference_capacity"):
                await client.post("chat/completions", json=PAYLOAD)
        assert isinstance(scope.error, AdmissionDeferred)


async def test_sdk_separates_internal_evidence_from_openai_chunks():
    evidence = {'type': 'started', 'attempt_id': 'a', 'logical_request_id': 'l',
                'generation': 0, 'compute_state': 'sent'}
    completion = 'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
    content = 'event: intramind.control\ndata: ' + json.dumps(evidence) + '\n\n' + completion + 'data: [DONE]\n\n'
    binding = DirectBinding('http://runtime', 'a' * 32, 'tool')
    observed = []

    async def started(frame):
        observed.append(frame)

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raw = content.encode()
            for offset in range(0, len(raw), 7):
                yield raw[offset:offset + 7]

    with inference_scope('tenant'), admission_scope(mode='try', on_event=started) as scope:
        async with httpx.AsyncClient(**binding.client_kwargs(observe_admission=True),
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream(),
                headers={'Content-Type': 'text/event-stream'}))) as client:
            async with client.stream('POST', 'chat/completions', json=PAYLOAD) as response:
                assert (await response.aread()).decode() == completion + 'data: [DONE]\n\n'
        assert len(observed) == 1 and observed[0]['attempt_id'] == 'a'
        assert scope.evidence['compute_state'] == 'terminated'
