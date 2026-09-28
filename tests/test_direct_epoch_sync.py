"""The first foreground call after an idle engine restart uses the current epoch."""

import asyncio
import json

import httpx
import pytest
from conftest import pool

from intramind_runtime.contracts import AdmissionDenied
from intramind_runtime.direct_proxy import DirectProxy
from intramind_runtime.memory_scheduler import MemoryScheduler, _Pool


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("already_started", [False, True])
async def test_first_wait_request_follows_qualified_restart(streaming, already_started):
    spec = pool(transport_limit=1, background_transport_limit=1)
    scheduler = MemoryScheduler(object())  # No database is needed on the direct path.
    scheduler._started = True
    scheduler.pools["p"] = _Pool(spec, "e1", "HEALTHY", 1, "gpu", "HEALTHY", 4)
    scheduler._ready_by_pool["p"] = 0
    calls = []

    def complete(request):
        calls.append(request)
        if streaming:
            return httpx.Response(200, content=(
                b'data: {"choices":[{"delta":{"content":"first call works"}}]}\n\n'
                b'data: [DONE]\n\n'))
        return httpx.Response(200, json={"choices": [
            {"message": {"content": "first call works"}, "finish_reason": "stop"}]})

    proxy = DirectProxy(object(), pool=spec, scheduler=scheduler,
        client=httpx.AsyncClient(base_url="http://engine/",
                                 transport=httpx.MockTransport(complete)))
    try:
        if already_started:
            await proxy.start()
            await asyncio.sleep(0)  # Let the idle dispatcher sleep on its local signal.
        current = scheduler.pools["p"]
        current.epoch = "e2"
        current.spec = spec.model_copy(update={"engine_epoch": "e2"})
        scheduler._notify()  # Control refresh does not wake the idle proxy.
        assert proxy.pool.engine_epoch == "e1"

        response = await proxy.open("tenant", {
            "model": "model-1", "messages": [{"role": "user", "content": "hello"}],
            "stream": streaming, "max_tokens": 20,
        }, request_bound=30, workload_class="qa")
        async with asyncio.timeout(2):
            body = b"".join([chunk async for chunk in response.body_iterator])
        assert response.status_code == 200, body
        assert b"first call works" in body, body
        if streaming:
            assert b"data: [DONE]" in body and b"intramind.error" not in body
        else:
            assert "error" not in json.loads(body)
        assert len(calls) == 1
        attempt = next(iter(scheduler.direct_attempts.values()))
        assert attempt.reservation.request.expected_engine_epoch == "e2"
        assert attempt.engine_epoch == "e2"
        assert not scheduler.permits and not scheduler.waiters
    finally:
        await proxy.close()


@pytest.mark.parametrize("changed_field", ["profile_id", "model_revision"])
async def test_wait_request_does_not_adopt_an_unqualified_restart(changed_field):
    spec = pool()
    scheduler = MemoryScheduler(object())
    scheduler._started = True
    scheduler.pools["p"] = _Pool(spec, "e1", "HEALTHY", 1, "gpu", "HEALTHY", 4)
    scheduler._ready_by_pool["p"] = 0
    calls = []
    proxy = DirectProxy(object(), pool=spec, scheduler=scheduler,
        client=httpx.AsyncClient(base_url="http://engine/",
            transport=httpx.MockTransport(lambda request: calls.append(request))))
    try:
        await proxy.start()
        await asyncio.sleep(0)
        scheduler.pools["p"].epoch = "e2"
        scheduler.pools["p"].spec = spec.model_copy(update={
            "engine_epoch": "e2", changed_field: "unqualified-replacement"})
        with pytest.raises(AdmissionDenied, match="qualified pool profile"):
            await proxy.open("tenant", {"model": "model-1", "stream": False,
                "messages": [{"role": "user", "content": "hello"}]}, request_bound=30)
        assert proxy.pool.engine_epoch == "e1"
        assert not calls and not scheduler.waiters and not scheduler.permits
    finally:
        await proxy.close()
