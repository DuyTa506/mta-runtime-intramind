"""Disposable rc18 mixed-admission latency benchmark; no model is contacted.

Run three times for each RUNTIME_BENCH_BACKLOG=0,1000,10000 with the guarded
PostgreSQL test URL on port 55440. The release gate uses the median run at
each backlog: dispatch p95 <= 50 ms, p99 <= 200 ms, 100/100 completed and
process CPU below 30 percent at 100 QA requests/second.
"""

import asyncio
import json
import os
import resource
from datetime import UTC, datetime, timedelta
from time import perf_counter

import httpx
import pytest
from conftest import pool

from intramind_runtime.direct import DirectRequest
from intramind_runtime.direct_proxy import DirectProxy
from intramind_runtime.memory_scheduler import MemoryScheduler


@pytest.mark.integration
async def test_synthetic_dispatcher_load(store):
    backlog = int(os.environ.get("RUNTIME_BENCH_BACKLOG", "1000"))
    if backlog not in {0, 1000, 10000}:
        raise ValueError("benchmark backlog must be 0, 1000 or 10000")
    pools = [pool(f"bench-{index}", target=4, transport_limit=64,
                  background_transport_limit=8) for index in range(4)]
    for spec in pools:
        await store.configure_pool(spec, 1)
    scheduler = MemoryScheduler(store, endpoint_limit=20000, tenant_limit=20000)
    await scheduler.start()
    deadline = datetime.now(UTC) + timedelta(minutes=30)
    preload_started = perf_counter()
    for index in range(backlog):
        spec = pools[index % 4]
        await scheduler.enqueue(DirectRequest(request_id=f"backlog-{index}",
            tenant_id="bench-backlog-tenant", payload_digest="a" * 64,
            model_profile="test", capacity_profile_id="test-v1", request_bound=30,
            deadline=deadline, workload_class="background"), spec.pool_id, "bench-backlog")
    preload_seconds = perf_counter() - preload_started
    queued = {}
    reserved = {}
    sent = {}
    backend_calls = 0

    async def backend(request):
        nonlocal backend_calls
        backend_calls += 1
        return httpx.Response(200, content=b"data: [DONE]\n\n",
                              headers={"Content-Type": "text/event-stream"})

    proxies = [DirectProxy(store, pool=spec, scheduler=scheduler, client=httpx.AsyncClient(
        base_url="http://engine/v1/", transport=httpx.MockTransport(backend))) for spec in pools]
    original_enqueue = scheduler.enqueue
    original_reserve = scheduler.reserve
    original_mark_send = scheduler.mark_send

    async def tracked_enqueue(request, *args, **kwargs):
        result = await original_enqueue(request, *args, **kwargs)
        if request.workload_class == "qa":
            queued[request.request_id] = perf_counter()
        return result

    async def tracked_reserve(request, *args, **kwargs):
        result = await original_reserve(request, *args, **kwargs)
        if result is not None and request.workload_class == "qa":
            reserved[request.request_id] = perf_counter()
        return result

    async def tracked_mark_send(reservation):
        await original_mark_send(reservation)
        if reservation.request.workload_class == "qa":
            sent[reservation.request.request_id] = perf_counter()

    scheduler.enqueue = tracked_enqueue
    scheduler.reserve = tracked_reserve
    scheduler.mark_send = tracked_mark_send
    for proxy in proxies:
        await proxy.start()
    payload = {"model": "model-1", "messages": [{"role": "user", "content": "test"}],
               "stream": True, "max_tokens": 1}

    # Exercise every endpoint once before the timed window. Otherwise the
    # backlog=0 run alone includes first-use HTTP/Pydantic initialization.
    for proxy in proxies:
        response = await proxy.open("bench-warmup", payload, request_bound=30,
                                    workload_class="qa")
        assert b"[DONE]" in b"".join([chunk async for chunk in response.body_iterator])
    while any(permit.kind == "direct" for permit in scheduler.permits.values()):
        await asyncio.sleep(0)
    queued.clear()
    reserved.clear()
    sent.clear()
    backend_calls = 0
    started = perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF)

    async def send(index):
        await asyncio.sleep(index / 100)
        response = await proxies[index % 4].open("bench-tenant", payload,
            request_bound=30, workload_class="qa")
        body = b"".join([chunk async for chunk in response.body_iterator])
        assert b"[DONE]" in body

    try:
        await asyncio.wait_for(asyncio.gather(*(send(index) for index in range(100))), timeout=60)
        elapsed = perf_counter() - started
        after = resource.getrusage(resource.RUSAGE_SELF)
        dispatch = sorted((sent[key] - queued[key]) * 1000 for key in queued if key in sent)
        admission = sorted((reserved[key] - queued[key]) * 1000 for key in queued if key in reserved)
        assert len(dispatch) == len(admission) == 100
        cpu_seconds = ((after.ru_utime - before.ru_utime) +
                       (after.ru_stime - before.ru_stime))
        print("BENCHMARK " + json.dumps({"backlog": backlog, "requests": 100,
            "endpoints": 4, "arrival_rps": 100,
            "preload_seconds": round(preload_seconds, 3),
            "elapsed_seconds": round(elapsed, 3),
            "dispatch_p50_ms": round(dispatch[49], 2),
            "dispatch_p95_ms": round(dispatch[94], 2),
            "dispatch_p99_ms": round(dispatch[98], 2),
            "admission_p95_ms": round(admission[94], 2),
            "backend_calls": backend_calls,
            "process_cpu_percent": round(cpu_seconds / elapsed * 100, 2)}))
    finally:
        for proxy in proxies:
            await proxy.close()
        await scheduler.close()
