"""Explicit disposable benchmark for the direct dispatcher with mock serving.

Run with the same guarded RUNTIME_TEST_DATABASE_URL / RUNTIME_TEST_ALLOW_RESET
used by integration tests, then `pytest -s tests/benchmarks/bench_dispatch.py`.
No inference engine or model is contacted. This is a measurement, not a pass
for a production latency or 30-minute soak gate.
"""

import asyncio
import json
import os
from datetime import UTC, datetime
from time import perf_counter

import httpx
import pytest
from conftest import pool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from intramind_runtime.direct import DirectAdmissions, DirectRequest
from intramind_runtime.direct_proxy import DirectProxy


@pytest.mark.integration
async def test_synthetic_dispatcher_load(store):
    backlog = int(os.environ.get("RUNTIME_BENCH_BACKLOG", "1000"))
    if not 0 <= backlog <= 1000:
        raise ValueError("benchmark backlog must be between 0 and 1000")
    pool_size = int(os.environ.get("RUNTIME_BENCH_DB_POOL", "5"))
    if pool_size != 5:
        await store.engine.dispose()
        store.engine = create_async_engine(store.database_url, pool_size=pool_size,
            max_overflow=pool_size, pool_pre_ping=True, hide_parameters=True)
    pools = [pool(f"bench-{index}", target=4, transport_limit=64,
                  background_transport_limit=8) for index in range(4)]
    for spec in pools:
        await store.configure_pool(spec, 1)
    await store.register_owner("bench-backlog", "bench-boot")
    admission = DirectAdmissions(store)
    deadline = datetime(3000, 1, 1, tzinfo=UTC)
    preload_started = perf_counter()
    for index in range(backlog):
        spec = pools[index % 4]
        await admission.enqueue(DirectRequest(request_id=f"backlog-{index}",
            tenant_id="bench-backlog-tenant", payload_digest="a" * 64,
            model_profile="test", capacity_profile_id="test-v1", request_bound=30,
            deadline=deadline, workload_class="background"), spec.pool_id, "bench-backlog")
    preload_seconds = perf_counter() - preload_started
    backend_calls = 0
    queued = {}
    reserved = {}
    sent = {}

    async def backend(request):
        nonlocal backend_calls
        backend_calls += 1
        return httpx.Response(200, content=b"data: [DONE]\n\n",
                              headers={"Content-Type": "text/event-stream"})

    proxies = [DirectProxy(store, pool=spec, client=httpx.AsyncClient(
        base_url="http://engine/v1/", transport=httpx.MockTransport(backend))) for spec in pools]
    for proxy in proxies:
        original_enqueue = proxy.admission.enqueue
        original_reserve = proxy.admission.reserve
        original_mark_send = proxy.admission.mark_send

        async def tracked_enqueue(request, *args, _original=original_enqueue, **kwargs):
            created_at = await _original(request, *args, **kwargs)
            if request.workload_class == "qa":
                queued[request.request_id] = perf_counter()
            return created_at

        async def tracked_mark_send(reservation, _original=original_mark_send):
            await _original(reservation)
            if reservation.request.workload_class == "qa":
                sent[reservation.request.request_id] = perf_counter()

        async def tracked_reserve(request, *args, _original=original_reserve, **kwargs):
            reservation = await _original(request, *args, **kwargs)
            if reservation is not None and request.workload_class == "qa":
                reserved[request.request_id] = perf_counter()
            return reservation

        proxy.admission.enqueue = tracked_enqueue
        proxy.admission.reserve = tracked_reserve
        proxy.admission.mark_send = tracked_mark_send
        await proxy.start()
    async with store.engine.connect() as connection:
        before = (await connection.execute(text("""SELECT xact_commit,xact_rollback
            FROM pg_stat_database WHERE datname='runtime_test'"""))).mappings().one()
    started = perf_counter()
    payload = {"model": "model-1", "messages": [{"role": "user", "content": "test"}],
               "stream": True, "max_tokens": 1}

    async def send(index):
        await asyncio.sleep(index / 100)
        waiting_since = perf_counter()
        response = await proxies[index % 4].open("bench-tenant", payload,
            request_bound=30, workload_class="qa")
        opened_at = perf_counter()
        body = b"".join([chunk async for chunk in response.body_iterator])
        assert b"[DONE]" in body
        completed_at = perf_counter()
        return ((opened_at - waiting_since) * 1000,
                (completed_at - opened_at) * 1000,
                (completed_at - waiting_since) * 1000)

    try:
        measurements = await asyncio.wait_for(asyncio.gather(
            *(send(index) for index in range(100))), timeout=60)
        opened = sorted(item[0] for item in measurements)
        streamed = sorted(item[1] for item in measurements)
        latencies = sorted(item[2] for item in measurements)
        dispatch = sorted((sent[key] - queued[key]) * 1000 for key in queued if key in sent)
        admission = sorted((reserved[key] - queued[key]) * 1000 for key in queued if key in reserved)
        send_intent = sorted((sent[key] - reserved[key]) * 1000 for key in reserved if key in sent)
        assert len(dispatch) == 100
        assert len(admission) == len(send_intent) == 100
        elapsed = perf_counter() - started
        async with store.engine.connect() as connection:
            await connection.execute(text("SELECT pg_stat_clear_snapshot()"))
            after = (await connection.execute(text("""SELECT xact_commit,xact_rollback
                FROM pg_stat_database WHERE datname='runtime_test'"""))).mappings().one()
        print("BENCHMARK " + json.dumps({"backlog": backlog, "requests": 100,
            "endpoints": 4, "arrival_rps": 100, "db_pool_size": pool_size,
            "preload_seconds": round(preload_seconds, 3),
            "elapsed_seconds": round(elapsed, 3), "p50_ms": round(latencies[49], 2),
            "p95_ms": round(latencies[94], 2), "p99_ms": round(latencies[98], 2),
            "enqueue_p95_ms": round(opened[94], 2),
            "stream_p95_ms": round(streamed[94], 2),
            "dispatch_p50_ms": round(dispatch[49], 2),
            "dispatch_p95_ms": round(dispatch[94], 2),
            "dispatch_p99_ms": round(dispatch[98], 2),
            "admission_p95_ms": round(admission[94], 2),
            "send_intent_p95_ms": round(send_intent[94], 2),
            "backend_calls": backend_calls,
            "db_commits_delta": after["xact_commit"] - before["xact_commit"],
            "db_rollbacks_delta": after["xact_rollback"] - before["xact_rollback"]}))
    finally:
        for proxy in proxies:
            await proxy.close()
