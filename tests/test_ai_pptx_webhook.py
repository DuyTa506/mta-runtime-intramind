"""SQLite handoff, real HTTP lost response, and Temporal recovery for PPTX hooks."""

import asyncio
import os
from contextlib import suppress
from hashlib import sha256
from types import SimpleNamespace
from uuid import uuid4

import pytest
from feature_harness import feature_environment

pytestmark = pytest.mark.integration


async def test_webhook_restart_preserves_delivery_identity_without_inference(store, monkeypatch, tmp_path):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    from aiohttp import web
    from api.background.pptx import webhooks
    from api.background.pptx.webhook_workflows import webhook
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlmodel import SQLModel, select
    from tools.pptx.engine.enums.webhook_event import WebhookEvent
    from tools.pptx.engine.models.sql.user import User
    from tools.pptx.engine.models.sql.webhook_outbox import WebhookOutbox
    from tools.pptx.engine.models.sql.webhook_subscription import WebhookSubscription
    from tools.pptx.engine.services.webhook_outbox import capture, relay_once

    received, lost_ack, release_ack = [], asyncio.Event(), asyncio.Event()
    presentation_id = uuid4().hex

    async def receive(request):
        received.append((request.path, dict(request.headers), await request.json()))
        if request.path == "/lost" and sum(path == "/lost" for path, _, _ in received) == 1:
            await release_ack.wait()
            request.transport.close()
            lost_ack.set()
        return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/{sink}", receive)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/native.db")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    run_ids = []
    try:
        async with engine.begin() as connection:
            await connection.run_sync(lambda conn: SQLModel.metadata.create_all(conn, tables=[
                User.__table__, WebhookSubscription.__table__, WebhookOutbox.__table__]))
        async with sessions() as session:
            for sink in ("lost", "ok"):
                session.add(WebhookSubscription(id=sink, owner_id=None, url=f"http://127.0.0.1:{port}/{sink}",
                    secret="private-" + sink, event=WebhookEvent.PRESENTATION_GENERATION_COMPLETED.value))
            await session.commit()
            await capture(session, presentation_id, WebhookEvent.PRESENTATION_GENERATION_COMPLETED,
                          {"path": "a.pptx", "edit_path": "/presentation?id=deck-1"})
            await session.commit()
        limits = {"max_batch_size": 1, "max_delay_seconds": 0, "max_item_bytes": 1_048_576,
                  "max_pending_per_tenant": 4, "max_pending_per_partition": 1}

        async def unexpected_inference(*args):
            pytest.fail("webhook delivery must not acquire inference capacity")

        async with feature_environment(store, name="pptx.webhook", workflows=[webhook], buffering=limits,
                build_activities=lambda factory: [webhooks.WebhookActivities(factory).deliver],
                respond=unexpected_inference) as env:
            monkeypatch.setattr(webhooks, "RuntimeClient", lambda url, token, tenant: env.runtime_client(tenant))
            monkeypatch.setattr(webhooks, "get_settings", lambda: SimpleNamespace(url="http://runtime",
                service_token=SimpleNamespace(get_secret_value=lambda: "x" * 32)))
            api = env.runtime_client("system:pptx")
            try:
                await relay_once(sessions, webhooks.accept)
                async with sessions() as session:
                    rows = list(await session.scalars(select(WebhookOutbox)))
                assert len(rows) == 2 and all(row.receipt and row.payload is None for row in rows)
                ok_id = sha256(f"{presentation_id}/{WebhookEvent.PRESENTATION_GENERATION_COMPLETED.value}/ok".encode()).hexdigest()
                for row in rows:
                    if row.id == ok_id:
                        async with asyncio.timeout(30):
                            while True:
                                buffered = await api.get_buffered(row.receipt)
                                ok_run = buffered.get("run_id")
                                if buffered.get("state") == "SUCCEEDED":
                                    break
                                await asyncio.sleep(0.05)
                        await asyncio.wait_for(env.temporal.get_workflow_handle(ok_run).result(), timeout=30)
                release_ack.set()
                await asyncio.wait_for(lost_ack.wait(), timeout=30)
                await asyncio.wait_for(env.restart_worker(), timeout=30)
                for row in rows:
                    run_id = (await api.get_buffered(row.receipt))["run_id"]
                    run_ids.append(run_id)
                    await asyncio.wait_for(env.temporal.get_workflow_handle(run_id).result(), timeout=30)
                    state = await api.get_run(run_id)
                    assert state["state"] == "SUCCEEDED" and state["spent"] == state["reserved"] == 0
                    result = await api.read_json(state["result"])
                    assert result == {"delivery_id": row.id, "status": 204}
                    history = await env.temporal.get_workflow_handle(run_id).fetch_history()
                    assert "private-" not in str(history) and f"127.0.0.1:{port}" not in str(history)
                    await env.replay(run_id)
                await relay_once(sessions, webhooks.accept)
                assert env.calls == []
                lost = [(headers, body) for path, headers, body in received if path == "/lost"]
                ok = [(headers, body) for path, headers, body in received if path == "/ok"]
                assert len(lost) == 2 and len(ok) == 1
                for path, headers, body in received:
                    assert headers["Authorization"] == "Bearer private-" + path[1:]
                    assert headers["Idempotency-Key"] == headers["X-Intramind-Delivery-ID"]
                    assert body["path"] == "a.pptx"
                assert lost[0] == lost[1]
            finally:
                for run_id in run_ids:
                    with suppress(Exception):
                        handle = env.temporal.get_workflow_handle(run_id)
                        if (await handle.describe()).close_time is None:
                            await handle.terminate("disposable webhook test cleanup")
                await api.close()
    finally:
        release_ack.set()
        await engine.dispose()
        await runner.cleanup()
