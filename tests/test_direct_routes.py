"""Foreground admission validates the pinned contract without workflow/artifact writes."""

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import httpx
import pytest
from conftest import pool
from fakes import MemoryArtifacts
from fastapi.responses import JSONResponse

from intramind_runtime.api import create_app
from intramind_runtime.preparation import LlamaCppPromptSizer

TOKEN = "direct-route-test-credential-123456789"
BODY = {"model": "model-1", "messages": [{"role": "user", "content": "hello"}]}


@pytest.fixture
async def direct_api():
    async def sizing(request):
        return httpx.Response(200, json={"prompt": "hello"} if request.url.path.endswith(
            "apply-template") else {"tokens": [1, 2, 3]})

    async with httpx.AsyncClient(base_url="http://engine/", transport=httpx.MockTransport(sizing)) as engine:
        sizer = LlamaCppPromptSizer(engine, model="model-1", capacity_profile_id="test-v1",
            context_limit=20, token_margin=2, expected_output_tokens=5)
        proxy = SimpleNamespace(pool=pool(), model="model-1", open=AsyncMock(
            return_value=JSONResponse({"choices": []})), close=AsyncMock())
        blobs = MemoryArtifacts()
        blobs.put = AsyncMock(side_effect=AssertionError("foreground payload must not be persisted"))
        app = create_app(object(), blobs, TOKEN, {}, preparers={"test": sizer},
                         direct_proxies={"test": proxy})
        async with httpx.AsyncClient(base_url="http://runtime", transport=httpx.ASGITransport(app),
            headers={"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "owner"}) as client:
            yield client, proxy


@pytest.mark.parametrize("limit, bound", [(None, 20), (8, 13)])
async def test_direct_route_preserves_payload_and_uses_qualified_context_bound(direct_api, limit, bound):
    client, proxy = direct_api
    body = BODY | {"stream": True, "stream_options": {"include_usage": True},
                   "chat_template_kwargs": {"enable_thinking": False}}
    if limit:
        body["max_tokens"] = limit
    result = await client.post("/v1/direct/test/chat/completions", json=body)
    assert result.status_code == 200
    proxy.open.assert_awaited_once_with("owner", body, request_bound=bound,
                                        workload_class="background", logical_request_id=None,
                                        deadline_seconds=None, started_at_monotonic=ANY)


@pytest.mark.parametrize("header,status", [("5", 200), ("0", 400), ("-1", 400),
    ("later", 400), ("1.5", 400)])
async def test_direct_route_accepts_only_positive_deadline_seconds(direct_api, header, status):
    client, proxy = direct_api
    result = await client.post("/v1/direct/test/chat/completions", json=BODY,
                               headers={"X-Intramind-Deadline-Seconds": header})
    assert result.status_code == status
    if status == 200:
        assert proxy.open.await_args.kwargs["deadline_seconds"] == 5
    else:
        proxy.open.assert_not_awaited()


async def test_prompt_sizing_is_bounded_by_route_entry_deadline():
    import asyncio

    async def slow_size(request):
        await asyncio.sleep(3)

    proxy = SimpleNamespace(pool=pool(), model="model-1", open=AsyncMock(
        return_value=JSONResponse({"choices": []})), close=AsyncMock(),
        _deadline_seconds=lambda workload, requested: min(600, requested or 600))
    app = create_app(object(), MemoryArtifacts(), TOKEN, {},
        preparers={"test": SimpleNamespace(size=slow_size)}, direct_proxies={"test": proxy})
    async with httpx.AsyncClient(base_url="http://runtime", transport=httpx.ASGITransport(app),
        headers={"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "owner",
                 "X-Intramind-Deadline-Seconds": "1"}) as client:
        response = await client.post("/v1/direct/test/chat/completions", json=BODY)
    assert response.status_code == 504
    assert response.json()["error"]["reason"] == "deadline_exceeded"
    proxy.open.assert_not_awaited()


@pytest.mark.parametrize("extra", [{"n": True}, {"n": 2}, {"model": "other"}, {"stream": "true"},
    {"max_tokens": 16}, {"max_tokens": 2, "max_completion_tokens": 2}, {"priority": "bypass"}])
async def test_direct_route_rejects_unsupported_or_oversized_requests_before_admission(direct_api, extra):
    client, proxy = direct_api
    assert (await client.post("/v1/direct/test/chat/completions", json=BODY | extra)).status_code == 422
    proxy.open.assert_not_awaited()


async def test_direct_route_requires_authenticated_tenant_and_available_profile(direct_api):
    client, proxy = direct_api
    assert (await client.post("/v1/direct/missing/chat/completions", json=BODY)).status_code == 503
    client.headers.pop("X-Tenant-ID")
    assert (await client.post("/v1/direct/test/chat/completions", json=BODY)).status_code == 400
    client.headers.pop("Authorization")
    assert (await client.post("/v1/direct/test/chat/completions", json=BODY)).status_code == 401
    proxy.open.assert_not_awaited()


async def test_direct_cli_requires_unique_qualified_sizing_and_is_opt_in(monkeypatch):
    from intramind_runtime.cli import direct_proxies

    monkeypatch.setenv("DIRECT_TEST_KEY", "fixture")
    spec = pool()
    item = {"admission": spec.model_dump(mode="json"), "base_url": "http://engine/v1/",
            "api_key_env": "DIRECT_TEST_KEY", "model": "model-1",
            "attempt_timeout_seconds": 45,
            "workload_deadline_seconds": {"qa": 90, "user_task": 120}}
    assert direct_proxies({"pools": [item]}, object(), {}) == {}
    item["direct_enabled"] = True
    with pytest.raises(ValueError, match="sizing"):
        direct_proxies({"pools": [item]}, object(), {})
    sizing = {"test": SimpleNamespace(profile_id=spec.profile_id, model="model-1")}
    with pytest.raises(ValueError, match="ambiguous"):
        direct_proxies({"pools": [item, item]}, object(), sizing)
    proxies = direct_proxies({"pools": [item]}, object(), sizing)
    assert set(proxies) == {"test"}
    assert proxies["test"].workload_deadline_seconds == {
        "qa": 90, "user_task": 120, "background": 45, "maintenance": 45}
    await proxies["test"].close()
