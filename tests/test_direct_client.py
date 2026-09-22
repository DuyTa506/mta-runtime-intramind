"""Caller binding keeps identity scoped and never retries or contacts the engine."""

import asyncio

import httpx
import pytest

from intramind_runtime.direct_client import DirectBinding, DirectRouting, inference_scope


def binding():
    return DirectBinding("http://runtime:8090", "secret-" * 6, "query-embedding")


def test_native_path_body_and_headers_survive_binding():
    requests = []
    with httpx.Client(**binding().client_kwargs(), transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json={})
    )) as client, inference_scope("user:one"):
        client.post("/api/v1/embed", json={"texts": ["hello"], "input_type": "query"})
    request, = requests
    assert str(request.url) == "http://runtime:8090/v1/direct/query-embedding/api/v1/embed"
    assert request.headers["X-Tenant-ID"] == "user:one"
    assert request.headers["Authorization"] == "Bearer " + "secret-" * 6
    assert request.content == b'{"texts":["hello"],"input_type":"query"}'


def test_missing_tenant_cannot_send_or_inherit_last_request():
    requests = []
    with httpx.Client(**binding().client_kwargs(), transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200)
    )) as client:
        with inference_scope("user:one"):
            client.post("/api/v1/embed")
        with pytest.raises(ValueError, match="identity"):
            client.post("/api/v1/embed")
    assert len(requests) == 1


async def test_one_shared_client_isolates_concurrent_tenants_and_threads():
    seen = []

    async def respond(request):
        await asyncio.sleep(0)
        seen.append(request.headers["X-Tenant-ID"])
        return httpx.Response(200)

    async with httpx.AsyncClient(**binding().client_kwargs(),
                                transport=httpx.MockTransport(respond)) as client:
        async def call(tenant):
            with inference_scope(tenant):
                await asyncio.sleep(0)
                await client.post("/api/v1/embed")
                return await asyncio.to_thread(lambda: _thread_tenant(binding()))

        assert await asyncio.gather(call("user:one"), call("user:two")) == ["user:one", "user:two"]
    assert sorted(seen) == ["user:one", "user:two"]


def _thread_tenant(config):
    with httpx.Client(**config.client_kwargs(), transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=request.headers["X-Tenant-ID"])
    )) as client:
        return client.post("/api/v1/embed").text


def test_redirect_and_timeout_do_not_retry_or_leak_credential():
    seen = []

    def respond(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(307, headers={"Location": "http://elsewhere/"})
        raise httpx.ReadTimeout("lost response", request=request)

    with httpx.Client(**binding().client_kwargs(), transport=httpx.MockTransport(respond)) as client:
        with inference_scope("user:one"):
            assert client.post("/api/v1/embed").status_code == 307
            with pytest.raises(httpx.ReadTimeout):
                client.post("/api/v1/embed")
    assert len(seen) == 2
    assert all(request.url.host == "runtime" for request in seen)


@pytest.mark.parametrize("profile", ["", "../other", "one/two", "one?x=y"])
def test_profile_cannot_change_route(profile):
    with pytest.raises(ValueError):
        DirectBinding("http://runtime", "s" * 32, profile)


def test_secret_is_not_in_representation():
    assert "secret-" not in repr(binding())


def test_routing_matches_original_endpoint_and_model_without_mutating_config():
    routing = DirectRouting(enabled=True, llm_routes=[{
        "base_url": "http://llama:8080/v1", "model": "answer", "model_profile": "chat",
    }], embedding_profile="embed")
    config = {"provider": "openai", "base_url": "http://llama:8080/v1/", "model_name": "answer"}
    original = dict(config)
    assert routing.llm_binding(config, "http://runtime", "s" * 32).model_profile == "chat"
    assert config == original
    assert routing.native_binding("embedding", "http://runtime", "s" * 32).model_profile == "embed"
    with pytest.raises(ValueError, match="qualified"):
        routing.llm_binding(config | {"model_name": "other"}, "http://runtime", "s" * 32)
    with pytest.raises(ValueError, match="qualified"):
        routing.native_binding("rerank", "http://runtime", "s" * 32)
    assert DirectRouting().llm_binding(config, "", "") is None


def test_routing_rejects_duplicate_backend_and_unsupported_provider():
    route = {"base_url": "http://llama/v1", "model": "answer", "model_profile": "chat"}
    with pytest.raises(ValueError):
        DirectRouting(enabled=True, llm_routes=[route, route | {"base_url": "http://llama/v1/"}])
    routing = DirectRouting(enabled=True, llm_routes=[route])
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        routing.llm_binding({"provider": "ollama", "base_url": "http://llama/v1",
                             "model_name": "answer"}, "http://runtime", "s" * 32)


def test_binding_refuses_absolute_foreign_url_before_send():
    def unexpected(request):
        pytest.fail("credentials must not reach an unbound route")

    with httpx.Client(**binding().client_kwargs(), transport=httpx.MockTransport(unexpected)) as client:
        with inference_scope("user:one"), pytest.raises(ValueError, match="outside"):
            client.post("http://engine/chat/completions")
