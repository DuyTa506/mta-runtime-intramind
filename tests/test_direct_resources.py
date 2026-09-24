"""Direct embedding/rerank preserve native payloads while validating qualified bounds."""

import json
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import httpx
import pytest
from conftest import operation, pool, root
from fakes import MemoryArtifacts
from fastapi.responses import JSONResponse
from test_embedding import embedding_pool, profile

from intramind_runtime.api import create_app
from intramind_runtime.cli import direct_proxies
from intramind_runtime.contracts import RerankPoolSpec, parse_pool
from intramind_runtime.direct import DirectAdmissions
from intramind_runtime.direct_proxy import DirectProxy
from intramind_runtime.rerank import RerankProfile


def rerank_pool():
    return RerankPoolSpec(**(pool("rerank").model_dump(exclude={"context_limit"}) | {
        "character_limit": 100, "max_batch_size": 2,
    }))


def rerank_profile():
    return RerankProfile(model_profile="test", capacity_profile_id="test-v1",
        model_revision="model-1", character_limit=100, max_batch_size=2,
        max_response_bytes=4096)


@pytest.mark.parametrize("kind", ["embedding", "rerank"])
async def test_direct_resource_route_forwards_native_payload_without_artifact_or_workflow(kind):
    qualification = profile() if kind == "embedding" else rerank_profile()
    spec = embedding_pool() if kind == "embedding" else rerank_pool()
    payload = ({"texts": ["", "xin chào"], "input_type": "query"} if kind == "embedding" else
               {"query": "q", "documents": ["a", "bbb", "omitted-by-native-cap"], "top_k": 1})
    proxy = SimpleNamespace(pool=spec, profile=qualification, open=AsyncMock(
        return_value=JSONResponse({"ok": True})), close=AsyncMock())
    blobs = MemoryArtifacts()
    blobs.put = AsyncMock(side_effect=AssertionError("no foreground artifacts"))
    app = create_app(object(), blobs, "t" * 32, {}, direct_proxies={spec.model_profile: proxy})
    async with httpx.AsyncClient(base_url="http://runtime", transport=httpx.ASGITransport(app),
        headers={"Authorization": "Bearer " + "t" * 32, "X-Tenant-ID": "owner"}) as client:
        path = "embed" if kind == "embedding" else "rerank"
        result = await client.post(f"/v1/direct/{spec.model_profile}/api/v1/{path}", json=payload)
        assert result.status_code == 200
        bound = 8 if kind == "embedding" else 5
        proxy.open.assert_awaited_once_with("owner", payload, request_bound=bound,
                                            path=f"api/v1/{path}", batch_size=2,
                                            workload_class="background", logical_request_id=None,
                                            deadline_seconds=None, started_at_monotonic=ANY)
        proxy.open.reset_mock()
        invalid = {"texts": ["x" * 25]} if kind == "embedding" else payload | {"query": "x" * 101}
        assert (await client.post(f"/v1/direct/{spec.model_profile}/api/v1/{path}", json=invalid)).status_code == 422
        proxy.open.assert_not_awaited()
        if kind == "rerank":
            empty = await client.post(f"/v1/direct/{spec.model_profile}/api/v1/rerank",
                                      json={"query": "q" * 101, "documents": []})
            assert empty.json() == {"results": []}
            proxy.open.assert_not_awaited()


async def test_direct_resource_configuration_pins_proof_profile_and_response_limits():
    embedding = embedding_pool()
    rerank = rerank_pool()
    pools = [
        {"admission": embedding.model_dump(mode="json"), "base_url": "http://serving",
         "direct_enabled": True, "embedding": profile().model_dump(exclude={"model_profile",
             "capacity_profile_id", "character_limit", "max_batch_size", "model_revision"}) | {
             "validated_profile_id": embedding.profile_id, "termination_contract": "termination-v1"}},
        {"admission": rerank.model_dump(mode="json"), "base_url": "http://serving",
         "direct_enabled": True, "rerank": {"validated_profile_id": rerank.profile_id,
              "termination_contract": "termination-v1", "max_response_bytes": 4096}},
    ]
    proxies = direct_proxies({"pools": pools}, object(), {})
    try:
        assert proxies[embedding.model_profile].profile == profile()
        assert proxies[rerank.model_profile].profile == rerank_profile()
        assert all(proxy.max_response_bytes == 4096 for proxy in proxies.values())
        assert parse_pool(rerank.model_dump()) == rerank
    finally:
        for proxy in proxies.values():
            await proxy.close()
    pools[1]["rerank"]["termination_contract"] = "unqualified"
    with pytest.raises(ValueError, match="contract"):
        direct_proxies({"pools": pools}, object(), {})


@pytest.mark.parametrize("kind,valid", [(kind, valid) for kind in ("embedding", "rerank") for valid in (True, False)])
@pytest.mark.integration
async def test_direct_proxy_rejects_invalid_native_output_before_publishing(store, kind, valid):
    spec = embedding_pool() if kind == "embedding" else rerank_pool()
    await store.configure_pool(spec, 2)
    qualification = profile() if kind == "embedding" else rerank_profile()
    payload = {"texts": ["a"]} if kind == "embedding" else {"query": "q", "documents": ["a"]}
    body = ({"embeddings": [[0.1, 0.2]] if valid else [[0.1]], "dimension": 2, "model": "native"}
            if kind == "embedding" else {"results": [{"index": 0 if valid else 9, "score": -2.5}]})

    async def upstream(request):
        return httpx.Response(200, json=body, headers={
            "X-Intramind-Attempt-ID": request.headers["X-Intramind-Attempt-ID"],
            f"X-Intramind-{kind.title()}-Contract": "termination-v1",
            "X-Intramind-Compute-State": "terminated", "X-Intramind-Model-Revision": spec.model_revision})

    proxy = DirectProxy(store, pool=spec, profile=qualification,
        client=httpx.AsyncClient(base_url="http://native/", transport=httpx.MockTransport(upstream)))
    try:
        response = await proxy.open("t", payload, request_bound=2)
        returned = json.loads(b"".join([chunk async for chunk in response.body_iterator]))
        assert response.status_code == (200 if valid else 502)
        if valid:
            assert returned == body
        else:
            assert "error" in returned
        assert (await store.drain_status())["compute_held"] == 0
    finally:
        await proxy.close()


@pytest.mark.integration
async def test_rerank_direct_does_not_serialize_llm_transport(store):
    from test_direct_admission import request

    await store.configure_pool(pool(), 1)
    await store.configure_pool(rerank_pool(), 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    direct = DirectAdmissions(store)
    direct_request = request(kind="rerank", batch_size=2)
    await direct.enqueue(direct_request, "rerank", "owner")
    held = await direct.reserve(direct_request, "rerank", "owner")
    assert held is not None
    assert await store.reserve_next("p", "background") is not None
    await direct.finish(held, evidence="not_sent")
    assert (await store.drain_status())["compute_held"] == 1
