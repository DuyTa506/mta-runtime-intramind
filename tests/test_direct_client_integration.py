"""Caller auth, direct routes and the shared ledger compose without a workflow."""

import httpx
import pytest
from fakes import MemoryArtifacts
from sqlalchemy import text
from test_direct_resources import rerank_pool, rerank_profile
from test_embedding import embedding_pool, profile

from intramind_runtime.api import create_app
from intramind_runtime.direct_client import DirectBinding, inference_scope
from intramind_runtime.direct_proxy import DirectProxy


@pytest.mark.integration
@pytest.mark.parametrize("kind,timeout", [("embedding", False), ("rerank", False), ("embedding", True)])
async def test_bound_client_uses_native_proof_and_persisted_quota(store, kind, timeout):
    spec = embedding_pool() if kind == "embedding" else rerank_pool()
    qualification = profile() if kind == "embedding" else rerank_profile()
    await store.configure_pool(spec, 1)
    calls = []
    body = ({"embeddings": [[0.1, 0.2]], "dimension": 2, "model": "native"}
            if kind == "embedding" else {"results": [{"index": 0, "score": -0.2}]})

    async def backend(request):
        calls.append(request)
        if timeout:
            raise httpx.ReadTimeout("backend still computing", request=request)
        return httpx.Response(200, json=body, headers={
            "X-Intramind-Attempt-ID": request.headers["X-Intramind-Attempt-ID"],
            f"X-Intramind-{kind.title()}-Contract": "termination-v1",
            "X-Intramind-Compute-State": "terminated",
            "X-Intramind-Model-Revision": spec.model_revision,
        })

    proxy = DirectProxy(store, pool=spec, profile=qualification,
                       client=httpx.AsyncClient(base_url="http://native/", transport=httpx.MockTransport(backend)))
    app = create_app(store, MemoryArtifacts(), "s" * 32, {}, direct_proxies={spec.model_profile: proxy})
    binding = DirectBinding("http://runtime", "s" * 32, spec.model_profile)
    payload = {"texts": ["hello"], "input_type": "query"} if kind == "embedding" else {
        "query": "hello", "documents": ["world"], "top_k": 1}
    endpoint = "embed" if kind == "embedding" else "rerank"
    try:
        async with httpx.AsyncClient(**binding.client_kwargs(), transport=httpx.ASGITransport(app)) as client:
            with inference_scope("user:verified"):
                response = await client.post(f"/api/v1/{endpoint}", json=payload)
        assert response.status_code == (502 if timeout else 200)
        if not timeout:
            assert response.json() == body
        assert len(calls) == 1
        async with store.engine.connect() as connection:
            row = (await connection.execute(text(
                "SELECT tenant_id, state, compute_held FROM runtime_direct_attempts"))).one()
            assert row == ("user:verified", "UNKNOWN" if timeout else "FINISHED", timeout)
            assert (await connection.execute(text("SELECT count(*) FROM runtime_roots"))).scalar_one() == 0
            assert (await connection.execute(text("SELECT count(*) FROM runtime_operations"))).scalar_one() == 0
    finally:
        await proxy.close()
