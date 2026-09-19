import httpx
from fakes import MemoryArtifacts

from intramind_runtime.api import create_app
from intramind_runtime.client import RuntimeClient
from intramind_runtime.preparation import LlamaCppPromptSizer

TOKEN = "isolated-test-credential-only-12345678"


async def test_planner_reads_pinned_context_capability_without_touching_inference_or_ledger():
    def forbidden(request):
        raise AssertionError("profile discovery must not invoke the engine")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as backend:
        sizer = LlamaCppPromptSizer(
            backend, model="tool", capacity_profile_id="gpu-profile-v7", context_limit=16384,
            token_margin=8, expected_output_tokens=1024,
            response_formats=frozenset({"json_schema", "text"}), allow_tool_calls=True,
        )
        app = create_app(object(), MemoryArtifacts(), TOKEN, {}, preparers={"review": sizer})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as http:
            assert (await http.get("/v1/llm/profiles/review")).status_code == 401
            http.headers.update({"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "t"})
            client = RuntimeClient("http://test", TOKEN, "t", client=http)
            profile = await client.llm_profile("review")
            assert profile == {
                "model_profile": "review", "capacity_profile_id": "gpu-profile-v7",
                "context_limit": 16384, "response_formats": ["json_schema", "text"],
                "allow_tool_calls": True,
            }
            assert (await http.get("/v1/llm/profiles/missing")).status_code == 503


async def test_profile_does_not_invent_tool_or_schema_qualification():
    async with httpx.AsyncClient() as backend:
        sizer = LlamaCppPromptSizer(
            backend, model="tool", capacity_profile_id="text-only", context_limit=4096,
            token_margin=8, expected_output_tokens=256,
        )
        app = create_app(object(), MemoryArtifacts(), TOKEN, {}, preparers={"text": sizer})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test",
            headers={"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "t"},
        ) as http:
            profile = (await http.get("/v1/llm/profiles/text")).json()
            assert profile["allow_tool_calls"] is False
            assert profile["response_formats"] == ["text"]
