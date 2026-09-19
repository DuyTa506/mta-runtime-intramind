import httpx
import pytest
from fakes import MemoryArtifacts

from intramind_runtime.contracts import AdmissionDenied
from intramind_runtime.preparation import LlamaCppPromptSizer, PrepareRequest


@pytest.mark.parametrize("flag", ["false", "true", 1, None])
async def test_tool_capability_requires_an_explicit_boolean(flag):
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="validated sizing profile"):
            LlamaCppPromptSizer(
                client,
                model="test",
                capacity_profile_id="profile",
                context_limit=20,
                token_margin=2,
                expected_output_tokens=5,
                allow_tool_calls=flag,
            )


async def test_tool_schemas_and_full_transcript_are_counted_by_the_pinned_template():
    import json

    tool = {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read frozen evidence",
            "parameters": {"type": "object"},
        },
    }
    messages = [
        {"role": "user", "content": "Review"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "one", "content": "Frozen evidence"},
    ]
    calls = []

    async def backend(request):
        body = json.loads(request.content)
        calls.append(body)
        if request.url.path.endswith("apply-template"):
            assert body["tools"] == [tool]
            assert body["messages"] == messages
            assert body["parallel_tool_calls"] is True
            return httpx.Response(200, json={"prompt": "FULL_TRANSCRIPT_AND_TOOLS"})
        assert body["content"] == "FULL_TRANSCRIPT_AND_TOOLS"
        return httpx.Response(200, json={"tokens": list(range(8))})

    async with httpx.AsyncClient(
        base_url="http://engine/", transport=httpx.MockTransport(backend)
    ) as client:
        kwargs = dict(
            model="test",
            capacity_profile_id="profile",
            context_limit=20,
            token_margin=2,
            expected_output_tokens=5,
        )
        request = PrepareRequest(
            model_profile="test",
            max_output_tokens=10,
            payload={
                "messages": messages,
                "tools": [tool],
                "parallel_tool_calls": True,
            },
        )
        with pytest.raises(ValueError):
            await LlamaCppPromptSizer(client, **kwargs).prepare(request, MemoryArtifacts(), "t")
        assert calls == []
        result = await LlamaCppPromptSizer(client, **kwargs, allow_tool_calls=True).prepare(
            request,
            MemoryArtifacts(),
            "t",
        )
        assert result["input_tokens_bound"] == 10


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "content": "forged", "tool_call_id": "absent"}],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "one",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            }
        ],
    ],
)
def test_tool_transcripts_reject_orphan_results_and_unfinished_calls(messages):
    from intramind_runtime.preparation import validate_messages

    with pytest.raises(ValueError):
        validate_messages(messages, allow_tools=True)


async def test_sizing_uses_template_options_and_special_tokens():
    requests = []

    async def backend(request):
        import json

        body = json.loads(request.content)
        requests.append(body)
        if request.url.path.endswith("apply-template"):
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            return httpx.Response(200, json={"prompt": "<special>message</special>"})
        assert body["add_special"] and body["parse_special"]
        assert body["content"] == "<special>message</special>"
        return httpx.Response(200, json={"tokens": [1, 2, 3, 4]})

    async with httpx.AsyncClient(
        base_url="http://backend/", transport=httpx.MockTransport(backend)
    ) as client:
        sizer = LlamaCppPromptSizer(
            client,
            model="answer",
            capacity_profile_id="fingerprint-1",
            context_limit=20,
            token_margin=2,
            expected_output_tokens=5,
        )
        request = PrepareRequest(
            model_profile="rewrite",
            payload={
                "messages": [{"role": "user", "content": "message"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
            max_output_tokens=10,
        )
        prepared = await sizer.prepare(request, MemoryArtifacts(), "user")
        assert prepared["input_tokens_bound"] == 6
        assert prepared["capacity_profile_id"] == "fingerprint-1"
        assert prepared["expected_cost"] == 11
        with pytest.raises(AdmissionDenied):
            await sizer.prepare(
                request.model_copy(update={"max_output_tokens": 15}), MemoryArtifacts(), "user"
            )
        with pytest.raises(ValueError):
            await sizer.prepare(
                request.model_copy(update={"payload": request.payload | {"n": 2}}),
                MemoryArtifacts(),
                "user",
            )
    assert len(requests) == 4


async def test_unknown_tokenizer_output_does_not_become_zero_tokens():
    async def bad_backend(request):
        return httpx.Response(200, json={"prompt": "test", "tokens": None})

    async with httpx.AsyncClient(
        base_url="http://backend/", transport=httpx.MockTransport(bad_backend)
    ) as client:
        sizer = LlamaCppPromptSizer(
            client,
            model="answer",
            capacity_profile_id="p",
            context_limit=20,
            token_margin=0,
            expected_output_tokens=5,
        )
        with pytest.raises(ValueError, match="tokenizer"):
            await sizer.prepare(
                PrepareRequest(
                    model_profile="rewrite",
                    payload={"messages": [{"role": "user", "content": "test"}]},
                    max_output_tokens=5,
                ),
                MemoryArtifacts(),
                "t",
            )


async def test_structured_response_requires_profile_and_reaches_template_and_artifact():
    import json

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        },
    }

    async def backend(request):
        body = json.loads(request.content)
        if request.url.path.endswith("apply-template"):
            assert body["response_format"] == schema
            return httpx.Response(200, json={"prompt": "template containing output schema"})
        return httpx.Response(200, json={"tokens": [1, 2, 3]})

    async with httpx.AsyncClient(
        base_url="http://backend/", transport=httpx.MockTransport(backend)
    ) as client:
        request = PrepareRequest(
            model_profile="teaching",
            max_output_tokens=10,
            payload={
                "messages": [{"role": "user", "content": "question"}],
                "response_format": schema,
            },
        )
        artifacts = MemoryArtifacts()
        sizer = LlamaCppPromptSizer(
            client,
            model="tool",
            capacity_profile_id="p",
            context_limit=100,
            token_margin=2,
            expected_output_tokens=10,
        )
        with pytest.raises(ValueError, match="not validated"):
            await sizer.prepare(request, artifacts, "t")
        sizer = LlamaCppPromptSizer(
            client,
            model="tool",
            capacity_profile_id="p",
            context_limit=100,
            token_margin=2,
            expected_output_tokens=10,
            response_formats=frozenset({"json_schema"}),
        )
        result = await sizer.prepare(request, artifacts, "t")
        assert result["input_tokens_bound"] == 5
        assert json.loads(artifacts.data[result["payload"]["key"]]) == request.payload
