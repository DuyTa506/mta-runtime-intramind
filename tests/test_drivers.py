import json

import httpx
import pytest
from conftest import operation, root

from intramind_runtime.contracts import Reservation
from intramind_runtime.drivers import DriverFailure, OpenAICompletionDriver


def reservation():
    return Reservation(attempt_id="attempt", operation=operation(), pool_id="pool", engine_epoch="e1",
                       model_revision="model-1", owner_id="executor", lease_epoch=1,
                       attempt_deadline=root().deadline)


async def test_transport_never_retries_and_timeout_is_unknown():
    count = 0

    async def handler(request):
        nonlocal count
        count += 1
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fake/v1/") as client:
        driver = OpenAICompletionDriver("http://fake/v1", "unused", "tool", client=client)
        with pytest.raises(DriverFailure) as error:
            await driver.execute(reservation(), {"messages": [{"role": "user", "content": "test"}]})
    assert count == 1
    assert not error.value.not_sent and not error.value.finished


async def test_caller_cannot_smuggle_multiple_completions_or_model_routing():
    async def handler(request):
        pytest.fail("invalid payload must never reach the engine")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fake/v1/") as client:
        driver = OpenAICompletionDriver("http://fake/v1", "unused", "tool", client=client)
        with pytest.raises(DriverFailure) as error:
            await driver.execute(reservation(), {"messages": [], "n": 2, "model": "unapproved"})
    assert error.value.not_sent


async def test_success_preserves_usage_and_partial_content_finish_reason():
    async def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 20}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fake/v1/") as client:
        driver = OpenAICompletionDriver("http://fake/v1", "unused", "tool", client=client)
        result = await driver.execute(reservation(), {"messages": [{"role": "user", "content": "test"}]})
    assert (result.input_tokens, result.output_tokens) == (7, 20)
    assert result.body["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("parallel", [True, False])
async def test_tool_request_reaches_http_once_with_its_transcript_and_parallel_policy(parallel):
    calls = []
    payload = {
        "messages": [
            {"role": "user", "content": "Review the selected evidence"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "search-1", "type": "function",
                "function": {"name": "search", "arguments": '{"query":"evidence"}'},
            }]},
            {"role": "tool", "tool_call_id": "search-1", "content": "Recorded evidence"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "search", "parameters": {"type": "object"},
        }}],
        "tool_choice": "auto",
        "parallel_tool_calls": parallel,
    }
    reply = {"choices": [{"message": {"role": "assistant", "content": "Reviewed"},
                           "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 3}}

    async def handler(request):
        calls.append(json.loads(request.content))
        assert request.headers["X-Intramind-Attempt-ID"] == "attempt"
        return httpx.Response(200, json=reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fake/v1/") as client:
        driver = OpenAICompletionDriver("http://fake/v1", "unused", "tool", client=client)
        result = await driver.execute(reservation(), payload)
    assert calls == [payload | {"model": "tool", "max_tokens": 20, "n": 1, "stream": False}]
    assert result.body == reply


@pytest.mark.parametrize("parallel", ["true", 1, None])
async def test_non_boolean_parallel_policy_is_rejected_before_http(parallel):
    async def handler(request):
        pytest.fail("invalid policy must not reach the backend")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fake/v1/") as client:
        driver = OpenAICompletionDriver("http://fake/v1", "unused", "tool", client=client)
        with pytest.raises(DriverFailure) as error:
            await driver.execute(reservation(), {
                "messages": [{"role": "user", "content": "test"}], "parallel_tool_calls": parallel,
            })
    assert error.value.not_sent
