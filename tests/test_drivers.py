import httpx
import pytest
from conftest import operation

from intramind_runtime.contracts import Reservation
from intramind_runtime.drivers import DriverFailure, OpenAICompletionDriver


def reservation():
    return Reservation(attempt_id="attempt", operation=operation(), pool_id="pool", engine_epoch="e1",
                       model_revision="model-1", owner_id="executor", lease_epoch=1)


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
