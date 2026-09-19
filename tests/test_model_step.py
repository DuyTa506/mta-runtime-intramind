"""Pure model-step replay never runs inference or silently rebinds an operation."""

import asyncio
import json

import pytest

from intramind_runtime.model_step import ModelRecord, StepReplayError, plan_model_step


def payload(content="question"):
    return {"messages": [{"role": "user", "content": content}], "temperature": 0.3}


@pytest.mark.asyncio
async def test_plan_resume_and_schema_repair_keep_distinct_calls():
    async def body(model):
        raw = await model.invoke(payload(), max_output_tokens=100)
        if raw["answer"] != "valid":
            raw = await model.invoke(payload("repair"), max_output_tokens=200)
        return {"validated": raw["answer"]}

    records = []
    first = await plan_model_step(body, records)
    assert not first.done and first.call.payload == payload()
    records.append(ModelRecord(first.call.request_digest, result={"answer": "invalid"}))
    records = [ModelRecord(**json.loads(json.dumps(vars(record)))) for record in records]
    second = await plan_model_step(body, records)
    assert second.call.payload == payload("repair")
    assert second.call.request_digest != first.call.request_digest
    records.append(ModelRecord(second.call.request_digest, result={"answer": "valid"}))
    final = await plan_model_step(body, records)
    assert final.done and final.result == {"validated": "valid"}


@pytest.mark.asyncio
async def test_same_call_with_changed_input_fails_instead_of_reusing_response():
    async def original(model):
        return await model.invoke(payload(), max_output_tokens=100)

    first = await plan_model_step(original, [])

    async def changed(model):
        return await model.invoke(payload("changed"), max_output_tokens=100)

    with pytest.raises(StepReplayError, match="changed"):
        await plan_model_step(changed, [ModelRecord(first.call.request_digest, result={})])


@pytest.mark.asyncio
async def test_unused_record_detects_changed_control_flow():
    async def body(model):
        return "short circuit"

    with pytest.raises(StepReplayError, match="unused"):
        await plan_model_step(body, [ModelRecord("a" * 64, result={})])


@pytest.mark.asyncio
async def test_recorded_terminal_model_failure_preserves_feature_fallback():
    async def body(model):
        try:
            await model.invoke(payload(), max_output_tokens=100)
        except RuntimeError:
            return {"fallback": True}

    first = await plan_model_step(body, [])
    final = await plan_model_step(body, [ModelRecord(first.call.request_digest, error="failed")])
    assert final.done and final.result == {"fallback": True}


@pytest.mark.asyncio
async def test_parallel_leaf_calls_are_rejected_use_durable_map_instead():
    async def body(model):
        return await asyncio.gather(
            model.invoke(payload("a"), max_output_tokens=100),
            model.invoke(payload("b"), max_output_tokens=100),
        )

    with pytest.raises(StepReplayError, match="sequential"):
        await plan_model_step(body, [])


@pytest.mark.parametrize(
    "record",
    [
        {"request_digest": "a" * 64},
        {"request_digest": "a" * 64, "result": {}, "error": "bad"},
        {"request_digest": "bad", "result": {}},
    ],
)
def test_invalid_result_record_rejected(record):
    with pytest.raises(ValueError):
        ModelRecord(**record)
