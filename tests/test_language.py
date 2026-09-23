import json

import pytest

from intramind_runtime.language import (
    LanguageGuardedPort,
    LanguagePolicy,
    LanguageValidationError,
    apply_repair,
    inspect_language,
    repair_payload,
)
from intramind_runtime.model_step import ModelRecord, plan_model_step


@pytest.mark.parametrize("text", ["Nội dung rõ ràng", "GPU NVIDIA", "Số 123 [Nguồn 1]"])
def test_clean_output(text):
    assert not inspect_language(text, LanguagePolicy("vietnamese"))


@pytest.mark.parametrize("text", ["Có 问题 trong câu", "中文内容", "ký tự 𠀀", "ký tự 𰀀"])
def test_han_across_unicode_planes(text):
    assert inspect_language(text, LanguagePolicy("vi"))


def test_source_literals_and_json_identity_are_not_translated():
    value = {"id": "中文", "text": "北京 có vấn đề 问题, 12 [Nguồn 1]"}
    policy = LanguagePolicy("vi", protected_paths=("id",), protected_terms=("北京",))
    issues = inspect_language(value, policy)
    assert [i.path for i in issues] == [("text",)]
    result = apply_repair(
        value, issues, json.dumps({"texts": ["北京 có vấn đề, 12 [Nguồn 1]"]}), policy
    )
    assert result["id"] == value["id"]
    assert value["text"].endswith("问题, 12 [Nguồn 1]")


@pytest.mark.parametrize("target", ["zh", "zh-CN", "ja", "ko"])
def test_requested_han_language(target):
    assert not inspect_language("中文", LanguagePolicy(target))


@pytest.mark.parametrize(
    "raw",
    [
        '{"texts": ["sửa 13"]}',
        '{"texts": ["问题 12"]}',
        '{"texts": []}',
        '{"texts": [""]}',
        '{"texts": ["sửa 12"], "id": "changed"}',
        "not json",
    ],
)
def test_invalid_repair_cannot_escape_validation(raw):
    policy = LanguagePolicy("vi")
    with pytest.raises(LanguageValidationError):
        apply_repair("问题 12", inspect_language("问题 12", policy), raw, policy)


def body(text, **extra):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop", **extra}]}


@pytest.mark.asyncio
async def test_repair_is_checkpointed_and_replayed_without_network():
    async def leaf(port):
        guard = LanguageGuardedPort(port, LanguagePolicy("vi"))
        return await guard.invoke(
            {"messages": [{"role": "user", "content": "source"}]}, max_output_tokens=100
        )

    first = await plan_model_step(leaf, [])
    records = [ModelRecord(first.call.request_digest, result=body("Nội dung 问题 12"))]
    repair = await plan_model_step(leaf, records)
    assert repair.call.payload["temperature"] == 0
    records.append(
        ModelRecord(repair.call.request_digest, result=body('{"texts":["Nội dung hợp lệ 12"]}'))
    )
    done = await plan_model_step(leaf, records)
    assert done.done
    assert done.result == body("Nội dung hợp lệ 12")
    assert (await plan_model_step(leaf, records)).result == done.result


@pytest.mark.asyncio
async def test_clean_output_never_requests_repair():
    async def leaf(port):
        return await LanguageGuardedPort(port, LanguagePolicy("vi")).invoke(
            {"messages": []}, max_output_tokens=50
        )

    first = await plan_model_step(leaf, [])
    assert (
        await plan_model_step(leaf, [ModelRecord(first.call.request_digest, result=body("Đúng"))])
    ).done


def test_policy_is_portable_and_does_not_accept_auto():
    policy = LanguagePolicy("english", text_paths=("items/*/text",))
    assert LanguagePolicy(**json.loads(json.dumps(policy.snapshot()))) == policy
    with pytest.raises(ValueError):
        LanguagePolicy("auto")
    assert repair_payload((), policy)["response_format"] == {"type": "json_object"}
