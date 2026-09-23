import json

import pytest

from intramind_runtime.language import (
    LanguageGuardedPort,
    LanguagePolicy,
    LanguageValidationError,
    apply_repair,
    completion_value,
    inspect_language,
    repair_payload,
    source_literals,
)
from intramind_runtime.model_step import ModelRecord, plan_model_step


@pytest.mark.parametrize("text", ["Nội dung rõ ràng", "GPU NVIDIA", "Số 123 [Nguồn 1]"])
def test_clean_output(text):
    assert not inspect_language(text, LanguagePolicy("vietnamese"))


VI = "Tài liệu này giải thích cách hệ thống xử lý dữ liệu và tạo ra kết quả cho người sử dụng."
EN = "The system processes the uploaded documents and produces a detailed summary of the available information."
FR = "Ce document décrit les principales étapes du traitement des données et présente les résultats obtenus."


@pytest.mark.parametrize(
    "target,text",
    [
        ("vi", EN),
        ("en", VI),
        ("vi", FR),
        ("fr", EN),
        ("zh", EN),
        ("vi", "Настройка сервера завершена"),
        ("en", "การประมวลผลข้อมูล"),
    ],
)
def test_target_language_mismatch_is_not_limited_to_han(target, text):
    assert inspect_language(text, LanguagePolicy(target))


@pytest.mark.parametrize(
    "target,text",
    [
        ("vi", VI),
        ("en", EN),
        ("fr", FR),
        ("vi", "Hệ thống dùng PostgreSQL và GPU NVIDIA để xử lý dữ liệu của người dùng."),
        ("vi", "Docker Kubernetes PostgreSQL Temporal Redis MinIO FastAPI NVIDIA"),
        ("vi", "Công thức $α + β + γ$ và mã `print('hello world')` được giữ nguyên."),
    ],
)
def test_target_prose_and_technical_literals_remain_untouched(target, text):
    assert not inspect_language(text, LanguagePolicy(target))


def test_mixed_sentence_is_found_and_declared_source_quote_is_preserved():
    assert inspect_language(VI + " " + EN, LanguagePolicy("vi"))
    assert not inspect_language(VI + " " + EN, LanguagePolicy("vi", protected_terms=(EN,)))


def test_saved_v1_policy_preserves_its_original_inspection_semantics():
    assert not inspect_language(EN, LanguagePolicy("vi", version=1))
    assert inspect_language("问题", LanguagePolicy("vi", version=1))


def test_confident_detector_is_deterministic_and_short_terms_are_uncertain():
    from intramind_runtime.language_detection import detect_language

    assert {detect_language(VI) for _ in range(5)} == {"vi"}
    assert detect_language(EN) == "en"
    assert detect_language("GPU RAG") is None


def test_non_han_repair_must_reach_the_requested_target():
    policy = LanguagePolicy("vi")
    issues = inspect_language(EN, policy)
    assert apply_repair(EN, issues, json.dumps({"texts": [VI]}), policy) == VI
    with pytest.raises(LanguageValidationError):
        apply_repair(EN, issues, json.dumps({"texts": [FR]}), policy)


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
    response_format = repair_payload((), policy)["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["texts"] and schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"texts"}


def test_repair_schema_cannot_echo_input_metadata():
    policy = LanguagePolicy("vi", protected_terms=("北京",))
    issues = inspect_language({"text": "北京 có 问题 12"}, policy)
    request = repair_payload(issues, policy)
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["texts"]["minItems"] == 1
    assert schema["properties"]["texts"]["maxItems"] == 1
    with pytest.raises(LanguageValidationError):
        apply_repair(
            {"text": "北京 có 问题 12"},
            issues,
            '{"texts":["北京 có vấn đề 12"],"preserve_verbatim":["北京"]}',
            policy,
        )


def test_only_explicit_source_literals_are_exempt():
    terms = source_literals("Văn bản 中文, trích “北京”, mã `变量`")
    assert terms == ("北京", "变量")
    policy = LanguagePolicy("vi", protected_terms=terms)
    assert not inspect_language("Trích 北京, mã 变量", policy)
    assert inspect_language("Tự sinh 中文", policy)


def test_completion_does_not_repair_truncated_json_or_tool_calls():
    assert (
        completion_value(body('{"text":"问题"}', finish_reason="length"), structured=True)[1]
        is False
    )
    assert completion_value(body(None, finish_reason="tool_calls"))[1] is False
    assert completion_value(body('```json\n{"text":"问题"}\n```')) == ({"text": "问题"}, True)


@pytest.mark.asyncio
async def test_failed_repair_prevents_hidden_leaf_retry():
    from unittest.mock import AsyncMock

    port = AsyncMock()
    port.invoke.side_effect = [body("问题 12"), body('{"texts":["sửa 13"]}')]
    guard = LanguageGuardedPort(port, LanguagePolicy("vi"))
    for _ in range(3):
        with pytest.raises(LanguageValidationError):
            await guard.invoke({"messages": []}, max_output_tokens=100)
    assert port.invoke.await_count == 2
