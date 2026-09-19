"""Temporal/Postgres exam phases, real DOCX export and render retry isolation."""

import os
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from feature_harness import feature_environment, peak_children

pytestmark = pytest.mark.integration


async def test_exam_replays_quality_gates_and_retries_export_without_new_inference(
    store, monkeypatch
):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment and opt-in required")
    from agents.exam_generation.durable.leaves import LEAVES
    from api.background.common.models import ModelActivities
    from api.background.exam import activities as exam_activities
    from api.background.exam.activities import ExamActivities
    from api.background.exam.policy import ExamPolicy
    from api.background.exam.workflows import WORKFLOWS
    from api.config import settings
    from api.services.exam.documents import ExamCorpus
    from tests.test_exam_generation.test_engine import DOC, _FakeLLM

    fake, sources, exports, publications = _FakeLLM(), [], [], []
    render = exam_activities._render

    def source(document_ids, **kwargs):
        sources.append(document_ids)
        return ExamCorpus(documents=[("d1", "Quy chế quản trị", DOC)])

    def export(plan, template):
        exports.append(fake.calls)
        if len(exports) == 1:
            raise OSError("temporary export storage failure")
        return render(plan, template)

    def publish(**kwargs):
        publications.append(kwargs)
        return "exam-artifact"

    monkeypatch.setattr(exam_activities, "_render", export)

    async def respond(reservation, payload):
        assert reservation.operation.attempt_timeout_seconds == 29
        text = await fake.agenerate(
            user_prompt=payload["messages"][-1]["content"],
            system_prompt=payload["messages"][0]["content"],
            response_format=payload.get("response_format"),
        )
        return {"choices": [{"message": {"content": text}}]}

    def activities(factory):
        exam = ExamActivities(
            factory,
            model_profile="test",
            source_loader=source,
            publication=publish,
            embedder_factory=lambda: None,
            template_directory=lambda: str(
                Path(__file__).resolve().parents[2] / "mta-ai-intramind/templates_docx"
            ),
        )
        return [*exam.registered(), ModelActivities(factory, leaves=LEAVES).plan]

    async with feature_environment(
        store,
        name="exam",
        workflows=WORKFLOWS,
        build_activities=activities,
        respond=respond,
    ) as env:
        monkeypatch.setattr(settings.exam, "generate_concurrency", 1)
        monkeypatch.setattr(settings.exam, "verify_concurrency", 1)
        monkeypatch.setattr(settings.exam.llm, "timeout_seconds", 29)
        policy = ExamPolicy.capture(settings.exam, "test").model_dump(mode="json")
        monkeypatch.setattr(settings.exam, "enabled", False)
        monkeypatch.setattr(settings.exam, "verify_concurrency", 8)
        monkeypatch.setattr(settings.exam.llm, "timeout_seconds", 1)
        status, result = await env.submit(
            {
                "document_ids": ["d1"],
                "mode": "quality",
                "n_questions": 6,
                "preset": "mcq_only",
            },
            configuration=policy,
        )
        assert status["state"] in {"SUCCEEDED", "PARTIAL"}
        assert sources == [["d1"]]
        assert len(exports) == 2 and exports[0] == exports[1] == fake.calls
        assert len(publications) == 1
        assert len(env.calls) == fake.calls == result["llm_call_count"]
        history = await env.temporal.get_workflow_handle(status["root_id"]).fetch_history()
        assert peak_children(history) == 1
        assert fake.by_stage["evidence_verdict"] > 0
        assert fake.by_stage["blind_solve"] > 0
        assert result["artifact_id"] == "exam-artifact"
        assert result["question_count"] > 0
        assert result["question_count"] <= result["requested_questions"]
        with ZipFile(BytesIO(env.blobs.data[result["object_key"]])) as package:
            assert package.testzip() is None
            assert "ĐỀ KIỂM TRA" in package.read("word/document.xml").decode()
        await env.replay(status["root_id"])
        assert len(env.calls) == result["llm_call_count"]
