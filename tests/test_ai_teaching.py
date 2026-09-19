"""Real Temporal teaching, structured requests, two DOCX files, and publication recovery."""

import os
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from feature_harness import feature_environment

pytestmark = pytest.mark.integration


async def test_teaching_both_retries_publication_without_repeating_generation_or_render(store, monkeypatch):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment and opt-in required")
    from agents.teaching_docs.durable.leaves import LEAVES
    from api.background.common.models import ModelActivities
    from api.background.teaching.activities import TeachingActivities
    from api.background.teaching.workflows import WORKFLOWS
    from api.config import settings
    from api.services.teaching.documents import TeachingCorpus
    from tests.test_teaching_docs.test_engine import DOCS, FakeLLM

    fake, source_reads, publication_attempts, saved = FakeLLM(), [], [], {}
    monkeypatch.setattr(settings.teaching, "enabled", True)

    def sources(document_ids, **kwargs):
        source_reads.append(document_ids)
        return TeachingCorpus(documents=DOCS)

    def publication(**kwargs):
        key = kwargs["publication_key"]
        publication_attempts.append(key)
        previous = saved.setdefault(key, kwargs["envelope"])
        assert previous == kwargs["envelope"]
        if len(publication_attempts) == 1:
            raise RuntimeError("Mongo acknowledgement lost after committing the first file")
        return "artifact-" + key

    async def respond(reservation, payload):
        messages = payload["messages"]
        text = await fake.agenerate(
            user_prompt=messages[-1]["content"],
            system_prompt=messages[0]["content"],
            response_format=payload.get("response_format"),
            max_tokens=reservation.operation.max_output_tokens,
        )
        return {"choices": [{"message": {"content": text}}]}

    def activities(factory):
        teaching = TeachingActivities(
            factory,
            model_profile="test",
            source_loader=sources,
            publication=publication,
            template_directory=lambda: str(
                Path(__file__).resolve().parents[2] / "mta-ai-intramind/templates_docx"
            ),
        )
        return [*teaching.registered(), ModelActivities(factory, leaves=LEAVES).plan]

    async with feature_environment(
        store, name="teaching", workflows=WORKFLOWS, build_activities=activities, respond=respond
    ) as env:
        status, result = await env.submit({"document_ids": ["d1"], "kind": "both"})
        assert status["state"] == "SUCCEEDED"
        assert len(result["artifacts"]) == 2
        assert len({item["artifact_id"] for item in result["artifacts"]}) == 2
        assert result["artifacts"][1]["derived_from_modules"]
        assert len(publication_attempts) == 3
        assert source_reads == [["d1"]]
        assert len(env.calls) == len(fake.calls) == result["llm_call_count"]
        for item in result["artifacts"]:
            data = env.blobs.data[item["object_key"]]
            with ZipFile(BytesIO(data)) as document:
                assert "word/document.xml" in document.namelist()
                assert "Điều lệnh" in document.read("word/document.xml").decode()
        await env.replay(status["root_id"])
        assert len(env.calls) == result["llm_call_count"]
