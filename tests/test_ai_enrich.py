"""Durable extraction/fill joins preserve prose and do not repeat inference on replay."""

import os

import pytest
from feature_harness import feature_environment

pytestmark = pytest.mark.integration


async def test_enrichment_preserves_prose_and_replays_completed_model_leaves(store):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment and opt-in required")
    from agents.context_synthesis.durable.leaves import LEAVES
    from api.background.common.models import ModelActivities
    from api.background.enrich.activities import EnrichActivities
    from api.background.enrich.workflows import WORKFLOWS
    from tests.test_context_synthesis.test_enrich_suggest import DRAFT, EnrichFakeLLM
    from tests.test_context_synthesis.test_linch_synthesis import FakeDocStore, FakeEmbedder

    model, reads = EnrichFakeLLM(), []

    def documents():
        reads.append(True)
        return FakeDocStore()

    async def respond(reservation, payload):
        text = await model.agenerate(
            user_prompt=payload["messages"][-1]["content"],
            system_prompt=payload["messages"][0]["content"],
        )
        return {"choices": [{"message": {"content": text}}]}

    def activities(factory):
        enrich = EnrichActivities(factory, model_profile="test", store_factory=documents,
                                  embedder_factory=FakeEmbedder)
        return [*enrich.registered(), ModelActivities(factory, leaves=LEAVES).plan]

    async with feature_environment(
        store, name="enrich", workflows=WORKFLOWS, build_activities=activities, respond=respond,
    ) as env:
        status, result = await env.submit({
            "draft_markdown": DRAFT, "document_ids": ["d1"], "options": {"dedup_threshold": 0.99},
        })
        assert status["state"] == "SUCCEEDED"
        assert len(result["insertions"]) == 2
        restored = result["enriched_markdown"]
        for item in sorted(result["insertions"], key=lambda item: item["gap_index"], reverse=True):
            restored = restored.replace(item["text"], item["placeholder"], 1)
        assert restored == DRAFT
        assert len(env.calls) == result["telemetry"]["llm_calls"]
        assert reads == [True]
        await env.replay(status["root_id"])
        assert len(env.calls) == result["telemetry"]["llm_calls"]
