"""Native source reading on real Temporal/Postgres with bounded children and rollover."""

import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from math import ceil

import pytest
from feature_harness import feature_environment, peak_children
from pptx_workflows import checkpoint_context
from temporalio.worker import Replayer

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("document_count", [1, 3])
async def test_pptx_context_retains_sources_policy_and_budget_across_rollover(store, monkeypatch, document_count):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    import llmai
    from api.background.common.models import ModelActivities
    from api.background.pptx.activities import PptxActivities
    from api.background.pptx.leaves import LEAVES
    from api.background.pptx.policy import PptxPolicy
    from api.background.pptx.workflows import WORKFLOWS
    from api.config import PptxLLMConfig, PptxSettings
    from tools.pptx.documents import LoadedCorpus

    def forbidden(*args, **kwargs):
        raise AssertionError("A durable reading leaf cannot create a native inference client")

    monkeypatch.setattr(llmai, "get_client", forbidden)
    config = PptxSettings(llm=PptxLLMConfig(base_url="http://fake/v1", model="test", context_tokens=16000,
                                          timeout_seconds=37, concurrency=2))
    policy = PptxPolicy.capture(config, "test", {
        "model_profile": "test", "capacity_profile_id": "test-v1", "model": "test",
        "context_limit": 16384, "response_formats": ["text", "json_schema"],
        "endpoint_fingerprint": sha256(b"http://fake").hexdigest(),
    }, environment={"DISABLE_THINKING": "true", "CONTENT_PIPELINE_CHUNK_TOKENS": "500"}).model_copy(update={"history_window": 1})
    sources = [("Primary", "# Primary\n## Evidence\nSource evidence is 42 approved units."),
               ("Reference", "# Reference\n## Evidence\nSupporting evidence is 42 approved units."),
               ("Excluded", "# Excluded\n## Evidence\nOFF-TOPIC-MARKER has no related evidence.")][:document_count]
    if document_count == 1:
        sources[0] = ("Primary", "# Primary\n## Evidence\n" + "Source evidence is 42 approved units. " * 300)
    loaded, calls, publications = [], [], []

    def load(ids, *, max_chars):
        loaded.append(ids)
        assert ids == [str(i) for i in range(document_count)]
        return LoadedCorpus(sources, [], {str(i): value[0] for i, value in enumerate(sources)})

    async def respond(reservation, payload):
        calls.append(payload)
        assert reservation.operation.attempt_timeout_seconds == 37
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        # Mutable worker defaults change after the first dispatch. Every later
        # leaf must still use the policy accepted with this root.
        monkeypatch.setenv("CUSTOM_MODEL", "new-worker-model")
        monkeypatch.setenv("CONTENT_PIPELINE_LLM_TIMEOUT_SECONDS", "1200")
        monkeypatch.setenv("CONTENT_PIPELINE_SUMMARY_CONCURRENCY", "8")
        if payload.get("response_format"):
            assert payload["response_format"]["json_schema"]["name"] == "corpus_selection"
            answer = json.dumps({"corpus_subject": "Approved source evidence", "primary_ref": "D2",
                "subjects": [{"name": "Approved source evidence", "refs": ["D1", "D2"]},
                             {"name": "Unrelated material", "refs": ["D3"]}],
                "decisions": [{"ref": "D1", "role": "reference", "rationale": "Related source D1"},
                              {"ref": "D2", "role": "primary", "rationale": "Main source D2"},
                              {"ref": "D3", "role": "excluded", "rationale": "Unrelated source D3"}]})
        else:
            answer = "# Source\n## Evidence\nSource evidence is 42 approved units."
        return {"choices": [{"message": {"content": answer}}]}

    def activities(factory):
        @durable_activity(name="test.pptx.context-publish/v1")
        async def publish(inputs):
            publications.append(inputs["input"])
            if len(publications) == 1:
                raise OSError("Publication acknowledgement lost after reading completed")
            return inputs["input"]

        return [*PptxActivities(factory, model_profile="test", source_loader=load).registered(),
                ModelActivities(factory, leaves=LEAVES).plan, publish]

    workflows = [checkpoint_context, *WORKFLOWS]
    async with feature_environment(store, name="pptx-context-checkpoint", workflows=workflows,
                                   build_activities=activities, respond=respond) as env:
        status, result = await env.submit({"document_ids": [str(i) for i in range(document_count)],
            "primary_document_id": "0", "language": "en", "instructions": "User intent: use the designated source",
            "n_slides": 6, "avoid_layout_repetition": True}, configuration=policy.model_dump(mode="json"))
        assert len(loaded) == 1 and len(publications) == 2 and publications[0] == publications[1]
        client = env.runtime_client("user:test")
        try:
            documents = [await client.read_json(ref) for ref in result["documents"]]
            states = [await client.read_json(doc["state"]) for doc in documents]
            expected_calls = sum(sum(state["counts"].values()) for state in states) + int(document_count > 1)
            expected_rollovers = sum(sum(ceil(count / 2) for count in state["counts"].values()) for state in states)
            assert len(calls) == len(env.calls) == result["model_calls"] == expected_calls
            assert status["operations"] == {"SUCCEEDED": expected_calls}
            assert status["reserved"] == 0 and status["spent"] == expected_calls * 30
            if document_count == 1:
                assert states[0]["counts"]["chunks"] > 2  # Rollover before the phase ends.
            prepared = await client.read_json(result["prepared"])
            assert "42" in prepared["document_map"]["distilled_markdown"]
            assert "OFF-TOPIC-MARKER" not in json.dumps(prepared, ensure_ascii=False)
            if document_count > 1:
                selection = (await client.read_json(result["selection"]))["value"]
                assert selection["trace"]["vote"]["applied"] is False
                primary = next(doc for doc in selection["distillation"]["plan"]["documents"] if doc["role"] == "primary")
                assert primary["doc_id"] == "Primary" and primary["role_source"] == "user"
        finally:
            await client.close()

        continuations, document_peaks = [], []

        async def replay_chain(workflow_id, run_id=None):
            history = await env.temporal.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
            await Replayer(workflows=workflows).replay_workflow(history)
            start = history.events[0].workflow_execution_started_event_attributes
            envelope = (await env.temporal.data_converter.decode(start.input.payloads))[0]
            assert envelope["root_id"] == status["root_id"]
            assert len(json.dumps(envelope)) < 64 * 1024
            if start.workflow_type.name == "pptx.context/v1":
                assert peak_children(history) == min(2, document_count)
            if start.workflow_type.name == "pptx.document/v1":
                document_peaks.append(peak_children(history))
            for event in history.events:
                if event.HasField("child_workflow_execution_started_event_attributes"):
                    child = event.child_workflow_execution_started_event_attributes.workflow_execution
                    await replay_chain(child.workflow_id, child.run_id)
                if event.HasField("workflow_execution_continued_as_new_event_attributes"):
                    continuation = event.workflow_execution_continued_as_new_event_attributes
                    checkpoint = (await env.temporal.data_converter.decode(continuation.input.payloads))[0]
                    assert checkpoint["configuration"] == envelope["configuration"]
                    assert checkpoint["deadline"] == envelope["deadline"]
                    if "started_at" in envelope:
                        assert checkpoint["started_at"] == envelope["started_at"]
                    else:
                        # A child's initial start is supplied by Temporal. Its
                        # first rollover freezes that value for later histories.
                        started = datetime.fromisoformat(checkpoint["started_at"])
                        assert history.events[0].event_time.ToDatetime(tzinfo=UTC) <= started
                        assert started <= history.events[-1].event_time.ToDatetime(tzinfo=UTC)
                    continuations.append(checkpoint)
                    await replay_chain(workflow_id, continuation.new_execution_run_id)

        await replay_chain(status["root_id"])
        assert len(continuations) == expected_rollovers
        assert document_peaks and max(document_peaks) <= 2
        if document_count == 1:
            assert max(document_peaks) == 2
        assert len(calls) == expected_calls  # Replay never repeats inference.
