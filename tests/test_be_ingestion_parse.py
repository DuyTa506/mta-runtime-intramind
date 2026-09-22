"""Native BE parsing survives downstream failure and Temporal rollover/replay."""

import os
from types import SimpleNamespace

import pytest
from be_ingestion_workflows import checkpoint_ingestion
from feature_harness import feature_environment
from temporalio.worker import Replayer

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


async def test_ingestion_parse_checkpoint_survives_downstream_retry_rollover_and_replay(store, monkeypatch):
    if os.environ.get("RUNTIME_TEST_BE_FEATURES") != "yes":
        pytest.skip("explicit BE dependency environment required")
    from api.background.ingestion.activities import ParseActivities
    from api.background.ingestion.parsing import run_parser
    from api.background.ingestion.policy import ParsePolicy
    from api.background.ingestion.snapshot import load_document
    from api.background.ingestion.workflows import WORKFLOWS
    from api.config import settings

    policy = ParsePolicy.capture(SimpleNamespace(
        liteparse_ocr_enabled=False, liteparse_ocr_server_url=None,
        liteparse_render_all_page_images=False,
    ))
    reads, parses, downstream = [], [], []
    source_bytes = "# Báo cáo\n\nSố học viên đã duyệt: 42.\n".encode()

    async def parse(job, output, *, timeout_seconds):
        parses.append(job)
        # Later worker defaults must not change the accepted native parser options.
        monkeypatch.setattr(settings, "liteparse_ocr_enabled", True)
        monkeypatch.setattr(settings, "liteparse_render_all_page_images", True)
        monkeypatch.setattr(settings, "liteparse_ocr_server_url", "http://must-not-be-called")
        await run_parser(job, output, timeout_seconds=timeout_seconds)

    def activities(factory):
        @durable_activity(name="test.ingestion-source/v1")
        async def source(inputs):
            reads.append(inputs)
            client = factory(inputs["tenant_id"])
            try:
                blob = await client.put_bytes(source_bytes, content_type="text/plain")
                return await client.put_json({
                    "document": {"id": "accepted-doc", "name": "Báo cáo.txt", "pages": [],
                        "created_at": "2026-09-19T00:00:00", "metadata": {
                            "user_id": "test", "conversation_id": "accepted-conversation",
                            "content_hash": blob["sha256"], "file_size": blob["size"],
                            "storage_key": "documents/accepted-original"}},
                    "source": blob, "skip_dedup": True,
                })
            finally:
                await client.close()

        @durable_activity(name="test.ingestion-after-parse/v1")
        async def after_parse(inputs):
            downstream.append(inputs["parsed"])
            if len(downstream) == 1:
                raise OSError("Downstream indexing acknowledgement lost")
            return inputs["parsed"]

        return [source, after_parse, *ParseActivities(factory, parser=parse).registered()]

    async def forbidden(*args):
        raise AssertionError("Text parsing cannot dispatch a model call")

    workflows = [checkpoint_ingestion, *WORKFLOWS]
    async with feature_environment(store, name="be-ingestion-parse-checkpoint",
                                   workflows=workflows, build_activities=activities,
                                   respond=forbidden) as env:
        status, result = await env.submit({}, configuration=policy.model_dump(mode="json"))
        assert len(reads) == len(parses) == 1 and len(downstream) == 2
        assert downstream[0] == downstream[1] == status["result"]
        assert status["reserved"] == status["spent"] == 0 and not env.calls
        assert result["skip_dedup"] is True
        client = env.runtime_client("user:test")
        try:
            document = await load_document(client, status["result"], images=True)
            assert document.id == "accepted-doc" and document.name == "Báo cáo.txt"
            assert "42" in document.doc_content and len(document.pages) == 1
            assert document.metadata["conversation_id"] == "accepted-conversation"
            assert await client.read_bytes(result["source"]) == source_bytes
            assert not hasattr(document.pages[0], "_image_bytes")
        finally:
            await client.close()

        replayed, configurations = [], []
        replayer = Replayer(workflows=workflows)

        async def replay_chain(workflow_id, run_id=None):
            history = await env.temporal.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
            await replayer.replay_workflow(history)
            started = history.events[0].workflow_execution_started_event_attributes
            envelope = (await env.temporal.data_converter.decode(started.input.payloads))[0]
            configurations.append(envelope["configuration"])
            replayed.append(history)
            if started.continued_execution_run_id:
                await replay_chain(workflow_id, started.continued_execution_run_id)
            for event in history.events:
                if event.HasField("child_workflow_execution_started_event_attributes"):
                    child = event.child_workflow_execution_started_event_attributes.workflow_execution
                    await replay_chain(child.workflow_id, child.run_id)

        await replay_chain(status["root_id"])
        assert len(replayed) == 3 and all(c == configurations[0] for c in configurations)
        assert len(reads) == len(parses) == 1 and len(downstream) == 2
