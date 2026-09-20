"""Native BE chunking preserves committed batches across lost ACKs and rollover."""

import json
import os
from datetime import datetime

import httpx
import pytest
from be_ingestion_chunk_workflows import checkpoint_ingestion_chunk
from feature_harness import feature_environment
from temporalio.worker import Replayer

from intramind_runtime.embedding import EmbeddingProfile
from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


async def test_native_chunks_resume_embedding_records_and_replay_every_history(store, tmp_path):
    if os.environ.get("RUNTIME_TEST_BE_FEATURES") != "yes":
        pytest.skip("explicit BE dependency environment required")
    from api.background.ingestion.chunk_activities import ChunkActivities
    from api.background.ingestion.chunk_policy import ChunkOptions, ChunkPolicy, dependency_versions
    from api.background.ingestion.chunk_workflows import WORKFLOWS
    from api.background.ingestion.parsing import write_result
    from api.background.ingestion.snapshot import persist_result
    from intramind_shared.models import Document, Page

    profile = EmbeddingProfile(model_profile="native-embedding", capacity_profile_id="native-embed-v1",
        model="native-test", model_revision="native-weights-v1", dimension=3,
        max_batch_size=3, character_limit=10000, max_text_characters=10000,
        max_response_bytes=1024 * 1024)
    policy = ChunkPolicy(native_versions=dependency_versions(), embedding=profile,
        options=ChunkOptions(strategy="hybrid_chunking", chunk_size=150, chunk_overlap=25,
                             min_chunk_size=50, table_row_chunk_size=100),
        rollover_every=2, enable_document_level_search=True)
    calls, sources, writes, published = [], [], [], []

    async def backend(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"model": "native-test", "dimension": 3,
            "embeddings": [[float(len(text)), float(sum(map(ord, text)) % 31), 1.]
                           for text in payload["texts"]]}, headers={
            "X-Intramind-Attempt-ID": request.headers["X-Intramind-Attempt-ID"],
            "X-Intramind-Embedding-Contract": "termination-v1",
            "X-Intramind-Compute-State": "terminated", "X-Intramind-Model-Revision": "native-weights-v1",
        })

    async def forbidden(*args):
        raise AssertionError("Native chunking cannot dispatch completion calls")

    def activities(factory):
        @durable_activity(name="test.ingestion-parsed-source/v1")
        async def source(inputs):
            sources.append(inputs)
            client = factory(inputs["tenant_id"])
            try:
                text = "# Findings\n\n" + " ".join(
                    f"Finding {i} records the approved count of 42 and its evidence." for i in range(8))
                document = Document("accepted-doc", "report.pdf", [Page(1, "placeholder", text)],
                    doc_content=text, summary="Approved count: 42.",
                    metadata={"user_id": "test", "conversation_id": "accepted-conversation"},
                    created_at=datetime(2026, 9, 19))
                output = tmp_path / "parsed"
                write_result(document, output)
                blob = await client.put_bytes(b"accepted-original", content_type="application/pdf")
                parsed = await persist_result(client, output, {
                    "source": blob, "document": document.to_dict(), "skip_dedup": True,
                    "configuration": await client.put_json({"parser": "fixture"}),
                })
                return await client.put_json({"parsed": parsed})
            finally:
                await client.close()

        native = ChunkActivities(factory)

        @durable_activity(name="ingestion.chunk-record/v1")
        async def record(inputs):
            result = await native.record(inputs)
            writes.append(result)
            if len(writes) == 1:
                raise OSError("Embedding record acknowledgement lost after immutable write")
            return result

        @durable_activity(name="test.ingestion-after-chunks/v1")
        async def after_chunks(inputs):
            published.append(inputs["chunked"])
            if len(published) == 1:
                raise OSError("Publication acknowledgement lost")
            return inputs["chunked"]

        return [source, after_chunks, native.prepare, native.plan, record]

    workflows = [checkpoint_ingestion_chunk, *WORKFLOWS]
    async with feature_environment(store, name="be-ingestion-chunk-checkpoint", workflows=workflows,
        build_activities=activities, respond=forbidden, embedding_profile=profile,
        respond_embedding=backend) as env:
        status, result = await env.submit({}, configuration=policy.model_dump(mode="json"))
        assert len(sources) == 1 and len(published) == 2 and published[0] == published[1]
        assert writes[0] == writes[1] and len(writes) == len(calls) + 1
        assert len(calls) > 2 and not env.calls
        assert status["spent"] == status["reserved"] == 0
        assert status["operations"] == {"SUCCEEDED": len(calls)}
        assert status["resource_budgets"]["embedding_characters"]["spent"] == sum(
            sum(map(len, call["texts"])) for call in calls)
        assert result["document_metadata"]["conversation_id"] == "accepted-conversation"
        assert result["summary_embedding"] and result["chunks"] > 1
        client = env.runtime_client("user:test")
        try:
            batches = [await client.read_json(ref) for ref in result["batches"]]
            chunks = [chunk for batch in batches for chunk in batch["chunks"]]
            assert len(chunks) == len({chunk["id"] for chunk in chunks}) == result["chunks"]
            assert all(chunk["metadata"]["page_number"] == 1 for chunk in chunks)
            assert (await client.read_json(result["parsed"]))["skip_dedup"] is True
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
                    await replay_chain(child.workflow_id)

        await replay_chain(status["root_id"])
        assert len(replayed) >= 4 and all(item == configurations[0] for item in configurations)
        assert len(writes) == len(calls) + 1 and len(published) == 2
