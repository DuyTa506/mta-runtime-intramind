"""Native publication survives lost ACKs, child/root rollovers and Temporal replay."""

import copy
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from be_ingestion_publication_workflows import checkpoint_ingestion_publication
from feature_harness import feature_environment
from temporalio.worker import Replayer

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


async def test_generation_publication_replays_without_rewriting_acknowledged_batches(store, tmp_path):
    if os.environ.get("RUNTIME_TEST_BE_FEATURES") != "yes":
        pytest.skip("explicit BE dependency environment required")
    names = ("INGESTION_TEST_MONGO_URI", "INGESTION_TEST_ELASTIC_URL", "INGESTION_TEST_QDRANT_URL")
    if not all(os.environ.get(name) for name in names):
        pytest.skip("explicit test stores required; only owned UUID targets will be mutated")
    from api.background.ingestion.indexing import GenerationIndexes
    from api.background.ingestion.parsing import write_result
    from api.background.ingestion.publication_activities import PublicationActivities
    from api.background.ingestion.publication_workflows import WORKFLOWS
    from api.background.ingestion.snapshot import persist_result
    from api.services.document_storage.publication import (
        PublicationIntent,
        PublicationStore,
        source_fingerprint,
    )
    from intramind_shared.models import Document
    from intramind_shared.storages.docstores.elasticsearch import ElasticStore
    from intramind_shared.storages.vectorstores.qdrant import QdrantStore
    from pymongo import MongoClient

    prefix = "ingestion_publication_test_" + uuid4().hex
    mongo = MongoClient(os.environ[names[0]], tz_aware=True, serverSelectionTimeoutMS=2000)
    assert mongo.admin.command("hello").get("setName")
    db = mongo[prefix]
    elastic = ElasticStore(prefix + "_docs", prefix + "_chunks", os.environ[names[1]], analyzer="standard")
    vectors = QdrantStore(os.environ[names[2]], prefix + "_vectors")
    calls, receipts, roots = [], [], []
    try:
        vectors.create_collection(3)
        indexes = GenerationIndexes(elastic.client, vectors.client, document_index=elastic.doc_index,
            chunk_index=elastic.chunk_index, collection=vectors.collection_name, dimension=3, storage_id=prefix)
        indexes.prepare()
        elastic.client.indices.put_settings(index=elastic.chunk_index, settings={"refresh_interval": "50ms"})
        authority = PublicationStore(db)
        db.documents.create_index("id", unique=True)

        class NoPages:
            # This case has zero page assets. Deploy tests exercise real page bytes in MinIO.
            def put(self, *args):
                raise AssertionError("No page asset was planned")

        def activities(factory):
            @durable_activity(name="test.ingestion-publication-source/v1")
            async def source(inputs):
                roots.append(inputs["root_id"])
                client = factory(inputs["tenant_id"])
                try:
                    document = Document("source", "accepted.txt", [], doc_content="Approved count is 42.",
                        metadata={"user_id": "test", "conversation_id": "scope"},
                        created_at=datetime(2026, 9, 20))
                    body = document.to_dict()
                    db.documents.insert_one(copy.deepcopy(body))
                    directory = tmp_path / "parsed"
                    write_result(document, directory)
                    blob = await client.put_bytes(b"accepted source", content_type="text/plain")
                    config = await client.put_json({"fixture": "sealed-parser-policy"})
                    parsed = await persist_result(client, directory, {"source": blob,
                        "document": body, "configuration": config, "skip_dedup": True})
                    batches = []
                    for ordinal in range(65):
                        batches.append(await client.put_json({"chunks": [{"id": f"chunk-{ordinal}",
                            "document_id": document.id, "document_name": document.name,
                            "chunk_index": ordinal, "content": f"Evidence {ordinal}: count 42.",
                            "metadata": {"page_number": 1}}], "embeddings": [[1., 2., 3.]]}))
                    chunked = await client.put_json({"ingestion_chunk_checkpoint": 1, "parsed": parsed,
                        "batches": batches, "chunks": 65, "document_metadata": document.metadata,
                        "summary_embedding": None, "embedding_profile": {"dimension": 3}})
                    intent = PublicationIntent(document_id=document.id, tenant_id=inputs["tenant_id"],
                        submission_key=inputs["root_id"], source_fingerprint=source_fingerprint(body),
                        input=parsed, configuration=config, deadline=datetime.now(UTC) + timedelta(minutes=5))
                    authority.reserve(intent)
                    return await client.put_json({"intent": intent.model_dump(mode="json"), "chunked": chunked})
                finally:
                    await client.close()

            native = PublicationActivities(factory, authority, indexes, NoPages())

            @durable_activity(name="ingestion.publication-batch/v1")
            async def batch(inputs):
                await native.batch(inputs)
                calls.append(inputs["ordinal"])
                if len(calls) == 1:
                    raise OSError("Batch activity ACK lost after both indexes and Mongo committed")

            @durable_activity(name="ingestion.publication-finish/v1")
            async def finish(inputs):
                receipt = await native.finish(inputs)
                receipts.append(receipt)
                if len(receipts) == 1:
                    raise OSError("Publication receipt acknowledgement lost")
                return receipt

            return [source, native.prepare, batch, native.assets, finish]

        async def forbidden(*args):
            raise AssertionError("Publishing committed vectors must not run inference")

        workflows = [checkpoint_ingestion_publication, *WORKFLOWS]
        async with feature_environment(store, name="be-ingestion-publication-checkpoint",
            workflows=workflows, build_activities=activities, respond=forbidden) as env:
            status, result = await env.submit({})
            assert len(roots) == 1 and len(calls) == 66 and calls[:2] == [0, 0]
            assert len(receipts) == 2 and receipts[0] == receipts[1]
            assert status["spent"] == status["reserved"] == 0 and not env.calls
            assert status["operations"] == {}
            visible = db.documents.find_one({"id": "source"})
            assert visible["ingestion"]["published"] == result
            rows = indexes.read_chunks("source", result["generation"], batch_size=7)
            assert len(rows) == vectors.client.count(vectors.collection_name).count == 65
            assert [row["chunk_index"] for row in rows] == list(range(65))
            histories = []
            replayer = Replayer(workflows=workflows)

            async def replay_chain(workflow_id, run_id=None):
                history = await env.temporal.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
                await replayer.replay_workflow(history)
                histories.append(history)
                started = history.events[0].workflow_execution_started_event_attributes
                if started.continued_execution_run_id:
                    await replay_chain(workflow_id, started.continued_execution_run_id)
                for event in history.events:
                    if event.HasField("child_workflow_execution_started_event_attributes"):
                        await replay_chain(event.child_workflow_execution_started_event_attributes.workflow_execution.workflow_id)

            await replay_chain(status["root_id"])
            assert len(histories) == 4 and len(calls) == 66 and len(receipts) == 2
    finally:
        mongo.drop_database(prefix)
        mongo.close()
        vectors.client.delete_collection(vectors.collection_name)
        vectors.client.close()
        for index in (elastic.doc_index, elastic.chunk_index):
            elastic.client.indices.delete(index=index)
        elastic.client.close()
