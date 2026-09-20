"""The actual BE root shares accounting and accepted flags across native phases."""

import copy
import json
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from feature_harness import feature_environment
from temporalio.worker import Replayer

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("alias", [False, True])
async def test_ingestion_root_replays_native_phases_and_committed_dedup(store, tmp_path, alias):
    if os.environ.get("RUNTIME_TEST_BE_FEATURES") != "yes":
        pytest.skip("explicit BE dependency environment required")
    names = ("INGESTION_TEST_MONGO_URI", "INGESTION_TEST_ELASTIC_URL", "INGESTION_TEST_QDRANT_URL")
    if not all(os.environ.get(name) for name in names):
        pytest.skip("explicit disposable Mongo replica set, Elasticsearch and Qdrant required")
    from types import SimpleNamespace

    from api.background.ingestion.activities import ParseActivities
    from api.background.ingestion.chunk_activities import ChunkActivities
    from api.background.ingestion.chunk_policy import ChunkOptions, ChunkPolicy, dependency_versions
    from api.background.ingestion.chunk_workflows import WORKFLOWS as CHUNKS
    from api.background.ingestion.indexing import GenerationIndexes
    from api.background.ingestion.parsing import run_parser
    from api.background.ingestion.policy import ParsePolicy
    from api.background.ingestion.publication_activities import PublicationActivities
    from api.background.ingestion.publication_workflows import WORKFLOWS as PUBLICATION
    from api.background.ingestion.root_activities import IngestionActivities
    from api.background.ingestion.root_policy import IngestionPolicy
    from api.background.ingestion.root_workflows import WORKFLOWS as ROOTS
    from api.background.ingestion.snapshot import load_document
    from api.background.ingestion.workflows import WORKFLOWS as PARSE
    from api.services.document_storage.dedup import compute_text_hash
    from api.services.document_storage.publication import (
        PublicationIntent,
        PublicationStore,
        source_fingerprint,
    )
    from intramind_shared.storages.blobstores.base import BlobStat
    from intramind_shared.storages.docstores.elasticsearch import ElasticStore
    from intramind_shared.storages.vectorstores.qdrant import QdrantStore
    from pymongo import MongoClient

    from intramind_runtime.embedding import EmbeddingProfile

    prefix = "ingestion_root_test_" + uuid4().hex
    mongo = MongoClient(os.environ[names[0]], tz_aware=True, serverSelectionTimeoutMS=3000)
    db = mongo[prefix]
    elastic = ElasticStore(prefix + "_docs", prefix + "_chunks", os.environ[names[1]], analyzer="standard")
    vectors = QdrantStore(os.environ[names[2]], prefix + "_vectors")
    profile = EmbeddingProfile(model_profile="native-embedding", capacity_profile_id="native-embed-v1",
        model="native-test", model_revision="native-weights-v1", dimension=3,
        max_batch_size=3, character_limit=10000, max_text_characters=10000, max_response_bytes=1024 * 1024)
    policy = IngestionPolicy(
        parse=ParsePolicy.capture(SimpleNamespace(liteparse_ocr_enabled=False, liteparse_ocr_server_url=None)),
        chunk=ChunkPolicy(native_versions=dependency_versions(), embedding=profile,
            options=ChunkOptions(strategy="recursive_character", chunk_size=150, chunk_overlap=25,
                                 min_chunk_size=50, table_row_chunk_size=100),
            enable_document_level_search=True, rollover_every=2))
    calls, parses, decisions, records, publications, owners = [], [], [], [], [], []
    blobs = {}

    class Pages:
        def put(self, key, body, content_type, size):
            data = body.read()
            assert len(data) == size and data.startswith(b"\xff\xd8")
            blobs[key] = data
            return key

        def stat(self, key):
            return BlobStat(len(blobs[key]), "image/jpeg")

    async def backend(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"model": "native-test", "dimension": 3,
            "embeddings": [[float(len(text)), float(sum(map(ord, text)) % 31), 1.]
                           for text in payload["texts"]]}, headers={
            "X-Intramind-Attempt-ID": request.headers["X-Intramind-Attempt-ID"],
            "X-Intramind-Embedding-Contract": "termination-v1",
            "X-Intramind-Compute-State": "terminated", "X-Intramind-Model-Revision": profile.model_revision})

    async def forbidden(*args):
        raise AssertionError("Document ingestion cannot dispatch completion calls or an implicit KG build")

    async def parse(job, output, *, timeout_seconds):
        parses.append(job)
        await run_parser(job, output, timeout_seconds=timeout_seconds)

    try:
        vectors.create_collection(3)
        indexes = GenerationIndexes(elastic.client, vectors.client, document_index=elastic.doc_index,
            chunk_index=elastic.chunk_index, collection=vectors.collection_name, dimension=3, storage_id=prefix)
        indexes.prepare()
        authority = PublicationStore(db)
        db.documents.create_index("id", unique=True)

        def activities(factory):
            root, chunk = IngestionActivities(factory, authority), ChunkActivities(factory)
            publication = PublicationActivities(factory, authority, indexes, Pages())

            @durable_activity(name="ingestion.root-dedup/v1")
            async def dedup(inputs):
                owners.append(inputs["root_id"])
                if alias and not decisions:
                    client = factory(inputs["tenant_id"])
                    try:
                        document = await load_document(client, inputs["parsed"])
                        target = document.to_dict()
                        target.update(id="canonical", status="COMPLETED", pages=[])
                        target["metadata"]["text_hash"] = compute_text_hash(document.doc_content)
                        db.documents.insert_one(target)
                    finally:
                        await client.close()
                result = await root.dedup(inputs)
                decisions.append(result)
                if len(decisions) == 1:
                    raise OSError("Dedup acknowledgement lost after its decision committed")
                return result

            @durable_activity(name="ingestion.chunk-record/v1")
            async def record(inputs):
                result = await chunk.record(inputs)
                records.append(result)
                if len(records) == 1:
                    raise OSError("Embedding record acknowledgement lost after immutable write")
                return result

            @durable_activity(name="ingestion.publication-finish/v1")
            async def finish(inputs):
                owners.append(inputs["root_id"])
                result = await publication.finish(inputs)
                publications.append(result)
                if len(publications) == 1:
                    raise OSError("Publication acknowledgement lost after pointer commit")
                return result

            return [root.prepare, dedup, root.publication, *ParseActivities(factory, parser=parse).registered(),
                    chunk.prepare, chunk.plan, record, publication.prepare, publication.batch,
                    publication.assets, finish]

        workflows = [*ROOTS, *PARSE, *CHUNKS, *PUBLICATION]
        async with feature_environment(store, name="ingestion.document", workflows=workflows,
            build_activities=activities, respond=forbidden, embedding_profile=profile,
            respond_embedding=backend) as env:
            client = env.runtime_client("user:test")
            try:
                data = ("# Findings\n\n" + " ".join(
                    f"Finding {i} records the approved count of 42 and its evidence." for i in range(10))).encode()
                blob = await client.put_bytes(data, content_type="text/plain")
                document = {"id": "source", "name": "accepted.txt", "pages": [], "status": "PENDING",
                    "created_at": "2026-09-20T00:00:00", "metadata": {"user_id": "test",
                        "conversation_id": "same-conversation", "content_hash": blob["sha256"],
                        "file_size": blob["size"], "storage_key": "documents/accepted-original"}}
                db.documents.insert_one(copy.deepcopy(document))
                original = await client.put_json({"document": document, "source": blob, "skip_dedup": False})
                configuration = await client.put_json(policy.model_dump(mode="json"))
                intent = PublicationIntent(document_id="source", tenant_id="user:test", submission_key=prefix,
                    source_fingerprint=source_fingerprint(document), input=original, configuration=configuration,
                    deadline=datetime.now(UTC) + timedelta(minutes=5))
                authority.reserve(intent)
            finally:
                await client.close()
            status, result = await env.submit({"intent": intent.model_dump(mode="json")},
                                               configuration=policy.model_dump(mode="json"))
            current = db.documents.find_one({"id": "source"})
            assert current["status"] == "COMPLETED" and current["ingestion"]["published"] == result
            assert current["metadata"]["conversation_id"] == "same-conversation"
            assert len(parses) == 1 and len(decisions) == 2 and decisions[0] == decisions[1]
            assert set(owners) == {status["root_id"]}
            assert status["reserved"] == status["spent"] == 0 and not env.calls
            assert status["resource_budgets"]["embedding_characters"]["spent"] == sum(
                sum(map(len, call["texts"])) for call in calls)
            if alias:
                assert result["canonical_id"] == "canonical"
                assert not calls and not records and not publications and not blobs
                assert vectors.client.count(vectors.collection_name).count == 0
            else:
                assert len(records) == len(calls) + 1 and records[0] == records[1]
                assert len(publications) == 2 and publications[0] == publications[1]
                assert len(calls) > 2 and blobs and vectors.client.count(vectors.collection_name).count > 1
            histories, configurations, phases = [], [], set()
            replayer = Replayer(workflows=workflows)

            async def replay_chain(workflow_id, run_id=None):
                history = await env.temporal.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
                await replayer.replay_workflow(history)
                histories.append(history)
                started = history.events[0].workflow_execution_started_event_attributes
                envelope = (await env.temporal.data_converter.decode(started.input.payloads))[0]
                configurations.append(envelope["configuration"])
                phases.add(started.workflow_type.name)
                assert envelope["root_id"] == status["root_id"]
                if started.continued_execution_run_id:
                    await replay_chain(workflow_id, started.continued_execution_run_id)
                for event in history.events:
                    if event.HasField("child_workflow_execution_started_event_attributes"):
                        child = event.child_workflow_execution_started_event_attributes.workflow_execution
                        await replay_chain(child.workflow_id)

            await replay_chain(status["root_id"])
            assert phases == ({"ingestion.document/v1", "ingestion.parse/v1"} if alias else {
                "ingestion.document/v1", "ingestion.parse/v1", "ingestion.chunk/v1", "ingestion.publish/v1"})
            assert len(histories) >= (2 if alias else 5)
            assert all(value == configuration for value in configurations)
            assert len(parses) == 1 and len(decisions) == 2  # Replay made no native calls.
    finally:
        elastic.client.indices.delete(index=elastic.doc_index, ignore_unavailable=True)
        elastic.client.indices.delete(index=elastic.chunk_index, ignore_unavailable=True)
        vectors.client.delete_collection(vectors.collection_name)
        vectors.client.close()
        elastic.client.close()
        mongo.drop_database(prefix)
        mongo.close()
