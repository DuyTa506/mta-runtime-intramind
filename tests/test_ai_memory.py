"""Real Temporal checkpoints with a local Qdrant index and fake model/embedding I/O."""

import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

import pytest
from feature_harness import feature_environment

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("hard_delete", [False, True])
async def test_memory_corrects_then_retracts_with_lost_write_ack_and_history_rollover(store, hard_delete):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    from api.background.common.models import ModelActivities
    from api.background.memory.activities import MemoryActivities
    from api.background.memory.policy import MemoryPolicy
    from api.background.memory.workflows import WORKFLOWS
    from memory.long_term.durable import LEAVES
    from memory.long_term.mutations import ConsolidationStore
    from qdrant_client import QdrantClient, models
    from temporalio.worker import Replayer

    class Embedding:
        def embed(self, texts):
            return [[1.0, 0.5] for _ in texts]

        def embed_query(self, query):
            return [1.0, 0.5]

        def get_embedding_dimension(self):
            return 2

    qdrant = QdrantClient(":memory:", force_disable_check_same_thread=True)

    class VectorStore:
        client = qdrant

        def search_with_filter(self, vector, conditions, limit, collection):
            return [point.payload | {"score": point.score} for point in qdrant.query_points(
                collection, query=vector, query_filter=models.Filter(must=conditions), limit=limit).points]

    writes = []

    class MemoryStore(ConsolidationStore):
        async def apply(self, mutation):
            result = await super().apply(mutation)
            writes.append(mutation)
            if len(writes) == 1:
                raise ConnectionError("Qdrant applied the mutation, activity response lost")
            return result

    memory = MemoryStore(VectorStore(), Embedding(), "memory_test_" + uuid4().hex)
    calls = []

    async def respond(reservation, payload):
        calls.append(payload)
        prompt = payload["messages"][-1]["content"]
        assert reservation.operation.attempt_timeout_seconds == 29
        if "CANDIDATES:" in prompt:
            target = prompt.split("id=")[1].split(" ::")[0]
            operation = "DELETE" if "retract" in prompt else "UPDATE"
            text = json.dumps([{"candidate": 0, "op": operation, "target_id": target,
                                "content": "Deadline day 9"}])
        else:
            fact = "Deadline day 9" if "correction" in prompt else "Deadline day 5"
            if "retract" in prompt:
                fact = "Explicitly retract the deadline"
            text = json.dumps([fact])
        return {"choices": [{"message": {"content": text}}]}

    def activities(factory):
        return [*MemoryActivities(factory, model_profile="test", store_factory=lambda: memory).registered(),
                ModelActivities(factory, leaves=LEAVES).plan]

    policy = MemoryPolicy(model_profile="test", chunk_turns=1, max_facts=3, max_candidates=3,
        search_limit=5, answer_char_cap=2000, hard_delete=hard_delete, history_window=1,
        attempt_timeout_seconds=29.0).model_dump(mode="json")
    namespace = "unit:test"
    try:
        async with feature_environment(store, name="memory", workflows=WORKFLOWS,
                                        build_activities=activities, respond=respond) as env:
            client = env.runtime_client("user:test")
            try:
                items = []
                for i, (query, trigger) in enumerate([
                    ("context deadline day 5", "context"),
                    ("correction deadline day 9", "correction"),
                    ("retract my deadline", "correction"),
                ]):
                    ref = await client.put_json({"namespace": namespace, "user_id": "test",
                        "query": query, "answer": "", "trigger_class": trigger})
                    items.append({"item_id": str(i), "input": ref})
                status, result = await env.submit({"items": items,
                    "partition_key": sha256(namespace.encode()).hexdigest(),
                    "accepted_at": datetime.now(UTC).isoformat()}, configuration=policy)
                assert status["state"] == "SUCCEEDED"
                assert result == {"namespaces": 1, "chunks": 3, "partial": False,
                                  "added": 1, "updated": 1, "deleted": 1, "noop": 0}
                assert len(calls) == 5  # 3 distill + 2 arbitrate; no LLM retry after lost write ACK
                assert len(writes) == 4 and writes[0] == writes[1]
                assert await memory.search("Deadline", namespace=namespace) == []
                after = await memory.get_by_id(writes[0]["id"])
                assert (after is None) is hard_delete
                history = await env.temporal.get_workflow_handle(status["root_id"]).fetch_history()
                rollovers = 0
                while previous := history.events[0].workflow_execution_started_event_attributes.continued_execution_run_id:
                    history = await env.temporal.get_workflow_handle(status["root_id"], run_id=previous).fetch_history()
                    await Replayer(workflows=WORKFLOWS).replay_workflow(history)
                    rollovers += 1
                assert rollovers == 2
                await env.replay(status["root_id"])
                assert len(calls) == 5 and len(writes) == 4
            finally:
                await client.close()
    finally:
        qdrant.close()
