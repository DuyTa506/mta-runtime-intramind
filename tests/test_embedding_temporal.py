import json

import httpx
import pytest
from embedding_workflows import embedding_checkpoint
from feature_harness import feature_environment
from temporalio.worker import Replayer
from test_embedding import profile, proof, vectors

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


async def test_batches_replay_after_rollover_and_lost_publication_without_inference(store):
    calls, published = [], []

    async def backend(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=vectors(), headers=proof(request))

    async def forbidden(*args):
        raise AssertionError("Embedding cannot consume a completion pool")

    def activities(factory):
        @durable_activity(name="test.embedding-prepare/v1")
        async def prepare(inputs):
            client = factory(inputs["tenant_id"])
            try:
                source = await client.read_json(inputs["input"])
                return [await client.prepare_embedding(model_profile="embedding-test",
                    capacity_profile_id="embed-v1", payload=payload,
                    attempt_timeout_seconds=20) for payload in source["batches"]]
            finally:
                await client.close()

        @durable_activity(name="test.embedding-publish/v1")
        async def publish(inputs):
            published.append(inputs["results"])
            if len(published) == 1:
                raise OSError("publication acknowledgement lost")
            client = factory(inputs["tenant_id"])
            try:
                results = [await client.read_json(ref) for ref in inputs["results"]]
                return await client.put_json({"batches": results})
            finally:
                await client.close()
        return [prepare, publish]

    batches = [{"texts": ["xin chào", ""], "input_type": "query"},
               {"texts": ["42", "ok"], "input_type": "document"}]
    async with feature_environment(store, name="embedding-checkpoint", workflows=[embedding_checkpoint],
        build_activities=activities, respond=forbidden, embedding_profile=profile(),
        respond_embedding=backend) as env:
        status, result = await env.submit({"batches": batches})
        assert calls == batches and len(published) == 2 and published[0] == published[1]
        assert [batch["body"] for batch in result["batches"]] == [vectors(), vectors()]
        assert status["reserved"] == status["spent"] == 0
        assert status["resource_budgets"]["embedding_characters"] == {
            "limit": 100000, "reserved": 0, "spent": 12,
        }
        assert status["operations"] == {"SUCCEEDED": 2} and not env.calls
        handle = env.temporal.get_workflow_handle(status["root_id"])
        history = await handle.fetch_history()
        previous = history.events[0].workflow_execution_started_event_attributes.continued_execution_run_id
        assert previous
        old = await env.temporal.get_workflow_handle(status["root_id"], run_id=previous).fetch_history()
        replayer = Replayer(workflows=[embedding_checkpoint])
        await replayer.replay_workflow(old)
        await replayer.replay_workflow(history)
        assert calls == batches and len(published) == 2
