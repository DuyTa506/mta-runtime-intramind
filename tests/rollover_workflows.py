"""Replayable contract fixture using only the feature facade."""

from intramind_runtime.sdk import TaskPolicy, durable_task


@durable_task(
    name="runtime-rollover-contract",
    version=1,
    policy=TaskPolicy(max_iterations=2, deadline_seconds=180, max_rollovers=1),
)
async def rollover_feature(ctx, inputs):
    answer = await ctx.llm(
        key="answer/v1",
        payload=inputs["payload"],
        model_profile="test",
        input_tokens_bound=10,
        max_output_tokens=20,
        expected_cost=10,
    )
    if not inputs.get("resumed"):
        await ctx.wait_event("rollover")
        ctx.continue_as_new(
            {
                "resumed": True,
                "payload": inputs["payload"],
                "answer": answer,
                "root_id": ctx.root_id,
                "deadline": ctx.deadline.isoformat(),
                "started_at": ctx.started_at.isoformat(),
                "elapsed_seconds": ctx.elapsed_seconds(),
            }
        )

    carried = await ctx.wait_event("carry")
    assert carried == {"approval": "accepted-before-rollover"}
    assert answer == inputs["answer"]
    assert ctx.root_id == inputs["root_id"]
    assert ctx.deadline.isoformat() == inputs["deadline"]
    assert ctx.started_at.isoformat() == inputs["started_at"]
    assert ctx.elapsed_seconds() >= inputs["elapsed_seconds"]
    assert ctx.rollovers == 1
    return answer
