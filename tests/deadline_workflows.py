"""Exercise a bounded phase without importing Temporal into feature code."""

from datetime import timedelta

from intramind_runtime.sdk import OperationDeadlineExceeded, TaskPolicy, durable_task


@durable_task(name="deadline-contract", version=1, policy=TaskPolicy(max_iterations=1))
async def deadline_feature(ctx, inputs):
    try:
        return await ctx.llm(
            key="assess", payload=inputs, model_profile="test", input_tokens_bound=10,
            max_output_tokens=20, expected_cost=10,
            deadline=ctx.started_at + timedelta(seconds=3),
        )
    except OperationDeadlineExceeded:
        return ctx.finish_partial(inputs, reason="assessment_timeout")
