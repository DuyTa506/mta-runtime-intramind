"""A contract-test feature: no Temporal imports or engine access in feature code."""

from .sdk import TaskPolicy, durable_task


@durable_task(name="runtime-contract", version=1, policy=TaskPolicy(max_iterations=1))
async def contract_feature(ctx, inputs):
    return await ctx.llm(key="answer/v1", payload=inputs["payload"], model_profile="test",
                         input_tokens_bound=10, max_output_tokens=20, expected_cost=10)
