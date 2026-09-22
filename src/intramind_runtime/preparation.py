"""Text prompt sizing against a pinned llama.cpp template/tokenizer.

The profile must certify template/token count parity with the deployed engine.
This adapter does not infer memory safety from token count.
"""

import asyncio
import json

import httpx
from pydantic import Field

from .contracts import AdmissionDenied, AttemptTimeout, Contract


class PrepareRequest(Contract):
    model_profile: str = Field(min_length=1, max_length=120)
    payload: dict
    max_output_tokens: int = Field(gt=0, le=1_000_000)
    attempt_timeout_seconds: AttemptTimeout | None = None


class LlamaCppPromptSizer:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        model: str,
        capacity_profile_id: str,
        context_limit: int,
        token_margin: int,
        expected_output_tokens: int,
        response_formats: frozenset[str] = frozenset({"text"}),
        allow_tool_calls: bool = False,
    ):
        if (
            not capacity_profile_id
            or context_limit <= 0
            or token_margin < 0
            or expected_output_tokens <= 0
            or not response_formats
            or response_formats - {"text", "json_object", "json_schema"}
            or type(allow_tool_calls) is not bool
        ):
            raise ValueError("a validated sizing profile is required")
        self.client, self.model = client, model
        self.profile_id, self.context_limit = capacity_profile_id, context_limit
        self.token_margin, self.expected_output = token_margin, expected_output_tokens
        self.response_formats = response_formats
        self.allow_tool_calls = allow_tool_calls
        self._io_window = asyncio.Semaphore(4)

    async def size(self, request: PrepareRequest):
        """Validate and size without persisting foreground conversation content."""
        payload = request.payload
        allowed = {
            "messages",
            "temperature",
            "top_p",
            "seed",
            "stop",
            "chat_template_kwargs",
            "response_format",
            "presence_penalty",
            "frequency_penalty",
        }
        if self.allow_tool_calls:
            allowed.update({"tools", "tool_choice", "parallel_tool_calls"})
        if payload.keys() - allowed:
            raise ValueError("unsupported parameter for the text completion sizing profile")
        response_format = payload.get("response_format", {"type": "text"})
        if (
            not isinstance(response_format, dict)
            or response_format.get("type") not in self.response_formats
        ):
            raise ValueError("response format is not validated for this sizing profile")
        if response_format["type"] == "json_schema":
            wrapper = response_format.get("json_schema")
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get("schema"), dict):
                raise ValueError("json_schema requires an explicit schema object")
        template_options = payload.get("chat_template_kwargs", {})
        if template_options not in ({}, {"enable_thinking": False}):
            raise ValueError("template options are outside this sizing contract")
        messages = payload.get("messages")
        validate_messages(messages, allow_tools=self.allow_tool_calls)
        if "tools" in payload:
            validate_tools(payload["tools"])
        if payload.get("tool_choice", "auto") not in ("auto", "none", "required"):
            raise ValueError("unsupported tool choice")
        if "parallel_tool_calls" in payload and type(payload["parallel_tool_calls"]) is not bool:
            raise ValueError("parallel_tool_calls must be boolean")
        raw = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        if len(raw) > 1024 * 1024:
            raise ValueError("prompt exceeds sizing request bound")
        async with self._io_window:
            template = await self.client.post(
                "apply-template",
                json={
                    "model": self.model,
                    "messages": messages,
                    "chat_template_kwargs": template_options,
                    **(
                        {"response_format": response_format} if "response_format" in payload else {}
                    ),
                    **{
                        key: payload[key]
                        for key in ("tools", "tool_choice", "parallel_tool_calls")
                        if key in payload
                    },
                },
            )
            template.raise_for_status()
            prompt = template.json().get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("invalid template response")
            response = await self.client.post(
                "tokenize",
                json={
                    "model": self.model,
                    "content": prompt,
                    "add_special": True,
                    "parse_special": True,
                    "with_pieces": False,
                },
            )
            response.raise_for_status()
            tokens = response.json().get("tokens")
            if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                raise ValueError("invalid tokenizer response")
        bound = len(tokens) + self.token_margin
        if bound + request.max_output_tokens > self.context_limit:
            raise AdmissionDenied("request exceeds compatible context; workflow must split input")
        return {
            "model_profile": request.model_profile,
            "input_tokens_bound": bound,
            "max_output_tokens": request.max_output_tokens,
            "expected_cost": bound + min(request.max_output_tokens, self.expected_output),
            "capacity_profile_id": self.profile_id,
            **(
                {"attempt_timeout_seconds": request.attempt_timeout_seconds}
                if request.attempt_timeout_seconds is not None else {}
            ),
        }

    async def prepare(self, request: PrepareRequest, artifacts, tenant_id: str):
        sized = await self.size(request)
        raw = json.dumps(request.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ref = await artifacts.put(tenant_id, raw)
        return {"payload": ref.model_dump(mode="json"), **sized}


def validate_tools(tools: list) -> None:
    if not isinstance(tools, list) or not 1 <= len(tools) <= 64:
        raise ValueError("tools must be a bounded list")
    names = set()
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or set(tool) != {"type", "function"}
            or tool["type"] != "function"
        ):
            raise ValueError("unsupported tool definition")
        definition = tool["function"]
        if (
            not isinstance(definition, dict)
            or definition.keys() - {"name", "description", "parameters", "strict"}
            or not isinstance(definition.get("name"), str)
            or not definition["name"]
            or not isinstance(definition.get("parameters"), dict)
        ):
            raise ValueError("invalid function definition")
        if definition["name"] in names:
            raise ValueError("duplicate tool name")
        names.add(definition["name"])


def validate_messages(messages: list, *, allow_tools: bool) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("a non-empty conversation is required")
    pending, seen = set(), set()
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("invalid chat message")
        role = message.get("role")
        if role == "tool" and allow_tools:
            if (
                set(message) != {"role", "content", "tool_call_id"}
                or not isinstance(message["content"], str)
                or not isinstance(message["tool_call_id"], str)
                or message["tool_call_id"] not in pending
            ):
                raise ValueError("tool result does not match an outstanding call")
            pending.remove(message["tool_call_id"])
            continue
        if pending:
            raise ValueError("tool calls require results before another turn")
        calls = message.get("tool_calls")
        if allow_tools and role == "assistant" and calls is not None:
            if (
                set(message) != {"role", "content", "tool_calls"}
                or message["content"] is not None
                and not isinstance(message["content"], str)
                or not isinstance(calls, list)
                or not 1 <= len(calls) <= 16
            ):
                raise ValueError("invalid assistant tool turn")
            for call in calls:
                if (
                    not isinstance(call, dict)
                    or set(call) != {"id", "type", "function"}
                    or call["type"] != "function"
                    or not isinstance(call["id"], str)
                    or not call["id"]
                    or call["id"] in seen
                ):
                    raise ValueError("invalid or duplicate tool call identity")
                function = call["function"]
                if (
                    not isinstance(function, dict)
                    or set(function) != {"name", "arguments"}
                    or not isinstance(function["name"], str)
                    or not isinstance(function["arguments"], str)
                ):
                    raise ValueError("invalid tool call")
                pending.add(call["id"])
                seen.add(call["id"])
        elif (
            set(message) != {"role", "content"}
            or role not in ("system", "user", "assistant")
            or not isinstance(message["content"], str)
        ):
            raise ValueError("unsupported chat message for this tokenizer contract")
    if pending:
        raise ValueError("conversation contains unfinished tool calls")
