"""Foreground HTTP contracts stay separate from durable workflow submission."""

import json

from fastapi import Depends, HTTPException, Request

from .contracts import AdmissionDenied
from .preparation import PrepareRequest


async def _payload(request):
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 1024*1024:
            raise HTTPException(413, "direct request exceeds transport bound")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(422, "invalid request JSON") from None
    if not isinstance(payload, dict):
        raise HTTPException(422, "request must be an object")
    return payload


def register(app, tenant, proxies, preparers):
    @app.post("/v1/direct/{model_profile}/chat/completions")
    async def completion(model_profile: str, request: Request, tenant_id=Depends(tenant)):
        proxy = proxies.get(model_profile)
        preparer = preparers.get(model_profile)
        if proxy is None or preparer is None or proxy.pool.kind != "llm":
            raise HTTPException(503, "qualified direct inference profile unavailable")
        payload = await _payload(request)
        if (payload.get("model") != proxy.model or type(payload.get("n", 1)) is not int
            or payload.get("n", 1) != 1):
            raise HTTPException(422, "model or completion count differs from the qualified profile")
        if type(payload.get("stream", False)) is not bool:
            raise HTTPException(422, "stream must be boolean")
        stream_options = payload.get("stream_options", {})
        if (not isinstance(stream_options, dict) or stream_options.keys()-{"include_usage"}
            or ("include_usage" in stream_options and type(stream_options["include_usage"]) is not bool)):
            raise HTTPException(422, "unsupported stream options")
        limit = payload.get("max_tokens", payload.get("max_completion_tokens"))
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise HTTPException(422, "output limit must be positive")
        if "max_tokens" in payload and "max_completion_tokens" in payload:
            raise HTTPException(422, "provide one output limit")
        options = {k: v for k, v in payload.items() if k not in {
            "model", "n", "stream", "stream_options", "max_tokens", "max_completion_tokens"}}
        try:
            sized = await preparer.size(PrepareRequest(model_profile=model_profile, payload=options,
                                                      max_output_tokens=limit or 1))
        except AdmissionDenied:
            raise
        except ValueError:
            raise HTTPException(422, "request is outside the qualified prompt contract") from None
        # An omitted output limit retains engine semantics and reserves a full context slot.
        bound = sized["input_tokens_bound"] + limit if limit else preparer.context_limit
        return await proxy.open(tenant_id, payload, request_bound=bound)
