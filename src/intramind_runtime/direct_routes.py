"""Foreground HTTP contracts stay separate from durable workflow submission."""

import asyncio
import json
import math
import re
from datetime import UTC, datetime, timedelta

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse

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
    def expired(proxy, streaming=False):
        if hasattr(proxy, "_deadline_response"):
            return proxy._deadline_response(streaming)
        return JSONResponse({"error": {"message": "inference request deadline exceeded",
            "type": "admission_timeout", "reason": "deadline_exceeded"}}, status_code=504)

    def until_for(proxy, workload_class, requested, started):
        policy = getattr(proxy, "_deadline_seconds", None)
        seconds = policy(workload_class, requested) if policy else min(1800, requested or 1800)
        return started+seconds

    def workload(request: Request):
        value = request.headers.get("x-intramind-workload-class", "background")
        if value not in {"qa", "user_task", "background", "maintenance"}:
            raise HTTPException(400, "invalid trusted inference workload")
        logical = request.headers.get("x-intramind-logical-request-id")
        if logical is not None and (len(logical) > 200 or not re.fullmatch(r"[A-Za-z0-9._:-]+", logical)):
            raise HTTPException(400, "invalid logical request identity")
        return value, logical

    def requested_deadline(request: Request):
        raw = request.headers.get("x-intramind-deadline-seconds")
        if raw is None:
            return None
        if len(raw) > 64 or not raw.isascii() or not raw.isdecimal() or int(raw) <= 0:
            raise HTTPException(400, "invalid direct inference deadline")
        return int(raw)

    def admission_options(request, workload_class):
        mode = request.headers.get("x-intramind-admission-mode", "wait")
        if mode not in {"wait", "try"} or (mode == "try" and workload_class != "background"):
            raise HTTPException(400, "invalid internal admission mode")
        cutoff = request.headers.get("x-intramind-dispatch-before")
        if cutoff is not None:
            try:
                if len(cutoff) > 64:
                    raise ValueError()
                cutoff = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
                if cutoff.tzinfo is None or cutoff.utcoffset() != timedelta(0):
                    raise ValueError()
            except ValueError:
                raise HTTPException(400, "dispatch cutoff must be a UTC timestamp") from None
        seconds = request.headers.get("x-intramind-step-timeout-seconds")
        if seconds is not None:
            try:
                seconds = float(seconds) if len(seconds) <= 64 else float("nan")
                if not math.isfinite(seconds) or not 0 < seconds <= 86400:
                    raise ValueError()
            except ValueError:
                raise HTTPException(400, "invalid step execution timeout") from None
        return {"admission_mode": mode, "dispatch_before": cutoff,
                "execution_timeout_seconds": seconds}

    @app.post("/v1/direct/{model_profile}/chat/completions")
    async def completion(model_profile: str, request: Request, tenant_id=Depends(tenant)):
        started = asyncio.get_running_loop().time()
        proxy = proxies.get(model_profile)
        preparer = preparers.get(model_profile)
        if proxy is None or preparer is None or proxy.pool.kind != "llm":
            raise HTTPException(503, "qualified direct inference profile unavailable")
        workload_class, logical_request_id = workload(request)
        requested = requested_deadline(request)
        options = admission_options(request, workload_class)
        trying = options["admission_mode"] == "try"
        if options["dispatch_before"] and datetime.now(UTC) >= options["dispatch_before"]:
            return proxy.deferred_response("dispatch_cutoff")
        until = until_for(proxy, workload_class, requested, started)
        if trying:
            seconds = min(30, proxy.timeout_seconds, options["execution_timeout_seconds"] or 30)
            requested = min(requested or 86400, math.ceil(seconds + 10))
            until = started + min(10, requested)
        try:
            async with asyncio.timeout_at(until):
                payload = await _payload(request)
        except TimeoutError:
            if trying:
                return proxy.deferred_response("model_not_ready")
            if until <= asyncio.get_running_loop().time():
                return expired(proxy)
            raise
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
        prompt_options = {k: v for k, v in payload.items() if k not in {
            "model", "n", "stream", "stream_options", "max_tokens", "max_completion_tokens"}}
        try:
            async with asyncio.timeout_at(until):
                sized = await preparer.size(PrepareRequest(model_profile=model_profile, payload=prompt_options,
                                                          max_output_tokens=limit or 1))
        except TimeoutError:
            if trying:
                return proxy.deferred_response("model_not_ready")
            if until <= asyncio.get_running_loop().time():
                return expired(proxy, payload.get("stream", False))
            raise
        except AdmissionDenied:
            raise
        except httpx.HTTPError:
            if trying:
                return proxy.deferred_response("model_not_ready")
            raise
        except ValueError:
            raise HTTPException(422, "request is outside the qualified prompt contract") from None
        # An omitted output limit retains engine semantics and reserves a full context slot.
        bound = sized["input_tokens_bound"] + limit if limit else preparer.context_limit
        return await proxy.open(tenant_id, payload, request_bound=bound,
                                workload_class=workload_class, logical_request_id=logical_request_id,
                                deadline_seconds=requested, started_at_monotonic=started, **options)

    @app.post("/v1/direct/{model_profile}/api/v1/embed")
    @app.post("/v1/direct/{model_profile}/api/v1/rerank")
    async def native(model_profile: str, request: Request, tenant_id=Depends(tenant)):
        started = asyncio.get_running_loop().time()
        path = request.url.path.rsplit("/", 1)[1]
        kind = {"embed": "embedding", "rerank": "rerank"}[path]
        proxy = proxies.get(model_profile)
        if proxy is None or proxy.pool.kind != kind or proxy.profile is None:
            raise HTTPException(503, "qualified direct inference profile unavailable")
        workload_class, logical_request_id = workload(request)
        requested = requested_deadline(request)
        options = admission_options(request, workload_class)
        trying = options['admission_mode'] == 'try'
        if options['dispatch_before'] and datetime.now(UTC) >= options['dispatch_before']:
            return proxy.deferred_response('dispatch_cutoff')
        until = until_for(proxy, workload_class, requested, started)
        if trying:
            seconds = min(30, proxy.timeout_seconds, options['execution_timeout_seconds'] or 30)
            requested = min(requested or 86400, math.ceil(seconds + 10))
            until = started + min(10, requested)
        try:
            async with asyncio.timeout_at(until):
                payload = await _payload(request)
        except TimeoutError:
            if trying:
                return proxy.deferred_response('model_not_ready')
            if until <= asyncio.get_running_loop().time():
                return expired(proxy)
            raise
        try:
            validated = proxy.profile.validate_payload(payload)
        except ValueError:
            raise HTTPException(422, "request is outside the qualified native contract") from None
        if kind == "embedding":
            bound, batch = validated.characters, len(validated.texts)
        else:
            bound, batch = proxy.profile.requirement(validated)
            if not batch:
                return JSONResponse({"results": []})
        return await proxy.open(tenant_id, payload, request_bound=max(1, bound),
                                path=f"api/v1/{path}", batch_size=batch,
                                workload_class=workload_class, logical_request_id=logical_request_id,
                                deadline_seconds=requested, started_at_monotonic=started, **options)
