"""One HTTP attempt, no invisible SDK retries or caller-controlled routing."""

from typing import Protocol

import httpx

from .contracts import CancelOutcome, EmbeddingResult, EngineResult, Reservation, SpeechResult


class DriverFailure(Exception):
    def __init__(self, reason: str, *, not_sent=False, finished=False, retry=False):
        super().__init__(reason)
        self.not_sent, self.finished, self.retry = not_sent, finished, retry


class EngineDriver(Protocol):
    async def execute(
        self, reservation: Reservation, payload: dict
    ) -> EngineResult | SpeechResult | EmbeddingResult: ...
    async def cancel(self, reservation: Reservation) -> CancelOutcome: ...


class OpenAICompletionDriver:
    def __init__(self, base_url: str, api_key: str, model: str, *, timeout=1800.0,
                 client: httpx.AsyncClient | None = None):
        self.model = model
        self.client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/")+"/", headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=10), transport=httpx.AsyncHTTPTransport(retries=0),
            follow_redirects=False)

    async def execute(self, reservation: Reservation, payload: dict) -> EngineResult:
        allowed = {"messages", "temperature", "top_p", "seed", "stop", "response_format",
                   "presence_penalty", "frequency_penalty", "tools", "tool_choice", "chat_template_kwargs",
                   "parallel_tool_calls"}
        if (payload.keys() - allowed or not isinstance(payload.get("messages"), list)
            or ("parallel_tool_calls" in payload and type(payload["parallel_tool_calls"]) is not bool)):
            raise DriverFailure("invalid_completion_payload", not_sent=True)
        request = payload | {"model": self.model, "max_tokens": reservation.operation.max_output_tokens,
                             "n": 1, "stream": False}
        try:
            response = await self.client.post("chat/completions", json=request,
                headers={"X-Intramind-Attempt-ID": reservation.attempt_id})
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise DriverFailure(type(exc).__name__, not_sent=True, retry=True) from exc
        except httpx.HTTPError as exc:
            raise DriverFailure(type(exc).__name__) from exc
        if response.status_code in (400, 401, 403, 404, 422):
            raise DriverFailure(f"backend_rejected_{response.status_code}", finished=True)
        if response.status_code != 200:
            # 429/5xx termination semantics require a pinned engine contract.
            raise DriverFailure(f"backend_status_{response.status_code}_unconfirmed")
        try:
            body = response.json()
            if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or len(body["choices"]) != 1:
                raise ValueError("expected one completion")
            usage = body.get("usage") or {}
            return EngineResult(body=body, input_tokens=usage.get("prompt_tokens"),
                                output_tokens=usage.get("completion_tokens"))
        except (ValueError, TypeError) as exc:
            raise DriverFailure("invalid_response", finished=True) from exc

    async def cancel(self, reservation: Reservation) -> CancelOutcome:
        # HTTP disconnection alone is not cancellation evidence.
        return CancelOutcome.UNSUPPORTED

    async def close(self):
        await self.client.aclose()
