"""Pinned serving TTS contract; one transport attempt and a separate character budget."""

import io
import json
import wave
from typing import Literal

import httpx
from pydantic import Field, field_serializer

from .contracts import (
    AttemptTimeout,
    CancelOutcome,
    Contract,
    Reservation,
    SpeechOperationSpec,
    SpeechResult,
)
from .drivers import DriverFailure

TERMINATION_CONTRACT = "termination-v1"


class SpeechVoice(Contract):
    voice_id: str | None = Field(default=None, min_length=1, max_length=120)
    gender: Literal["male", "female"] | None = None
    speaker_id: int | None = Field(default=None, ge=0, strict=True)


class SpeechPayload(Contract):
    text: str = Field(min_length=1, max_length=100_000)
    language: Literal["vi"] = "vi"
    voice: SpeechVoice = Field(default_factory=SpeechVoice)
    speed: float = Field(default=1.0, ge=0.5, le=2, allow_inf_nan=False)
    pause_scale: float | None = Field(default=None, ge=0, le=3, allow_inf_nan=False)


class SpeechProfile(Contract):
    model_profile: str = Field(min_length=1)
    capacity_profile_id: str = Field(min_length=1)
    character_limit: int = Field(gt=0, le=100_000)
    sample_rate: int = Field(gt=0)
    max_audio_bytes: int = Field(gt=44, le=128 * 1024 * 1024)
    voices: frozenset[str] = Field(min_length=1)

    @field_serializer("voices")
    def ordered_voices(self, value):
        return sorted(value)

    def validate_payload(self, payload: dict) -> SpeechPayload:
        """Reject incompatible input; normalization and splitting belong to the workflow."""
        request = SpeechPayload.model_validate(payload)
        voice = request.voice.voice_id or f"vi_{request.voice.gender or 'female'}"
        if not request.text.strip() or len(request.text) > self.character_limit:
            raise ValueError("speech text exceeds the qualified character limit or is empty")
        if voice not in self.voices:
            raise ValueError("speech voice is outside the qualified profile")
        return request


class SpeechPrepareRequest(Contract):
    model_profile: str = Field(min_length=1)
    capacity_profile_id: str = Field(min_length=1)
    payload: dict
    attempt_timeout_seconds: AttemptTimeout


class SpeechPreparer:
    """Persist qualified input without calling the synthesis endpoint."""

    def __init__(self, profile: SpeechProfile):
        self.profile = profile

    async def prepare(self, request: SpeechPrepareRequest, artifacts, tenant_id: str) -> dict:
        if (request.model_profile != self.profile.model_profile
            or request.capacity_profile_id != self.profile.capacity_profile_id):
            raise ValueError("speech profile changed; accepted work requires its pinned profile")
        payload = self.profile.validate_payload(request.payload)
        raw = json.dumps(request.payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
        artifact = await artifacts.put(tenant_id, raw)
        return {
            "payload": artifact.model_dump(mode="json"),
            "model_profile": request.model_profile,
            "capacity_profile_id": request.capacity_profile_id,
            "characters_bound": len(payload.text),
            "expected_cost": len(payload.text),
            "attempt_timeout_seconds": request.attempt_timeout_seconds,
        }


class ServingSpeechDriver:
    """Only an echoed attempt and the pinned termination contract can release compute."""

    def __init__(self, base_url: str, profile: SpeechProfile, *, api_key: str | None = None,
                 client: httpx.AsyncClient | None = None):
        self.profile = profile
        self.client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(None, connect=10),
            transport=httpx.AsyncHTTPTransport(retries=0), follow_redirects=False,
        )

    async def execute(self, reservation: Reservation, payload: dict) -> SpeechResult:
        spec = reservation.operation
        try:
            request = self.profile.validate_payload(payload)
            if (not isinstance(spec, SpeechOperationSpec)
                or spec.characters_bound != len(request.text)
                or spec.model_profile != self.profile.model_profile
                or spec.capacity_profile_id != self.profile.capacity_profile_id):
                raise ValueError("speech reservation does not match payload/profile")
        except ValueError as exc:
            raise DriverFailure("invalid_speech_payload", not_sent=True) from exc
        state = None
        try:
            async with self.client.stream("POST", "api/v1/tts", json=payload,
                headers={"X-Intramind-Attempt-ID": reservation.attempt_id}) as response:
                if (response.headers.get("X-Intramind-TTS-Contract") == TERMINATION_CONTRACT
                    and response.headers.get("X-Intramind-Attempt-ID") == reservation.attempt_id):
                    state = response.headers.get("X-Intramind-Compute-State")
                if state not in {"not_started", "terminated"}:
                    raise DriverFailure("speech_response_unconfirmed")
                if response.status_code != 200:
                    raise DriverFailure(f"speech_backend_{response.status_code}",
                        not_sent=state == "not_started", finished=state == "terminated",
                        retry=response.status_code in {429, 500})
                if state != "terminated":
                    raise DriverFailure("speech_success_without_termination")
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.profile.max_audio_bytes:
                        raise DriverFailure("speech_audio_exceeds_profile", finished=True)
                    chunks.append(chunk)
                audio = b"".join(chunks)
                try:
                    if response.headers.get("Content-Type", "").split(";")[0] != "audio/wav":
                        raise ValueError("expected audio/wav")
                    with wave.open(io.BytesIO(audio), "rb") as wav:
                        frames, rate = wav.getnframes(), wav.getframerate()
                        if (wav.getnchannels() != 1 or wav.getsampwidth() != 2
                            or rate != self.profile.sample_rate or not frames
                            or len(wav.readframes(frames)) != frames * 2):
                            raise ValueError("audio differs from qualified PCM profile")
                    duration_ms = round(1000 * frames / rate)
                except (ValueError, wave.Error, EOFError) as exc:
                    raise DriverFailure("speech_invalid_audio", finished=True) from exc
                return SpeechResult(body={"duration_ms": duration_ms}, audio=audio,
                                    characters=len(request.text))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise DriverFailure(type(exc).__name__, not_sent=True, retry=True) from exc
        except httpx.HTTPError as exc:
            raise DriverFailure(type(exc).__name__, finished=state == "terminated",
                                retry=state == "terminated") from exc

    async def cancel(self, reservation: Reservation) -> CancelOutcome:
        return CancelOutcome.UNSUPPORTED

    async def close(self):
        await self.client.aclose()
