"""Trusted-service HTTP client; public identity verification remains at gateway."""

import json
from contextlib import asynccontextmanager
from hashlib import sha256
from tempfile import TemporaryFile
from urllib.parse import quote

import httpx

from .artifacts import IO_CHUNK_BYTES, MAX_ARTIFACT_BYTES, SpoolBudget, file_io
from .contracts import Artifact


class RuntimeClient:
    def __init__(self, base_url: str, service_token: str, tenant_id: str, *, client=None, spool_directory=None):
        self.spool_directory = spool_directory
        self.spool_budget = SpoolBudget()
        self.client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=15,
            headers={"Authorization": f"Bearer {service_token}", "X-Tenant-ID": tenant_id},
        )

    async def submit(self, submission: dict):
        response = await self.client.post("/v1/runs", json=submission)
        response.raise_for_status()
        return response.json()

    async def get_run(self, run_id: str):
        response = await self.client.get(f"/v1/runs/{run_id}")
        response.raise_for_status()
        return response.json()

    async def buffer(self, submission: dict):
        response = await self.client.post("/v1/buffers", json=submission)
        response.raise_for_status()
        return response.json()

    async def get_buffered(self, item_id: str):
        response = await self.client.get(f"/v1/buffers/{item_id}")
        response.raise_for_status()
        return response.json()

    async def cancel(self, run_id: str):
        response = await self.client.post(f"/v1/runs/{run_id}/cancel")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()

    async def put_json(self, value: dict):
        return await self.put_bytes(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode(),
            content_type="application/json",
        )

    async def put_bytes(self, value: bytes, *, content_type: str):
        response = await self.client.post(
            "/v1/artifacts",
            content=value,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()
        return response.json()

    async def put_file(self, source, *, content_type: str):
        """Stream a seekable caller-owned file without constructing a bytes copy."""
        def measure():
            source.seek(0, 2)
            size = source.tell()
            source.seek(0)
            return size
        size = await file_io(measure)
        if size > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact exceeds configured limit")
        checksum, received = sha256(), 0
        async def chunks():
            nonlocal received
            while chunk := await file_io(source.read, IO_CHUNK_BYTES):
                received += len(chunk)
                if received > size:
                    raise ValueError("artifact file changed during upload")
                checksum.update(chunk)
                yield chunk
            if received != size:
                raise ValueError("artifact file changed during upload")
        response = await self.client.post("/v1/artifacts", content=chunks(),
            headers={"Content-Type": content_type, "Content-Length": str(size)})
        response.raise_for_status()
        result = response.json()
        ref = Artifact.model_validate(result)
        if received != size or ref.size != size or ref.sha256 != checksum.hexdigest():
            raise ValueError("artifact upload receipt checksum/length mismatch")
        return result

    async def read_json(self, ref: dict):
        response = await self.client.post("/v1/artifacts/read", json=ref)
        response.raise_for_status()
        return response.json()

    async def read_bytes(self, ref: dict) -> bytes:
        response = await self.client.post("/v1/artifacts/read", json=ref)
        response.raise_for_status()
        return response.content

    @asynccontextmanager
    async def open_verified(self, ref: dict):
        """Download to disk and verify before exposing any bytes to the consumer."""
        artifact = Artifact.model_validate(ref)
        if artifact.size > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact exceeds configured limit")
        with self.spool_budget.reserve(artifact.size, read=True):
            with TemporaryFile(dir=self.spool_directory) as staged:
                checksum, size = sha256(), 0
                async with self.client.stream("POST", "/v1/artifacts/read", json=ref) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(IO_CHUNK_BYTES):
                        size += len(chunk)
                        if size > artifact.size:
                            raise ValueError("artifact checksum/length mismatch")
                        checksum.update(chunk)
                        await file_io(staged.write, chunk)
                if size != artifact.size or checksum.hexdigest() != artifact.sha256:
                    raise ValueError("artifact checksum/length mismatch")
                await file_io(staged.seek, 0)
                yield staged

    async def read_to_file(self, ref: dict, destination):
        """Copy verified bytes to a caller-owned binary file at its current position."""
        async with self.open_verified(ref) as source:
            def copy():
                while chunk := source.read(IO_CHUNK_BYTES):
                    destination.write(chunk)
            await file_io(copy)

    async def prepare(
        self, *, model_profile: str, payload: dict, max_output_tokens: int,
        attempt_timeout_seconds: float | None = None,
    ):
        response = await self.client.post(
            "/v1/requests/prepare",
            json={
                "model_profile": model_profile,
                "payload": payload,
                "max_output_tokens": max_output_tokens,
                **(
                    {"attempt_timeout_seconds": attempt_timeout_seconds}
                    if attempt_timeout_seconds is not None else {}
                ),
            },
        )
        response.raise_for_status()
        return response.json()

    async def llm_profile(self, model_profile: str):
        """Read the configured context/tool contract; actual admission remains atomic at dispatch."""
        response = await self.client.get(f"/v1/llm/profiles/{quote(model_profile, safe='')}")
        response.raise_for_status()
        return response.json()

    async def speech_profile(self, model_profile: str):
        response = await self.client.get(f"/v1/speech/profiles/{quote(model_profile, safe='')}")
        response.raise_for_status()
        return response.json()

    async def prepare_speech(self, *, model_profile: str, capacity_profile_id: str,
                             payload: dict, attempt_timeout_seconds: float):
        response = await self.client.post("/v1/speech/prepare", json={
            "model_profile": model_profile, "capacity_profile_id": capacity_profile_id,
            "payload": payload, "attempt_timeout_seconds": attempt_timeout_seconds,
        })
        response.raise_for_status()
        return response.json()

    async def embedding_profile(self, model_profile: str):
        response = await self.client.get(f"/v1/embedding/profiles/{quote(model_profile, safe='')}")
        response.raise_for_status()
        return response.json()

    async def prepare_embedding(self, *, model_profile: str, capacity_profile_id: str,
                                payload: dict, attempt_timeout_seconds: float):
        response = await self.client.post("/v1/embedding/prepare", json={
            "model_profile": model_profile, "capacity_profile_id": capacity_profile_id,
            "payload": payload, "attempt_timeout_seconds": attempt_timeout_seconds,
        })
        response.raise_for_status()
        return response.json()
