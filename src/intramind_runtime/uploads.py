"""Bound temporary disk and storage I/O before accepting artifact bodies."""

import asyncio
from tempfile import TemporaryFile

from fastapi import HTTPException, Request

from .artifacts import MAX_ARTIFACT_BYTES, ArtifactPort, ArtifactTooLarge, file_io
from .contracts import Artifact

JSON_LIMIT_BYTES = 16 * 1024 * 1024


class ArtifactUploads:
    """Reject excess transfers before allocating temporary storage or consuming bodies."""

    def __init__(self, artifacts: ArtifactPort, *, max_bytes: int = MAX_ARTIFACT_BYTES,
                 concurrency: int = 2) -> None:
        if max_bytes <= 0 or concurrency <= 0:
            raise ValueError("artifact size and upload concurrency must be positive")
        self.artifacts, self.max_bytes = artifacts, max_bytes
        self.capacity = asyncio.Semaphore(concurrency)

    async def receive(self, request: Request, tenant_id: str) -> Artifact:
        """Spool a bounded body, then publish its verified immutable reference."""
        content_type = request.headers.get("content-type", "application/json")
        media_type = content_type.partition(";")[0].strip().lower()
        limit = self.max_bytes
        if media_type == "application/json" or media_type.endswith("+json"):
            limit = min(limit, JSON_LIMIT_BYTES)
        declared = request.headers.get("content-length")
        if declared is not None:
            if not declared.isascii() or not declared.isdecimal() or len(declared) > 20:
                raise HTTPException(400, "invalid artifact content length")
            if int(declared) > limit:
                raise HTTPException(413, f"artifact exceeds {limit} byte upload limit")
        if self.capacity.locked():
            raise HTTPException(503, "artifact upload capacity busy", headers={"Retry-After": "1"})
        async with self.capacity:
            with TemporaryFile() as staged:
                size = 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"artifact exceeds {limit} byte upload limit")
                    if chunk:
                        await file_io(staged.write, chunk)
                if declared is not None and size != int(declared):
                    raise HTTPException(400, "artifact content length mismatch")
                try:
                    return await self.artifacts.put_file(tenant_id, staged, content_type)
                except ArtifactTooLarge as exc:
                    raise HTTPException(413, str(exc)) from exc
