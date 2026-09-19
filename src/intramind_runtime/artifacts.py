"""MinIO adapter: verified bytes first, durable database reference second."""

import asyncio
import io
from hashlib import sha256
from typing import BinaryIO, Protocol

from minio import Minio

from .contracts import Artifact, digest

MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
IO_CHUNK_BYTES = 64 * 1024


class ArtifactTooLarge(ValueError):
    """The upload exceeds the configured storage limit."""


async def file_io(function, *args, **kwargs):
    """Keep file ownership until the blocking thread exits, even on cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


def tenant_prefix(tenant_id: str) -> str:
    return f"tenants/{digest(tenant_id.encode())}/"


class ArtifactPort(Protocol):
    async def put(self, tenant_id: str, data: bytes, content_type="application/json") -> Artifact: ...
    async def put_file(self, tenant_id: str, data: BinaryIO,
                       content_type="application/json") -> Artifact: ...
    async def get(self, artifact: Artifact) -> bytes: ...


class MinioArtifacts:
    def __init__(self, client: Minio, bucket: str, max_bytes: int = MAX_ARTIFACT_BYTES):
        self.client, self.bucket, self.max_bytes = client, bucket, max_bytes

    async def ready(self):
        # Provision explicitly; a typo must not silently create a new bucket.
        if not await asyncio.to_thread(self.client.bucket_exists, self.bucket):
            raise RuntimeError("configured artifact bucket does not exist")

    async def put(self, tenant_id: str, data: bytes, content_type="application/json") -> Artifact:
        if len(data) > self.max_bytes:
            raise ArtifactTooLarge("artifact exceeds configured limit")
        with io.BytesIO(data) as source:
            return await self.put_file(tenant_id, source, content_type)

    async def put_file(self, tenant_id: str, data: BinaryIO,
                       content_type="application/json") -> Artifact:
        """Read a seekable caller-owned file; return only after checksum verification."""
        def write():
            data.seek(0, io.SEEK_END)
            size = data.tell()
            if size > self.max_bytes:
                raise ArtifactTooLarge("artifact exceeds configured limit")
            data.seek(0)
            checksum = sha256()
            while chunk := data.read(IO_CHUNK_BYTES):
                checksum.update(chunk)
            value = checksum.hexdigest()
            artifact = Artifact(key=f"{tenant_prefix(tenant_id)}sha256/{value}",
                sha256=value, size=size, content_type=content_type)
            data.seek(0)
            self.client.put_object(self.bucket, artifact.key, data, size,
                content_type=content_type, metadata={"sha256": value},
                part_size=5 * 1024 * 1024, num_parallel_uploads=1)
            self._verify(artifact)
            return artifact

        return await file_io(write)

    def _verify(self, artifact: Artifact, destination: BinaryIO | None = None):
        response = self.client.get_object(self.bucket, artifact.key)
        try:
            checksum, size = sha256(), 0
            while chunk := response.read(IO_CHUNK_BYTES):
                size += len(chunk)
                if size > artifact.size:
                    raise ValueError("artifact checksum/length mismatch")
                checksum.update(chunk)
                if destination is not None:
                    destination.write(chunk)
            if size != artifact.size or checksum.hexdigest() != artifact.sha256:
                raise ValueError("artifact checksum/length mismatch")
        finally:
            response.close()
            response.release_conn()

    async def get(self, artifact: Artifact) -> bytes:
        if artifact.size > self.max_bytes:
            raise ArtifactTooLarge("artifact exceeds configured limit")

        def read():
            with io.BytesIO() as destination:
                self._verify(artifact, destination)
                return destination.getvalue()

        return await file_io(read)
