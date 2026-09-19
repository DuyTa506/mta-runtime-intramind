"""MinIO adapter: verified bytes first, durable database reference second."""

import asyncio
import io
from typing import Protocol

from minio import Minio

from .contracts import Artifact, digest


def tenant_prefix(tenant_id: str) -> str:
    return f"tenants/{digest(tenant_id.encode())}/"


class ArtifactPort(Protocol):
    async def put(self, tenant_id: str, data: bytes, content_type="application/json") -> Artifact: ...
    async def get(self, artifact: Artifact) -> bytes: ...


class MinioArtifacts:
    def __init__(self, client: Minio, bucket: str, max_bytes: int = 64 * 1024 * 1024):
        self.client, self.bucket, self.max_bytes = client, bucket, max_bytes

    async def ready(self):
        # Provision explicitly; a typo must not silently create a new bucket.
        if not await asyncio.to_thread(self.client.bucket_exists, self.bucket):
            raise RuntimeError("configured artifact bucket does not exist")

    async def put(self, tenant_id: str, data: bytes, content_type="application/json") -> Artifact:
        if len(data) > self.max_bytes:
            raise ValueError("artifact exceeds configured limit")
        checksum = digest(data)
        artifact = Artifact(key=f"{tenant_prefix(tenant_id)}sha256/{checksum}",
                            sha256=checksum, size=len(data), content_type=content_type)
        await asyncio.to_thread(self.client.put_object, self.bucket, artifact.key,
                                io.BytesIO(data), len(data), content_type=content_type,
                                metadata={"sha256": checksum})
        await self.get(artifact)
        return artifact

    async def get(self, artifact: Artifact) -> bytes:
        if artifact.size > self.max_bytes:
            raise ValueError("artifact exceeds configured limit")

        def read():
            response = self.client.get_object(self.bucket, artifact.key)
            try:
                data = response.read(artifact.size + 1)
            finally:
                response.close()
                response.release_conn()
            if len(data) != artifact.size or digest(data) != artifact.sha256:
                raise ValueError("artifact checksum/length mismatch")
            return data

        return await asyncio.to_thread(read)
