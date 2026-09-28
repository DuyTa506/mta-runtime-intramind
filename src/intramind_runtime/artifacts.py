"""MinIO adapter: verified bytes first, durable database reference second."""

import asyncio
import io
import os
from contextlib import asynccontextmanager, contextmanager
from hashlib import sha256
from tempfile import TemporaryFile, gettempdir
from typing import BinaryIO, Protocol

from minio import Minio

from .contracts import Artifact, digest

MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
IO_CHUNK_BYTES = 64 * 1024
MIN_SPOOL_FREE_BYTES = 1024 * 1024 * 1024


class ArtifactTooLarge(ValueError):
    """The upload exceeds the configured storage limit."""


class ArtifactIntegrityError(ValueError):
    """Stored content does not match the committed digest or length."""


class ArtifactCapacityBusy(RuntimeError):
    """Retryable exhaustion of the process-local artifact spool budget."""


class SpoolBudget:
    def __init__(self, *, concurrency=2, read_bytes=256 * 1024 * 1024,
                 total_bytes=512 * 1024 * 1024, spool_directory=None,
                 min_free_bytes=MIN_SPOOL_FREE_BYTES):
        if min(concurrency, read_bytes, total_bytes) <= 0:
            raise ValueError("artifact budgets must be positive")
        if min_free_bytes < 0:
            raise ValueError("artifact minimum free space must not be negative")
        self.spool_directory = spool_directory or gettempdir()
        self.min_free_bytes = min_free_bytes
        self.concurrency, self.read_limit, self.total_limit = concurrency, read_bytes, total_bytes
        self.reads = self.read_bytes = self.total_bytes = 0

    @contextmanager
    def reserve(self, size, *, read=False):
        # Used only on the owning event loop; admission has no suspension/queue.
        if type(size) is not int or size < 0:
            raise ValueError("artifact reservation size must be a nonnegative integer")
        if (self.total_bytes + size > self.total_limit or
                (read and (self.reads >= self.concurrency or
                           self.read_bytes + size > self.read_limit))):
            raise ArtifactCapacityBusy("artifact spool capacity busy")
        disk = os.statvfs(self.spool_directory)
        # Conservatively subtract all outstanding local reservations, even if
        # some of their bytes already appear in disk usage. This is not a
        # cross-process quota: deployment must also budget shared disk users.
        available = disk.f_bavail * disk.f_frsize
        if available - self.total_bytes - size < self.min_free_bytes:
            raise ArtifactCapacityBusy("artifact spool disk headroom exhausted")
        self.total_bytes += size
        if read:
            self.reads += 1
            self.read_bytes += size
        try:
            yield
        finally:
            self.total_bytes -= size
            if read:
                self.reads -= 1
                self.read_bytes -= size


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
    def open_verified(self, artifact: Artifact): ...


class MinioArtifacts:
    def __init__(self, client: Minio, bucket: str, max_bytes: int = MAX_ARTIFACT_BYTES,
                 *, spool_directory=None, read_concurrency=2,
                 read_spool_bytes=256 * 1024 * 1024, spool_bytes=512 * 1024 * 1024,
                 spool_min_free_bytes=MIN_SPOOL_FREE_BYTES):
        self.client, self.bucket, self.max_bytes = client, bucket, max_bytes
        self.spool_directory = spool_directory
        self.spool_budget = SpoolBudget(concurrency=read_concurrency,
            read_bytes=read_spool_bytes, total_bytes=spool_bytes,
            spool_directory=spool_directory, min_free_bytes=spool_min_free_bytes)

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
                    raise ArtifactIntegrityError("artifact checksum/length mismatch")
                checksum.update(chunk)
                if destination is not None:
                    destination.write(chunk)
            if size != artifact.size or checksum.hexdigest() != artifact.sha256:
                raise ArtifactIntegrityError("artifact checksum/length mismatch")
        finally:
            response.close()
            response.release_conn()

    @asynccontextmanager
    async def open_verified(self, artifact: Artifact):
        """Hold admission and disk until the consumer finishes, including cancellation."""
        if artifact.size > self.max_bytes:
            raise ArtifactTooLarge("artifact exceeds configured limit")
        with self.spool_budget.reserve(artifact.size, read=True):
            with TemporaryFile(dir=self.spool_directory) as destination:
                await file_io(self._verify, artifact, destination)
                await file_io(destination.seek, 0)
                yield destination

    async def get(self, artifact: Artifact) -> bytes:
        """Compatibility for small structured callers; binary readers use open_verified."""
        async with self.open_verified(artifact) as source:
            return await file_io(source.read)

    async def read_to_file(self, artifact: Artifact, destination: BinaryIO):
        async with self.open_verified(artifact) as source:
            def copy():
                while chunk := source.read(IO_CHUNK_BYTES):
                    destination.write(chunk)
            await file_io(copy)


async def verify_artifact(artifacts: ArtifactPort, artifact: Artifact):
    """Validate references without materializing a binary payload in memory."""
    if hasattr(artifacts, "open_verified"):
        async with artifacts.open_verified(artifact):
            pass
    else:
        await artifacts.get(artifact)
