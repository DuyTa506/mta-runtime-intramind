import asyncio
import hashlib
import io
import threading
from contextlib import asynccontextmanager

import httpx
import pytest

from intramind_runtime.api import create_app
from intramind_runtime.artifacts import MinioArtifacts

TOKEN = "isolated-test-credential-only-12345678"
MIB = 1024 * 1024


class ObjectResponse(io.BytesIO):
    def release_conn(self):
        pass


class StreamingMinio:
    def __init__(self):
        self.objects = {}
        self.read_sizes = []
        self.sources = []
        self.corrupt = False

    def put_object(self, bucket, key, data, length, **kwargs):
        self.sources.append(data)
        chunks = []
        while chunk := data.read(64 * 1024):
            chunks.append(chunk)
        body = b"".join(chunks)
        assert len(body) == length
        self.objects[key] = body

    def get_object(self, bucket, key):
        owner = self

        class BoundedResponse(ObjectResponse):
            def read(self, size=-1):
                owner.read_sizes.append(size)
                assert 0 < size <= MIB, "verification must not read an entire audio artifact"
                return super().read(size)

        return BoundedResponse(b"corrupt" if self.corrupt else self.objects[key])


@asynccontextmanager
async def api(blobs, **options):
    app = create_app(None, blobs, TOKEN, {}, **options)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test",
        headers={"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "owner"},
    ) as client:
        yield client


async def payload(size, chunk_size=64 * 1024):
    for start in range(0, size, chunk_size):
        yield b"a" * min(chunk_size, size - start)


async def test_large_wav_upload_is_spooled_and_verified_without_full_body_read():
    backend = StreamingMinio()
    async with api(MinioArtifacts(backend, "test")) as client:
        response = await client.post(
            "/v1/artifacts", content=payload(17 * MIB), headers={"Content-Type": "audio/wav"}
        )
    assert response.status_code == 200, response.text
    ref = response.json()
    assert ref["size"] == 17 * MIB
    assert ref["sha256"] == hashlib.sha256(b"a" * (17 * MIB)).hexdigest()
    assert ref["content_type"] == "audio/wav"
    assert backend.read_sizes and max(backend.read_sizes) <= MIB
    assert backend.sources[0].closed
    assert not isinstance(backend.sources[0], io.BytesIO)


@pytest.mark.parametrize("content_type", ["application/json", "application/problem+json; charset=utf-8"])
async def test_structured_payload_limit_is_not_raised_for_large_audio(content_type):
    backend = StreamingMinio()
    async with api(MinioArtifacts(backend, "test")) as client:
        response = await client.post(
            "/v1/artifacts", content=payload(16 * MIB + 1),
            headers={"Content-Type": content_type},
        )
    assert response.status_code == 413
    assert not backend.objects


async def test_binary_limit_checks_received_bytes_without_content_length():
    backend = StreamingMinio()
    async with api(MinioArtifacts(backend, "test"), artifact_max_bytes=1024) as client:
        response = await client.post(
            "/v1/artifacts", content=payload(1025, 128), headers={"Content-Type": "audio/wav"}
        )
    assert response.status_code == 413
    assert not backend.objects


async def test_declared_oversize_and_unauthorized_upload_do_not_consume_body():
    backend = StreamingMinio()

    async def unreadable():
        pytest.fail("a rejected upload must not consume the body")
        yield b"unreachable"

    async with api(MinioArtifacts(backend, "test"), artifact_max_bytes=1024) as client:
        response = await client.post(
            "/v1/artifacts", content=unreadable(),
            headers={"Content-Type": "audio/wav", "Content-Length": "1025"},
        )
        assert response.status_code == 413
        response = await client.post(
            "/v1/artifacts", content=unreadable(), headers={"Authorization": "Bearer wrong"}
        )
        assert response.status_code == 401
    assert not backend.objects


async def test_upload_concurrency_rejects_excess_work_before_reading_body():
    backend = StreamingMinio()
    receiving, proceed = asyncio.Event(), asyncio.Event()

    async def slow_upload():
        receiving.set()
        await proceed.wait()
        yield b"first"

    async def unreadable():
        pytest.fail("busy upload capacity must reject before consuming the body")
        yield b"unreachable"

    async with api(MinioArtifacts(backend, "test"), artifact_upload_concurrency=1) as client:
        first = asyncio.create_task(client.post("/v1/artifacts", content=slow_upload()))
        try:
            await asyncio.wait_for(receiving.wait(), 2)
            busy = await client.post("/v1/artifacts", content=unreadable())
            assert busy.status_code == 503
            assert busy.headers["retry-after"]
        finally:
            proceed.set()
            done = await asyncio.wait_for(first, 5)
        assert done.status_code == 200
        assert (await client.post("/v1/artifacts", content=b"next")).status_code == 200


async def test_disconnected_upload_never_publishes_partial_object():
    backend = StreamingMinio()

    async def interrupted():
        yield b"partial"
        raise ConnectionError("client disconnected")

    async with api(MinioArtifacts(backend, "test"), artifact_upload_concurrency=1) as client:
        with pytest.raises(ConnectionError, match="disconnected"):
            await client.post("/v1/artifacts", content=interrupted())
        assert not backend.objects
        assert (await client.post("/v1/artifacts", content=b"retry")).status_code == 200
    assert all(source.closed for source in backend.sources)


async def test_cancelled_upload_keeps_file_and_capacity_until_storage_thread_finishes():
    backend = StreamingMinio()
    started, proceed = threading.Event(), threading.Event()
    original = backend.put_object

    def blocked_put(*args, **kwargs):
        started.set()
        assert proceed.wait(5)
        original(*args, **kwargs)

    backend.put_object = blocked_put
    async with api(MinioArtifacts(backend, "test"), artifact_upload_concurrency=1) as client:
        first = asyncio.create_task(client.post("/v1/artifacts", content=b"first"))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            first.cancel()
            await asyncio.sleep(0)
            assert not first.done()
            busy = await client.post("/v1/artifacts", content=b"second")
            assert busy.status_code == 503
        finally:
            proceed.set()
            with pytest.raises(asyncio.CancelledError):
                await first
        assert all(source.closed for source in backend.sources)
        assert (await client.post("/v1/artifacts", content=b"retry")).status_code == 200


async def test_minio_file_put_enforces_limit_before_object_write_and_detects_corruption():
    backend = StreamingMinio()
    blobs = MinioArtifacts(backend, "test", max_bytes=10)
    with pytest.raises(ValueError, match="limit"):
        await blobs.put_file("t", io.BytesIO(b"a" * 11), "audio/wav")
    assert not backend.objects
    backend.corrupt = True
    with pytest.raises(ValueError, match="checksum|length"):
        await blobs.put_file("t", io.BytesIO(b"audio"), "audio/wav")


async def test_file_upload_retry_has_stable_identity_and_preserves_tenant_isolation():
    backend = StreamingMinio()
    blobs = MinioArtifacts(backend, "test")
    first = await blobs.put_file("a", io.BytesIO(b"audio"), "audio/wav")
    again = await blobs.put_file("a", io.BytesIO(b"audio"), "audio/wav")
    other = await blobs.put_file("b", io.BytesIO(b"audio"), "audio/wav")
    assert first == again
    assert other.key != first.key
