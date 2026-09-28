import asyncio
import io
import threading

import httpx
import pytest
from test_artifact_uploads import TOKEN, StreamingMinio, api

from intramind_runtime.api import create_app
from intramind_runtime.artifacts import ArtifactCapacityBusy, MinioArtifacts
from intramind_runtime.client import RuntimeClient


async def fixture_blob(**kwargs):
    backend = StreamingMinio()
    blobs = MinioArtifacts(backend, "test", **kwargs)
    ref = await blobs.put("owner", b"verified payload", "application/octet-stream")
    return backend, blobs, ref


async def test_disk_file_reservation_covers_slow_consumer_and_internal_reads(tmp_path):
    backend, blobs, ref = await fixture_blob(spool_directory=tmp_path, read_concurrency=1)
    async with blobs.open_verified(ref) as source:
        assert not isinstance(source, io.BytesIO)
        assert source.read(1) == b"v"
        assert blobs.spool_budget.read_bytes == ref.size
        with pytest.raises(ArtifactCapacityBusy):
            await blobs.get(ref)
        assert source.read() == b"erified payload"
    assert source.closed
    assert blobs.spool_budget.total_bytes == 0
    assert await blobs.get(ref) == b"verified payload"


async def test_corrupt_download_exposes_no_body_and_releases_budget():
    backend, blobs, ref = await fixture_blob()
    backend.corrupt = True
    destination = io.BytesIO()
    with pytest.raises(ValueError, match="checksum/length"):
        await blobs.read_to_file(ref, destination)
    assert destination.getvalue() == b""
    assert blobs.spool_budget.total_bytes == 0
    app = create_app(None, blobs, TOKEN, {})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app, raise_app_exceptions=False),
        base_url="http://test", headers={"Authorization": f"Bearer {TOKEN}",
                                       "X-Tenant-ID": "owner"}) as client:
        response = await client.post("/v1/artifacts/read", json=ref.model_dump())
        assert response.json() == {"detail": "artifact integrity verification failed"}
        backend.corrupt = False
        recovered = await client.post("/v1/artifacts/read", json=ref.model_dump())
        assert recovered.status_code == 200 and recovered.content == b"verified payload"
    assert response.status_code == 500
    assert b"corrupt" not in response.content


async def test_shared_spool_budget_rejects_upload_and_read_before_allocation():
    backend, blobs, ref = await fixture_blob(spool_bytes=20)
    async with blobs.open_verified(ref):
        async with api(blobs) as client:
            busy = await client.post("/v1/artifacts", content=b"x" * 10)
            assert busy.status_code == 503
            assert busy.headers["retry-after"] == "1"
            busy = await client.post("/v1/artifacts/read", json=ref.model_dump())
            assert busy.status_code == 503
    assert blobs.spool_budget.total_bytes == 0


async def test_cancelled_read_drains_blocking_thread_before_releasing_capacity():
    backend, blobs, ref = await fixture_blob(read_concurrency=1)
    started, proceed = threading.Event(), threading.Event()
    original = backend.get_object

    def blocked(*args):
        started.set()
        assert proceed.wait(5)
        return original(*args)

    backend.get_object = blocked
    task = asyncio.create_task(blobs.get(ref))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # Repeated cancellation must not release a live thread's file.
        await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(ArtifactCapacityBusy):
            await blobs.get(ref)
    finally:
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert blobs.spool_budget.total_bytes == 0
    assert await blobs.get(ref) == b"verified payload"


async def test_http_lifetime_holds_budget_until_last_send_and_disconnect():
    _, blobs, ref = await fixture_blob(read_concurrency=1)
    app = create_app(None, blobs, TOKEN, {})
    body = ref.model_dump_json().encode()
    blocked, proceed = asyncio.Event(), asyncio.Event()
    received = False
    messages = []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body}
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            blocked.set()
            await proceed.wait()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
             "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/artifacts/read", "raw_path": b"/v1/artifacts/read",
             "query_string": b"", "headers": [(b"authorization", f"Bearer {TOKEN}".encode()),
                 (b"x-tenant-id", b"owner"), (b"content-type", b"application/json")],
             "server": ("test", 80), "client": ("test", 123)}
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(blocked.wait(), 3)
        assert blobs.spool_budget.reads == 1
        with pytest.raises(ArtifactCapacityBusy):
            await blobs.get(ref)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        proceed.set()
    assert blobs.spool_budget.total_bytes == 0
    assert messages[0]["status"] == 200


async def test_client_file_api_preserves_wire_and_verifies_before_copy(tmp_path):
    _, blobs, ref = await fixture_blob()
    async with api(blobs) as http:
        client = RuntimeClient("unused", TOKEN, "owner", client=http, spool_directory=tmp_path)
        output = io.BytesIO()
        await client.read_to_file(ref.model_dump(), output)
        assert output.getvalue() == b"verified payload"
        assert await client.read_bytes(ref.model_dump()) == output.getvalue()
        forbidden = ref.model_copy(update={"key": "tenants/other/object"})
        response = await http.post("/v1/artifacts/read", json=forbidden.model_dump())
        assert response.status_code == 404
    assert client.spool_budget.total_bytes == 0


async def test_disk_full_releases_budget_and_connection(monkeypatch):
    import intramind_runtime.artifacts as module
    _, blobs, ref = await fixture_blob()

    class FullDisk(io.BytesIO):
        def write(self, data):
            raise OSError(28, "No space left on device")

    staged = FullDisk()
    monkeypatch.setattr(module, "TemporaryFile", lambda **kwargs: staged)
    with pytest.raises(OSError, match="No space"):
        await blobs.get(ref)
    assert staged.closed
    assert blobs.spool_budget.total_bytes == 0


@pytest.mark.parametrize("mib", [16, 64, 128])
async def test_large_generated_objects_use_bounded_chunks(tmp_path, mib):
    from hashlib import sha256

    from intramind_runtime.contracts import Artifact

    chunk = b"a" * (64 * 1024)
    checksum = sha256()
    for _ in range(mib * 16):
        checksum.update(chunk)

    class GeneratedResponse:
        remaining = mib * 1024 * 1024
        closed = released = False

        def read(self, size):
            assert size == 64 * 1024
            count = min(size, self.remaining)
            self.remaining -= count
            return chunk[:count]

        def close(self):
            self.closed = True

        def release_conn(self):
            self.released = True

    response = GeneratedResponse()

    class GeneratedMinio:
        def get_object(self, *_):
            return response

    blobs = MinioArtifacts(GeneratedMinio(), "test", spool_directory=tmp_path)
    ref = Artifact(key="test", sha256=checksum.hexdigest(), size=mib * 1024 * 1024)
    async with blobs.open_verified(ref) as source:
        assert source.seek(0, io.SEEK_END) == ref.size
        assert blobs.spool_budget.read_bytes == ref.size
    assert response.closed and response.released
    assert blobs.spool_budget.total_bytes == 0


async def test_read_byte_budget_independent_of_combined_budget():
    _, blobs, ref = await fixture_blob(read_spool_bytes=20, spool_bytes=100)
    async with blobs.open_verified(ref):
        with pytest.raises(ArtifactCapacityBusy):
            async with blobs.open_verified(ref):
                pytest.fail("read byte budget must reject")
        assert blobs.spool_budget.total_bytes == ref.size
    assert blobs.spool_budget.total_bytes == 0


async def test_client_rejects_same_length_corruption_before_copy():
    _, _, ref = await fixture_blob()
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"x" * ref.size)),
            base_url="http://test") as http:
        client = RuntimeClient("unused", TOKEN, "owner", client=http)
        destination = io.BytesIO()
        with pytest.raises(ValueError, match="checksum/length"):
            await client.read_to_file(ref.model_dump(), destination)
        assert destination.getvalue() == b""
        assert client.spool_budget.total_bytes == 0


async def test_response_construction_failure_releases_open_file():
    _, blobs, ref = await fixture_blob()
    invalid_header = ref.model_copy(update={"content_type": "application/\u2603"})
    async with api(blobs) as client:
        with pytest.raises(UnicodeEncodeError):
            await client.post("/v1/artifacts/read", json=invalid_header.model_dump())
    assert blobs.spool_budget.total_bytes == 0


async def test_client_put_file_streams_and_preserves_caller_ownership():
    _, blobs, _ = await fixture_blob()
    async with api(blobs) as http:
        client = RuntimeClient("unused", TOKEN, "owner", client=http)
        class BoundedFile(io.BytesIO):
            def read(self, size=-1):
                assert size == 64 * 1024
                return super().read(size)
        source = BoundedFile(b"upload" * 100000)
        result = await client.put_file(source, content_type="application/octet-stream")
        assert result["size"] == 600000
        assert not source.closed
        assert await client.read_bytes(result) == b"upload" * 100000


async def test_client_put_file_rejects_unverified_upload_receipt():
    from hashlib import sha256
    async def wrong(request):
        await request.aread()
        return httpx.Response(200, json={"key": "ref", "size": 3,
            "sha256": sha256(b"bad").hexdigest()})
    async with httpx.AsyncClient(transport=httpx.MockTransport(wrong), base_url="http://test") as http:
        client = RuntimeClient("unused", TOKEN, "owner", client=http)
        with pytest.raises(ValueError, match="receipt checksum/length"):
            await client.put_file(io.BytesIO(b"good"), content_type="application/octet-stream")


async def test_http_disk_exhaustion_is_retryable_before_response_headers(monkeypatch):
    _, blobs, ref = await fixture_blob()
    original = blobs._verify
    def no_space(*args):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(blobs, "_verify", no_space)
    app = create_app(None, blobs, TOKEN, {})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test",
        headers={"Authorization":f"Bearer {TOKEN}","X-Tenant-ID":"owner"}) as client:
        response = await client.post("/v1/artifacts/read", json=ref.model_dump())
        assert response.status_code == 503 and response.headers['retry-after'] == '1'
        assert blobs.spool_budget.total_bytes == 0
        monkeypatch.setattr(blobs, "_verify", original)
        response = await client.post("/v1/artifacts/read", json=ref.model_dump())
        assert response.status_code == 200
