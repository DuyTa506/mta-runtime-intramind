import asyncio
import os
from uuid import uuid4

import pytest
from minio import Minio

from intramind_runtime.artifacts import MinioArtifacts

pytestmark = pytest.mark.integration


async def test_minio_roundtrip_checksum_and_tenant_isolation():
    endpoint = os.environ.get("RUNTIME_TEST_MINIO_ENDPOINT")
    if not endpoint:
        pytest.skip("explicit disposable RUNTIME_TEST_MINIO_ENDPOINT required")
    if endpoint not in {"127.0.0.1:19009", "127.0.0.1:19010"}:
        pytest.fail("refusing to mutate an unrecognized MinIO endpoint")
    client = Minio(endpoint, access_key="runtime_test", secret_key="runtime_test_only", secure=False)
    bucket = "runtime-test-"+uuid4().hex
    await asyncio.to_thread(client.make_bucket, bucket)
    blobs = MinioArtifacts(client, bucket)
    try:
        await blobs.ready()
        data = "Kiểm thử checkpoint — tiếng Việt".encode()
        ref = await blobs.put("tenant-a", data, "text/plain; charset=utf-8")
        again = await blobs.put("tenant-a", data, "text/plain; charset=utf-8")
        other = await blobs.put("tenant-b", data, "text/plain; charset=utf-8")
        assert ref == again and ref.key != other.key
        assert await blobs.get(ref) == data
        with pytest.raises(ValueError, match="checksum"):
            await blobs.get(ref.model_copy(update={"sha256": "0"*64}))
        # Exercise MinIO's multipart path with a payload above its part size.
        large = b"payload-"*(1024*1024)
        big = await blobs.put("tenant-a", large)
        assert await blobs.get(big) == large
        response = await asyncio.to_thread(client.get_object, bucket, big.key, offset=3, length=25)
        try:
            assert response.read() == large[3:28]
        finally:
            response.close()
            response.release_conn()
    finally:
        def cleanup():
            for obj in client.list_objects(bucket, recursive=True):
                client.remove_object(bucket, obj.object_name)
            client.remove_bucket(bucket)
        await asyncio.to_thread(cleanup)
