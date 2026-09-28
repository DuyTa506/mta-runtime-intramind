import httpx
import pytest

from intramind_runtime.direct_client import DirectBinding, current_inference_tenant, inference_scope

PROFILE = dict(model_profile='embedding',capacity_profile_id='qualified-v1',model='embed',
    model_revision='revision1',dimension=2,max_batch_size=32,character_limit=262144,
    max_text_characters=32768,max_response_bytes=16777216)


@pytest.mark.asyncio
async def test_profile_uses_verified_scope_and_reflects_revision_changes():
    calls=[]
    def handle(request):
        calls.append(request)
        return httpx.Response(200,json=PROFILE | {'model_revision':f'revision{len(calls)}'})
    binding=DirectBinding('http://runtime.test','x'*32,'embedding')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError,match='verified inference identity'):
            await binding.embedding_profile(client=client)
        with inference_scope('tenant-a'):
            assert current_inference_tenant()=='tenant-a'
            assert (await binding.embedding_profile(client=client)).model_revision=='revision1'
            assert (await binding.embedding_profile(client=client)).model_revision=='revision2'
        assert current_inference_tenant() is None
    assert len(calls)==2
    assert all(r.headers['x-tenant-id']=='tenant-a' for r in calls)
    assert all(str(r.url)=='http://runtime.test/v1/embedding/profiles/embedding' for r in calls)
