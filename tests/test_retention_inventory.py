from contextlib import asynccontextmanager
from datetime import datetime, timezone
import pytest
from intramind_runtime.retention_inventory import inventory, COUNTS


@pytest.mark.asyncio
async def test_inventory_has_no_deletion_authority_even_for_old_rows():
    queries=[]
    class Result:
        def mappings(self): return self
        def one(self): return {'aged_terminal':100}
    class Connection:
        @asynccontextmanager
        async def begin(self): yield
        async def execute(self, query, params=None):
            queries.append(str(query));return Result()
    class Engine:
        @asynccontextmanager
        async def connect(self):yield Connection()
    result=await inventory(Engine(),now=datetime(2026,9,28,tzinfo=timezone.utc))
    assert result['delete_authorized'] is False and result['object_deletion_candidates'] is None
    assert set(result['tables'])==set(COUNTS)
    assert queries[0].endswith('READ ONLY')
    assert all(q.startswith(('SELECT','SET')) for q in queries)
    assert result['cutoff'].startswith('2026-08-29')


def test_object_inventory_streams_counts_without_classifying_deletions():
    from types import SimpleNamespace
    from intramind_runtime.retention_inventory import object_inventory
    now=datetime(2026,9,28,tzinfo=timezone.utc)
    old=datetime(2020,1,1,tzinfo=timezone.utc)
    class Client:
        def list_objects(self,bucket,**kwargs):
            assert kwargs=={'prefix':'tenants/','recursive':True}
            yield SimpleNamespace(size=10,last_modified=old)
            yield SimpleNamespace(size=20,last_modified=now)
    result=object_inventory(Client(),'fixture',now)
    assert result['bytes']==30 and result['aged_bytes']==10
    assert result['objects']==2 and result['deletion_candidates'] is None
