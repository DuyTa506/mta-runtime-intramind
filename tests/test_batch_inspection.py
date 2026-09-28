import subprocess
import sys
from contextlib import asynccontextmanager
import pytest
from intramind_runtime.inspection import inspect_engines


@pytest.mark.asyncio
async def test_batch_uses_one_readonly_connection_and_one_metadata_query():
    calls=[]
    class Result:
        def mappings(self): return [{'pool_id':'a','known_inflight':2},{'pool_id':'b','known_inflight':0}]
    class Connection:
        @asynccontextmanager
        async def begin(self): yield
        async def execute(self, sql, params=None):
            calls.append((str(sql),params)); return Result()
    class Engine:
        connects=0
        @asynccontextmanager
        async def connect(self): self.connects+=1;yield Connection()
    engine=Engine()
    report=await inspect_engines(engine,['a','b','a'])
    assert engine.connects==1 and len(calls)==3
    assert calls[0][0]=='SET TRANSACTION READ ONLY'
    assert calls[-1][1]=={'pools':['a','b']}
    assert report['a']['known_inflight']==2
    assert 'spec' not in calls[-1][0] and 'task_token' not in calls[-1][0]
    with pytest.raises(ValueError,match='not found'): await inspect_engines(engine,['a','missing'])


def test_inspection_import_does_not_load_temporal_sdk():
    subprocess.run([sys.executable,'-c',
        'import sys;import intramind_runtime.inspection;assert not any(k.startswith("temporalio") for k in sys.modules)'],check=True)
