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


@pytest.mark.asyncio
async def test_real_postgres_aggregates_preserve_known_unknown_and_held_counts():
    import os
    from urllib.parse import urlparse
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    url=os.environ.get('PERF_INSPECTION_POSTGRES_DSN')
    if not url:pytest.skip('isolated PostgreSQL DSN required')
    parsed=urlparse(url)
    if (parsed.hostname,parsed.port)!=('127.0.0.1',15438):pytest.skip('only isolated fixture port allowed')
    engine=create_async_engine(url.replace('postgresql://','postgresql+asyncpg://',1),pool_size=1,max_overflow=0,hide_parameters=True)
    try:
        async with engine.begin() as c:
            await c.execute(text('CREATE TEMP TABLE runtime_pools(pool_id text,engine_epoch text,health text,target int,quiesce_token text)'))
            await c.execute(text('CREATE TEMP TABLE runtime_inflight_attempts(pool_id text,state text,compute_held boolean,budget_held boolean)'))
            await c.execute(text("INSERT INTO runtime_pools VALUES ('a','epoch','HEALTHY',1,NULL),('b','epoch','DEGRADED',0,'owner')"))
            await c.execute(text("INSERT INTO runtime_inflight_attempts VALUES ('a','ACTIVE',true,true),('a','UNKNOWN',true,false),('a','FINISHED',false,true),('b','FINISHED',false,false)"))
        report=await inspect_engines(engine,['a','b'])
        assert (report['a']['known_inflight'],report['a']['unknown_inflight'],report['a']['unsettled_inflight'])==(1,1,3)
        assert report['b']['unsettled_inflight']==0 and report['b']['quiesce_owned'] is True
    finally:await engine.dispose()
