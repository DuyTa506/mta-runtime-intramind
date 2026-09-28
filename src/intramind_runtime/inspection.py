"""Read-only batched engine status without importing the Temporal SDK."""
import argparse
import asyncio
import json
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

SQL = """
SELECT p.pool_id,p.engine_epoch,p.health,p.target,
       p.quiesce_token IS NOT NULL AS quiesce_owned,
       COALESCE(a.known_inflight,0) AS known_inflight,
       COALESCE(a.unknown_inflight,0) AS unknown_inflight,
       COALESCE(a.unsettled_inflight,0) AS unsettled_inflight
FROM runtime_pools p LEFT JOIN (
    SELECT pool_id,
        count(*) FILTER (WHERE state<>'UNKNOWN' AND compute_held) AS known_inflight,
        count(*) FILTER (WHERE state='UNKNOWN' AND compute_held) AS unknown_inflight,
        count(*) FILTER (WHERE compute_held OR budget_held) AS unsettled_inflight
    FROM runtime_inflight_attempts WHERE pool_id=ANY(CAST(:pools AS text[]))
    GROUP BY pool_id
) a USING(pool_id)
WHERE p.pool_id=ANY(CAST(:pools AS text[]))
ORDER BY p.pool_id
"""


async def inspect_engines(engine, pool_ids):
    ids = list(dict.fromkeys(pool_ids))
    if not ids or len(ids) > 64 or any(not isinstance(i, str) or not i or len(i) > 200 for i in ids):
        raise ValueError('One to64 valid pool IDs required')
    async with engine.connect() as connection:
        async with connection.begin():
            await connection.execute(text('SET TRANSACTION READ ONLY'))
            await connection.execute(text("SET LOCAL statement_timeout = '5000ms'"))
            result = await connection.execute(text(SQL), {'pools': ids})
            report = {row['pool_id']: dict(row) for row in result.mappings()}
            if set(report) != set(ids):
                raise ValueError('Configured pool not found')
            return report


async def execute(args):
    engine = create_async_engine(os.environ['RUNTIME_DATABASE_URL'])
    try:
        print(json.dumps({'pools': await inspect_engines(engine, args.pool)}, sort_keys=True))
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', action='append', required=True)
    asyncio.run(execute(parser.parse_args()))


if __name__ == '__main__':
    main()
