# Intramind Runtime

[Repository](https://github.com/DuyTa506/mta-runtime-intramind) · development branch: `dev`.
The package and Intramind migration are in active development; no production release has been declared.

A Python package for durable agent workflows and shared inference admission.
Application code declares the next step; the runtime accounts for inference
attempts; Temporal persists workflow decisions; the serving engine executes them.

## Boundaries

| Repository/component | Owns |
| --- | --- |
| This package | Task facade, operation ledger, admission, budgets, adapters and runtime services |
| Consuming application | Prompts, validators, workflow steps, source access and rendering |
| Infrastructure | Temporal/PostgreSQL deployment, MinIO, credentials, engine placement and capacity profiles |

The package is maintained independently and consumed as a versioned wheel.
It does not vendor AI/BE application code or start application services on import.
The existing MinIO artifact backend remains supported; installing this library
neither replaces storage nor migrates buckets.

## Components

| Module | Responsibility |
| --- | --- |
| `sdk` | Durable activities, LLM operations, child/map/wait and Continue-As-New |
| `client` / `api` | Authenticated submission, status, cancellation and artifact access |
| `temporal_adapter` | Idempotent submit/attach, asynchronous completion and outbox delivery |
| `store` | Atomic tenant/root admission, attempts, reservations, fencing and budgets |
| `executor` / `drivers` | One transport attempt and separate result persistence |
| `artifacts` | Tenant-scoped immutable objects and checksum verification |
| `admin` | Namespace and pinned worker deployment administration |
| `controller` | Pure feedback policy, requiring installation-specific telemetry/profiles |
| `release_gate` | Application-supplied migration and qualification evidence validation |

UNKNOWN reservations remain accounted for until compute termination is known.
No DB transaction spans inference. The completion driver does not promise
reliable remote cancellation, backend status lookup or exactly-once inference.

## Install and use

Use Python 3.12. Applications install the release wheel from the maintainer's
release artifacts or internal package index and pin its version and hash.

```bash
python -m pip install /release/intramind_runtime-0.1.0-py3-none-any.whl
```

Declare features with `@durable_task` and call `TaskContext.activity`, `llm`,
`run_child`, `map_children`, `sleep` and `wait_event`. See
[the example workflow](src/intramind_runtime/example_workflow.py).
Feature modules do not import Temporal primitives directly. Keep HTTP/database
client initialization in activity modules, outside replayable workflow imports.
Use stable item keys and immutable artifacts; paginate large plans rather than
embedding source documents or unbounded child lists in workflow history.

Service commands are `intramind-runtime api|worker|executor|outbox|reconciler|configure`.
They require explicit `RUNTIME_*` configuration. An application installs matching
schema migrations and provisions its namespace, storage bucket and pools before
starting workers. The API never initializes tables at startup.

## Development and verification

Python 3.12, `uv` and Make are sufficient for unit tests. Docker Compose v2 is
needed for the isolated integration services. No sibling application checkout,
GPU, production credentials or existing Intramind deployment is required.

```bash
git clone --branch dev https://github.com/DuyTa506/mta-runtime-intramind.git
cd mta-runtime-intramind
make setup check test
make dev-up
make integration
make build
make dev-down
```

`dev-up` creates only the `intramind-runtime-dev` Compose project. `dev-down`
retains its data volumes. The development credentials in `dev/` are fixtures
for loopback services, and must not be used for a shared deployment.

| Local service | Address |
| --- | --- |
| PostgreSQL | `127.0.0.1:55440` |
| Temporal | `127.0.0.1:17234` |
| Temporal UI | `http://127.0.0.1:18088` |
| MinIO API / console | `127.0.0.1:19010` / `http://127.0.0.1:19011` |

Tests use a dedicated `runtime_test` database, a test namespace, unique
workflow queues and temporary buckets. They never reset the development
`runtime_dev` database. Run `make integration` serially against a given test DB.
Application feature tests stay opt-in in the consuming application's environment.

The MinIO image is a compatibility test fixture pinned to the existing installation's
binary, fetched from the official Quay mirror. It is not a production version recommendation:
MinIO published a later [security release](https://github.com/minio/minio/releases/tag/RELEASE.2025-10-15T17-29-55Z)
with source-build instructions. Production storage upgrades and recovery validation belong
to the infrastructure release and do not run as part of this development setup.

The integration baseline is PostgreSQL 17.11, Temporal server 1.32.0 and Python
SDK 1.33.0. The lockfile pins Python dependencies. Full integration tests use
explicitly configured disposable services and reset the test database schema:

```bash
RUNTIME_TEST_DATABASE_URL=postgresql+asyncpg://runtime_test:runtime_test_only@127.0.0.1:55439/runtime_test \
RUNTIME_TEST_ALLOW_RESET=yes \
RUNTIME_TEST_TEMPORAL_ADDRESS=127.0.0.1:17233 \
RUNTIME_TEST_MINIO_ENDPOINT=127.0.0.1:19009 \
.venv/bin/python -m pytest -q --junitxml=.test-data/runtime-tests.xml
```

Missing external endpoints are explicit skips. Cross-application feature tests
also require their dependency environment and opt-in. CI/library tests do not
certify a GPU configuration, disaster recovery or application cutover.

`config/pools.example.json` intentionally registers no deployment capacity.
An installation supplies measured context, concurrency, model/template and
resource-group profiles. The pure controller is not automatically enabled.
`config/migration-status.json` is the Intramind migration checklist snapshot;
other consumers supply their own evidence to `intramind_runtime.release_gate`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contract, compatibility and release rules,
and [CHANGELOG.md](CHANGELOG.md) for the unreleased package contents.
