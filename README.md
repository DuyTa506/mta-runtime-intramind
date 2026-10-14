# Intramind Runtime

[Repository](https://github.com/DuyTa506/mta-runtime-intramind) · development branch: `dev`.
The package and Intramind migration are in active development; no production release has been declared.

A Python package for durable agent workflows and shared inference admission.
Application code declares the next step; the runtime accounts for inference
attempts; Temporal persists workflow decisions; the serving engine executes them.
Current version: `0.2.0rc20` (release candidate).

## Roles

`intramind-runtime <role>` starts one process. Each role needs explicit `RUNTIME_*` settings.

| Role | Does |
| --- | --- |
| `api` | Authenticated submit/status/cancel, artifacts, direct inference and capacity owner |
| `worker` | Temporal broker activities, pinned to `RUNTIME_BUILD_ID` |
| `executor` | Sends one admitted attempt to the engine and persists the result |
| `outbox` | Delivers committed submissions to Temporal and seals buffered batches |
| `reconciler` | Settles expired leases and UNKNOWN attempts |
| `configure` | Writes the pools from `RUNTIME_POOL_CONFIG` into the ledger, then exits |

## Quick start

1. Use Python 3.12, `uv` and Make.
2. Install dependencies: `make setup`.
3. Run lint and unit tests: `make check test`.
4. Start disposable PostgreSQL/Temporal/MinIO: `make dev-up`.
5. Run integration tests: `make integration`.
6. Build the wheel: `make build`.
7. Stop services (volumes are kept): `make dev-down`.

Applications install the pinned release wheel, not a Git branch:

```bash
uv pip install /release/intramind_runtime-0.2.0rc10-py3-none-any.whl
```

Before starting workers: apply migrations (`python -m alembic upgrade head`),
then provision the namespace, storage bucket and pools. The API never creates tables.

## Commands

| Task | Command |
| --- | --- |
| Lint | `make check` |
| Unit tests (no services) | `make test` |
| Integration tests (PostgreSQL on 55440) | `make dev-up && make integration` |
| Build wheel | `make build` |
| Run a role | `intramind-runtime api\|worker\|executor\|outbox\|reconciler\|configure` |
| Build image | `scripts/build-image.sh` |
| Offline bundle | `deploy/04-make-bundle.sh --out /path/bundle` |

## Docs

| File | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Boundaries, components, UNKNOWN settlement |
| [docs/contracts.md](docs/contracts.md) | Night warehouse admission; install and use, per-version contracts |
| [docs/development.md](docs/development.md) | Local services, integration tests, language and artifact-download contracts |
| [docs/operations.md](docs/operations.md) | Rollout gate, image build and offline bundle |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contract, compatibility and release rules |
| [CHANGELOG.md](CHANGELOG.md) | Unreleased package contents |
