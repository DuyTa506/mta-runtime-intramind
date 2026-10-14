# Architecture

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
| `buffering` | Bounded durable inputs, accepted policy, delayed batches and partition serialization |
| `executor` / `drivers` | One transport attempt and separate result persistence |
| `speech` | Qualified TTS preparation, termination contract and bounded WAV transport |
| `embedding` | Pinned document/query batches, vector validation and independent input accounting |
| `direct` / `direct_proxy` | Shared compute reservations and direct HTTP streaming without Temporal |
| `memory_scheduler` / `permits` | Runtime-api endpoint owner, RAM direct queue and service-token permits for durable executors |
| `artifacts` | Tenant-scoped immutable objects and checksum verification |
| `uploads` | Bounded temporary files and concurrency for artifact ingestion |
| `admin` | Namespace and pinned worker deployment administration |
| `controller` | Pure feedback policy, requiring installation-specific telemetry/profiles |
| `release_gate` | Application-supplied migration and qualification evidence validation |

UNKNOWN reservations remain accounted for until compute termination is known.
Termination is known when the epoch is confirmed stopped, or when two consecutive
runtime-api probes of the same epoch prove the attempt is no longer computing:

| Pool kind | Probe | Proves |
|---|---|---|
| `llm` | llama.cpp `/slots` | every UNKNOWN attempt, when every slot is idle |
| `embedding` · `rerank` · `speech` | serving `GET /api/v1/attempts/{kind}` | each UNKNOWN attempt absent from `inflight`, both probes from one `boot_id` |

Attempts recorded UNKNOWN before the first probe are then settled
(`error_class=engine_idle_verified`), durable work is requeued, RAM direct
permits are released and their callers may retry once on the same epoch. A
serving without the attempts endpoint (404) or an unreachable engine proves
nothing. Pool health and target stay with their owner (watchdog or operator).
`intramind_runtime_direct_unknown_oldest_seconds{pool}` exports the age of the
oldest unsettled RAM direct permit.
A durable permit stays held from grant through `SEND_INTENT` (a `RESERVED`
attempt holds no compute yet but its permit covers the send). The executor
retries the ledger write that ends an attempt until it lands: a live executor
keeps its lease, so an unwritten end is never reconciled by anyone else. After
4 inline tries (1/2/4 s backoff) the retry moves to a background task, capped at
30 s between tries, so the executor slot is freed and the end still lands.
A result that can no longer be committed does not wait for a probe. Once compute
has ended and nothing can commit the output (`BACKEND_FINISHED` under an owner that
lost its lease, or `UNKNOWN` with no compute held), the reconciler fails the attempt
with `error_class=result_lost_after_backend_finished`, charges the bound as
estimated usage and requeues the operation
(`wait_reason=result_lost_after_backend_finished`) within its attempt, budget and
deadline limits, else fails it. The late executor is fenced by its next ledger
write; its in-memory result is discarded.
No DB transaction spans inference. The completion driver does not promise
reliable remote cancellation, backend status lookup or exactly-once inference.
