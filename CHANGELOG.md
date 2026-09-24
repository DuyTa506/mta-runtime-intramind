# Changelog

## 0.2.0rc17 — unreleased

- Remove a direct waiter if its enqueue committed but the caller was cancelled
  before a producer could take ownership of cleanup.

## 0.2.0rc16 — unreleased

- Use PostgreSQL's committed waiter timestamp to prevent direct endpoint
  dispatch starvation when concurrent enqueues return out of order.
- Add an explicit mock-serving benchmark for concurrent QA with a bounded
  background queue; it is separate from default tests.

## 0.2.0rc15 — unreleased

- Migration `0006` adds process owner leases, endpoint-scoped admission and bounded
  direct waiters without a global lock on every inference grant.
- Trusted QA/workload routing, shared lower-class transport caps and SSE
  wait/recovery controls let chat wait its turn without silent inference retry.
- Token-fenced host quiesce and structured stopped-epoch proof authorize bounded
  generation recovery; unknown compute remains held until proof.

## 0.2.0rc6 — unreleased

- `TaskContext.embedding_outcome` exposes only confirmed terminal backend failures
  to an application's accepted fallback. Budget, revision/vector corruption,
  deadline, cancellation and uncertain compute still propagate through recovery.

## 0.2.0rc5 — unreleased

- Embedding operations share admission, root identities and attempt limits while
  keeping a separate character budget. Migration `0004` retains existing reservations.
- Pinned embedding profiles bound batches and validate model revision, dimension,
  finite vectors and response bytes. Timeout/unconfirmed compute remains UNKNOWN.
- `TaskContext.embedding`, service preparation and broker completion keep vectors
  through persistence retries, downstream failure, Continue-As-New and history replay.

## 0.1.0 — unreleased

- Bounded file-based artifact uploads support 128 MiB binary results while retaining
  the 16 MiB JSON limit, with incremental checksum verification and cancellation cleanup.
- Speech operations with independent character budgets, shared root/group accounting,
  bounded WAV artifacts and a serving termination contract; migration `0002` retains
  legacy token reservations. Audio integration covers confirmed fallback and UNKNOWN cancellation.
- Per-session test namespaces avoid accumulating worker deployments against one namespace's limit.
- Accepted configuration artifacts retained across retries, children and rollover.
- Per-feature child windows bounded by the workflow's hard cap.
- Total attempt deadlines derived from the ledger, preserving UNKNOWN accounting
  after send and retaining completed output through persistence failures.
- Durable task facade for activities, LLM operations, child workflows and waits.
- Stable identities and root accounting across children and Continue-As-New.
- PostgreSQL operation, attempt, reservation and budget ledger.
- Temporal asynchronous completion and durable outbox delivery.
- Immutable artifact adapter for existing MinIO/S3 storage.
- Admission, cancellation, timeout reconciliation and graceful executor drain.
- Authenticated service API, client and worker deployment administration.
- Registered pure model continuations with request-digest replay checks and bounded repair calls.
- Explicit structured-output and tool-call sizing profiles; tool schemas and transcript results
  are included in chat-template token accounting. Profiles remain deployment-qualified inputs.
- Independent development setup on `dev`, with pinned disposable PostgreSQL/Temporal/MinIO,
  Make targets, real integration CI and wheel installation checks.

The first release is being integrated into Intramind. Feature migrations and
deployment qualification are tracked by the application, not inferred from this
package version.
