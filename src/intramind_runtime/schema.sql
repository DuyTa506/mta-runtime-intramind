CREATE TABLE runtime_authority (
    id integer PRIMARY KEY CHECK (id = 1),
    dispatch_clock bigint NOT NULL DEFAULT 0,
    max_roots integer NOT NULL CHECK (max_roots > 0),
    max_pending integer NOT NULL CHECK (max_pending > 0)
);
INSERT INTO runtime_authority VALUES (1, 0, 128, 2048);
CREATE TABLE runtime_roots (
    root_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    spec jsonb NOT NULL,
    state text NOT NULL DEFAULT 'RUNNING',
    deadline timestamptz NOT NULL,
    budget_limit bigint NOT NULL CHECK (budget_limit > 0),
    reserved bigint NOT NULL DEFAULT 0 CHECK (reserved >= 0),
    spent bigint NOT NULL DEFAULT 0 CHECK (spent >= 0),
    attempts integer NOT NULL DEFAULT 0,
    operation_count integer NOT NULL DEFAULT 0,
    last_served bigint NOT NULL DEFAULT 0,
    priority text NOT NULL,
    cancel_requested boolean NOT NULL DEFAULT false,
    result jsonb,
    terminal_reason text,
    finished_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX runtime_root_tenant ON runtime_roots(tenant_id);
CREATE TABLE runtime_groups (
    group_id text PRIMARY KEY,
    hard_ceiling integer NOT NULL CHECK (hard_ceiling > 0),
    health text NOT NULL DEFAULT 'HEALTHY'
);
CREATE TABLE runtime_pools (
    pool_id text PRIMARY KEY,
    group_id text NOT NULL REFERENCES runtime_groups,
    spec jsonb NOT NULL,
    engine_epoch text NOT NULL,
    envelope_version bigint NOT NULL DEFAULT 1,
    target integer NOT NULL CHECK (target >= 0),
    hard_ceiling integer NOT NULL CHECK (hard_ceiling > 0),
    context_limit integer NOT NULL,
    model_profile text NOT NULL,
    health text NOT NULL DEFAULT 'HEALTHY',
    valid_until timestamptz NOT NULL
);
CREATE TABLE runtime_operations (
    operation_id text PRIMARY KEY,
    root_id text NOT NULL REFERENCES runtime_roots,
    tenant_id text NOT NULL,
    spec jsonb NOT NULL,
    state text NOT NULL DEFAULT 'READY',
    attempts integer NOT NULL DEFAULT 0,
    active_attempt text,
    result jsonb,
    wait_reason text,
    retry_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX runtime_ready ON runtime_operations(state, retry_at, created_at);
CREATE INDEX runtime_operation_root ON runtime_operations(root_id, state);
CREATE TABLE runtime_attempts (
    attempt_id text PRIMARY KEY,
    operation_id text NOT NULL REFERENCES runtime_operations,
    pool_id text NOT NULL REFERENCES runtime_pools,
    engine_epoch text NOT NULL,
    attempt_number integer NOT NULL,
    owner_id text NOT NULL,
    lease_epoch bigint NOT NULL,
    lease_expires_at timestamptz NOT NULL,
    state text NOT NULL DEFAULT 'RESERVED',
    compute_held boolean NOT NULL DEFAULT true,
    budget_held boolean NOT NULL DEFAULT true,
    budget_bound bigint NOT NULL CHECK (budget_bound > 0),
    usage bigint,
    usage_estimated boolean,
    error_class text,
    send_intent_at timestamptz,
    backend_finished_at timestamptz,
    result_committed_at timestamptz,
    unknown_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(operation_id, attempt_number)
);
CREATE INDEX runtime_attempt_compute ON runtime_attempts(pool_id) WHERE compute_held;
CREATE INDEX runtime_attempt_reconcile ON runtime_attempts(lease_expires_at)
    WHERE state IN ('RESERVED','SEND_INTENT','ACTIVE');
CREATE TABLE runtime_artifacts (
    object_key text NOT NULL,
    tenant_id text NOT NULL,
    operation_id text NOT NULL REFERENCES runtime_operations,
    attempt_id text NOT NULL REFERENCES runtime_attempts,
    manifest jsonb NOT NULL,
    disposition text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(object_key, attempt_id)
);
CREATE TABLE runtime_outbox (
    event_id text PRIMARY KEY,
    kind text NOT NULL,
    aggregate_id text NOT NULL,
    payload jsonb NOT NULL,
    available_at timestamptz NOT NULL DEFAULT now(),
    owner_id text,
    lease_expires_at timestamptz,
    delivered_at timestamptz,
    deliveries integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX runtime_outbox_pending ON runtime_outbox(available_at) WHERE delivered_at IS NULL;
CREATE TABLE runtime_completion_bindings (
    binding_id text PRIMARY KEY,
    operation_id text NOT NULL REFERENCES runtime_operations,
    task_token bytea NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE runtime_submissions (
    run_id text PRIMARY KEY REFERENCES runtime_roots(root_id),
    tenant_id text NOT NULL,
    submission_key text NOT NULL,
    input_digest text NOT NULL,
    spec jsonb NOT NULL,
    UNIQUE(tenant_id, submission_key)
);
CREATE TABLE runtime_controller_updates (
    update_id bigserial PRIMARY KEY,
    pool_id text NOT NULL REFERENCES runtime_pools,
    envelope_version bigint NOT NULL,
    target integer NOT NULL,
    reason text NOT NULL,
    observed_at timestamptz NOT NULL DEFAULT now()
);
