CREATE TABLE IF NOT EXISTS runtime_direct_attempts (
    attempt_id text PRIMARY KEY,
    request_id text NOT NULL,
    tenant_id text NOT NULL,
    spec jsonb NOT NULL,
    pool_id text NOT NULL REFERENCES runtime_pools,
    engine_epoch text NOT NULL,
    owner_id text NOT NULL,
    state text NOT NULL DEFAULT 'RESERVED',
    compute_held boolean NOT NULL DEFAULT true,
    lease_expires_at timestamptz NOT NULL,
    deadline timestamptz NOT NULL,
    send_intent_at timestamptz,
    backend_finished_at timestamptz,
    unknown_at timestamptz,
    error_class text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(tenant_id, request_id),
    CHECK (state IN ('RESERVED','SEND_INTENT','UNKNOWN','FINISHED','FAILED_NOT_SENT','FAILED'))
);
CREATE INDEX IF NOT EXISTS runtime_direct_compute ON runtime_direct_attempts(pool_id) WHERE compute_held;
CREATE INDEX IF NOT EXISTS runtime_direct_reconcile ON runtime_direct_attempts(lease_expires_at) WHERE compute_held;
CREATE OR REPLACE VIEW runtime_inflight_attempts AS
    SELECT attempt_id,pool_id,engine_epoch,state,compute_held,budget_held,unknown_at
    FROM runtime_attempts
    UNION ALL
    SELECT attempt_id,pool_id,engine_epoch,state,compute_held,false AS budget_held,unknown_at
    FROM runtime_direct_attempts;
