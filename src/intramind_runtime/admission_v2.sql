-- Additive admission state. Neither a token nor a streamed progress frame writes here.
CREATE TABLE IF NOT EXISTS runtime_owners (
    owner_id text PRIMARY KEY,
    boot_generation text NOT NULL,
    lease_expires_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS runtime_owner_expiry ON runtime_owners(lease_expires_at);

ALTER TABLE runtime_pools ADD COLUMN IF NOT EXISTS quiesce_token text;

CREATE TABLE IF NOT EXISTS runtime_admission_models (
    model_profile text PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS runtime_direct_waiters (
    request_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    pool_id text NOT NULL REFERENCES runtime_pools,
    owner_id text NOT NULL,
    workload_class text NOT NULL CHECK (workload_class IN ('qa','user_task','background','maintenance')),
    deadline timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS runtime_direct_waiters_pool
    ON runtime_direct_waiters(pool_id,workload_class,created_at);
CREATE INDEX IF NOT EXISTS runtime_direct_waiters_owner
    ON runtime_direct_waiters(owner_id);

ALTER TABLE runtime_direct_attempts ADD COLUMN IF NOT EXISTS generation integer NOT NULL DEFAULT 0;
ALTER TABLE runtime_direct_attempts
    DROP CONSTRAINT IF EXISTS runtime_direct_attempts_tenant_id_request_id_key;
ALTER TABLE runtime_direct_attempts
    DROP CONSTRAINT IF EXISTS runtime_direct_attempts_state_check;
ALTER TABLE runtime_direct_attempts
    ADD CONSTRAINT runtime_direct_attempts_state_check
    CHECK (state IN ('RESERVED','SEND_INTENT','UNKNOWN','FINISHED','FAILED_NOT_SENT',
                    'FAILED','FAILED_RECOVERABLE'));
CREATE UNIQUE INDEX IF NOT EXISTS runtime_direct_identity_generation
    ON runtime_direct_attempts(tenant_id,request_id,generation);

CREATE INDEX IF NOT EXISTS runtime_operations_model_pending
    ON runtime_operations ((spec->>'model_profile'),state,created_at)
    WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED');
