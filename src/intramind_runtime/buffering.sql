CREATE TABLE runtime_buffer_batches (
    batch_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    task_type text NOT NULL,
    partition_key text NOT NULL,
    state text NOT NULL DEFAULT 'PREPARING' CHECK (state IN ('PREPARING','SUBMITTED','FAILED')),
    definition jsonb NOT NULL,
    configuration jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    deadline timestamptz NOT NULL,
    retry_at timestamptz NOT NULL DEFAULT now(),
    terminal_reason text
);
CREATE INDEX runtime_buffer_batch_partition ON runtime_buffer_batches(tenant_id,task_type,partition_key);
CREATE TABLE runtime_buffer_items (
    ordinal bigserial PRIMARY KEY,
    item_id text NOT NULL UNIQUE,
    tenant_id text NOT NULL,
    task_type text NOT NULL,
    partition_key text NOT NULL,
    submission_key text NOT NULL,
    input jsonb NOT NULL,
    configuration jsonb NOT NULL,
    definition jsonb NOT NULL,
    batch_size integer NOT NULL CHECK (batch_size BETWEEN 1 AND 256),
    due_at timestamptz NOT NULL,
    batch_id text REFERENCES runtime_buffer_batches,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(tenant_id,submission_key)
);
CREATE INDEX runtime_buffer_pending ON runtime_buffer_items(due_at,ordinal) WHERE batch_id IS NULL;
