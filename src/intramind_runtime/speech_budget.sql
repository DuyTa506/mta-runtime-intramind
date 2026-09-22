CREATE TABLE runtime_resource_budgets (
    root_id text NOT NULL REFERENCES runtime_roots,
    unit text NOT NULL CHECK (unit IN ('speech_characters')),
    budget_limit bigint NOT NULL CHECK (budget_limit > 0),
    reserved bigint NOT NULL DEFAULT 0 CHECK (reserved >= 0),
    spent bigint NOT NULL DEFAULT 0 CHECK (spent >= 0),
    PRIMARY KEY (root_id, unit)
);
ALTER TABLE runtime_attempts ADD COLUMN budget_unit text NOT NULL DEFAULT 'tokens'
    CHECK (budget_unit IN ('tokens', 'speech_characters'));
