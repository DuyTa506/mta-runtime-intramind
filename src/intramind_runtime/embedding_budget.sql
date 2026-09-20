ALTER TABLE runtime_resource_budgets DROP CONSTRAINT runtime_resource_budgets_unit_check;
ALTER TABLE runtime_resource_budgets ADD CONSTRAINT runtime_resource_budgets_unit_check
    CHECK (unit IN ('speech_characters', 'embedding_characters'));
ALTER TABLE runtime_attempts DROP CONSTRAINT runtime_attempts_budget_unit_check;
ALTER TABLE runtime_attempts ADD CONSTRAINT runtime_attempts_budget_unit_check
    CHECK (budget_unit IN ('tokens', 'speech_characters', 'embedding_characters'));
