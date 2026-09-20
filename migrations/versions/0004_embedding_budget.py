"""Keep embedding input accounting separate from completion and speech budgets."""

from importlib.resources import files

from alembic import op

revision = "0004"
down_revision = "0003"


def upgrade():
    for statement in files("intramind_runtime").joinpath("embedding_budget.sql").read_text().split(";"):
        if statement.strip():
            op.execute(statement)


def downgrade():
    raise RuntimeError("Embedding accounting requires a reviewed recovery migration; data is retained")
