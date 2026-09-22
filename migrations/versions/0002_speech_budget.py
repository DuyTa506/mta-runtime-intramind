"""Keep speech character accounting separate from the existing token ledger."""

from importlib.resources import files

from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade():
    for statement in files("intramind_runtime").joinpath("speech_budget.sql").read_text().split(";"):
        if statement.strip():
            op.execute(statement)


def downgrade():
    raise RuntimeError("Resource budget recovery requires a reviewed migration; data is retained")
