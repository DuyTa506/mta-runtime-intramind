"""Authoritative operation, attempt and reservation ledger."""
from importlib.resources import files

from alembic import op

revision = "0001"
down_revision = None


def upgrade():
    for statement in files("intramind_runtime").joinpath("schema.sql").read_text().split(";"):
        if statement.strip():
            op.execute(statement)


def downgrade():
    raise RuntimeError("Ledger downgrade requires a reviewed recovery migration; data is retained")
