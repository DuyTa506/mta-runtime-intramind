"""Retain deferred inputs until a durable, ordered workflow batch owns them."""

from importlib.resources import files

from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade():
    for statement in files("intramind_runtime").joinpath("buffering.sql").read_text().split(";"):
        if statement.strip():
            op.execute(statement)


def downgrade():
    raise RuntimeError("Buffered inputs require a reviewed recovery migration; data is retained")
