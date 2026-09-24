"""Endpoint-scoped admission, process ownership, and fenced direct generations."""

from importlib.resources import files

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade():
    for statement in files("intramind_runtime").joinpath("admission_v2.sql").read_text().split(";"):
        if statement.strip():
            op.execute(statement)


def downgrade():
    raise RuntimeError("Drain active inference and archive admission generations before downgrade")
