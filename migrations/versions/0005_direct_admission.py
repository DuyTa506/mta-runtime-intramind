"""Share pool accounting with direct foreground requests."""

from importlib.resources import files

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    for statement in files("intramind_runtime").joinpath("direct.sql").read_text().split(";"):
        if statement.strip():
            op.execute(statement)


def downgrade():
    raise RuntimeError("Drain and archive direct inference attempts before removing shared admission")
