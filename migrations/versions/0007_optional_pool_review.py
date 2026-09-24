"""Make pool validity a review signal rather than a dispatch cutoff."""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE runtime_pools ALTER COLUMN valid_until DROP NOT NULL")


def downgrade():
    raise RuntimeError("rc17 requires a populated valid_until on every pool; downgrade is unsafe")
