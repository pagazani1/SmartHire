"""add screening idempotency key

Revision ID: 7b31f0d9c4a2
Revises: c68042771911
Create Date: 2026-11-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7b31f0d9c4a2"
down_revision: Union[str, None] = "c68042771911"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("screening_results") as batch_op:
        batch_op.add_column(sa.Column("idempotency_key", sa.String(length=255)))
        batch_op.create_unique_constraint(
            "uq_screening_results_application_idempotency_key",
            ["application_id", "idempotency_key"],
        )


def downgrade() -> None:
    with op.batch_alter_table("screening_results") as batch_op:
        batch_op.drop_constraint(
            "uq_screening_results_application_idempotency_key",
            type_="unique",
        )
        batch_op.drop_column("idempotency_key")
