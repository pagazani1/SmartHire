"""add screening requirement snapshots

Revision ID: 2d8e6a4b91c3
Revises: 7b31f0d9c4a2
Create Date: 2026-11-13
"""

from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "2d8e6a4b91c3"
down_revision: Union[str, None] = "7b31f0d9c4a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("screening_requirement_results") as batch_op:
        batch_op.drop_constraint(
            "pk_screening_requirement_results",
            type_="primary",
        )
        batch_op.add_column(sa.Column("id", sa.Uuid(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "requirement_snapshot",
                sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
                nullable=True,
            )
        )
        batch_op.alter_column(
            "requirement_group_id",
            existing_type=sa.Uuid(),
            nullable=True,
        )

    connection = op.get_bind()
    table = sa.table(
        "screening_requirement_results",
        sa.column("id", sa.Uuid()),
        sa.column("screening_result_id", sa.Uuid()),
        sa.column("requirement_group_id", sa.Uuid()),
    )
    legacy_rows = connection.execute(
        sa.select(
            table.c.screening_result_id,
            table.c.requirement_group_id,
        )
    ).all()
    for row in legacy_rows:
        connection.execute(
            table.update()
            .where(
                table.c.screening_result_id == row.screening_result_id,
                table.c.requirement_group_id == row.requirement_group_id,
            )
            .values(id=uuid.uuid4())
        )

    with op.batch_alter_table("screening_requirement_results") as batch_op:
        batch_op.alter_column("id", existing_type=sa.Uuid(), nullable=False)
        batch_op.create_primary_key("pk_screening_requirement_results", ["id"])
        batch_op.create_unique_constraint(
            "uq_screening_requirement_results_result_group",
            ["screening_result_id", "requirement_group_id"],
        )


def downgrade() -> None:
    connection = op.get_bind()
    table = sa.table(
        "screening_requirement_results",
        sa.column("requirement_group_id", sa.Uuid()),
        sa.column("requirement_snapshot", sa.JSON()),
    )
    fallback_count = connection.scalar(
        sa.select(sa.func.count())
        .select_from(table)
        .where(table.c.requirement_group_id.is_(None))
    )
    snapshot_count = connection.scalar(
        sa.select(sa.func.count())
        .select_from(table)
        .where(table.c.requirement_snapshot.is_not(None))
    )
    if fallback_count or snapshot_count:
        raise RuntimeError(
            "Cannot downgrade screening requirement snapshots while fallback "
            "requirements or snapshots exist."
        )

    with op.batch_alter_table("screening_requirement_results") as batch_op:
        batch_op.drop_constraint(
            "uq_screening_requirement_results_result_group",
            type_="unique",
        )
        batch_op.drop_constraint(
            "pk_screening_requirement_results",
            type_="primary",
        )
        batch_op.alter_column(
            "requirement_group_id",
            existing_type=sa.Uuid(),
            nullable=False,
        )
        batch_op.drop_column("requirement_snapshot")
        batch_op.drop_column("id")
        batch_op.create_primary_key(
            "pk_screening_requirement_results",
            ["screening_result_id", "requirement_group_id"],
        )
