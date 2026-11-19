"""add durable audit_events table

Revision ID: a1c5e7d3b9f4
Revises: 9c4f1e2a7b80
Create Date: 2026-11-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1c5e7d3b9f4"
down_revision: Union[str, Sequence[str], None] = "9c4f1e2a7b80"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("event", sa.String(length=100), nullable=False),
        sa.Column("outcome", sa.String(length=20), nullable=False),
        sa.Column("recruiter_id", sa.Uuid(), nullable=True),
        sa.Column("resource_type", sa.String(length=50), nullable=True),
        sa.Column("resource_id", sa.Uuid(), nullable=True),
        sa.Column("reason", sa.String(length=200), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_events")),
    )
    op.create_index("ix_audit_events_occurred_at", "audit_events", ["occurred_at"])
    op.create_index(
        "ix_audit_events_recruiter_occurred",
        "audit_events",
        ["recruiter_id", "occurred_at"],
    )
    op.create_index(
        "ix_audit_events_event_occurred", "audit_events", ["event", "occurred_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_audit_events_event_occurred", table_name="audit_events")
    op.drop_index("ix_audit_events_recruiter_occurred", table_name="audit_events")
    op.drop_index("ix_audit_events_occurred_at", table_name="audit_events")
    op.drop_table("audit_events")
