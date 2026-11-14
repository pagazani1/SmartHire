"""add recruiter role and password hash

Revision ID: 9c4f1e2a7b80
Revises: 2d8e6a4b91c3
Create Date: 2026-11-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "9c4f1e2a7b80"
down_revision: Union[str, Sequence[str], None] = "2d8e6a4b91c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("recruiters") as batch_op:
        batch_op.add_column(
            sa.Column(
                "role",
                sa.Enum(
                    "recruiter",
                    "admin",
                    name="recruiter_role",
                    native_enum=False,
                    create_constraint=True,
                ),
                nullable=False,
                server_default="recruiter",
            )
        )
        batch_op.add_column(
            sa.Column(
                "password_hash",
                sa.String(length=500),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("recruiters") as batch_op:
        batch_op.drop_constraint("recruiter_role", type_="check")
        batch_op.drop_column("password_hash")
        batch_op.drop_column("role")
