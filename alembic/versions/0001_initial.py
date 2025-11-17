"""initial schema

Revision ID: 0001_initial
Revises: 
Create Date: 2025-11-17

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
    CREATE TABLE IF NOT EXISTS sessions(
        id TEXT PRIMARY KEY,
        user_id TEXT,
        created_at DOUBLE PRECISION,
        metadata TEXT
    );
    """)
    op.execute("""
    CREATE TABLE IF NOT EXISTS messages(
        id SERIAL PRIMARY KEY,
        session_id TEXT,
        role TEXT,
        content TEXT,
        created_at DOUBLE PRECISION
    );
    """)
    op.execute("""
    CREATE TABLE IF NOT EXISTS feedback(
        id SERIAL PRIMARY KEY,
        message_id INTEGER,
        vote TEXT,
        comment TEXT,
        created_at DOUBLE PRECISION
    );
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS feedback;")
    op.execute("DROP TABLE IF EXISTS messages;")
    op.execute("DROP TABLE IF EXISTS sessions;")


