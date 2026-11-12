# SmartHire Database Setup (Local Development)

This guide configures the initial SQLAlchemy/Alembic foundation. The existing
`POST /api/analyze/` endpoint remains stateless: adding this infrastructure
does not store candidates, documents, or screening results. Do not use real
candidate data in local development.

## 1. Install PostgreSQL on Windows

1. Install a supported PostgreSQL release for Windows using the official
   PostgreSQL Windows installer (or an approved package manager).
2. During setup, record the local PostgreSQL administrator password securely;
   do not put it in source control or chat.
3. Ensure the PostgreSQL Windows service is running. The default local port is
   `5432`; use the port selected during installation if different.
4. Open SQL Shell (`psql`) or PowerShell and connect as the database
   administrator:

   ```powershell
   psql -U postgres -h localhost -p 5432
   ```

5. Create a dedicated local role and database. Replace the placeholder
   password with a unique local value and enter it only in the interactive
   prompt:

   ```sql
   CREATE ROLE smarthire_user WITH LOGIN;
   \password smarthire_user
   CREATE DATABASE smarthire OWNER smarthire_user;
   ```

   Do not run these commands against a production database. If the role or
   database already exists, inspect it rather than dropping or overwriting it.

## 2. Configure the local environment

From the repository root, copy the example and edit the local-only file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `DATABASE_URL` in `.env`:

```dotenv
DATABASE_URL=postgresql+psycopg://smarthire_user:YOUR_LOCAL_PASSWORD@localhost:5432/smarthire
```

URL-encode special characters in a password. `.env` is ignored by Git; keep
credentials out of `.env.example`, source files, command history, and logs.
Environment variables already set in the process take precedence over `.env`.
The engine is created lazily, so imports and the legacy analysis endpoint do
not require a configured database.

## 3. Install Python dependencies

Activate the project's virtual environment, then install the requirements:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The database foundation uses SQLAlchemy 2.x, Alembic, `psycopg[binary]`, and
`python-dotenv`. The binary psycopg extra is convenient for local Windows
development; review production driver/build policy separately.

## 4. Create or inspect the schema

First inspect the migration plan and current database state. The migration
does not drop existing tables, but it is intended for a new SmartHire schema.
Back up any database that already contains data and review migration SQL
before applying changes.

```powershell
python -m alembic current
python -m alembic history
python -m alembic upgrade head
```

To inspect generated SQL without applying it:

```powershell
python -m alembic upgrade head --sql
```

Rollback of this initial revision is available only when the resulting tables
are empty or after an explicit backup and data-retention review:

```powershell
python -m alembic downgrade base
```

Do not run downgrade against a database containing records you need; it drops
the initial schema tables.

## 5. Check connectivity

After setting `.env`, verify the connection without printing credentials:

```powershell
python -c "from sqlalchemy import text; from backend.db.session import get_engine; print(get_engine().connect().execute(text('SELECT 1')).scalar_one())"
```

Expected output is `1`. This confirms the configured database accepts a
connection; it does not verify backup, access-control, or production readiness.

## 6. Run tests

Ordinary database infrastructure tests use temporary SQLite databases and do
not require a running PostgreSQL server or a production database URL:

```powershell
python -m unittest tests.test_database
python -m unittest tests.test_decision_policy tests.test_predict tests.test_skill_extractor tests.test_structured_classifier tests.test_hybrid_predict tests.test_resume_analyzer tests.test_api
```

SQLite tests cover model mapping, relationships, constraints, dependency
session cleanup, and Alembic upgrade/downgrade. PostgreSQL-only behavior,
driver connectivity, and deployment settings require validation against an
isolated PostgreSQL database before deployment.
