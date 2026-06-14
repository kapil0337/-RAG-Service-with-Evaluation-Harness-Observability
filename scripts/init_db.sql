-- Run automatically by the postgres container on first startup
-- (mounted into /docker-entrypoint-initdb.d/).
-- Application tables are created by the app itself via SQLAlchemy
-- (see app/db.py::init_db), but the pgvector extension must exist first.

CREATE EXTENSION IF NOT EXISTS vector;
