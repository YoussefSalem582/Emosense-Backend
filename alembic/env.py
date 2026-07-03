import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add the project root to the path so we can import the app package.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base, GUID
from app.models import user, emotion  # noqa: F401  (register models on Base.metadata)

# Alembic Config object, providing access to values in alembic.ini.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata used for 'autogenerate' support.
target_metadata = Base.metadata


def _sync_database_url() -> str:
    """Return a synchronous SQLAlchemy URL for Alembic.

    The application configures an async driver (asyncpg / aiosqlite); Alembic
    runs synchronously, so strip the async driver suffix.
    """
    from app.config import get_settings

    url = get_settings().DATABASE_URL
    return url.replace("+asyncpg", "").replace("+aiosqlite", "")


def _render_item(type_, obj, autogen_context):
    """Render our custom GUID type into migration scripts.

    Without this, Alembic would emit the dialect-specific impl (e.g. CHAR(36)),
    hard-coding one backend. Emitting ``app.database.GUID()`` keeps migrations
    portable across PostgreSQL and SQLite.
    """
    if type_ == "type" and isinstance(obj, GUID):
        autogen_context.imports.add("import app.database")
        return "app.database.GUID()"
    return False


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DBAPI connection)."""
    context.configure(
        url=_sync_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=_render_item,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live connection."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _sync_database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_item=_render_item,
            compare_type=True,
            # SQLite cannot ALTER most columns; batch mode rebuilds tables.
            render_as_batch=is_sqlite,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
