"""
Database Configuration and Setup for EmoSense Backend API

Handles SQLAlchemy database configuration, connection management,
and table creation for PostgreSQL database.
"""

import uuid as _uuid
from typing import AsyncGenerator

from sqlalchemy import MetaData, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy.types import CHAR, TypeDecorator

from app.config import get_settings


# Get application settings
settings = get_settings()

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

# Build engine kwargs. SQLite (used for local dev/tests) uses NullPool, which
# does not accept pool_size/max_overflow; only pass those for real pooled DBs.
_engine_kwargs = {"echo": settings.DEBUG, "future": True}
if _is_sqlite:
    _engine_kwargs["poolclass"] = NullPool
else:
    _engine_kwargs["pool_size"] = settings.DATABASE_POOL_SIZE
    _engine_kwargs["max_overflow"] = settings.DATABASE_MAX_OVERFLOW

# Create async database engine
engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

# Create async session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

class GUID(TypeDecorator):
    """Platform-independent UUID column type.

    Uses PostgreSQL's native ``UUID`` type when available and falls back to
    ``CHAR(36)`` on other backends (e.g. SQLite for local dev/tests). Values are
    always exposed to Python as ``uuid.UUID`` objects.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, _uuid.UUID):
            value = _uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, _uuid.UUID):
            return value
        return _uuid.UUID(str(value))


# Create declarative base for models
Base = declarative_base()

# Define consistent naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

Base.metadata = MetaData(naming_convention=convention)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function to get database session.
    
    Yields:
        AsyncSession: Database session for dependency injection
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables() -> None:
    """
    Create all database tables.
    
    This function creates all tables defined in the models.
    Should be called during application startup.
    """
    # Import all models to ensure they are registered with Base.metadata
    from app.models import user, emotion  # noqa: F401
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables() -> None:
    """
    Drop all database tables.
    
    WARNING: This will delete all data! Use only for testing or development.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def get_engine():
    """Get the database engine instance."""
    return engine


class DatabaseManager:
    """
    Database manager class for handling advanced database operations.
    
    Provides methods for connection testing, health checks, and maintenance.
    """
    
    def __init__(self):
        """Initialize database manager."""
        self.engine = engine
        self.session_factory = async_session_factory
    
    async def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    async def get_connection_info(self) -> dict:
        """
        Get database connection information.
        
        Returns:
            dict: Database connection details
        """
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT version()"))
                version = result.scalar()
                
                return {
                    "connected": True,
                    "version": version,
                    "url": str(self.engine.url).split("@")[-1],  # Hide credentials
                    "pool_size": self.engine.pool.size(),
                    "checked_out": self.engine.pool.checkedout(),
                }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "url": str(self.engine.url).split("@")[-1],
            }
    
    async def execute_raw_sql(self, sql: str) -> list:
        """
        Execute raw SQL query.
        
        Args:
            sql: SQL query string
            
        Returns:
            list: Query results
            
        Warning:
            Use with caution. Only for administrative tasks.
        """
        async with self.engine.begin() as conn:
            result = await conn.execute(text(sql) if isinstance(sql, str) else sql)
            return result.fetchall()


# Global database manager instance
db_manager = DatabaseManager()
