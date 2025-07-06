"""
Database Configuration and Setup for EmoSense Backend API

Handles SQLAlchemy database configuration, connection management,
and table creation for PostgreSQL database.
"""

from typing import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool

from app.config import get_settings


# Get application settings
settings = get_settings()

# Create async database engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,  # Log SQL queries in debug mode
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    poolclass=NullPool if "sqlite" in settings.DATABASE_URL else None,
    future=True,
)

# Create async session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

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
    from app.models import user, emotion, analysis  # noqa: F401
    
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
                await conn.execute("SELECT 1")
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
                result = await conn.execute("SELECT version()")
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
            result = await conn.execute(sql)
            return result.fetchall()


# Global database manager instance
db_manager = DatabaseManager()
