"""
Database Connection Pooling
Thread-safe connection pool for PostgreSQL and SQLite
"""

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, Dict

try:
    import psycopg2  # type: ignore

    # Ensure a pool attribute is available so tests can patch
    try:  # pragma: no cover - trivial attribute wiring
        from psycopg2 import pool as _psycopg2_pool  # type: ignore

        setattr(psycopg2, "pool", _psycopg2_pool)
    except Exception:
        pass

    PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via SQLite path in tests
    psycopg2 = None  # type: ignore
    PSYCOPG2_AVAILABLE = False


class DatabasePool:
    """Thread-safe database connection pool"""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        sqlite_path: Optional[str] = None,
        min_connections: int = 5,
        max_connections: int = 20
    ):
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        self.sqlite_path = sqlite_path or os.getenv("CHAT_DB_PATH", "./chat.db")
        self.min_connections = min_connections
        self.max_connections = max_connections

        # Underlying pool / connections
        self._pg_pool: Optional["psycopg2.pool.ThreadedConnectionPool"] = None  # type: ignore[attr-defined]
        self._sqlite_connections: Dict[int, sqlite3.Connection] = {}
        self._sqlite_lock = threading.RLock()

        # Simple connection statistics used by tests
        self._connections_created: int = 0
        self._connections_reused: int = 0

        self._initialize_pool()

    @property
    def is_postgres(self) -> bool:
        """Public flag used in tests to distinguish backends."""
        return self._is_postgres()

    def _is_postgres(self) -> bool:
        return self.database_url.startswith(("postgres://", "postgresql://"))
    
    def _initialize_pool(self):
        """Initialize connection pool based on database type."""
        if self._is_postgres():
            if not PSYCOPG2_AVAILABLE:
                raise RuntimeError("psycopg2 is required for PostgreSQL connections")

            self._pg_pool = psycopg2.pool.ThreadedConnectionPool(  # type: ignore[attr-defined]
                minconn=self.min_connections,
                maxconn=self.max_connections,
                dsn=self.database_url,
            )
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool.

        Usage:
            with db_pool.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
        """
        if self._is_postgres():
            conn = None
            try:
                conn = self._pg_pool.getconn()  # type: ignore[union-attr]
                self._connections_created += 1
                yield conn
                conn.commit()
            except Exception as e:
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    self._pg_pool.putconn(conn)  # type: ignore[union-attr]
        else:
            # SQLite - one connection per thread
            thread_id = threading.get_ident()
            
            with self._sqlite_lock:
                if thread_id not in self._sqlite_connections:
                    conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    self._sqlite_connections[thread_id] = conn
                    self._connections_created += 1
                else:
                    self._connections_reused += 1

                conn = self._sqlite_connections[thread_id]
            
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise
    
    def close_all(self):
        """Close all connections in the pool."""
        if self._pg_pool:
            self._pg_pool.closeall()

        with self._sqlite_lock:
            for conn in self._sqlite_connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._sqlite_connections.clear()

    def get_stats(self) -> dict:
        """Get connection pool statistics."""
        if self._is_postgres() and self._pg_pool:
            return {
                "type": "postgresql",
                "min_connections": self.min_connections,
                "max_connections": self.max_connections,
                "connections_created": self._connections_created,
                "connections_reused": self._connections_reused,
            }
        with self._sqlite_lock:
            return {
                "type": "sqlite",
                "active_threads": len(self._sqlite_connections),
                "path": self.sqlite_path,
                "connections_created": self._connections_created,
                "connections_reused": self._connections_reused,
            }


# Global pool instance
_pool_instance: Optional[DatabasePool] = None


def initialize_pool(
    database_url: Optional[str] = None,
    sqlite_path: Optional[str] = None,
    min_connections: int = 5,
    max_connections: int = 20
) -> DatabasePool:
    """Initialize the global database pool"""
    global _pool_instance
    _pool_instance = DatabasePool(
        database_url=database_url,
        sqlite_path=sqlite_path,
        min_connections=min_connections,
        max_connections=max_connections
    )
    return _pool_instance


def get_pool() -> DatabasePool:
    """Get the global database pool instance"""
    if _pool_instance is None:
        raise RuntimeError("Database pool has not been initialized. Call initialize_pool() first.")
    return _pool_instance


def close_pool():
    """Close the global database pool"""
    global _pool_instance
    if _pool_instance:
        _pool_instance.close_all()
        _pool_instance = None


def get_pool_stats() -> dict:
    """
    Get connection statistics for the global pool.

    When no pool has been initialized yet, a zeroed-out stats dict is
    returned, which matches test expectations.
    """
    if _pool_instance is None:
        return {"connections_created": 0, "connections_reused": 0}
    stats = _pool_instance.get_stats()
    return {
        "connections_created": stats.get("connections_created", 0),
        "connections_reused": stats.get("connections_reused", 0),
    }
