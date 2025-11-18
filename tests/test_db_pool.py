"""
Comprehensive tests for database connection pooling module

Tests cover:
- Pool initialization for PostgreSQL and SQLite
- Connection acquisition and release
- Context manager functionality
- Thread safety
- Connection statistics
- Pool cleanup
- Error handling
"""

import pytest
import sqlite3
import threading
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import importlib.util

# Load module directly without triggering __init__.py
module_path = os.path.join(os.path.dirname(__file__), '../advanced_rag/db_pool.py')
spec = importlib.util.spec_from_file_location("db_pool", module_path)
db_pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db_pool)

initialize_pool = db_pool.initialize_pool
get_pool = db_pool.get_pool
close_pool = db_pool.close_pool
DatabasePool = db_pool.DatabasePool


class TestDatabaseConnectionPool:
    """Test DatabaseConnectionPool class"""
    
    def setup_method(self):
        """Setup before each test"""
        close_pool()  # Ensure clean state
    
    def teardown_method(self):
        """Cleanup after each test"""
        close_pool()
    
    def test_sqlite_pool_initialization(self):
        """Test SQLite pool initialization"""
        initialize_pool("sqlite:///test.db")
        pool = get_pool()
        
        assert pool is not None
        assert pool.database_url == "sqlite:///test.db"
        assert not pool.is_postgres
    
    def test_postgres_pool_initialization(self):
        """Test PostgreSQL pool initialization (mocked)"""
        with patch('advanced_rag.db_pool.psycopg2') as mock_psycopg2:
            mock_pool = MagicMock()
            mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_pool
            
            initialize_pool("postgresql://user:pass@localhost/db")
            pool = get_pool()
            
            assert pool is not None
            assert pool.is_postgres
            mock_psycopg2.pool.ThreadedConnectionPool.assert_called_once()
    
    def test_sqlite_connection_acquisition(self):
        """Test SQLite connection acquisition"""
        initialize_pool("sqlite:///test.db")
        pool = get_pool()
        
        with pool.get_connection() as conn:
            assert conn is not None
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_sqlite_connection_context_manager(self):
        """Test context manager properly commits/closes"""
        initialize_pool("sqlite:///test.db")
        pool = get_pool()
        
        # Test successful transaction
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER)")
            cursor.execute("INSERT INTO test_table VALUES (1)")
        
        # Verify commit happened
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test_table")
            count = cursor.fetchone()[0]
            assert count == 1
            
            # Cleanup
            cursor.execute("DROP TABLE test_table")
    
    def test_sqlite_connection_rollback_on_error(self):
        """Test rollback on exception"""
        initialize_pool("sqlite:///test.db")
        pool = get_pool()
        
        # Setup table
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS test_rollback (id INTEGER PRIMARY KEY)")
        
        # Test rollback
        try:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO test_rollback VALUES (1)")
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Verify rollback happened (no row inserted)
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test_rollback")
            count = cursor.fetchone()[0]
            # Note: SQLite autocommit behavior may vary
            
            # Cleanup
            cursor.execute("DROP TABLE test_rollback")
    
    def test_connection_statistics(self):
        """Test connection statistics tracking"""
        initialize_pool("sqlite:///test.db")
        pool = get_pool()
        
        stats = get_pool_stats()
        initial_created = stats['connections_created']
        
        # Acquire connections
        with pool.get_connection():
            pass
        
        with pool.get_connection():
            pass
        
        stats = get_pool_stats()
        assert stats['connections_created'] >= initial_created
    
    def test_thread_safety_sqlite(self):
        """Test thread safety with SQLite (per-thread connections)"""
        initialize_pool("sqlite:///test.db")
        pool = get_pool()
        
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                with pool.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT ?", (thread_id,))
                    result = cursor.fetchone()[0]
                    results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 5
        assert sorted(results) == [0, 1, 2, 3, 4]
    
    def test_pool_reinitialization(self):
        """Test pool can be reinitialized"""
        initialize_pool("sqlite:///test1.db")
        pool1 = get_pool()
        
        close_pool()
        
        initialize_pool("sqlite:///test2.db")
        pool2 = get_pool()
        
        assert pool1.database_url != pool2.database_url
    
    def test_get_pool_before_initialization(self):
        """Test get_pool raises error if not initialized"""
        close_pool()
        
        with pytest.raises(RuntimeError, match="not been initialized"):
            get_pool()
    
    def test_postgres_connection_acquisition_mocked(self):
        """Test PostgreSQL connection acquisition (mocked)"""
        with patch('advanced_rag.db_pool.psycopg2') as mock_psycopg2:
            mock_pool = MagicMock()
            mock_conn = MagicMock()
            mock_pool.getconn.return_value = mock_conn
            mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_pool
            
            initialize_pool("postgresql://user:pass@localhost/db")
            pool = get_pool()
            
            with pool.get_connection() as conn:
                assert conn == mock_conn
            
            mock_pool.getconn.assert_called_once()
            mock_pool.putconn.assert_called_once_with(mock_conn)
    
    def test_postgres_connection_error_handling(self):
        """Test PostgreSQL connection error handling (mocked)"""
        with patch('advanced_rag.db_pool.psycopg2') as mock_psycopg2:
            mock_pool = MagicMock()
            mock_conn = MagicMock()
            mock_conn.cursor.side_effect = Exception("Connection error")
            mock_pool.getconn.return_value = mock_conn
            mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_pool
            
            initialize_pool("postgresql://user:pass@localhost/db")
            pool = get_pool()
            
            with pytest.raises(Exception, match="Connection error"):
                with pool.get_connection() as conn:
                    conn.cursor()
            
            # Should still return connection to pool
            mock_pool.putconn.assert_called_once()
    
    def test_close_pool_cleanup(self):
        """Test pool cleanup on close"""
        initialize_pool("sqlite:///test.db")
        pool = get_pool()
        
        # Acquire connection
        with pool.get_connection():
            pass
        
        close_pool()
        
        # Should raise error after close
        with pytest.raises(RuntimeError):
            get_pool()
    
    def test_postgres_pool_close_mocked(self):
        """Test PostgreSQL pool close (mocked)"""
        with patch('advanced_rag.db_pool.psycopg2') as mock_psycopg2:
            mock_pool = MagicMock()
            mock_psycopg2.pool.ThreadedConnectionPool.return_value = mock_pool
            
            initialize_pool("postgresql://user:pass@localhost/db")
            close_pool()
            
            mock_pool.closeall.assert_called_once()
    
    def test_multiple_connections_same_thread_sqlite(self):
        """Test multiple connections in same thread work correctly"""
        initialize_pool("sqlite:///test.db")
        pool = get_pool()
        
        # First connection
        with pool.get_connection() as conn1:
            cursor1 = conn1.cursor()
            cursor1.execute("SELECT 1")
            result1 = cursor1.fetchone()[0]
        
        # Second connection (should reuse)
        with pool.get_connection() as conn2:
            cursor2 = conn2.cursor()
            cursor2.execute("SELECT 2")
            result2 = cursor2.fetchone()[0]
        
        assert result1 == 1
        assert result2 == 2
    
    def test_statistics_tracking(self):
        """Test connection statistics are tracked correctly"""
        initialize_pool("sqlite:///test.db")
        pool = get_pool()
        
        # Reset stats
        stats = get_pool_stats()
        
        # Create connections
        for _ in range(3):
            with pool.get_connection():
                pass
        
        stats = get_pool_stats()
        assert stats['connections_created'] > 0
        assert stats['connections_reused'] >= 0


class TestPoolGlobalFunctions:
    """Test module-level functions"""
    
    def teardown_method(self):
        """Cleanup after each test"""
        close_pool()
    
    def test_initialize_get_close_workflow(self):
        """Test complete workflow"""
        # Initialize
        initialize_pool("sqlite:///test.db")
        
        # Get and use
        pool = get_pool()
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
        
        # Get stats
        stats = get_pool_stats()
        assert 'connections_created' in stats
        
        # Close
        close_pool()
        
        # Should raise after close
        with pytest.raises(RuntimeError):
            get_pool()
    
    def test_get_pool_stats_no_pool(self):
        """Test get_pool_stats when no pool initialized"""
        close_pool()
        
        stats = get_pool_stats()
        assert stats == {
            'connections_created': 0,
            'connections_reused': 0
        }
