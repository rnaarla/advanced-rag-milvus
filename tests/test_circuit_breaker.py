"""
Comprehensive tests for circuit breaker module

Tests cover:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold detection
- Timeout and recovery
- Thread safety
- Statistics tracking
- Configuration validation
"""

import pytest
import time
import threading
import sys
import os
import importlib.util

# Load module directly without triggering __init__.py
module_path = os.path.join(os.path.dirname(__file__), '../src/advanced_rag/circuit_breaker.py')
spec = importlib.util.spec_from_file_location("circuit_breaker", module_path)
circuit_breaker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(circuit_breaker)

CircuitBreaker = circuit_breaker.CircuitBreaker
CircuitBreakerConfig = circuit_breaker.CircuitBreakerConfig
CircuitState = circuit_breaker.CircuitState
CircuitBreakerOpenError = circuit_breaker.CircuitBreakerOpenError


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.timeout_seconds == 60
        assert config.success_threshold == 2
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            timeout_seconds=120,
            success_threshold=3
        )
        
        assert config.failure_threshold == 10
        assert config.timeout_seconds == 120
        assert config.success_threshold == 3


class TestCircuitBreaker:
    """Test CircuitBreaker functionality"""
    
    def test_initial_state_closed(self):
        """Test circuit breaker starts in CLOSED state"""
        cb = CircuitBreaker()
        
        assert not cb.is_open()
        stats = cb.get_stats()
        assert stats['state'] == 'closed'
        assert stats['failures'] == 0
        assert stats['successes'] == 0
    
    def test_record_success(self):
        """Test recording successful operations"""
        cb = CircuitBreaker()
        
        cb.record_success()
        cb.record_success()
        
        stats = cb.get_stats()
        assert stats['successes'] == 2
        assert stats['state'] == 'closed'
    
    def test_record_failure(self):
        """Test recording failed operations"""
        cb = CircuitBreaker()
        
        cb.record_failure()
        cb.record_failure()
        
        stats = cb.get_stats()
        assert stats['failures'] == 2
        assert stats['state'] == 'closed'  # Not reached threshold yet
    
    def test_open_on_failure_threshold(self):
        """Test circuit opens after failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)
        
        # Record failures
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open()
        
        cb.record_failure()  # This should open the circuit
        
        assert cb.is_open()
        stats = cb.get_stats()
        assert stats['state'] == 'open'
        assert stats['failures'] == 3
    
    def test_open_state_rejects_calls(self):
        """Test that is_open() returns True when circuit is open"""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config)
        
        cb.record_failure()
        cb.record_failure()
        
        assert cb.is_open()
    
    def test_half_open_after_timeout(self):
        """Test transition to HALF_OPEN after timeout"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=1
        )
        cb = CircuitBreaker(config)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open()
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Should transition to HALF_OPEN
        assert not cb.is_open()  # HALF_OPEN allows requests
        stats = cb.get_stats()
        assert stats['state'] == 'half_open'
    
    def test_half_open_success_closes_circuit(self):
        """Test successful calls in HALF_OPEN close the circuit"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=1,
            success_threshold=2
        )
        cb = CircuitBreaker(config)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Record successes in HALF_OPEN
        cb.record_success()
        stats = cb.get_stats()
        assert stats['state'] == 'half_open'
        
        cb.record_success()  # Should close circuit
        
        stats = cb.get_stats()
        assert stats['state'] == 'closed'
        assert not cb.is_open()
    
    def test_half_open_failure_reopens_circuit(self):
        """Test failure in HALF_OPEN reopens circuit"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=1
        )
        cb = CircuitBreaker(config)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        
        # Wait for timeout to enter HALF_OPEN
        time.sleep(1.1)
        assert not cb.is_open()  # In HALF_OPEN
        
        # Record failure in HALF_OPEN
        cb.record_failure()
        
        # Should reopen circuit
        assert cb.is_open()
        stats = cb.get_stats()
        assert stats['state'] == 'open'
    
    def test_statistics_tracking(self):
        """Test statistics are tracked correctly"""
        cb = CircuitBreaker()
        
        cb.record_success()
        cb.record_success()
        cb.record_failure()
        
        stats = cb.get_stats()
        assert stats['total_requests'] == 3
        assert stats['successes'] == 2
        assert stats['failures'] == 1
        assert stats['state_transitions'] == 0  # No state changes yet
    
    def test_state_transition_tracking(self):
        """Test state transitions are tracked"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=1
        )
        cb = CircuitBreaker(config)
        
        # Cause state transition to OPEN
        cb.record_failure()
        cb.record_failure()
        
        stats = cb.get_stats()
        assert stats['state_transitions'] == 1  # CLOSED -> OPEN
        
        # Wait for HALF_OPEN
        time.sleep(1.1)
        cb.is_open()  # Trigger state check
        
        stats = cb.get_stats()
        assert stats['state_transitions'] == 2  # OPEN -> HALF_OPEN
    
    def test_thread_safety(self):
        """Test circuit breaker is thread-safe"""
        config = CircuitBreakerConfig(failure_threshold=50)
        cb = CircuitBreaker(config)
        
        errors = []
        
        def worker_success():
            try:
                for _ in range(10):
                    cb.record_success()
            except Exception as e:
                errors.append(str(e))
        
        def worker_failure():
            try:
                for _ in range(10):
                    cb.record_failure()
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=worker_success))
            threads.append(threading.Thread(target=worker_failure))
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        stats = cb.get_stats()
        assert stats['successes'] == 50
        assert stats['failures'] == 50
        assert stats['total_requests'] == 100
    
    def test_concurrent_state_transitions(self):
        """Test state transitions under concurrent load"""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=1
        )
        cb = CircuitBreaker(config)
        
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    if cb.is_open():
                        time.sleep(0.1)
                    else:
                        cb.record_failure()
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=worker) for _ in range(3)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have no errors even with concurrent access
        assert len(errors) == 0
    
    def test_reset_on_close(self):
        """Test counters reset when circuit closes"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=1,
            success_threshold=2
        )
        cb = CircuitBreaker(config)
        
        # Open circuit
        cb.record_failure()
        cb.record_failure()
        
        stats = cb.get_stats()
        assert stats['failures'] == 2
        
        # Wait and close circuit
        time.sleep(1.1)
        cb.record_success()
        cb.record_success()
        
        stats = cb.get_stats()
        assert stats['state'] == 'closed'
        assert stats['failures'] == 0  # Should reset
        assert stats['successes'] == 0  # Should reset
    
    def test_multiple_open_close_cycles(self):
        """Test multiple open/close cycles"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.5,
            success_threshold=2
        )
        cb = CircuitBreaker(config)
        
        for cycle in range(3):
            # Open circuit
            cb.record_failure()
            cb.record_failure()
            assert cb.is_open()
            
            # Wait for HALF_OPEN
            time.sleep(0.6)
            
            # Close circuit
            cb.record_success()
            cb.record_success()
            assert not cb.is_open()
            
            stats = cb.get_stats()
            assert stats['state'] == 'closed'
    
    def test_long_timeout_maintains_open_state(self):
        """Test circuit stays open during timeout period"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=5
        )
        cb = CircuitBreaker(config)
        
        # Open circuit
        cb.record_failure()
        cb.record_failure()
        
        # Check multiple times during timeout
        for _ in range(3):
            time.sleep(0.5)
            assert cb.is_open()
    
    def test_edge_case_zero_failures(self):
        """Test circuit with zero failures"""
        cb = CircuitBreaker()
        
        # Only successes
        for _ in range(10):
            cb.record_success()
        
        assert not cb.is_open()
        stats = cb.get_stats()
        assert stats['state'] == 'closed'
        assert stats['failures'] == 0
    
    def test_edge_case_threshold_boundary(self):
        """Test exact threshold boundary"""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(config)
        
        # Exactly threshold - 1
        for _ in range(4):
            cb.record_failure()
        
        assert not cb.is_open()
        
        # One more should open
        cb.record_failure()
        assert cb.is_open()
    
    def test_stats_format(self):
        """Test statistics return format"""
        cb = CircuitBreaker()
        stats = cb.get_stats()
        
        required_keys = [
            'state',
            'failures',
            'successes',
            'total_requests',
            'state_transitions'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert isinstance(stats['state'], str)
        assert isinstance(stats['failures'], int)
        assert isinstance(stats['successes'], int)


class TestCircuitBreakerOpenError:
    """Test CircuitBreakerOpenError exception"""
    
    def test_exception_creation(self):
        """Test exception can be created and raised"""
        with pytest.raises(CircuitBreakerOpenError, match="Circuit is open"):
            raise CircuitBreakerOpenError("Circuit is open")
    
    def test_exception_inheritance(self):
        """Test exception inherits from base exception"""
        assert issubclass(CircuitBreakerOpenError, Exception)
