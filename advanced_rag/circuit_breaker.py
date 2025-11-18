"""
Thread-safe Circuit Breaker Pattern
Prevents cascading failures by temporarily blocking requests when error threshold is exceeded
"""

import time
import threading
from collections import deque
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    max_failures: int = 10
    window_seconds: int = 30
    open_duration_seconds: int = 15
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Thread-safe circuit breaker implementation
    
    Usage:
        breaker = CircuitBreaker()
        
        if breaker.is_open():
            raise ServiceUnavailableError()
        
        try:
            result = do_operation()
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            raise
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failures = deque()
        self._open_until: float = 0.0
        self._half_open_calls: int = 0
        self._lock = threading.RLock()
    
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if time.time() >= self._open_until:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    return False
                return True
            
            if self._state == CircuitState.HALF_OPEN:
                # Limit calls in half-open state
                if self._half_open_calls >= self.config.half_open_max_calls:
                    return True
            
            return False
    
    def record_failure(self):
        """Record a failure and potentially open the circuit"""
        with self._lock:
            now = time.time()
            
            # Remove old failures outside the window
            while self._failures and now - self._failures[0] > self.config.window_seconds:
                self._failures.popleft()
            
            # Add new failure
            self._failures.append(now)
            
            # Check if we should open the circuit
            if len(self._failures) >= self.config.max_failures:
                self._state = CircuitState.OPEN
                self._open_until = now + self.config.open_duration_seconds
                self._failures.clear()
            
            # If in half-open state, failure means back to open
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._open_until = now + self.config.open_duration_seconds
                self._half_open_calls = 0
    
    def record_success(self):
        """Record a success and potentially close the circuit"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                
                # After enough successful calls in half-open, close the circuit
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failures.clear()
                    self._half_open_calls = 0
            
            elif self._state == CircuitState.CLOSED:
                # Decay failures on success
                if self._failures:
                    # Remove oldest failures (reward good behavior)
                    remove_count = max(1, len(self._failures) // 2)
                    for _ in range(remove_count):
                        if self._failures:
                            self._failures.popleft()
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            return self._state
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        with self._lock:
            now = time.time()
            
            # Clean old failures
            while self._failures and now - self._failures[0] > self.config.window_seconds:
                self._failures.popleft()
            
            return {
                "state": self._state.value,
                "failures_in_window": len(self._failures),
                "max_failures": self.config.max_failures,
                "window_seconds": self.config.window_seconds,
                "open_until": self._open_until if self._state == CircuitState.OPEN else None,
                "half_open_calls": self._half_open_calls if self._state == CircuitState.HALF_OPEN else None
            }
    
    def reset(self):
        """Manually reset the circuit breaker"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures.clear()
            self._open_until = 0.0
            self._half_open_calls = 0


def with_circuit_breaker(breaker: CircuitBreaker):
    """
    Decorator to wrap functions with circuit breaker protection
    
    Usage:
        @with_circuit_breaker(my_breaker)
        async def my_function():
            # function code
    """
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs) -> Any:
            if breaker.is_open():
                raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        def sync_wrapper(*args, **kwargs) -> Any:
            if breaker.is_open():
                raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
