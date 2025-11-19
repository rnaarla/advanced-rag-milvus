"""
Thread-safe Circuit Breaker Pattern
Prevents cascading failures by temporarily blocking requests when error threshold is exceeded.

This implementation matches the richer semantics exercised in tests:
- CLOSED -> OPEN when failures reach a threshold
- OPEN -> HALF_OPEN after a timeout
- HALF_OPEN -> CLOSED after a number of successful probes
It also tracks basic statistics for observability.
"""

import time
import threading
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration.

    Supports both the newer names (failure_threshold, timeout_seconds,
    success_threshold) and the legacy names (max_failures, window_seconds,
    open_duration_seconds, half_open_max_calls) used in some tests.
    """

    # Primary names used by the main test-suite
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    success_threshold: int = 2

    # Backwards-compatible aliases expected by older tests
    max_failures: int | None = None
    window_seconds: float | None = None
    open_duration_seconds: float | None = None
    half_open_max_calls: int | None = None

    def __post_init__(self):
        # Map legacy fields to primary ones when provided.
        if self.max_failures is not None:
            self.failure_threshold = self.max_failures
        if self.window_seconds is not None:
            self.timeout_seconds = self.window_seconds
        if self.open_duration_seconds is not None:
            # For legacy configs we treat "open_duration_seconds" as the
            # timeout period after which we transition to HALF_OPEN.
            self.timeout_seconds = self.open_duration_seconds
        if self.half_open_max_calls is not None:
            self.success_threshold = self.half_open_max_calls


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and calls are rejected."""


class CircuitBreaker:
    """
    Thread-safe circuit breaker implementation.

    Usage:
        breaker = CircuitBreaker()

        if breaker.is_open():
            raise CircuitBreakerOpenError("Circuit is open")

        try:
            result = do_operation()
            breaker.record_success()
            return result
        except Exception:
            breaker.record_failure()
            raise
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()

        self._state: CircuitState = CircuitState.CLOSED
        self._failures: int = 0
        self._successes: int = 0
        self._total_requests: int = 0
        self._state_transitions: int = 0

        self._opened_at: Optional[float] = None
        self._half_open_successes: int = 0

        self._lock = threading.RLock()

    def _transition_to(self, new_state: CircuitState) -> None:
        """Internal helper to change state and track transitions."""
        if new_state is self._state:
            return
        self._state = new_state
        self._state_transitions += 1

        if new_state is CircuitState.OPEN:
            self._opened_at = time.time()
            self._half_open_successes = 0
        elif new_state is CircuitState.CLOSED:
            # Reset counters when fully closed again
            self._opened_at = None
            self._half_open_successes = 0
            self._failures = 0
            self._successes = 0

    def is_open(self) -> bool:
        """
        Check if the circuit is open (blocking requests).

        - In OPEN: returns True until timeout elapses, then transitions to HALF_OPEN.
        - In HALF_OPEN or CLOSED: returns False (requests allowed).
        """
        with self._lock:
            if self._state is CircuitState.OPEN:
                if self._opened_at is None:
                    return True
                if time.time() - self._opened_at >= self.config.timeout_seconds:
                    # Move to HALF_OPEN and allow a limited number of probes
                    self._transition_to(CircuitState.HALF_OPEN)
                    return False
                return True
            # CLOSED and HALF_OPEN allow requests
            return False

    def record_failure(self) -> None:
        """Record a failure and potentially open/reopen the circuit."""
        with self._lock:
            self._total_requests += 1

            # If we've been OPEN long enough, move to HALF_OPEN even if
            # is_open() hasn't been called explicitly.
            if (
                self._state is CircuitState.OPEN
                and self._opened_at is not None
                and time.time() - self._opened_at >= self.config.timeout_seconds
            ):
                self._transition_to(CircuitState.HALF_OPEN)

            # Failure during HALF_OPEN immediately re-opens the circuit.
            if self._state is CircuitState.HALF_OPEN:
                self._failures += 1
                self._transition_to(CircuitState.OPEN)
                return

            # Normal failure accounting in CLOSED/OPEN
            self._failures += 1

            if self._state is CircuitState.CLOSED and self._failures >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        with self._lock:
            self._total_requests += 1

            # Similar timeout check as in record_failure to ensure we leave
            # OPEN even if is_open() was not consulted.
            if (
                self._state is CircuitState.OPEN
                and self._opened_at is not None
                and time.time() - self._opened_at >= self.config.timeout_seconds
            ):
                self._transition_to(CircuitState.HALF_OPEN)

            if self._state is CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                self._successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                return

            # In CLOSED, simply accumulate successes. In OPEN, successes should
            # not normally be recorded since calls are blocked by is_open().
            if self._state is CircuitState.CLOSED:
                self._successes += 1

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    def get_stats(self) -> dict:
        """Get circuit breaker statistics for monitoring."""
        with self._lock:
            stats = {
                "state": self._state.value,
                "failures": self._failures,
                "successes": self._successes,
                "total_requests": self._total_requests,
                "state_transitions": self._state_transitions,
            }
            # Backwards-compatible key used in older tests
            stats["failures_in_window"] = self._failures
            return stats

    def reset(self) -> None:
        """Manually reset the circuit breaker to a clean CLOSED state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._total_requests = 0
            self._state_transitions = 0


def with_circuit_breaker(breaker: CircuitBreaker):
    """
    Decorator to wrap functions with circuit breaker protection.

    Usage:
        @with_circuit_breaker(my_breaker)
        async def my_function():
            ...
    """

    def decorator(func: Callable) -> Callable:
        import asyncio

        async def async_wrapper(*args, **kwargs) -> Any:
            if breaker.is_open():
                raise CircuitBreakerOpenError("Circuit is open")

            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception:
                breaker.record_failure()
                raise

        def sync_wrapper(*args, **kwargs) -> Any:
            if breaker.is_open():
                raise CircuitBreakerOpenError("Circuit is open")

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception:
                breaker.record_failure()
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
