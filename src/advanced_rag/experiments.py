"""
Experiment and bandit utilities for adaptive configuration selection.

This module provides a lightweight, in-memory ExperimentManager that can
be used to choose among different retrieval / ranking configurations and
update their weights based on feedback or rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random


@dataclass
class VariantStats:
    """Simple statistics for a single experiment variant."""

    trials: int = 0
    reward_sum: float = 0.0

    @property
    def average_reward(self) -> float:
        return self.reward_sum / self.trials if self.trials > 0 else 0.0


@dataclass
class ExperimentState:
    """State for an experiment with multiple variants."""

    variants: Dict[str, VariantStats] = field(default_factory=dict)


class ExperimentManager:
    """
    In-memory epsilon-greedy bandit for experiment management.

    Intended use:
        mgr = ExperimentManager(epsilon=0.1)
        mgr.register_experiment("retrieval_profile", ["baseline", "mmr", "learned_ranker"])
        variant = mgr.choose_variant("retrieval_profile")
        ...
        mgr.record_outcome("retrieval_profile", variant, reward=1.0)
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon
        self._experiments: Dict[str, ExperimentState] = {}

    def register_experiment(self, name: str, variants: List[str]) -> None:
        """Register a new experiment or extend variants for an existing one."""
        state = self._experiments.setdefault(name, ExperimentState())
        for v in variants:
            if v not in state.variants:
                state.variants[v] = VariantStats()

    def choose_variant(self, name: str) -> Optional[str]:
        """
        Choose a variant for the given experiment.

        Uses epsilon-greedy strategy:
        - With probability epsilon, chooses a random variant.
        - Otherwise chooses the variant with the highest average reward.
        If the experiment is unknown or has no variants, returns None.
        """
        state = self._experiments.get(name)
        if not state or not state.variants:
            return None

        variants = list(state.variants.keys())

        # Ensure deterministic behaviour in tests by allowing epsilon=0
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(variants)

        # Greedy selection by average reward, ties broken lexicographically
        best_variant = None
        best_reward = float("-inf")
        for v in sorted(variants):
            avg = state.variants[v].average_reward
            if avg > best_reward:
                best_reward = avg
                best_variant = v
        return best_variant

    def record_outcome(self, name: str, variant: str, reward: float) -> None:
        """Record a scalar reward outcome for the given experiment/variant."""
        state = self._experiments.get(name)
        if state is None or variant not in state.variants:
            # Auto-register unseen experiment/variant for robustness.
            self.register_experiment(name, [variant])
            state = self._experiments[name]
        stats = state.variants[variant]
        stats.trials += 1
        stats.reward_sum += float(reward)

    def get_stats(self, name: str) -> Dict[str, Dict[str, float]]:
        """Return a serializable view of stats for the given experiment."""
        state = self._experiments.get(name)
        if not state:
            return {}
        return {
            variant: {
                "trials": float(stats.trials),
                "reward_sum": float(stats.reward_sum),
                "average_reward": float(stats.average_reward),
            }
            for variant, stats in state.variants.items()
        }



