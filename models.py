from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class KnapsackInstance:
    """0-1 knapsack test instance."""

    task_id: int
    weights: Sequence[int]
    values: Sequence[int]
    capacity: int

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.values):
            raise ValueError("weights and values must have the same length")
        if len(self.weights) == 0:
            raise ValueError("instance must contain at least one item")
        if self.capacity < 0:
            raise ValueError("capacity must be non-negative")

    @property
    def n_items(self) -> int:
        return len(self.weights)


@dataclass(frozen=True)
class KnapsackSolution:
    chromosome: Sequence[int]
    total_weight: int
    total_value: int


@dataclass(frozen=True)
class ExperimentRow:
    task_id: int
    f_opt: int
    ega1_value: int
    ega1_delta_percent: float
    ega2_value: int
    ega2_delta_percent: float


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 60
    offspring_size: int = 60
    generations: int = 200
    tournament_k: int = 3
    mutation_prob: float = 0.2
    random_seed: int | None = None
    pairing_mode: str = "random"
