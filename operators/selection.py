from __future__ import annotations

import random
from typing import Sequence


def tournament_select(population: Sequence[list[int]], fitnesses: Sequence[int], k: int, rng: random.Random) -> list[int]:
    """B-tournament selection: sample k and return best by fitness."""
    if not population:
        raise ValueError("population must not be empty")
    k = max(1, min(k, len(population)))
    candidate_indices = rng.sample(range(len(population)), k=k)
    best_idx = max(candidate_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx][:]