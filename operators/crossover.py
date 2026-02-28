from __future__ import annotations

import random


def two_point_crossover(parent1: list[int], parent2: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
    """Two-point crossover swapping middle segment."""
    n = len(parent1)
    if n != len(parent2):
        raise ValueError("parents must have equal chromosome lengths")
    if n < 2:
        return parent1[:], parent2[:]

    i, j = sorted(rng.sample(range(n), 2))
    child1 = parent1[:]
    child2 = parent2[:]
    child1[i : j + 1] = parent2[i : j + 1]
    child2[i : j + 1] = parent1[i : j + 1]
    return child1, child2