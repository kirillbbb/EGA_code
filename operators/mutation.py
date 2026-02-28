from __future__ import annotations

import random


def bitflip_mutation(chromosome: list[int], mutation_prob: float, rng: random.Random) -> list[int]:
    """With probability p, flip one random bit."""
    mutated = chromosome[:]
    if mutated and rng.random() < mutation_prob:
        idx = rng.randrange(len(mutated))
        mutated[idx] = 1 - mutated[idx]
    return mutated
