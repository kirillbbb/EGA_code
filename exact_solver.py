from __future__ import annotations

from itertools import product

from models import KnapsackInstance, KnapsackSolution


def solve_exact(instance: KnapsackInstance) -> KnapsackSolution:
    """Solve 0-1 knapsack by full enumeration of all 2^n binary vectors."""
    best_value = -1
    best_weight = 0
    best_chromosome = None

    for chromosome in product((0, 1), repeat=instance.n_items):
        total_weight = sum(w * bit for w, bit in zip(instance.weights, chromosome))
        if total_weight > instance.capacity:
            continue
        total_value = sum(v * bit for v, bit in zip(instance.values, chromosome))
        if total_value > best_value:
            best_value = total_value
            best_weight = total_weight
            best_chromosome = chromosome

    if best_chromosome is None:
        best_chromosome = tuple(0 for _ in range(instance.n_items))
        best_value = 0

    return KnapsackSolution(
        chromosome=best_chromosome,
        total_weight=best_weight,
        total_value=best_value,
    )