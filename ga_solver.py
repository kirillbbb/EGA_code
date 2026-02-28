from __future__ import annotations

import random
from typing import Literal

from models import GAConfig, KnapsackInstance, KnapsackSolution
from operators.crossover import two_point_crossover
from operators.mutation import bitflip_mutation
from operators.selection import tournament_select

PairingMode = Literal[
    "random",
    "negative_associative_fitness",
]


def evaluate(instance: KnapsackInstance, chromosome: list[int]) -> tuple[int, int]:
    total_weight = sum(w * bit for w, bit in zip(instance.weights, chromosome))
    total_value = sum(v * bit for v, bit in zip(instance.values, chromosome))
    return total_weight, total_value


def repair(instance: KnapsackInstance, chromosome: list[int]) -> list[int]:
    repaired = chromosome[:]
    total_weight, _ = evaluate(instance, repaired)

    if total_weight <= instance.capacity:
        return repaired

    selected_indices = [i for i, bit in enumerate(repaired) if bit == 1]
    selected_indices.sort(key=lambda i: instance.values[i] / instance.weights[i])

    for idx in selected_indices:
        repaired[idx] = 0
        total_weight -= instance.weights[idx]
        if total_weight <= instance.capacity:
            break

    return repaired


def select_partner(
    parent: list[int],
    parent_fitness: int,
    mating_pool: list[list[int]],
    fitnesses: list[int],
    mode: PairingMode,
    rng: random.Random,
) -> list[int]:

    if mode == "random":
        return rng.choice(mating_pool)[:]

    if mode == "negative_associative_fitness":
        paired = list(zip(mating_pool, fitnesses))
        return max(
            paired,
            key=lambda pair: abs(parent_fitness - pair[1])
        )[0][:]

    raise ValueError(f"Unknown pairing mode: {mode}")


def create_initial_population(instance: KnapsackInstance, cfg: GAConfig, rng: random.Random):
    population = []
    for _ in range(cfg.population_size):
        chromosome = [rng.randint(0, 1) for _ in range(instance.n_items)]
        chromosome = repair(instance, chromosome)
        population.append(chromosome)
    return population


def solve_ga(instance: KnapsackInstance, cfg: GAConfig) -> KnapsackSolution:

    rng = random.Random(cfg.random_seed)
    population = create_initial_population(instance, cfg, rng)

    for _ in range(cfg.generations):

        fitnesses = [evaluate(instance, c)[1] for c in population]
        offspring = []

        while len(offspring) < cfg.offspring_size:

            parent1 = tournament_select(population, fitnesses, cfg.tournament_k, rng)
            parent1_fitness = evaluate(instance, parent1)[1]

            parent2 = select_partner(
                parent1,
                parent1_fitness,
                population,
                fitnesses,
                cfg.pairing_mode,
                rng,
            )

            child1, child2 = two_point_crossover(parent1, parent2, rng)

            child1 = bitflip_mutation(child1, cfg.mutation_prob, rng)
            child2 = bitflip_mutation(child2, cfg.mutation_prob, rng)

            offspring.append(repair(instance, child1))
            if len(offspring) < cfg.offspring_size:
                offspring.append(repair(instance, child2))

        combined = population + offspring
        combined.sort(key=lambda c: evaluate(instance, c)[1], reverse=True)

        population = combined[: cfg.population_size]

    best = max(population, key=lambda c: evaluate(instance, c)[1])
    total_weight, total_value = evaluate(instance, best)

    return KnapsackSolution(
        chromosome=best,
        total_weight=total_weight,
        total_value=total_value,
    )