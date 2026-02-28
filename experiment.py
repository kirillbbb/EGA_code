from __future__ import annotations

import random
from statistics import mean

from exact_solver import solve_exact
from ga_solver import solve_ga
from models import ExperimentRow, GAConfig, KnapsackInstance


def generate_test_instances(n_tasks: int = 18, n_items: int = 15, seed: int = 42) -> list[KnapsackInstance]:
    rng = random.Random(seed)
    instances: list[KnapsackInstance] = []

    for task_id in range(1, n_tasks + 1):
        weights = [rng.randint(1, 30) for _ in range(n_items)]
        values = [rng.randint(5, 100) for _ in range(n_items)]
        capacity = int(sum(weights) * rng.uniform(0.35, 0.55))
        instances.append(
            KnapsackInstance(task_id=task_id, weights=weights, values=values, capacity=capacity)
        )
    return instances


def relative_deviation(approx: int, optimum: int) -> float:
    if optimum == 0:
        return 0.0
    return (approx - optimum) / optimum * 100.0


def run_experiment(instances: list[KnapsackInstance], ga_cfg: GAConfig) -> tuple[list[ExperimentRow], float, float]:
    rows: list[ExperimentRow] = []

    for instance in instances:
        opt = solve_exact(instance)

        ega1_cfg = GAConfig(**{**ga_cfg.__dict__, "random_seed": (ga_cfg.random_seed or 0) + instance.task_id})
        ega2_cfg = GAConfig(**{**ga_cfg.__dict__, "random_seed": (ga_cfg.random_seed or 0) + 1000 + instance.task_id})

        ega1 = solve_ga(instance, ega1_cfg, pairing_mode="negative_associative")
        ega2 = solve_ga(instance, ega2_cfg, pairing_mode="random")

        d1 = relative_deviation(ega1.total_value, opt.total_value)
        d2 = relative_deviation(ega2.total_value, opt.total_value)

        rows.append(
            ExperimentRow(
                task_id=instance.task_id,
                f_opt=opt.total_value,
                ega1_value=ega1.total_value,
                ega1_delta_percent=d1,
                ega2_value=ega2.total_value,
                ega2_delta_percent=d2,
            )
        )

    avg_d1 = mean(r.ega1_delta_percent for r in rows)
    avg_d2 = mean(r.ega2_delta_percent for r in rows)
    return rows, avg_d1, avg_d2


def format_report(rows: list[ExperimentRow], avg_d1: float, avg_d2: float) -> str:
    header = (
        f"{'Task':<6}{'F_opt':<8}{'EGA1':<8}{'δ1, %':<10}{'EGA2':<8}{'δ2, %':<10}\n"
        + "-" * 50
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.task_id:<6}{r.f_opt:<8}{r.ega1_value:<8}{r.ega1_delta_percent:<10.2f}{r.ega2_value:<8}{r.ega2_delta_percent:<10.2f}"
        )
    lines.append("-" * 50)
    lines.append(f"Average δ1: {avg_d1:.2f}%")
    lines.append(f"Average δ2: {avg_d2:.2f}%")
    return "\n".join(lines)
