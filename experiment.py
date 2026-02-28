from __future__ import annotations

import csv
import random
from statistics import mean

from exact_solver import solve_exact
from ga_solver import PairingMode, solve_ga
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


def run_experiment(
    instances: list[KnapsackInstance],
    ga_cfg: GAConfig,
    ega1_mode: PairingMode = "outbreeding",
    ega2_mode: PairingMode = "random",
) -> tuple[list[ExperimentRow], float, float]:
    rows: list[ExperimentRow] = []

    for instance in instances:
        opt = solve_exact(instance)

        ega1_cfg = GAConfig(**{**ga_cfg.__dict__, "pairing_mode": ega1_mode, "random_seed": (ga_cfg.random_seed or 0) + instance.task_id})
        ega2_cfg = GAConfig(**{**ga_cfg.__dict__, "pairing_mode": ega2_mode, "random_seed": (ga_cfg.random_seed or 0) + 1000 + instance.task_id})

        ega1 = solve_ga(instance, ega1_cfg)
        ega2 = solve_ga(instance, ega2_cfg)

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


def format_report(
    rows: list[ExperimentRow],
    avg_d1: float,
    avg_d2: float,
    ega1_mode: PairingMode,
    ega2_mode: PairingMode,
) -> str:
    strategy_names = {
        "random": "Панмиксия (случайный выбор)",
        "inbreeding": "Инбридинг",
        "outbreeding": "Аутбридинг",
        "positive_associative": "Положительное ассоциативное",
        "negative_associative_fitness": "Отрицательное ассоциативное (по fitness)",
    }

    lines = [
        "Результаты вычислительного эксперимента",
        f"ЭГА1: {strategy_names[ega1_mode]}",
        f"ЭГА2: {strategy_names[ega2_mode]}",
        "",
        f"{'Задача':<8}{'F_опт':<8}{'ЭГА1':<8}{'δ1, %':<10}{'ЭГА2':<8}{'δ2, %':<10}",
        "-" * 52,
    ]

    for r in rows:
        lines.append(
            f"{r.task_id:<8}{r.f_opt:<8}{r.ega1_value:<8}{r.ega1_delta_percent:<10.2f}{r.ega2_value:<8}{r.ega2_delta_percent:<10.2f}"
        )

    lines.append("-" * 52)
    lines.append("Среднее относительное отклонение")
    lines.append(f"ЭГА1: {avg_d1:.2f}%")
    lines.append(f"ЭГА2: {avg_d2:.2f}%")
    return "\n".join(lines)


def save_results_csv(rows: list[ExperimentRow], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Задача", "F_опт", "ЭГА1", "δ1, %", "ЭГА2", "δ2, %"])
        for r in rows:
            writer.writerow(
                [
                    r.task_id,
                    r.f_opt,
                    r.ega1_value,
                    f"{r.ega1_delta_percent:.2f}",
                    r.ega2_value,
                    f"{r.ega2_delta_percent:.2f}",
                ]
            )
