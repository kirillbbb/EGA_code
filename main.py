from __future__ import annotations

from experiment import format_report, generate_test_instances, run_experiment
from models import GAConfig


def main() -> None:
    instances = generate_test_instances(n_tasks=18, n_items=15, seed=42)
    cfg = GAConfig(
        population_size=60,
        offspring_size=60,
        generations=200,
        tournament_k=3,
        mutation_prob=0.2,
        random_seed=123,
    )

    rows, avg_d1, avg_d2 = run_experiment(instances, cfg)
    print(format_report(rows, avg_d1, avg_d2))


if __name__ == "__main__":
    main()
