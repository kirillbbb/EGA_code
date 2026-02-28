from __future__ import annotations

from experiment import format_report, generate_test_instances, run_experiment, save_results_csv
from ga_solver import PairingMode
from models import GAConfig


def main() -> None:
    instances = generate_test_instances(n_tasks=18, n_items=15, seed=42)

    ega1_mode: PairingMode = "outbreeding"
    ega2_mode: PairingMode = "random"

    cfg = GAConfig(
        population_size=60,
        offspring_size=60,
        generations=200,
        tournament_k=3,
        mutation_prob=0.2,
        random_seed=123,
        pairing_mode=ega1_mode,
    )

    rows, avg_d1, avg_d2 = run_experiment(instances, cfg, ega1_mode=ega1_mode, ega2_mode=ega2_mode)
    print(format_report(rows, avg_d1, avg_d2, ega1_mode=ega1_mode, ega2_mode=ega2_mode))
    save_results_csv(rows, "results.csv")


if __name__ == "__main__":
    main()
