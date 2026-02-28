from experiment import (
    generate_test_instances,
    run_experiment,
    format_report,
    save_results_csv,
)
from models import GAConfig


def main() -> None:

    instances = generate_test_instances(n_tasks=18, n_items=15, seed=42)

    # ЭГА1 — отрицательное ассоциативное
    ega1_mode = "negative_associative_fitness"

    # ЭГА2 — случайное
    ega2_mode = "random"

    # Ослабленные параметры
    cfg = GAConfig(
        population_size=30,
        offspring_size=30,
        generations=60,
        tournament_k=2,
        mutation_prob=0.1,
        random_seed=123,
        pairing_mode=ega1_mode,
    )

    rows, avg_d1, avg_d2 = run_experiment(
        instances,
        cfg,
        ega1_mode=ega1_mode,
        ega2_mode=ega2_mode,
    )

    print(format_report(rows, avg_d1, avg_d2, ega1_mode, ega2_mode))
    save_results_csv(rows, "results.csv")


if __name__ == "__main__":
    main()