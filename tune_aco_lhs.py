"""
Parameter tuning using Latin Hypercube Sampling (DOE approach) for ACO
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import qmc

from run_aco_tuning import run_aco_for_tuning

def latin_hypercube_sampling_aco(n_samples: int = 80):
    """
    Parameters to tune (typical ACO ranges):
    - n_ants: [8, 60]
    - max_iters: [60, 500]
    - alpha: [0.5, 2.5]
    - beta: [1.5, 6.0]
    - evap: [0.10, 0.60]
    - Q: [50, 500]
    - q0: [0.00, 0.50]
    - subset_extra_frac: [0.00, 0.15]
    - local_search_rate: [0.00, 0.35]
    - early_stop_iters: [10, 180]
    """

    bounds = {
        "n_ants": (8, 60),
        "max_iters": (60, 500),
        "alpha": (0.5, 2.5),
        "beta": (1.5, 6.0),
        "evap": (0.10, 0.60),
        "Q": (50.0, 500.0),
        "q0": (0.0, 0.5),
        "subset_extra_frac": (0.0, 0.15),
        "local_search_rate": (0.0, 0.35),
        "early_stop_iters": (10, 180),
    }

    names = list(bounds.keys())
    sampler = qmc.LatinHypercube(d=len(names), seed=42)
    samples = sampler.random(n=n_samples)

    configs = []
    for s in samples:
        cfg = {}
        for i, name in enumerate(names):
            lo, hi = bounds[name]
            val = lo + s[i] * (hi - lo)
            if name in ["n_ants", "max_iters", "early_stop_iters"]:
                val = int(round(val))
            cfg[name] = val
        # Keep elite deposit always on (like your EA uses elitism)
        cfg["elite_deposit"] = True
        configs.append(cfg)
    return configs


def evaluate_configuration(config: dict, instance_files: list, n_runs: int = 3, fairness_type: str = "jain"):
    objectives = []
    for instance_file in instance_files:
        for seed in range(n_runs):
            obj = run_aco_for_tuning(instance_file, seed, fairness_type=fairness_type, **config)
            objectives.append(obj)
    return float(np.mean(objectives))


def tune_aco_parameters_lhs(instance_size: int, n_samples: int = 80, n_train_instances: int = 5,
                            fairness_type: str = "jain"):
    print(f"\n{'='*80}")
    print(f"ACO TUNING FOR SIZE {instance_size} - Latin Hypercube Sampling")
    print(f"{'='*80}")
    print(f"Configurations: {n_samples} | Training instances: {n_train_instances} | Fairness: {fairness_type}")

    train_dir = f"./instances/{instance_size}/train"
    instance_files = [
        os.path.join(train_dir, f)
        for f in sorted(os.listdir(train_dir))
        if f.endswith(".txt")
    ][:n_train_instances]

    if not instance_files:
        raise RuntimeError(f"No training instances found in {train_dir}")

    configs = latin_hypercube_sampling_aco(n_samples=n_samples)
    results = []

    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{n_samples}] Testing config: ants={cfg['n_ants']}, iters={cfg['max_iters']}, "
              f"alpha={cfg['alpha']:.3f}, beta={cfg['beta']:.3f}, evap={cfg['evap']:.3f}, q0={cfg['q0']:.3f}")

        avg_obj = evaluate_configuration(cfg, instance_files, n_runs=3, fairness_type=fairness_type)
        print(f"  → Average objective: {avg_obj:.2f}")

        row = cfg.copy()
        row["avg_objective"] = avg_obj
        row["instance_size"] = instance_size
        row["fairness_type"] = fairness_type
        results.append(row)

    df = pd.DataFrame(results).sort_values("avg_objective")
    out_dir = f"./tuning_aco/{instance_size}"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/lhs_results.csv", index=False)
    print(f"\nResults saved to {out_dir}/lhs_results.csv")

    best = df.iloc[0]
    print(f"\n{'='*80}")
    print("BEST ACO CONFIG")
    print(f"{'='*80}")
    for k in ["n_ants", "max_iters", "alpha", "beta", "evap", "Q", "q0", "subset_extra_frac", "local_search_rate", "early_stop_iters"]:
        print(f"{k:18s}: {best[k]}")
    print(f"Average objective: {best['avg_objective']:.2f}")
    print(f"{'='*80}\n")

    return df


if __name__ == "__main__":
    tuning_config = {
        50:   {"samples": 80, "instances": 8},
        100:  {"samples": 80, "instances": 8},
        200:  {"samples": 70, "instances": 6},
        500:  {"samples": 60, "instances": 5},
        1000: {"samples": 50, "instances": 4},
        2000: {"samples": 40, "instances": 3},
        5000: {"samples": 30, "instances": 2},
        10000: {"samples": 20, "instances": 2},
    }

    START_SIZE = 2000

    all_results = {}
    for size, cfg in tuning_config.items():
        if size < START_SIZE:
            continue

        train_dir = f"./instances/{size}/train"
        if not os.path.exists(train_dir):
            print(f"\n⚠ Skipping size {size}: directory {train_dir} not found")
            continue

        try:
            df = tune_aco_parameters_lhs(
                instance_size=size,
                n_samples=cfg["samples"],
                n_train_instances=cfg["instances"],
                fairness_type="jain"
            )
            all_results[size] = df
            print(f"\n✓ Completed ACO tuning for size {size}")
        except Exception as e:
            print(f"\n✗ Error tuning size {size}: {e}")
            import traceback

            traceback.print_exc()

    # summary
    if all_results:
        summary = []
        for size, df in all_results.items():
            best = df.iloc[0]
            summary.append({
                "size": size,
                "n_ants": int(best["n_ants"]),
                "max_iters": int(best["max_iters"]),
                "alpha": round(float(best["alpha"]), 4),
                "beta": round(float(best["beta"]), 4),
                "evap": round(float(best["evap"]), 4),
                "Q": round(float(best["Q"]), 2),
                "q0": round(float(best["q0"]), 4),
                "subset_extra_frac": round(float(best["subset_extra_frac"]), 4),
                "local_search_rate": round(float(best["local_search_rate"]), 4),
                "early_stop_iters": int(best["early_stop_iters"]),
                "avg_objective": round(float(best["avg_objective"]), 2),
            })
        sdf = pd.DataFrame(summary).sort_values("size")
        os.makedirs("./tuning_aco", exist_ok=True)
        sdf.to_csv("./tuning_aco/tuning_summary.csv", index=False)
        print("\n" + sdf.to_string(index=False))
        print("\n✓ Summary saved to ./tuning_aco/tuning_summary.csv")
