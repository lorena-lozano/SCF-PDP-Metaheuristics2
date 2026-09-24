# quick_compare_all_methods_v3.py
import time
import random
from pathlib import Path
import importlib
import pandas as pd

from SCFDP import SCFPDPInstance, get_instance_name

# A1 fixed constructors
from a1_constructors_fixed import greedy_construction_fixed, randomized_greedy_fixed

# A2 algorithms
from experiment_runner import run_ea_with_tracking, get_ea_config
from aco_alg import run_aco_with_tracking, get_aco_config


# -------------------------
# QUICK SETTINGS
# -------------------------
FAIRNESS = "jain"
SIZES = [50, 100, 200]
N_INSTANCES_PER_SIZE = 3
REPEATS_STOCH = 3                   # best-of-k for stochastic
INSTANCES_ROOT = Path("./instances2")
USE_TEST_SET = True

OUT_DIR = Path("./quick_compare_out_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Safe imports for A1 metaheuristics
# -------------------------
def try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"[SKIP] Could not import {module_name}: {e}")
        return None


GRASP_mod = try_import("GRASP")
VND_mod = try_import("VND")
TABU_mod = try_import("tabu_search")


def pick_instances(size: int):
    sub = "test" if USE_TEST_SET else "train"
    d = INSTANCES_ROOT / str(size) / sub
    files = sorted([p for p in d.iterdir() if p.suffix == ".txt"])
    if not files:
        raise FileNotFoundError(f"No instances found in {d}")
    return files[: min(N_INSTANCES_PER_SIZE, len(files))]


def eval_solution(sol):
    feas, _msg = sol.is_feasible()
    obj = sol.objective_value() if feas else float("inf")
    return feas, obj


def run_once(runner, inst_file: Path, seed: int | None):
    inst = SCFPDPInstance(str(inst_file))
    if seed is not None:
        random.seed(seed)

    t0 = time.time()
    sol = runner(inst, seed)
    rt = time.time() - t0

    feas, obj = eval_solution(sol)
    return feas, obj, rt


def run_best_of(runner, inst_file: Path, repeats: int, base_seed: int):
    best_obj = float("inf")
    best_feas = False
    best_rt = None

    for r in range(repeats):
        feas, obj, rt = run_once(runner, inst_file, base_seed + r)
        if feas and obj < best_obj:
            best_obj = obj
            best_feas = True
            best_rt = rt

    if not best_feas:
        # at least return runtime from first attempt
        feas, obj, rt = run_once(runner, inst_file, base_seed)
        return feas, obj, rt

    return True, best_obj, best_rt


def build_algorithm_registry():
    """
    Returns list of dicts:
      {name, stochastic, runner(inst, seed)->Solution}
    """
    algs = []

    # A1 fixed constructors
    algs.append({
        "name": "A1_greedy_fixed",
        "stochastic": False,
        "runner": lambda inst, seed: greedy_construction_fixed(inst, fairness_type=FAIRNESS)
    })
    algs.append({
        "name": "A1_rand_greedy_fixed",
        "stochastic": True,
        "runner": lambda inst, seed: randomized_greedy_fixed(inst, fairness_type=FAIRNESS, alpha=0.3, seed=seed)
    })

    # A1 GRASP
    if GRASP_mod is not None:
        for fn_name in ["grasp_N1_first", "grasp_N2_first", "grasp_N3_first"]:
            if hasattr(GRASP_mod, fn_name):
                fn = getattr(GRASP_mod, fn_name)
                algs.append({
                    "name": f"A1_{fn_name}",
                    "stochastic": True,
                    "runner": lambda inst, seed, f=fn: f(inst)  # wrapper ignores seed (GRASP has internal seed)
                })

    # A1 VND
    if VND_mod is not None:
        for fn_name in ["vnd_standard_first", "vnd_standard_best", "vnd_light_first"]:
            if hasattr(VND_mod, fn_name):
                fn = getattr(VND_mod, fn_name)
                algs.append({
                    "name": f"A1_{fn_name}",
                    "stochastic": False,
                    "runner": lambda inst, seed, f=fn: f(inst)
                })

    # A1 Tabu
    if TABU_mod is not None:
        for fn_name in ["greedy_tabu_first", "greedy_tabu_best", "random_tabu_first", "random_tabu_best"]:
            if hasattr(TABU_mod, fn_name):
                fn = getattr(TABU_mod, fn_name)
                # tabu can be deterministic depending on implementation; treat as stochastic if randomized constructor
                is_stoch = fn_name.startswith("random_")
                algs.append({
                    "name": f"A1_{fn_name}",
                    "stochastic": is_stoch,
                    "runner": lambda inst, seed, f=fn: f(inst)
                })

    # EA (A2)
    def ea_runner(inst, seed):
        cfg = get_ea_config(inst.n)
        sol, _track = run_ea_with_tracking(inst, fairness_type=FAIRNESS, **cfg)
        return sol

    algs.append({
        "name": "EA",
        "stochastic": True,
        "runner": ea_runner
    })

    # ACO (A2)
    def aco_runner(inst, seed):
        cfg = get_aco_config(inst.n)
        sol, _track = run_aco_with_tracking(inst, fairness_type=FAIRNESS, **cfg)
        return sol

    algs.append({
        "name": "ACO",
        "stochastic": True,
        "runner": aco_runner
    })

    return algs


def main():
    algs = build_algorithm_registry()
    print("\nAlgorithms included:")
    for a in algs:
        print(f" - {a['name']} (stochastic={a['stochastic']})")

    rows = []

    for size in SIZES:
        files = pick_instances(size)
        print(f"\n=== SIZE {size} | instances={len(files)} ===")

        for f in files:
            inst_name = get_instance_name(str(f))
            base_seed = abs(hash((size, inst_name))) % 10_000_000

            for alg in algs:
                if alg["stochastic"]:
                    feas, obj, rt = run_best_of(alg["runner"], f, REPEATS_STOCH, base_seed)
                else:
                    feas, obj, rt = run_once(alg["runner"], f, None)

                rows.append({
                    "size": size,
                    "instance": inst_name,
                    "algorithm": alg["name"],
                    "feasible": bool(feas),
                    "objective": float(obj),
                    "runtime_sec": float(rt),
                })

                print(f"{inst_name:>18} | {alg['name']:<22} | feas={feas} | obj={obj:.2f} | rt={rt:.2f}s")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "quick_compare_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Summary ranking: feasibility rate first, then avg objective among feasible, then avg runtime
    summary = []
    for alg_name in sorted(df["algorithm"].unique()):
        sub = df[df["algorithm"] == alg_name]
        feas_sub = sub[sub["feasible"] == True]
        feas_rate = 100.0 * len(feas_sub) / len(sub) if len(sub) else 0.0
        avg_obj = feas_sub["objective"].mean() if len(feas_sub) else float("inf")
        avg_rt = sub["runtime_sec"].mean() if len(sub) else float("inf")

        summary.append({
            "algorithm": alg_name,
            "feas_rate_%": feas_rate,
            "avg_obj_feasible": avg_obj,
            "avg_runtime_sec": avg_rt
        })

    sdf = pd.DataFrame(summary).sort_values(
        ["feas_rate_%", "avg_obj_feasible", "avg_runtime_sec"],
        ascending=[False, True, True]
    )
    out_sum = OUT_DIR / "quick_compare_summary.csv"
    sdf.to_csv(out_sum, index=False)

    print(f"Saved: {out_sum}")
    print("\n=== RANKING (feasibility, objective, runtime) ===")
    print(sdf.to_string(index=False))

    print("\nTop-2 candidates (for statistical test):")
    print(sdf.head(2)[["algorithm", "feas_rate_%", "avg_obj_feasible", "avg_runtime_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
