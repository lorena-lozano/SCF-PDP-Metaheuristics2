"""
Simple ACO Runner - Runs all instances with ALL fairness measures
Writes results incrementally with fairness type in folder structure

Usage:
    python run_aco_simple.py
"""

import os
import sys
import csv
import time
from typing import Dict, List, Optional

import pandas as pd

from SCFDP import SCFPDPInstance, write_solution, get_instance_name
from aco_alg import run_aco_with_tracking, get_aco_config

# ============================================================
# CONFIG
# ============================================================

FAIRNESS_MEASURES = ["jain", "maxmin", "gini"]
INSTANCES_ROOT = "./instances"
OUTPUT_DIR = "./aco_results"
SAVE_SOLUTIONS = True
SAVE_CONVERGENCE = True

# Tuning summary (best params per size)
TUNING_SUMMARY_PATH = "./tuning_aco/tuning_summary.csv"

# Columns in tuning_summary that we want to apply to the config
TUNED_COLS = [
    "n_ants",
    "max_iters",
    "alpha",
    "beta",
    "evap",
    "Q",
    "q0",
    "subset_extra_frac",
    "local_search_rate",
    "early_stop_iters",
]

INSTANCES_CONFIG = {
    "50": None,
    "100": None,
    "200": None,
    "500": None,
    "1000": None,
    "2000": 15,
    "5000": 10,
    "10000": 5,
}

# ============================================================
# Tuning summary helpers
# ============================================================

def _safe_int(x) -> int:
    return int(round(float(x)))

def _safe_float(x) -> float:
    return float(x)

def load_tuning_summary(path: str) -> Dict[int, Dict]:
    """
    Reads tuning_summary.csv and returns:
        { size_int: {param: value, ... , "avg_objective": ...}, ... }

    Handles size being stored as 50.0 etc.
    """
    if not os.path.exists(path):
        print(f"[TUNING] No tuning summary found at: {path} (will use get_aco_config defaults)")
        return {}

    df = pd.read_csv(path)

    if "size" not in df.columns:
        print(f"[TUNING] '{path}' has no 'size' column (will use defaults)")
        return {}

    # Build mapping
    tuning_map: Dict[int, Dict] = {}
    for _, row in df.iterrows():
        try:
            size_int = _safe_int(row["size"])
        except Exception:
            continue

        entry = {}
        if "avg_objective" in df.columns and pd.notna(row.get("avg_objective", None)):
            entry["avg_objective"] = _safe_float(row["avg_objective"])

        for c in TUNED_COLS:
            if c in df.columns and pd.notna(row.get(c, None)):
                entry[c] = row[c]
        tuning_map[size_int] = entry

    print(f"[TUNING] Loaded tuning summary rows: {len(tuning_map)} from {path}")
    return tuning_map

def apply_tuned_params(base_config: Dict, tuned_entry: Dict) -> Dict:
    """
    Returns a new config = base_config overridden with tuned params.
    Casts to expected types.
    """
    cfg = dict(base_config)

    if not tuned_entry:
        return cfg

    # ints
    for k in ["n_ants", "max_iters", "early_stop_iters"]:
        if k in tuned_entry:
            cfg[k] = _safe_int(tuned_entry[k])

    # floats
    for k in ["alpha", "beta", "evap", "Q", "q0", "subset_extra_frac", "local_search_rate"]:
        if k in tuned_entry:
            cfg[k] = _safe_float(tuned_entry[k])

    return cfg

# ============================================================
# Incremental CSV writer
# ============================================================

class IncrementalCSVWriter:
    def __init__(self, filename: str, fieldnames: List[str]):
        self.filename = filename
        self.fieldnames = fieldnames
        self.is_new_file = not os.path.exists(filename)
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        if self.is_new_file:
            with open(self.filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def append_row(self, row: Dict):
        formatted = row.copy()
        for key in ["objective", "total_duration", "fairness", "runtime", "avg_duration",
                    "min_duration", "max_duration", "avg_stops_per_route"]:
            if key in formatted and isinstance(formatted[key], float):
                if formatted[key] == float("inf"):
                    formatted[key] = "INF"
                else:
                    formatted[key] = f"{formatted[key]:.4f}"

        with open(self.filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(formatted)

        print(f"  ✓ Result appended to: {self.filename}")


def load_completed_instances(results_file: str) -> set:
    completed = set()
    if not os.path.exists(results_file):
        return completed
    with open(results_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "instance" not in reader.fieldnames:
            return completed
        for row in reader:
            inst = row.get("instance")
            if inst:
                completed.add(inst)
    return completed


def select_instances(instances_root: str) -> Dict[str, List[str]]:
    instances_by_size = {}
    for size, limit in INSTANCES_CONFIG.items():
        train_dir = os.path.join(instances_root, size, "train")
        if not os.path.exists(train_dir):
            print(f"  ⚠ Not found: {train_dir}")
            continue

        instance_files = sorted(
            os.path.join(train_dir, f)
            for f in os.listdir(train_dir)
            if f.endswith(".txt")
        )
        if limit is None:
            instances_by_size[size] = instance_files
            print(f"  ✓ Size {size}: ALL ({len(instance_files)}) instances")
        else:
            instances_by_size[size] = instance_files[:limit]
            print(f"  ✓ Size {size}: {limit} instances (out of {len(instance_files)})")
    return instances_by_size


def save_convergence_data(convergence_data: List[Dict], instance_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    conv_file = os.path.join(output_dir, f"{instance_name}_convergence.csv")
    if not convergence_data:
        return
    fieldnames = ["iteration", "objective", "total_duration", "fairness",
                  "num_served", "num_active_routes", "total_stops", "feasible", "iter_feasible_count"]
    with open(conv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in convergence_data:
            formatted = row.copy()
            for key in ["objective", "total_duration", "fairness"]:
                if key in formatted and isinstance(formatted[key], float):
                    formatted[key] = "INF" if formatted[key] == float("inf") else f"{formatted[key]:.4f}"
            writer.writerow(formatted)


def run_instance(instance_file: str, fairness_type: str, config: Dict,
                 output_dir: str, results_writer: IncrementalCSVWriter):
    inst_name = get_instance_name(instance_file)
    print(f"\nProcessing: {inst_name} [{fairness_type.upper()}]")

    instance = SCFPDPInstance(instance_file)
    print(f"  {instance.n} req, {instance.n_K} veh, cap={instance.C}, gamma={instance.gamma}")

    start = time.time()
    solution, tracking = run_aco_with_tracking(instance, fairness_type=fairness_type, **config)
    runtime = time.time() - start

    feasible, _ = solution.is_feasible()
    stats = solution.get_statistics()
    total_stops = sum(len(route.stops) for route in solution.routes)

    result = {
        "instance": inst_name,
        "fairness_measure": fairness_type,
        "instance_size": instance.n,
        "num_vehicles": instance.n_K,
        "capacity": instance.C,
        "min_requests": instance.gamma,
        "rho": instance.rho,
        "feasible": feasible,
        "objective": stats["objective"] if feasible else float("inf"),
        "total_duration": stats["total_duration"],
        "fairness": stats["fairness"],
        "num_served": stats["num_served"],
        "num_active_routes": stats["num_active_routes"],
        "total_stops": total_stops,
        "avg_stops_per_route": total_stops / instance.n_K if instance.n_K > 0 else 0,
        "min_duration": stats["min_duration"],
        "max_duration": stats["max_duration"],
        "avg_duration": stats["avg_duration"],
        "runtime": runtime,
        "total_iters": tracking.get("total_iters", 0),
    }

    print(f"  Result: {'FEASIBLE' if feasible else 'INFEASIBLE'}, "
          f"Obj={'%.2f' % result['objective'] if feasible else 'INF'}, "
          f"Fairness={result['fairness']:.4f}, Runtime={runtime:.1f}s")

    results_writer.append_row(result)

    if SAVE_SOLUTIONS and feasible:
        sol_file = os.path.join(output_dir, "solutions", f"{inst_name}_solution.txt")
        write_solution(solution, sol_file, inst_name)

    if SAVE_CONVERGENCE:
        conv_dir = os.path.join(output_dir, "convergence")
        save_convergence_data(tracking.get("best_per_iteration", []), inst_name, conv_dir)


def main():
    print("=" * 80)
    print("ANT COLONY OPTIMIZATION - ALL FAIRNESS MEASURES")
    print("=" * 80)
    print(f"Fairness measures: {', '.join(FAIRNESS_MEASURES)}")
    print("=" * 80)

    tuning_map = load_tuning_summary(TUNING_SUMMARY_PATH)

    instances_by_size = select_instances(INSTANCES_ROOT)
    if not instances_by_size:
        print("✗ No instances found!")
        return

    result_fieldnames = [
        "instance", "fairness_measure", "instance_size", "num_vehicles",
        "capacity", "min_requests", "rho", "feasible", "objective",
        "total_duration", "fairness", "num_served", "num_active_routes",
        "total_stops", "avg_stops_per_route", "min_duration", "max_duration",
        "avg_duration", "runtime", "total_iters"
    ]

    for size in sorted(instances_by_size.keys(), key=int):
        size_int = int(size)

        # Base config from code defaults
        base_config = get_aco_config(size_int)

        # Override with tuned params if available
        tuned_entry = tuning_map.get(size_int, {})
        aco_config = apply_tuned_params(base_config, tuned_entry)

        if tuned_entry:
            avg_obj = tuned_entry.get("avg_objective", None)
            if avg_obj is not None:
                print(f"\n[TUNING] size={size_int}: applying tuned params (best avg_objective={avg_obj:.4f})")
            else:
                print(f"\n[TUNING] size={size_int}: applying tuned params")
            print("[TUNING] overrides:", {k: aco_config[k] for k in TUNED_COLS if k in aco_config})
        else:
            print(f"\n[TUNING] size={size_int}: no tuned entry found -> using get_aco_config defaults")

        for fairness_type in FAIRNESS_MEASURES:
            print(f"\n{'='*80}")
            print(f"SIZE: {size} | FAIRNESS: {fairness_type.upper()}")
            print(f"{'='*80}")

            size_fair_dir = os.path.join(OUTPUT_DIR, size, fairness_type)
            os.makedirs(size_fair_dir, exist_ok=True)
            os.makedirs(os.path.join(size_fair_dir, "solutions"), exist_ok=True)
            os.makedirs(os.path.join(size_fair_dir, "convergence"), exist_ok=True)

            results_file = os.path.join(size_fair_dir, "results.csv")

            print(f"  Looking for: {results_file}")
            print(f"  File exists: {os.path.exists(results_file)}")

            completed = load_completed_instances(results_file)
            print(f"  Completed instances found: {len(completed)}")

            writer = IncrementalCSVWriter(results_file, result_fieldnames)

            size_instances = instances_by_size[size]
            print(f"Instances: {len(size_instances)}, Completed: {len(completed)}\n")

            for i, inst_file in enumerate(size_instances, 1):
                inst_name = get_instance_name(inst_file)
                if inst_name in completed:
                    print(f"[{i}/{len(size_instances)}] SKIP: {inst_name}")
                    continue

                print(f"[{i}/{len(size_instances)}]", end=" ")
                try:
                    run_instance(inst_file, fairness_type, aco_config, size_fair_dir, writer)
                except KeyboardInterrupt:
                    print("\n\nINTERRUPTED")
                    sys.exit(0)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
