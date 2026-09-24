# aco_test_config_comparison.py
"""
Compare Rank 1 vs Rank 2 tuned ACO configurations on INDEPENDENT TEST instances.

What it does:
- For each size in SIZES:
  * Loads ./tuning_aco/<size>/lhs_results.csv (from LHS tuning)
  * Selects Rank 1 and Rank 2 by lowest avg_objective
  * Runs both configs on N_TEST test instances from ./instances/<size>/test
  * Aggregates: feasible_count, avg objective (feasible only), avg runtime (minutes)
  * Computes Diff% = (R1 - R2) / R2 * 100   (negative => R1 better)
  * Declares Winner: R1 / R2 / TIE (if |Diff| < TIE_THRESHOLD_PCT)

Outputs:
- ./aco_config_comparison/results.csv
- ./aco_config_comparison/latex_table.tex
- ./aco_config_comparison/test_comparison.png

Run:
    python aco_test_config_comparison.py
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from SCFDP import SCFPDPInstance
from aco_alg import run_aco_with_tracking, get_aco_config

# ----------------------------
# Settings
# ----------------------------

SIZES = [50, 100, 200, 500, 1000, 2000, 5000]
N_TEST = 10                                     # how many test instances per size
FAIRNESS_TYPE = "jain"

# If abs(diff%) < this => TIE
TIE_THRESHOLD_PCT = 1.0

# Where your LHS tuning outputs are:
#   ./tuning_aco/<size>/lhs_results.csv
TUNING_DIR = Path("./tuning_aco")

INSTANCES_ROOT = Path("./instances2")

OUT_DIR = Path("./aco_config_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns to read from lhs_results.csv for config reconstruction
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

# ----------------------------
# Helpers
# ----------------------------

def _safe_int(x) -> int:
    return int(round(float(x)))

def _safe_float(x) -> float:
    return float(x)

def load_rank_configs(size: int) -> Tuple[Dict, Dict]:
    """
    Returns (rank1_cfg, rank2_cfg) from ./tuning_aco/<size>/lhs_results.csv
    """
    f = TUNING_DIR / str(size) / "lhs_results.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing tuning file: {f}")

    df = pd.read_csv(f)
    if "avg_objective" not in df.columns:
        raise ValueError(f"{f} has no avg_objective column")

    df = df.sort_values("avg_objective", ascending=True).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f"{f} has < 2 rows; cannot build Rank 1 vs Rank 2 comparison")

    def row_to_cfg(row) -> Dict:
        cfg = {}
        for k in TUNED_COLS:
            if k not in df.columns:
                continue
            v = row[k]
            if k in ["n_ants", "max_iters", "early_stop_iters"]:
                cfg[k] = _safe_int(v)
            else:
                cfg[k] = _safe_float(v)
        # keep elitist deposit like tuning did
        cfg["elite_deposit"] = True
        return cfg

    rank1 = row_to_cfg(df.iloc[0])
    rank2 = row_to_cfg(df.iloc[1])
    return rank1, rank2

def select_test_instances(size: int, n_test: int) -> List[Path]:
    test_dir = INSTANCES_ROOT / str(size) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test dir: {test_dir}")

    files = sorted([p for p in test_dir.iterdir() if p.suffix == ".txt"])
    if not files:
        raise FileNotFoundError(f"No .txt instances found in: {test_dir}")

    return files[: min(n_test, len(files))]

def merge_with_base_config(size: int, tuned_cfg: Dict) -> Dict:
    """
    Start from size-dependent base config (includes greedy seeding options etc.)
    and override with tuned parameters.
    """
    base = get_aco_config(size)
    merged = dict(base)
    merged.update(tuned_cfg)
    return merged

def run_on_instances(instance_files: List[Path], size: int, cfg: Dict) -> Dict:
    """
    Runs ACO on instance_files with config cfg.
    Returns aggregated metrics.
    """
    feasible_count = 0
    objectives = []
    runtimes_min = []

    for f in instance_files:
        inst = SCFPDPInstance(str(f))

        t0 = time.time()
        sol, _tracking = run_aco_with_tracking(inst, fairness_type=FAIRNESS_TYPE, **cfg)
        rt = (time.time() - t0) / 60.0

        feas, _msg = sol.is_feasible()
        runtimes_min.append(rt)

        if feas:
            feasible_count += 1
            objectives.append(sol.objective_value())

    avg_obj = float("inf") if feasible_count == 0 else float(sum(objectives) / len(objectives))
    avg_rt = float(sum(runtimes_min) / len(runtimes_min)) if runtimes_min else float("inf")

    return {
        "feasible_count": feasible_count,
        "total": len(instance_files),
        "avg_objective": avg_obj,
        "avg_runtime_min": avg_rt,
    }

def diff_percent(r1_obj: float, r2_obj: float) -> float:
    """
    Diff% = (R1 - R2) / R2 * 100
    Negative => R1 better (lower objective).
    """
    if r2_obj == 0 or r2_obj == float("inf"):
        return 0.0
    return (r1_obj - r2_obj) / r2_obj * 100.0

def pick_winner(d: float) -> str:
    if abs(d) < TIE_THRESHOLD_PCT:
        return "TIE"
    return "R1" if d < 0 else "R2"

# ----------------------------
# Main
# ----------------------------

def main():
    rows = []

    for size in SIZES:
        print(f"\n{'='*80}\nSIZE {size}\n{'='*80}")

        # load Rank1/Rank2 tuned configs
        r1_tuned, r2_tuned = load_rank_configs(size)

        # include base defaults (greedy seed flags etc.) + override tuned fields
        r1_cfg = merge_with_base_config(size, r1_tuned)
        r2_cfg = merge_with_base_config(size, r2_tuned)

        # choose test instances
        test_instances = select_test_instances(size, N_TEST)
        print(f"Test instances: {len(test_instances)}")

        # run both configs
        print("Running Rank 1 ...")
        r1_res = run_on_instances(test_instances, size, r1_cfg)

        print("Running Rank 2 ...")
        r2_res = run_on_instances(test_instances, size, r2_cfg)

        d = diff_percent(r1_res["avg_objective"], r2_res["avg_objective"])
        w = pick_winner(d)

        rows.append({
            "size": size,
            "config": "Rank 1",
            "feasible": f'{r1_res["feasible_count"]}/{r1_res["total"]}',
            "objective": r1_res["avg_objective"],
            "runtime_min": r1_res["avg_runtime_min"],
            "diff_pct": None,
            "winner": None,
        })
        rows.append({
            "size": size,
            "config": "Rank 2",
            "feasible": f'{r2_res["feasible_count"]}/{r2_res["total"]}',
            "objective": r2_res["avg_objective"],
            "runtime_min": r2_res["avg_runtime_min"],
            "diff_pct": d,
            "winner": w,
        })

        print(f"Rank1 obj={r1_res['avg_objective']:.2f} | Rank2 obj={r2_res['avg_objective']:.2f} "
              f"| Diff%={d:+.2f} | Winner={w}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Make LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Test results: Configuration comparison (Jain fairness) -- ACO}")
    latex_lines.append(r"\label{tab:test_config_comparison_aco}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{lcccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Size} & \textbf{Config} & \textbf{Feasible} & "
                       r"\textbf{Objective} & \textbf{Runtime (min)} & \textbf{Diff (\%)} & \textbf{Winner} \\")
    latex_lines.append(r"\midrule")

    for size in SIZES:
        r1 = df[(df["size"] == size) & (df["config"] == "Rank 1")].iloc[0]
        r2 = df[(df["size"] == size) & (df["config"] == "Rank 2")].iloc[0]

        latex_lines.append(
            f"{size} & Rank 1 & {r1['feasible']} & {int(round(r1['objective'])) if r1['objective'] != float('inf') else 'INF'}"
            f" & {r1['runtime_min']:.1f} & --- & ---\\\\"
        )
        diff_str = f"{r2['diff_pct']:+.1f}"
        latex_lines.append(
            f"{size} & Rank 2 & {r2['feasible']} & {int(round(r2['objective'])) if r2['objective'] != float('inf') else 'INF'}"
            f" & {r2['runtime_min']:.1f} & {diff_str} & {r2['winner']}\\\\"
        )
        latex_lines.append(r"\midrule")

    latex_lines[-1] = r"\bottomrule"  # replace last midrule
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    out_tex = OUT_DIR / "latex_table.tex"
    out_tex.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"Saved: {out_tex}")

    # Plot objective and runtime across sizes (Rank1 vs Rank2)
    # We'll aggregate to one row per size per config
    df_plot = df.copy()

    # Objectives plot
    plt.figure(figsize=(10, 4))
    for cfg_name in ["Rank 1", "Rank 2"]:
        sub = df_plot[df_plot["config"] == cfg_name].sort_values("size")
        plt.plot(sub["size"], sub["objective"], marker="o", label=cfg_name)
    plt.xlabel("Instance size")
    plt.ylabel("Average objective (feasible only)")
    plt.title("ACO: Rank 1 vs Rank 2 (Objective)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_obj = OUT_DIR / "objective_vs_size.png"
    plt.tight_layout()
    plt.savefig(out_obj, dpi=300)
    plt.close()

    # Runtime plot
    plt.figure(figsize=(10, 4))
    for cfg_name in ["Rank 1", "Rank 2"]:
        sub = df_plot[df_plot["config"] == cfg_name].sort_values("size")
        plt.plot(sub["size"], sub["runtime_min"], marker="o", label=cfg_name)
    plt.xlabel("Instance size")
    plt.ylabel("Average runtime (minutes)")
    plt.title("ACO: Rank 1 vs Rank 2 (Runtime)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_rt = OUT_DIR / "runtime_vs_size.png"
    plt.tight_layout()
    plt.savefig(out_rt, dpi=300)
    plt.close()

    print(f"Saved: {out_obj}")
    print(f"Saved: {out_rt}")
    print("\nDONE.")

if __name__ == "__main__":
    main()
