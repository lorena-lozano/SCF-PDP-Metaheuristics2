"""
Comprehensive Fairness Measures Comparison Analysis (ACO)
FINAL - Adapted to ACO results folder structure and CSV schema

Reads:
  ./aco_results/<size>/<fairness>/results.csv
  ./aco_results/<size>/<fairness>/convergence/*_convergence.csv

Produces:
  ./fairness_analysis_aco/
  - objective_value_distribution.(png/pdf)
  - fairness_values.(png/pdf)
  - convergence_speed_comparison.(png/pdf)
  - stops_analysis.(png/pdf)
  - relative_performance.(png/pdf)
  - statistics_summary.csv, relative_performance.csv
  - aco_fairness_table.tex
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import glob
import math

# -----------------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

OUTPUT_DIR = Path("./fairness_analysis_aco")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

ACO_RESULTS_DIR = Path("./aco_results")
FAIRNESS_MEASURES = ["jain", "maxmin", "gini"]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a column to numeric, handling 'INF'/'inf' strings."""
    if s.dtype == object:
        s = s.replace(
            {
                "INF": np.inf, "inf": np.inf, "Inf": np.inf,
                "INFINITY": np.inf, "Infinity": np.inf,
                "nan": np.nan, "NaN": np.nan,
            }
        )
    return pd.to_numeric(s, errors="coerce")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure expected columns exist and are numeric where needed."""
    if "size" not in df.columns:
        if "instance_size" in df.columns:
            df["size"] = _to_numeric_series(df["instance_size"]).astype("Int64")
        else:
            df["size"] = pd.NA

    # Normalize feasibility column
    if "feasible" in df.columns:
        if df["feasible"].dtype == object:
            df["feasible"] = df["feasible"].astype(str).str.lower().map(
                {"true": True, "false": False}
            )

    # Numeric columns
    numeric_cols = [
        "objective", "total_duration", "fairness", "runtime",
        "avg_duration", "min_duration", "max_duration",
        "total_stops", "avg_stops_per_route", "num_served",
        "num_active_routes",
        "total_iters",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = _to_numeric_series(df[c])

    # Compatibility with EA naming in some summaries
    if "total_generations" not in df.columns:
        if "total_iters" in df.columns:
            df["total_generations"] = df["total_iters"]
        else:
            df["total_generations"] = np.nan

    return df


def _fairness_pretty_label(x: str) -> str:
    m = {
        "jain": "Jain",
        "maxmin": "Max-Min",
        "gini": "Gini"
    }
    return m.get(x, str(x))


# -----------------------------------------------------------------------------
# DATA LOADING (ACO)
# -----------------------------------------------------------------------------
def load_all_results_aco():
    """
    Load ACO results from ./aco_results/<size>/<fairness>/results.csv
    Robust to missing folders/files.
    """
    all_data = []

    print("=" * 80)
    print("LOADING ACO DATA")
    print("=" * 80)

    if not ACO_RESULTS_DIR.exists():
        raise FileNotFoundError(f"ACO results folder not found: {ACO_RESULTS_DIR.resolve()}")

    # sizes present in folder
    size_dirs = []
    for p in ACO_RESULTS_DIR.iterdir():
        if p.is_dir():
            try:
                size_dirs.append(int(p.name))
            except Exception:
                pass
    size_dirs = sorted(size_dirs)

    if not size_dirs:
        size_dirs = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    for size in size_dirs:
        base_folder = ACO_RESULTS_DIR / str(size)

        for fairness in FAIRNESS_MEASURES:
            results_file = base_folder / fairness / "results.csv"
            if not results_file.exists():
                continue

            try:
                df = pd.read_csv(results_file, on_bad_lines="skip")
                df = df.dropna(axis=1, how="all")
                if len(df) == 0:
                    continue

                df = _ensure_columns(df)

                # Force truth from folder
                df["fairness_measure"] = fairness
                df["size"] = size

                if "runtime" not in df.columns:
                    df["runtime"] = np.nan

                all_data.append(df)
                print(f"✓ Loaded {len(df):3d} rows for size={size}, fairness={fairness}")

            except Exception as e:
                print(f"✗ Error loading {results_file}: {e}")

    if not all_data:
        raise FileNotFoundError(
            "No ACO results found. Expected: ./aco_results/<size>/<fairness>/results.csv"
        )

    combined_df = pd.concat(all_data, ignore_index=True)

    if "feasible" in combined_df.columns:
        feasible_df = combined_df[combined_df["feasible"] == True].copy()
    else:
        feasible_df = combined_df.copy()

    print(f"\n{'=' * 80}")
    print(f"Total rows: {len(combined_df)}, Feasible rows: {len(feasible_df)}")
    print(f"Sizes: {sorted(feasible_df['size'].dropna().unique().tolist())}")
    print(f"Fairness measures: {sorted(feasible_df['fairness_measure'].unique().tolist())}")
    print(f"{'=' * 80}\n")

    # Quick verification
    print("DATA VERIFICATION (sample sizes):")
    print("-" * 80)
    for size in sorted(feasible_df["size"].unique())[:2]:
        size_data = feasible_df[feasible_df["size"] == size]
        print(f"\nSize {size}:")
        for fairness in FAIRNESS_MEASURES:
            fair_data = size_data[size_data["fairness_measure"] == fairness]
            if len(fair_data) > 0:
                obj_mean = fair_data["objective"].mean()
                stops_mean = fair_data["total_stops"].mean() if "total_stops" in fair_data.columns else np.nan
                fair_mean = fair_data["fairness"].mean()
                iters_mean = fair_data["total_iters"].mean() if "total_iters" in fair_data.columns else np.nan
                print(
                    f"  {fairness:8s}: {len(fair_data):2d} inst, "
                    f"Obj={obj_mean:10.1f}, Stops={stops_mean:7.1f}, "
                    f"Fair={fair_mean:.4f}, Iters={iters_mean:6.1f}"
                )
    print()

    return feasible_df


# -----------------------------------------------------------------------------
# CONVERGENCE LOADING (ACO)
# -----------------------------------------------------------------------------
def load_convergence_curves_aco():
    """
    Loads convergence curves from:
      ./aco_results/<size>/<fairness>/convergence/*_convergence.csv

    Returns:
      dict[(size, fairness)] -> DataFrame with columns:
         iteration, objective_mean
    """
    curves = {}

    if not ACO_RESULTS_DIR.exists():
        return curves

    # detect sizes
    size_dirs = []
    for p in ACO_RESULTS_DIR.iterdir():
        if p.is_dir():
            try:
                size_dirs.append(int(p.name))
            except Exception:
                pass
    size_dirs = sorted(size_dirs)

    for size in size_dirs:
        for fairness in FAIRNESS_MEASURES:
            conv_dir = ACO_RESULTS_DIR / str(size) / fairness / "convergence"
            if not conv_dir.exists():
                continue

            files = sorted(glob.glob(str(conv_dir / "*_convergence.csv")))
            if not files:
                continue

            all_runs = []
            for f in files:
                try:
                    dfi = pd.read_csv(f)
                    if "iteration" not in dfi.columns or "objective" not in dfi.columns:
                        continue
                    dfi = dfi.copy()
                    dfi["iteration"] = _to_numeric_series(dfi["iteration"]).astype(int)
                    dfi["objective"] = _to_numeric_series(dfi["objective"])
                    # keep feasible-ish points only if column exists
                    if "feasible" in dfi.columns:
                        if dfi["feasible"].dtype == object:
                            dfi["feasible"] = dfi["feasible"].astype(str).str.lower().map({"true": True, "false": False})
                        # if feasible present, prefer feasible; if none feasible, keep all
                        if dfi["feasible"].notna().any():
                            dfi = dfi[dfi["feasible"] == True].copy()

                    dfi = dfi.dropna(subset=["iteration", "objective"])
                    if len(dfi) == 0:
                        continue
                    all_runs.append(dfi[["iteration", "objective"]])
                except Exception:
                    continue

            if not all_runs:
                continue

            merged = pd.concat(all_runs, ignore_index=True)
            curve = merged.groupby("iteration")["objective"].mean().reset_index()
            curve.rename(columns={"objective": "objective_mean"}, inplace=True)
            curves[(size, fairness)] = curve

    return curves


# -----------------------------------------------------------------------------
# STATISTICS
# -----------------------------------------------------------------------------
def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive statistics (ACO-friendly)."""
    agg_map = {
        "objective": ["mean", "std", "min", "median", "max"],
        "total_duration": ["mean", "std", "min", "median", "max"],
        "fairness": ["mean", "std", "min", "median", "max"],
        "total_stops": ["mean", "std", "min", "median", "max"],
        "avg_stops_per_route": ["mean", "std"],
        "num_active_routes": ["mean", "std"],
        "runtime": ["mean", "std", "min", "median", "max"],
        "instance": "count",
    }

    if "total_iters" in df.columns:
        agg_map["total_iters"] = ["mean", "std", "min", "median", "max"]

    stats_df = (
        df.groupby(["size", "fairness_measure"])
          .agg(agg_map)
          .reset_index()
    )

    stats_df.columns = ["_".join(col).strip("_") for col in stats_df.columns.values]
    stats_df.rename(columns={"instance_count": "n_instances"}, inplace=True)
    return stats_df


def compute_relative_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute relative performance vs Jain (objective, stops, fairness value)."""
    results = []

    for size in sorted(df["size"].unique()):
        df_size = df[df["size"] == size]
        jain_data = df_size[df_size["fairness_measure"] == "jain"]
        if len(jain_data) == 0:
            continue

        jain_obj = jain_data["objective"].mean()
        jain_stops = jain_data["total_stops"].mean() if "total_stops" in jain_data.columns else np.nan
        jain_fairness = jain_data["fairness"].mean()

        for fairness in ["maxmin", "gini"]:
            fair_data = df_size[df_size["fairness_measure"] == fairness]
            if len(fair_data) == 0:
                continue

            obj_diff = 100 * (fair_data["objective"].mean() - jain_obj) / jain_obj
            if np.isfinite(jain_stops) and jain_stops != 0:
                stops_diff = 100 * (fair_data["total_stops"].mean() - jain_stops) / jain_stops
            else:
                stops_diff = np.nan

            fairness_diff = 100 * (fair_data["fairness"].mean() - jain_fairness) / jain_fairness

            try:
                _, p_value = stats.ttest_ind(
                    jain_data["objective"].dropna(),
                    fair_data["objective"].dropna(),
                    equal_var=False
                )
            except Exception:
                p_value = 1.0

            results.append(
                {
                    "size": size,
                    "fairness_measure": fairness,
                    "objective_diff_pct": obj_diff,
                    "stops_diff_pct": stops_diff,
                    "fairness_value_diff_pct": fairness_diff,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
            )

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# FAIRNESS VALUES ANALYSIS (console text, like your teammate)
# -----------------------------------------------------------------------------
def analyze_fairness_values(df: pd.DataFrame):
    """Detailed analysis of fairness values achieved."""
    print("\n" + "=" * 80)
    print("FAIRNESS VALUES ANALYSIS (ACO)")
    print("=" * 80)
    print()

    for size in sorted(df["size"].unique()):
        size_data = df[df["size"] == size]

        print(f"SIZE {size}:")
        print("-" * 80)

        fairness_values = {}
        for fairness in FAIRNESS_MEASURES:
            fair_data = size_data[size_data["fairness_measure"] == fairness]
            if len(fair_data) == 0:
                continue

            mean_val = fair_data["fairness"].mean()
            std_val = fair_data["fairness"].std()
            min_val = fair_data["fairness"].min()
            max_val = fair_data["fairness"].max()

            fairness_values[fairness] = mean_val
            print(f"  {fairness.capitalize():8s}: {mean_val:.4f} ± {std_val:.4f} [{min_val:.4f}, {max_val:.4f}]")

        if len(fairness_values) >= 2:
            fair_range = max(fairness_values.values()) - min(fairness_values.values())
            fair_cv = 100 * fair_range / max(fairness_values.values())

            print(f"\n  Range: {fair_range:.4f} ({fair_cv:.2f}%)")
            if fair_cv < 1:
                print("  → IDENTICAL fairness values (< 1% difference)")
            elif fair_cv < 5:
                print("  → SIMILAR fairness values (< 5% difference)")
            else:
                print("  → DIFFERENT fairness values (> 5% difference)")
        print()

    print("=" * 80)
    print("OVERALL FAIRNESS VALUES (ACO):")
    print("=" * 80)
    overall = df.groupby("fairness_measure")["fairness"].agg(["mean", "std"])

    for fairness in FAIRNESS_MEASURES:
        if fairness in overall.index:
            print(f"  {fairness.capitalize():8s}: {overall.loc[fairness, 'mean']:.4f} ± {overall.loc[fairness, 'std']:.4f}")

    overall_range = overall["mean"].max() - overall["mean"].min()
    overall_cv = 100 * overall_range / overall["mean"].max()
    print(f"\n  Overall Range: {overall_range:.4f} ({overall_cv:.2f}%)")

    if overall_cv < 1:
        print("\n  ✓ CONCLUSION: All fairness measures achieve IDENTICAL values")
    elif overall_cv < 5:
        print("\n  ✓ CONCLUSION: All fairness measures achieve SIMILAR values")
    else:
        print("\n  ⚠ CONCLUSION: Fairness measures achieve DIFFERENT values")

    print()


# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------
def plot_objective_value_distribution(df: pd.DataFrame):
    """
    Grid of boxplots: for each size, objective distribution across fairness measures.
    """
    sizes = sorted(df["size"].unique())
    if len(sizes) == 0:
        print("⚠ No sizes for objective distribution.")
        return

    # Choose grid similar to teammate: 3 columns, multiple rows
    n_cols = 3
    n_rows = int(math.ceil(len(sizes) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # order + labels
    order = ["jain", "maxmin", "gini"]
    label_map = {_k: _fairness_pretty_label(_k) for _k in order}

    for idx, size in enumerate(sizes):
        ax = axes[idx]
        sdf = df[df["size"] == size].copy()
        if len(sdf) == 0:
            ax.axis("off")
            continue

        # seaborn boxplot
        sns.boxplot(
            data=sdf,
            x="fairness_measure",
            y="objective",
            order=order,
            ax=ax
        )
        ax.set_title(f"Size {size}", fontweight="bold")
        ax.set_xlabel("Fairness Measure", fontweight="bold")
        ax.set_ylabel("Objective Value", fontweight="bold")
        ax.set_xticklabels([label_map.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
        ax.grid(True, alpha=0.25, axis="y")

    # hide unused axes
    for j in range(len(sizes), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Objective Value Distribution by Size and Fairness Measure", fontweight="bold", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(OUTPUT_DIR / "objective_value_distribution.png", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "objective_value_distribution.pdf", bbox_inches="tight")
    print("✓ Saved: objective_value_distribution.(png/pdf)")
    plt.close()


def plot_fairness_values(df: pd.DataFrame, stats_df: pd.DataFrame):
    """
    Left: bars by size
    Right: distribution for size=100 (or first available)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sizes = sorted(stats_df["size"].unique())
    x_positions = np.arange(len(sizes))
    width = 0.25

    ax1 = axes[0]
    for idx, fairness in enumerate(FAIRNESS_MEASURES):
        fair_data = stats_df[stats_df["fairness_measure"] == fairness].sort_values("size")
        means = []
        for s in sizes:
            row = fair_data[fair_data["size"] == s]
            means.append(row["fairness_mean"].values[0] if len(row) > 0 else 0)
        ax1.bar(x_positions + idx * width, means, width, label=_fairness_pretty_label(fairness), alpha=0.85)

    ax1.set_xlabel("Instance Size", fontweight="bold")
    ax1.set_ylabel("Fairness Value", fontweight="bold")
    ax1.set_title("Fairness Values by Measure", fontweight="bold")
    ax1.set_xticks(x_positions + width)
    ax1.set_xticklabels(sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim([0, 1.05])

    ax2 = axes[1]
    sel = 100 if 100 in df["size"].unique() else sizes[0]
    size_sel = df[df["size"] == sel]

    data_to_plot, labels = [], []
    for fairness in FAIRNESS_MEASURES:
        fdf = size_sel[size_sel["fairness_measure"] == fairness]
        if len(fdf) > 0:
            data_to_plot.append(fdf["fairness"].dropna().values)
            labels.append(_fairness_pretty_label(fairness))

    if data_to_plot:
        ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        ax2.set_ylabel("Fairness Value", fontweight="bold")
        ax2.set_title(f"Fairness Values Distribution (Size {sel})", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fairness_values.png", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fairness_values.pdf", bbox_inches="tight")
    print("✓ Saved: fairness_values.(png/pdf)")
    plt.close()


def plot_convergence_speed_comparison(convergence_curves: dict):
    """
    3 measures (Jain / Max-Min / Gini), lines per size.
    """
    if not convergence_curves:
        print("⚠ No convergence curves found.")
        return

    sizes = sorted({k[0] for k in convergence_curves.keys()})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)

    for ax_idx, fairness in enumerate(FAIRNESS_MEASURES):
        ax = axes[ax_idx]

        for size in sizes:
            key = (size, fairness)
            if key not in convergence_curves:
                continue

            curve = convergence_curves[key].sort_values("iteration").copy()
            ax.plot(
                curve["iteration"].values,
                curve["objective_mean"].values,
                linewidth=2,
                label=f"Size {size}"
            )

        ax.set_title(_fairness_pretty_label(fairness), fontweight="bold")
        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_ylabel("Objective Value", fontweight="bold")
        ax.grid(True, alpha=0.25)

        ax.set_yscale("log")

        ax.legend(fontsize=8, frameon=True)

    plt.suptitle("Convergence Speed Comparison (log scale)", fontweight="bold", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.savefig(OUTPUT_DIR / "convergence_speed_comparison.png", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "convergence_speed_comparison.pdf", bbox_inches="tight")
    print("✓ Saved: convergence_speed_comparison.(png/pdf)")
    plt.close()


def plot_stops_analysis(df: pd.DataFrame, stats_df: pd.DataFrame):
    """
    For the same argument as your teammate:
    - show total stops by size & fairness
    - show that stops are (almost) identical across measures
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sizes = sorted(stats_df["size"].unique())
    x_positions = np.arange(len(sizes))
    width = 0.25

    # 1) Bar chart stops by size
    ax1 = axes[0, 0]
    for idx, fairness in enumerate(FAIRNESS_MEASURES):
        data = stats_df[stats_df["fairness_measure"] == fairness].sort_values("size")
        means = []
        for s in sizes:
            row = data[data["size"] == s]
            means.append(row["total_stops_mean"].values[0] if len(row) > 0 else 0)
        ax1.bar(x_positions + idx * width - width, means, width, label=_fairness_pretty_label(fairness), alpha=0.85)

    ax1.set_xlabel("Instance Size", fontweight="bold")
    ax1.set_ylabel("Total Stops", fontweight="bold")
    ax1.set_title("Total Stops by Size", fontweight="bold")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 2) Single size comparison
    ax2 = axes[0, 1]
    sel = 100 if 100 in df["size"].unique() else sizes[0]
    size_sel = df[df["size"] == sel]

    stops_means, stops_stds, labels = [], [], []
    for fairness in FAIRNESS_MEASURES:
        fair_data = size_sel[size_sel["fairness_measure"] == fairness]
        if len(fair_data) > 0:
            stops_means.append(fair_data["total_stops"].mean())
            stops_stds.append(fair_data["total_stops"].std())
            labels.append(_fairness_pretty_label(fairness))

    x = np.arange(len(labels))
    bars = ax2.bar(x, stops_means, yerr=stops_stds, capsize=5, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Total Stops", fontweight="bold")
    ax2.set_title(f"Total Stops (Size {sel})", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h, f"{h:.1f}", ha="center", va="bottom", fontweight="bold")

    # 3) Stops per served request
    ax3 = axes[1, 0]
    for fairness in FAIRNESS_MEASURES:
        fair_data = df[df["fairness_measure"] == fairness].copy()
        if len(fair_data) == 0:
            continue
        fair_data["stops_per_request"] = fair_data["total_stops"] / fair_data["num_served"].replace({0: np.nan})

        means = []
        for s in sizes:
            sd = fair_data[fair_data["size"] == s]
            means.append(sd["stops_per_request"].mean() if len(sd) > 0 else np.nan)

        ax3.plot(x_positions, means, marker="o", linewidth=2, label=_fairness_pretty_label(fairness))

    ax3.set_xlabel("Instance Size", fontweight="bold")
    ax3.set_ylabel("Stops per Served Request", fontweight="bold")
    ax3.set_title("Routing Efficiency (Lower = Better)", fontweight="bold")
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(sizes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4) Text panel
    ax4 = axes[1, 1]
    ax4.axis("off")

    stops_by_size = {}
    for s in sizes:
        sd = df[df["size"] == s]
        if len(sd) > 0:
            stops_by_size[s] = sd["total_stops"].mean()

    explanation = "ROUTING STRUCTURE CHECK\n\n"
    explanation += "Mean total stops by size:\n"
    explanation += "Size   Stops\n"
    explanation += "----------------\n"
    for s in sizes:
        if s in stops_by_size:
            explanation += f"{s:5d}  {stops_by_size[s]:7.1f}\n"
    explanation += "\nIf these values overlap across measures,\nthen fairness changes temporal balance,\nnot the spatial structure.\n"

    ax4.text(
        0.05, 0.5, explanation,
        fontsize=9, va="center", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stops_analysis.png", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "stops_analysis.pdf", bbox_inches="tight")
    print("✓ Saved: stops_analysis.(png/pdf)")
    plt.close()


def plot_relative_performance(rel_perf_df: pd.DataFrame):
    """Relative performance vs Jain (objective + stops)."""
    if len(rel_perf_df) == 0:
        print("⚠ No relative performance data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sizes = sorted(rel_perf_df["size"].unique())
    x_positions = np.arange(len(sizes))

    # Objective diff
    ax1 = axes[0]
    for fairness in ["maxmin", "gini"]:
        data = rel_perf_df[rel_perf_df["fairness_measure"] == fairness].sort_values("size")
        if len(data) == 0:
            continue
        diffs = []
        for s in sizes:
            row = data[data["size"] == s]
            diffs.append(row["objective_diff_pct"].values[0] if len(row) > 0 else np.nan)
        ax1.plot(x_positions, diffs, marker="o", linewidth=2, markersize=8, label=_fairness_pretty_label(fairness))

        sig = data[data["significant"] == True]
        if len(sig) > 0:
            sig_x = [sizes.index(s) for s in sig["size"].values if s in sizes]
            ax1.scatter(sig_x, sig["objective_diff_pct"].values, s=200, marker="*", color="red", zorder=5,
                        label="Significant" if fairness == "maxmin" else "")

    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Instance Size", fontweight="bold")
    ax1.set_ylabel("Objective Difference from Jain (%)", fontweight="bold")
    ax1.set_title("Relative Objective Performance", fontweight="bold")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(sizes)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Stops diff
    ax2 = axes[1]
    for fairness in ["maxmin", "gini"]:
        data = rel_perf_df[rel_perf_df["fairness_measure"] == fairness].sort_values("size")
        if len(data) == 0:
            continue
        diffs = []
        for s in sizes:
            row = data[data["size"] == s]
            diffs.append(row["stops_diff_pct"].values[0] if len(row) > 0 else np.nan)
        ax2.plot(x_positions, diffs, marker="s", linewidth=2, markersize=8, label=_fairness_pretty_label(fairness))

    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Instance Size", fontweight="bold")
    ax2.set_ylabel("Stops Difference from Jain (%)", fontweight="bold")
    ax2.set_title("Relative Stops (Near Zero = Identical)", fontweight="bold")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(sizes)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "relative_performance.png", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "relative_performance.pdf", bbox_inches="tight")
    print("✓ Saved: relative_performance.(png/pdf)")
    plt.close()


# -----------------------------------------------------------------------------
# LaTeX TABLE (kept as in your script, useful for report)
# -----------------------------------------------------------------------------
def generate_latex_table(stats_df: pd.DataFrame):
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{ACO comparison of fairness measures across instance sizes}")
    latex_lines.append(r"\label{tab:aco_fairness_comparison}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{lccccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(
        r"\textbf{Size} & \textbf{Fairness} & \textbf{Objective} & \textbf{Fairness Value} & \textbf{Total Stops} & \textbf{Iters} & \textbf{Runtime} & \textbf{n} \\"
    )
    latex_lines.append(r"\midrule")

    for size in sorted(stats_df["size"].unique()):
        for fairness in FAIRNESS_MEASURES:
            row = stats_df[(stats_df["size"] == size) & (stats_df["fairness_measure"] == fairness)]
            if len(row) == 0:
                continue
            row = row.iloc[0]

            iters = row["total_iters_mean"] if "total_iters_mean" in row else np.nan
            rtime = row["runtime_mean"] if "runtime_mean" in row else np.nan

            latex_lines.append(
                f"{int(size)} & {_fairness_pretty_label(fairness)} & "
                f"{row['objective_mean']:.0f} $\\pm$ {row['objective_std']:.0f} & "
                f"{row['fairness_mean']:.3f} & "
                f"{row['total_stops_mean']:.0f} & "
                f"{iters:.0f} & "
                f"{rtime:.1f} & "
                f"{row['n_instances']:.0f} \\\\"
            )
        latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    content = "\n".join(latex_lines)
    with open(OUTPUT_DIR / "aco_fairness_table.tex", "w", encoding="utf-8") as f:
        f.write(content)

    print("✓ Saved: aco_fairness_table.tex")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("\n" + "=" * 100)
    print("COMPREHENSIVE FAIRNESS ANALYSIS - ACO (MATCHING TEAMMATE FIGURES)")
    print("=" * 100 + "\n")

    # Load data
    df = load_all_results_aco()

    # Compute statistics
    print("Computing statistics...")
    stats_df = compute_statistics(df)
    rel_perf_df = compute_relative_performance(df)

    # Console fairness analysis (for your report text)
    analyze_fairness_values(df)

    # Save CSVs (useful for writing the report)
    stats_df.to_csv(OUTPUT_DIR / "statistics_summary.csv", index=False)
    rel_perf_df.to_csv(OUTPUT_DIR / "relative_performance.csv", index=False)
    print("✓ Saved statistics_summary.csv and relative_performance.csv")

    # ---------------------------------------------------------
    # FIGURES (ONLY the key ones your teammate used)
    # ---------------------------------------------------------
    print("\nGenerating figures...")

    # Figure like teammate's first image
    plot_objective_value_distribution(df)

    # Fairness values figure (you already had; keep)
    plot_fairness_values(df, stats_df)

    # Figure like teammate's third image
    convergence_curves = load_convergence_curves_aco()
    plot_convergence_speed_comparison(convergence_curves)

    # Routing structure argument (stops)
    plot_stops_analysis(df, stats_df)

    # Relative performance vs Jain
    plot_relative_performance(rel_perf_df)

    # LaTeX summary table (optional but helpful)
    print("\nGenerating LaTeX table...")
    generate_latex_table(stats_df)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
    print("\nGenerated:")
    print("  • objective_value_distribution.(png/pdf)")
    print("  • fairness_values.(png/pdf)")
    print("  • convergence_speed_comparison.(png/pdf)")
    print("  • stops_analysis.(png/pdf)")
    print("  • relative_performance.(png/pdf)")
    print("  • statistics_summary.csv, relative_performance.csv")
    print("  • aco_fairness_table.tex")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
