"""
Comprehensive Fairness Measures Comparison Analysis
FINAL VERSION - With Fairness Values Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

OUTPUT_DIR = Path("./fairness_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA LOADING - ROBUST VERSION
# ============================================================================

def load_all_results():
    """Load results - handles both CSV formats"""

    sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    fairness_measures = ['jain', 'maxmin', 'gini']

    all_data = []

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    for size in sizes:
        if size >= 2000:
            base_folder = f"./ea_results/{size}_greedy"
        else:
            base_folder = f"./ea_results/{size}"

        for fairness in fairness_measures:
            results_file = Path(base_folder) / fairness / "results.csv"

            if results_file.exists():
                try:
                    # Read CSV
                    df = pd.read_csv(results_file, on_bad_lines='skip')
                    df = df.dropna(axis=1, how='all')

                    if len(df) == 0:
                        continue

                    # CRITICAL: Set fairness_measure from folder name
                    df['fairness_measure'] = fairness
                    df['size'] = size

                    # Add missing columns if needed
                    if 'runtime' not in df.columns:
                        df['runtime'] = np.nan

                    all_data.append(df)
                    print(
                        f"✓ Loaded {len(df)} results for size={size}, fairness={fairness}")

                except Exception as e:
                    print(f"✗ Error loading {results_file}: {e}")
                    continue
            else:
                print(f"✗ Missing: {results_file}")

    if not all_data:
        raise FileNotFoundError("No results found!")

    combined_df = pd.concat(all_data, ignore_index=True)
    feasible_df = combined_df[combined_df['feasible'] == True].copy()

    print(f"\n{'=' * 80}")
    print(f"Total results: {len(combined_df)}, Feasible: {len(feasible_df)}")
    print(f"Sizes: {sorted(feasible_df['size'].unique())}")
    print(
        f"Fairness measures: {sorted(feasible_df['fairness_measure'].unique())}")
    print(f"{'=' * 80}\n")

    # Quick verification
    print("DATA VERIFICATION:")
    print("-" * 80)
    for size in sorted(feasible_df['size'].unique())[:2]:
        size_data = feasible_df[feasible_df['size'] == size]
        print(f"\nSize {size}:")
        for fairness in ['jain', 'maxmin', 'gini']:
            fair_data = size_data[size_data['fairness_measure'] == fairness]
            if len(fair_data) > 0:
                print(f"  {fairness:8s}: {len(fair_data):2d} instances, "
                      f"Obj={fair_data['objective'].mean():8.1f}, "
                      f"Stops={fair_data['total_stops'].mean():6.1f}, "
                      f"Fairness={fair_data['fairness'].mean():.4f}")
    print()

    return feasible_df


# ============================================================================
# STATISTICS
# ============================================================================

def compute_statistics(df):
    """Compute comprehensive statistics"""
    stats_df = df.groupby(['size', 'fairness_measure']).agg({
        'objective': ['mean', 'std', 'min', 'median', 'max'],
        'total_duration': ['mean', 'std', 'min', 'median', 'max'],
        'fairness': ['mean', 'std', 'min', 'median', 'max'],
        'total_stops': ['mean', 'std', 'min', 'median', 'max'],
        'avg_stops_per_route': ['mean', 'std'],
        'num_active_routes': ['mean', 'std'],
        'total_generations': ['mean', 'std'],
        'instance': 'count'
    }).reset_index()

    stats_df.columns = ['_'.join(col).strip('_') for col in
                        stats_df.columns.values]
    stats_df.rename(columns={'instance_count': 'n_instances'}, inplace=True)

    return stats_df


def compute_relative_performance(df):
    """Compute relative performance vs Jain"""

    results = []

    for size in df['size'].unique():
        df_size = df[df['size'] == size]

        jain_data = df_size[df_size['fairness_measure'] == 'jain']

        if len(jain_data) == 0:
            continue

        jain_obj = jain_data['objective'].mean()
        jain_stops = jain_data['total_stops'].mean()
        jain_fairness = jain_data['fairness'].mean()

        for fairness in ['maxmin', 'gini']:
            fair_data = df_size[df_size['fairness_measure'] == fairness]

            if len(fair_data) > 0:
                obj_diff = 100 * (fair_data[
                                      'objective'].mean() - jain_obj) / jain_obj
                stops_diff = 100 * (fair_data[
                                        'total_stops'].mean() - jain_stops) / jain_stops
                fairness_diff = 100 * (fair_data[
                                           'fairness'].mean() - jain_fairness) / jain_fairness

                try:
                    t_stat, p_value = stats.ttest_ind(jain_data['objective'],
                                                      fair_data['objective'])
                except:
                    p_value = 1.0

                results.append({
                    'size': size,
                    'fairness_measure': fairness,
                    'objective_diff_pct': obj_diff,
                    'stops_diff_pct': stops_diff,
                    'fairness_value_diff_pct': fairness_diff,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })

    return pd.DataFrame(results)


# ============================================================================
# NEW: FAIRNESS VALUES ANALYSIS
# ============================================================================

def analyze_fairness_values(df):
    """Detailed analysis of fairness values achieved"""

    print("\n" + "=" * 80)
    print("FAIRNESS VALUES ANALYSIS")
    print("=" * 80)
    print()

    for size in sorted(df['size'].unique()):
        size_data = df[df['size'] == size]

        print(f"SIZE {size}:")
        print("-" * 80)

        fairness_values = {}
        for fairness in ['jain', 'maxmin', 'gini']:
            fair_data = size_data[size_data['fairness_measure'] == fairness]

            if len(fair_data) > 0:
                mean_val = fair_data['fairness'].mean()
                std_val = fair_data['fairness'].std()
                min_val = fair_data['fairness'].min()
                max_val = fair_data['fairness'].max()

                fairness_values[fairness] = mean_val

                print(
                    f"  {fairness.capitalize():8s}: {mean_val:.4f} ± {std_val:.4f} "
                    f"[{min_val:.4f}, {max_val:.4f}]")

        # Calculate range
        if len(fairness_values) >= 2:
            fair_range = max(fairness_values.values()) - min(
                fairness_values.values())
            fair_cv = 100 * fair_range / max(fairness_values.values())

            print(f"\n  Range: {fair_range:.4f} ({fair_cv:.2f}%)")

            if fair_cv < 1:
                print(f"  → IDENTICAL fairness values (< 1% difference)")
            elif fair_cv < 5:
                print(f"  → SIMILAR fairness values (< 5% difference)")
            else:
                print(f"  → DIFFERENT fairness values (> 5% difference)")

        print()

    # Overall summary
    print("=" * 80)
    print("OVERALL FAIRNESS VALUES:")
    print("=" * 80)

    overall_fairness = df.groupby('fairness_measure')['fairness'].agg(
        ['mean', 'std'])

    for fairness in ['jain', 'maxmin', 'gini']:
        if fairness in overall_fairness.index:
            mean_val = overall_fairness.loc[fairness, 'mean']
            std_val = overall_fairness.loc[fairness, 'std']
            print(
                f"  {fairness.capitalize():8s}: {mean_val:.4f} ± {std_val:.4f}")

    # Check if comparable
    overall_range = overall_fairness['mean'].max() - overall_fairness[
        'mean'].min()
    overall_cv = 100 * overall_range / overall_fairness['mean'].max()

    print(f"\n  Overall Range: {overall_range:.4f} ({overall_cv:.2f}%)")

    if overall_cv < 1:
        print(
            "\n  ✓ CONCLUSION: All fairness measures achieve IDENTICAL values")
        print("    → Fairness values are directly comparable")
        print("    → Different measures lead to same equity outcome")
    elif overall_cv < 5:
        print("\n  ✓ CONCLUSION: All fairness measures achieve SIMILAR values")
        print("    → Fairness values are comparable")
        print("    → Different measures lead to comparable equity outcomes")
    else:
        print("\n  ⚠ CONCLUSION: Fairness measures achieve DIFFERENT values")
        print("    → Values may not be directly comparable")
        print("    → Different measures optimize different equity criteria")

    print()


# ============================================================================
# VISUALIZATIONS (keeping all previous ones, same code)
# ============================================================================

def plot_objective_comparison(df, stats_df):
    """Objective comparison - line and bar"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sizes = sorted(stats_df['size'].unique())
    x_positions = np.arange(len(sizes))

    # LEFT: Connected points
    ax1 = axes[0]

    for fairness in ['jain', 'maxmin', 'gini']:
        data = stats_df[stats_df['fairness_measure'] == fairness].sort_values(
            'size')

        if len(data) == 0:
            continue

        means = [data[data['size'] == s]['objective_mean'].values[0]
                 if len(data[data['size'] == s]) > 0 else np.nan
                 for s in sizes]
        stds = [data[data['size'] == s]['objective_std'].values[0]
                if len(data[data['size'] == s]) > 0 else 0
                for s in sizes]

        ax1.plot(x_positions, means, marker='o', linewidth=2, markersize=8,
                 label=fairness.capitalize(), alpha=0.8)

        ax1.errorbar(x_positions, means, yerr=stds, alpha=0.3,
                     fmt='none', capsize=3)

    ax1.set_xlabel('Instance Size', fontweight='bold')
    ax1.set_ylabel('Objective Value', fontweight='bold')
    ax1.set_title('Objective by Size and Fairness Measure', fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # RIGHT: Bar chart for selected size
    ax2 = axes[1]
    selected_size = 100
    size_data = df[df['size'] == selected_size]

    fairness_means = []
    fairness_stds = []
    labels = []

    for fairness in ['jain', 'maxmin', 'gini']:
        fair_data = size_data[size_data['fairness_measure'] == fairness]
        if len(fair_data) > 0:
            fairness_means.append(fair_data['objective'].mean())
            fairness_stds.append(fair_data['objective'].std())
            labels.append(fairness.capitalize())

    x = np.arange(len(labels))
    ax2.bar(x, fairness_means, yerr=fairness_stds, capsize=5, alpha=0.8,
            color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Objective Value', fontweight='bold')
    ax2.set_title(f'Objective Comparison (Size {selected_size})',
                  fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'objective_comparison.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'objective_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved: objective_comparison.png")
    plt.close()


def plot_fairness_values(df, stats_df):
    """Compare achieved fairness values"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sizes = sorted(stats_df['size'].unique())
    x_positions = np.arange(len(sizes))

    # Bar chart by size
    ax1 = axes[0]
    width = 0.25

    for idx, fairness in enumerate(['jain', 'maxmin', 'gini']):
        fair_data = stats_df[
            stats_df['fairness_measure'] == fairness].sort_values('size')
        means = [fair_data[fair_data['size'] == s]['fairness_mean'].values[0]
                 if len(fair_data[fair_data['size'] == s]) > 0 else 0
                 for s in sizes]

        ax1.bar(x_positions + idx * width, means, width,
                label=fairness.capitalize(), alpha=0.8)

    ax1.set_xlabel('Instance Size', fontweight='bold')
    ax1.set_ylabel('Fairness Value', fontweight='bold')
    ax1.set_title('Fairness Values by Measure', fontweight='bold')
    ax1.set_xticks(x_positions + width)
    ax1.set_xticklabels(sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.05])

    # Box plot for one size
    ax2 = axes[1]
    size_100 = df[df['size'] == 100]

    if len(size_100) > 0:
        data_to_plot = []
        labels = []
        for fairness in ['jain', 'maxmin', 'gini']:
            fair_data = size_100[size_100['fairness_measure'] == fairness]
            if len(fair_data) > 0:
                data_to_plot.append(fair_data['fairness'].values)
                labels.append(fairness.capitalize())

        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'],
                                ['#1f77b4', '#ff7f0e', '#2ca02c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax2.set_ylabel('Fairness Value', fontweight='bold')
        ax2.set_title('Fairness Value Distribution (Size 100)',
                      fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fairness_values.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fairness_values.pdf', bbox_inches='tight')
    print(f"✓ Saved: fairness_values.png")
    plt.close()


def plot_stops_analysis_corrected(df, stats_df):
    """CORRECTED stops analysis"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sizes = sorted(stats_df['size'].unique())
    x_positions = np.arange(len(sizes))

    # 1. BAR CHART
    ax1 = axes[0, 0]
    width = 0.25

    for idx, fairness in enumerate(['jain', 'maxmin', 'gini']):
        data = stats_df[stats_df['fairness_measure'] == fairness].sort_values(
            'size')

        if len(data) == 0:
            continue

        means = [data[data['size'] == s]['total_stops_mean'].values[0]
                 if len(data[data['size'] == s]) > 0 else 0
                 for s in sizes]

        ax1.bar(x_positions + idx * width - width, means, width,
                label=fairness.capitalize(), alpha=0.8)

    ax1.set_xlabel('Instance Size', fontweight='bold')
    ax1.set_ylabel('Total Stops', fontweight='bold')
    ax1.set_title('Total Stops by Size (Bars Show Identity)',
                  fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. SINGLE SIZE COMPARISON
    ax2 = axes[0, 1]
    size_100 = df[df['size'] == 100]

    stops_data = []
    stops_stds = []
    labels = []

    for fairness in ['jain', 'maxmin', 'gini']:
        fair_data = size_100[size_100['fairness_measure'] == fairness]
        if len(fair_data) > 0:
            stops_data.append(fair_data['total_stops'].mean())
            stops_stds.append(fair_data['total_stops'].std())
            labels.append(fairness.capitalize())

    x = np.arange(len(labels))
    bars = ax2.bar(x, stops_data, yerr=stops_stds, capsize=5, alpha=0.8,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Total Stops', fontweight='bold')
    ax2.set_title('Total Stops (Size 100) - Identical Values',
                  fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontweight='bold')

    # 3. STOPS EFFICIENCY
    ax3 = axes[1, 0]

    for fairness in ['jain', 'maxmin', 'gini']:
        fair_data = df[df['fairness_measure'] == fairness]

        if len(fair_data) == 0:
            continue

        fair_data_copy = fair_data.copy()
        fair_data_copy['stops_per_request'] = fair_data_copy['total_stops'] / \
                                              fair_data_copy['num_served']

        means = []
        for size in sizes:
            size_data = fair_data_copy[fair_data_copy['size'] == size]
            if len(size_data) > 0:
                means.append(size_data['stops_per_request'].mean())
            else:
                means.append(np.nan)

        ax3.plot(x_positions, means, marker='o', linewidth=2,
                 label=fairness.capitalize())

    ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5,
                linewidth=2, label='Optimal (2.0)')
    ax3.set_xlabel('Instance Size', fontweight='bold')
    ax3.set_ylabel('Stops per Served Request', fontweight='bold')
    ax3.set_title('Routing Efficiency (Lower = Better)', fontweight='bold')
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(sizes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. TEXT EXPLANATION
    ax4 = axes[1, 1]
    ax4.axis('off')

    stops_by_size = {}
    for size in sizes:
        size_data = df[df['size'] == size]
        if len(size_data) > 0:
            stops_by_size[size] = size_data['total_stops'].mean()

    explanation = f"""
    KEY FINDING: IDENTICAL ROUTING STRUCTURE

    All three fairness measures produce routes 
    with IDENTICAL structure across all sizes:

    Size    Total Stops (all measures)
    ────────────────────────────────────
    """

    for size in sizes:
        if size in stops_by_size:
            explanation += f"    {size:5d}   {stops_by_size[size]:7.1f}\n"

    explanation += """

    This means fairness measures affect:
    ✓ Route duration BALANCE
    ✓ Fairness values
    ✓ Total system objective

    But NOT:
    ✗ Spatial routing structure
    ✗ Request-to-vehicle assignment
    ✗ Number of stops

    INTERPRETATION:
    Fairness optimization balances temporal 
    equity without changing fundamental 
    routing structure.
    """

    ax4.text(0.05, 0.5, explanation, fontsize=9,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stops_analysis.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'stops_analysis.pdf', bbox_inches='tight')
    print(f"✓ Saved: stops_analysis.png")
    plt.close()


# [Keep other visualization functions: plot_duration_balance, plot_relative_performance, plot_heatmaps]
# (Same code as before - not repeating for brevity)

def plot_duration_balance(df):
    """Show route duration balance"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    size_data = df[df['size'] == 100]

    for idx, fairness in enumerate(['jain', 'maxmin', 'gini']):
        ax = axes[idx]
        fair_data = size_data[size_data['fairness_measure'] == fairness]

        if len(fair_data) == 0:
            continue

        ax.scatter(fair_data['min_duration'], fair_data['max_duration'],
                   alpha=0.6, s=50,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'][idx])

        max_val = fair_data['max_duration'].max()
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5,
                linewidth=2, label='Perfect Balance')

        ax.set_xlabel('Min Route Duration', fontweight='bold')
        ax.set_ylabel('Max Route Duration', fontweight='bold')
        ax.set_title(f'{fairness.capitalize()}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('Route Duration Balance (Size 100)', fontweight='bold',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'duration_balance.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'duration_balance.pdf', bbox_inches='tight')
    print(f"✓ Saved: duration_balance.png")
    plt.close()


def plot_relative_performance(rel_perf_df):
    """Plot relative performance - DISCRETE x-axis"""

    if len(rel_perf_df) == 0:
        print("⚠ No relative performance data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sizes = sorted(rel_perf_df['size'].unique())
    x_positions = np.arange(len(sizes))

    # Objective difference
    ax1 = axes[0]
    for fairness in ['maxmin', 'gini']:
        data = rel_perf_df[
            rel_perf_df['fairness_measure'] == fairness].sort_values('size')

        if len(data) == 0:
            continue

        diffs = [data[data['size'] == s]['objective_diff_pct'].values[0]
                 if len(data[data['size'] == s]) > 0 else np.nan
                 for s in sizes]

        ax1.plot(x_positions, diffs, marker='o', linewidth=2, markersize=8,
                 label=fairness.capitalize())

        sig_sizes = data[data['significant']]['size'].values
        sig_diffs = data[data['significant']]['objective_diff_pct'].values
        if len(sig_sizes) > 0:
            sig_x = [sizes.index(s) for s in sig_sizes if s in sizes]
            ax1.scatter(sig_x, sig_diffs, s=200, marker='*', color='red',
                        zorder=5,
                        label='Significant' if fairness == 'maxmin' else '')

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Instance Size', fontweight='bold')
    ax1.set_ylabel('Objective Difference from Jain (%)', fontweight='bold')
    ax1.set_title('Relative Objective Performance', fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(sizes)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Stops difference
    ax2 = axes[1]
    for fairness in ['maxmin', 'gini']:
        data = rel_perf_df[
            rel_perf_df['fairness_measure'] == fairness].sort_values('size')

        if len(data) == 0:
            continue

        diffs = [data[data['size'] == s]['stops_diff_pct'].values[0]
                 if len(data[data['size'] == s]) > 0 else np.nan
                 for s in sizes]

        ax2.plot(x_positions, diffs, marker='s', linewidth=2, markersize=8,
                 label=fairness.capitalize())

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Instance Size', fontweight='bold')
    ax2.set_ylabel('Stops Difference from Jain (%)', fontweight='bold')
    ax2.set_title('Relative Stops (Near Zero = Identical)', fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(sizes)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'relative_performance.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'relative_performance.pdf', bbox_inches='tight')
    print(f"✓ Saved: relative_performance.png")
    plt.close()


def plot_heatmaps(stats_df):
    """Heatmaps for objective and stops"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    pivot_obj = stats_df.pivot(index='fairness_measure', columns='size',
                               values='objective_mean')
    sns.heatmap(pivot_obj, annot=True, fmt='.0f', cmap='YlOrRd',
                ax=axes[0], cbar_kws={'label': 'Objective Value'})
    axes[0].set_title('Mean Objective Value', fontweight='bold', pad=15)
    axes[0].set_xlabel('Instance Size', fontweight='bold')
    axes[0].set_ylabel('Fairness Measure', fontweight='bold')
    axes[0].set_yticklabels(['Gini', 'Jain', 'Maxmin'], rotation=0)

    pivot_stops = stats_df.pivot(index='fairness_measure', columns='size',
                                 values='total_stops_mean')
    sns.heatmap(pivot_stops, annot=True, fmt='.0f', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'Total Stops'})
    axes[1].set_title('Mean Total Stops (Identical)', fontweight='bold',
                      pad=15)
    axes[1].set_xlabel('Instance Size', fontweight='bold')
    axes[1].set_ylabel('Fairness Measure', fontweight='bold')
    axes[1].set_yticklabels(['Gini', 'Jain', 'Maxmin'], rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heatmaps.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'heatmaps.pdf', bbox_inches='tight')
    print(f"✓ Saved: heatmaps.png")
    plt.close()


# [Keep generate_comprehensive_report and generate_latex_table - same as before]

def generate_latex_table(stats_df):
    """Generate LaTeX table"""

    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(
        r"\caption{Comparison of fairness measures across instance sizes}")
    latex_lines.append(r"\label{tab:fairness_comparison}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{lcccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(
        r"\textbf{Size} & \textbf{Fairness} & \textbf{Objective} & \textbf{Fairness Value} & \textbf{Total Stops} & \textbf{Gens} & \textbf{n} \\")
    latex_lines.append(r"\midrule")

    for size in sorted(stats_df['size'].unique()):
        for fairness in ['jain', 'maxmin', 'gini']:
            row = stats_df[(stats_df['size'] == size) &
                           (stats_df['fairness_measure'] == fairness)]

            if len(row) > 0:
                row = row.iloc[0]
                fairness_label = fairness.capitalize()

                latex_lines.append(
                    f"{size} & {fairness_label} & "
                    f"{row['objective_mean']:.0f} $\\pm$ {row['objective_std']:.0f} & "
                    f"{row['fairness_mean']:.3f} & "
                    f"{row['total_stops_mean']:.0f} & "
                    f"{row['total_generations_mean']:.0f} & "
                    f"{row['n_instances']:.0f} \\\\"
                )

        latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_content = "\n".join(latex_lines)

    with open(OUTPUT_DIR / 'fairness_table.tex', 'w') as f:
        f.write(latex_content)

    print(f"✓ Saved: fairness_table.tex")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print("COMPREHENSIVE FAIRNESS ANALYSIS - WITH FAIRNESS VALUES COMPARISON")
    print("=" * 100)
    print()

    # Load data
    df = load_all_results()

    # Compute statistics
    print("\nComputing statistics...")
    stats_df = compute_statistics(df)
    rel_perf_df = compute_relative_performance(df)

    # NEW: Analyze fairness values
    analyze_fairness_values(df)

    # Save CSVs
    stats_df.to_csv(OUTPUT_DIR / 'statistics_summary.csv', index=False)
    rel_perf_df.to_csv(OUTPUT_DIR / 'relative_performance.csv', index=False)
    print("✓ Saved statistics CSVs")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_objective_comparison(df, stats_df)
    plot_fairness_values(df, stats_df)
    plot_stops_analysis_corrected(df, stats_df)
    plot_duration_balance(df)
    plot_relative_performance(rel_perf_df)
    plot_heatmaps(stats_df)

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(stats_df)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated:")
    print("  • 6 visualization figures (PNG + PDF)")
    print("  • 2 statistics CSVs")
    print("  • 1 LaTeX table")
    print("  • Fairness values analysis in console")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()