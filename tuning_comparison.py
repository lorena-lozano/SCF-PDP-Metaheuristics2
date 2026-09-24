"""
Analyze parameter tuning results from Latin Hypercube Sampling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path("./tuning_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_tuning_results():
    """Load tuning results for all sizes"""

    sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    all_results = {}

    print("=" * 80)
    print("LOADING TUNING RESULTS")
    print("=" * 80)

    for size in sizes:
        results_file = f"./tuning/{size}/lhs_results.csv"

        if Path(results_file).exists():
            df = pd.read_csv(results_file)
            all_results[size] = df
            print(f"✓ Size {size:5d}: {len(df)} configurations")
        else:
            print(f"✗ Size {size:5d}: No results found")

    print("=" * 80 + "\n")
    return all_results


def analyze_best_parameters(all_results):
    """Extract and analyze best parameters for each size"""

    print("=" * 80)
    print("BEST PARAMETERS BY INSTANCE SIZE")
    print("=" * 80)
    print()

    best_configs = []

    for size in sorted(all_results.keys()):
        df = all_results[size]

        # Sort by objective
        df_sorted = df.sort_values('avg_objective')
        best = df_sorted.iloc[0]

        print(f"SIZE {size}:")
        print("-" * 80)
        print(f"  Best config ID: {int(best['config_id'])}")
        print(f"  Population size: {int(best['pop_size'])}")
        print(f"  Generations: {int(best['generations'])}")
        print(f"  Crossover rate: {best['crossover_rate']:.4f}")
        print(f"  Mutation rate: {best['mutation_rate']:.4f}")
        print(f"  Elite size: {int(best['elite_size'])}")
        print(f"  Tournament size: {int(best['tournament_size'])}")
        print(f"  Early stop gens: {int(best['early_stop_gens'])}")

        if 'use_greedy_init' in best:
            init_type = "Greedy" if best['use_greedy_init'] else "Random"
            print(f"  Initialization: {init_type}")

        print(f"  Avg objective: {best['avg_objective']:.2f}")
        print()

        # Store for summary
        config = {
            'size': size,
            'pop_size': int(best['pop_size']),
            'generations': int(best['generations']),
            'crossover_rate': round(best['crossover_rate'], 4),
            'mutation_rate': round(best['mutation_rate'], 4),
            'elite_size': int(best['elite_size']),
            'tournament_size': int(best['tournament_size']),
            'early_stop_gens': int(best['early_stop_gens']),
            'avg_objective': round(best['avg_objective'], 2)
        }

        if 'use_greedy_init' in best:
            config['use_greedy_init'] = bool(best['use_greedy_init'])

        best_configs.append(config)

    return pd.DataFrame(best_configs)


def analyze_parameter_trends(all_results, best_configs_df):
    """Analyze how optimal parameters change with instance size"""

    print("=" * 80)
    print("PARAMETER TRENDS ANALYSIS")
    print("=" * 80)
    print()

    # Population size trend
    print("POPULATION SIZE:")
    for _, row in best_configs_df.iterrows():
        print(f"  Size {row['size']:5d}: {row['pop_size']:3d}")

    pop_corr = \
    np.corrcoef(best_configs_df['size'], best_configs_df['pop_size'])[0, 1]
    print(f"  → Correlation with size: {pop_corr:.3f}")

    if pop_corr > 0.7:
        print(f"  → STRONG POSITIVE: Larger instances need more population")
    elif pop_corr < -0.7:
        print(f"  → STRONG NEGATIVE: Larger instances need less population")
    else:
        print(f"  → WEAK: No clear trend")
    print()

    # Generations trend
    print("GENERATIONS:")
    for _, row in best_configs_df.iterrows():
        print(f"  Size {row['size']:5d}: {row['generations']:4d}")

    gen_corr = \
    np.corrcoef(best_configs_df['size'], best_configs_df['generations'])[0, 1]
    print(f"  → Correlation with size: {gen_corr:.3f}")

    if gen_corr > 0.7:
        print(f"  → STRONG POSITIVE: Larger instances need more generations")
    elif gen_corr < -0.7:
        print(f"  → STRONG NEGATIVE: Larger instances need fewer generations")
    else:
        print(f"  → WEAK: No clear trend")
    print()

    # Mutation rate trend
    print("MUTATION RATE:")
    for _, row in best_configs_df.iterrows():
        print(f"  Size {row['size']:5d}: {row['mutation_rate']:.4f}")

    mut_corr = \
    np.corrcoef(best_configs_df['size'], best_configs_df['mutation_rate'])[
        0, 1]
    print(f"  → Correlation with size: {mut_corr:.3f}")

    if mut_corr > 0.7:
        print(
            f"  → STRONG POSITIVE: Larger instances benefit from more mutation")
    elif mut_corr < -0.7:
        print(
            f"  → STRONG NEGATIVE: Larger instances benefit from less mutation")
    else:
        print(f"  → WEAK: Mutation rate doesn't scale predictably")
    print()

    # Initialization strategy (if available)
    if 'use_greedy_init' in best_configs_df.columns:
        print("INITIALIZATION STRATEGY:")
        for _, row in best_configs_df.iterrows():
            init = "Greedy" if row['use_greedy_init'] else "Random"
            print(f"  Size {row['size']:5d}: {init}")

        # Count greedy vs random for small/large
        small_greedy = best_configs_df[best_configs_df['size'] <= 500][
            'use_greedy_init'].sum()
        large_greedy = best_configs_df[best_configs_df['size'] >= 1000][
            'use_greedy_init'].sum()

        print(f"  → Small instances (≤500): {small_greedy}/5 use greedy")
        print(
            f"  → Large instances (≥1000): {large_greedy}/{len(best_configs_df[best_configs_df['size'] >= 1000])} use greedy")
        print()


def plot_parameter_trends(best_configs_df):
    """Visualize how parameters change with size"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    sizes = best_configs_df['size'].values

    # 1. Population size
    ax1 = axes[0, 0]
    ax1.plot(sizes, best_configs_df['pop_size'], marker='o', linewidth=2,
             markersize=8)
    ax1.set_xlabel('Instance Size', fontweight='bold')
    ax1.set_ylabel('Population Size', fontweight='bold')
    ax1.set_title('Optimal Population Size', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Generations
    ax2 = axes[0, 1]
    ax2.plot(sizes, best_configs_df['generations'], marker='s', linewidth=2,
             markersize=8)
    ax2.set_xlabel('Instance Size', fontweight='bold')
    ax2.set_ylabel('Generations', fontweight='bold')
    ax2.set_title('Optimal Generations', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Mutation rate
    ax3 = axes[0, 2]
    ax3.plot(sizes, best_configs_df['mutation_rate'], marker='d', linewidth=2,
             markersize=8)
    ax3.set_xlabel('Instance Size', fontweight='bold')
    ax3.set_ylabel('Mutation Rate', fontweight='bold')
    ax3.set_title('Optimal Mutation Rate', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 0.6])

    # 4. Crossover rate
    ax4 = axes[1, 0]
    ax4.plot(sizes, best_configs_df['crossover_rate'], marker='^', linewidth=2,
             markersize=8)
    ax4.set_xlabel('Instance Size', fontweight='bold')
    ax4.set_ylabel('Crossover Rate', fontweight='bold')
    ax4.set_title('Optimal Crossover Rate', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])

    # 5. Elite size
    ax5 = axes[1, 1]
    ax5.plot(sizes, best_configs_df['elite_size'], marker='v', linewidth=2,
             markersize=8)
    ax5.set_xlabel('Instance Size', fontweight='bold')
    ax5.set_ylabel('Elite Size', fontweight='bold')
    ax5.set_title('Optimal Elite Size', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Early stopping
    ax6 = axes[1, 2]
    ax6.plot(sizes, best_configs_df['early_stop_gens'], marker='*',
             linewidth=2, markersize=10)
    ax6.set_xlabel('Instance Size', fontweight='bold')
    ax6.set_ylabel('Early Stop Generations', fontweight='bold')
    ax6.set_title('Optimal Early Stopping', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'parameter_trends.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'parameter_trends.pdf', bbox_inches='tight')
    print(f"✓ Saved: parameter_trends.png")
    plt.close()


def plot_initialization_comparison(best_configs_df):
    """Compare greedy vs random initialization if available"""

    if 'use_greedy_init' not in best_configs_df.columns:
        print("⚠ No initialization data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = best_configs_df['size'].values
    colors = ['#2ca02c' if x else '#d62728' for x in
              best_configs_df['use_greedy_init']]
    labels = ['Greedy' if x else 'Random' for x in
              best_configs_df['use_greedy_init']]

    bars = ax.bar(range(len(sizes)), [1] * len(sizes), color=colors, alpha=0.7)

    ax.set_xlabel('Instance Size', fontweight='bold')
    ax.set_ylabel('Optimal Initialization', fontweight='bold')
    ax.set_title('Best Initialization Strategy by Size', fontweight='bold')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_yticks([])

    # Add labels on bars
    for i, (bar, label) in enumerate(zip(bars, labels)):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.5, label,
                ha='center', va='center', fontweight='bold', fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.7, label='Greedy'),
        Patch(facecolor='#d62728', alpha=0.7, label='Random')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'initialization_strategy.png',
                bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'initialization_strategy.pdf',
                bbox_inches='tight')
    print(f"✓ Saved: initialization_strategy.png")
    plt.close()


def generate_latex_table(best_configs_df):
    """Generate LaTeX table of best parameters"""

    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(
        r"\caption{Optimal EA parameters by instance size (Latin Hypercube Sampling)}")
    latex_lines.append(r"\label{tab:tuned_parameters}")
    latex_lines.append(r"\small")

    if 'use_greedy_init' in best_configs_df.columns:
        latex_lines.append(r"\begin{tabular}{lcccccccc}")
        latex_lines.append(r"\toprule")
        latex_lines.append(
            r"\textbf{Size} & \textbf{Init} & \textbf{Pop} & \textbf{Gen} & \textbf{Crossover} & \textbf{Mutation} & \textbf{Elite} & \textbf{Early Stop} & \textbf{Obj} \\")
    else:
        latex_lines.append(r"\begin{tabular}{lccccccc}")
        latex_lines.append(r"\toprule")
        latex_lines.append(
            r"\textbf{Size} & \textbf{Pop} & \textbf{Gen} & \textbf{Crossover} & \textbf{Mutation} & \textbf{Elite} & \textbf{Early Stop} & \textbf{Obj} \\")

    latex_lines.append(r"\midrule")

    for _, row in best_configs_df.iterrows():
        if 'use_greedy_init' in row:
            init = "G" if row['use_greedy_init'] else "R"
            latex_lines.append(
                f"{int(row['size'])} & {init} & {int(row['pop_size'])} & {int(row['generations'])} & "
                f"{row['crossover_rate']:.3f} & {row['mutation_rate']:.3f} & {int(row['elite_size'])} & "
                f"{int(row['early_stop_gens'])} & {row['avg_objective']:.0f} \\\\"
            )
        else:
            latex_lines.append(
                f"{int(row['size'])} & {int(row['pop_size'])} & {int(row['generations'])} & "
                f"{row['crossover_rate']:.3f} & {row['mutation_rate']:.3f} & {int(row['elite_size'])} & "
                f"{int(row['early_stop_gens'])} & {row['avg_objective']:.0f} \\\\"
            )

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")

    if 'use_greedy_init' in best_configs_df.columns:
        latex_lines.append(r"\\\vspace{0.2cm}")
        latex_lines.append(r"\footnotesize Init: G = Greedy, R = Random")

    latex_lines.append(r"\end{table}")

    latex_content = "\n".join(latex_lines)

    with open(OUTPUT_DIR / 'tuned_parameters_table.tex', 'w') as f:
        f.write(latex_content)

    print(f"✓ Saved: tuned_parameters_table.tex")


def main():
    print("\n" + "=" * 100)
    print("PARAMETER TUNING ANALYSIS")
    print("=" * 100)
    print()

    # Load results
    all_results = load_tuning_results()

    if not all_results:
        print("✗ No tuning results found!")
        return

    # Analyze best parameters
    best_configs_df = analyze_best_parameters(all_results)

    # Save best configs
    best_configs_df.to_csv(OUTPUT_DIR / 'best_parameters.csv', index=False)
    print(f"✓ Saved best parameters to: {OUTPUT_DIR / 'best_parameters.csv'}")
    print()

    # Analyze trends
    analyze_parameter_trends(all_results, best_configs_df)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_parameter_trends(best_configs_df)
    plot_initialization_comparison(best_configs_df)

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(best_configs_df)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated:")
    print("  • best_parameters.csv")
    print("  • parameter_trends.png/pdf")
    print("  • initialization_strategy.png/pdf")
    print("  • tuned_parameters_table.tex")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()