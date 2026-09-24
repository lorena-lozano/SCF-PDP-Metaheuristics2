"""
Parameter tuning using Latin Hypercube Sampling (DOE approach)
Offline tuning method from the lecture slides
"""

import numpy as np
from scipy.stats import qmc
import pandas as pd
import os
from SCFDP import SCFPDPInstance
from run_ea_tuning import run_ea_for_tuning

def latin_hypercube_sampling(n_samples: int = 100):
    """
    Generate parameter configurations using Latin Hypercube Sampling
    
    Parameters to tune:
    - pop_size: [30, 150]
    - generations: [200, 1000]
    - crossover_rate: [0.5, 0.95]
    - mutation_rate: [0.1, 0.5]
    - elite_size: [2, 10]
    - tournament_size: [2, 5]
    - early_stop_gens: [20, 150]
    """
    
    # Define parameter bounds
    param_bounds = {
        'pop_size': (30, 150),
        'generations': (200, 1000),
        'crossover_rate': (0.5, 0.95),
        'mutation_rate': (0.1, 0.5),
        'elite_size': (2, 10),
        'tournament_size': (2, 5),
        'early_stop_gens': (20, 150),
    }
    
    n_params = len(param_bounds)
    
    # Create Latin Hypercube Sampler
    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    
    # Generate samples in [0, 1]^d
    samples = sampler.random(n=n_samples)
    
    # Scale to parameter ranges
    configurations = []
    param_names = list(param_bounds.keys())
    
    for sample in samples:
        config = {}
        for i, param_name in enumerate(param_names):
            lower, upper = param_bounds[param_name]
            
            # Scale to range
            value = lower + sample[i] * (upper - lower)
            
            # Round integers
            if param_name in ['pop_size', 'generations', 'elite_size', 'tournament_size', 'early_stop_gens']:
                value = int(round(value))
            
            config[param_name] = value
        
        configurations.append(config)
    
    return configurations


def evaluate_configuration(config: dict, instance_files: list, n_runs: int = 3):
    """
    Evaluate a configuration on multiple instances with multiple seeds
    Returns average objective value
    """
    objectives = []
    
    for instance_file in instance_files:
        for seed in range(n_runs):
            obj = run_ea_for_tuning(instance_file, seed, **config)
            objectives.append(obj)
    
    # Return average objective
    return np.mean(objectives)


def tune_parameters_lhs(instance_size: int, n_samples: int = 100, n_train_instances: int = 5):
    """
    Tune parameters using Latin Hypercube Sampling for a given instance size
    """
    
    print(f"\n{'='*80}")
    print(f"TUNING FOR SIZE {instance_size} - Latin Hypercube Sampling")
    print(f"{'='*80}")
    print(f"Number of configurations to test: {n_samples}")
    print(f"Training instances: {n_train_instances}")
    
    # Get training instances
    train_dir = f"./instances/{instance_size}/train"
    instance_files = [
        os.path.join(train_dir, f) 
        for f in sorted(os.listdir(train_dir)) 
        if f.endswith('.txt')
    ][:n_train_instances]
    
    print(f"Loaded {len(instance_files)} training instances")
    
    # Generate configurations using LHS
    print("\nGenerating parameter configurations with LHS...")
    configurations = latin_hypercube_sampling(n_samples=n_samples)
    
    # Evaluate each configuration
    results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n[{i}/{n_samples}] Testing configuration:")
        print(f"  pop_size={config['pop_size']}, gen={config['generations']}, "
              f"cx={config['crossover_rate']:.3f}, mut={config['mutation_rate']:.3f}")
        
        avg_obj = evaluate_configuration(config, instance_files, n_runs=3)
        
        print(f"  → Average objective: {avg_obj:.2f}")
        
        result = config.copy()
        result['avg_objective'] = avg_obj
        result['instance_size'] = instance_size
        results.append(result)
    
    # Convert to DataFrame and sort by objective
    df = pd.DataFrame(results)
    df = df.sort_values('avg_objective')
    
    # Save results
    output_dir = f"./tuning/{instance_size}"
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(f"{output_dir}/lhs_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/lhs_results.csv")
    
    # Print best configuration
    best_config = df.iloc[0]
    print(f"\n{'='*80}")
    print(f"BEST CONFIGURATION FOR SIZE {instance_size}")
    print(f"{'='*80}")
    print(f"pop_size:         {int(best_config['pop_size'])}")
    print(f"generations:      {int(best_config['generations'])}")
    print(f"crossover_rate:   {best_config['crossover_rate']:.4f}")
    print(f"mutation_rate:    {best_config['mutation_rate']:.4f}")
    print(f"elite_size:       {int(best_config['elite_size'])}")
    print(f"tournament_size:  {int(best_config['tournament_size'])}")
    print(f"early_stop_gens:  {int(best_config['early_stop_gens'])}")
    print(f"Average objective: {best_config['avg_objective']:.2f}")
    print(f"{'='*80}\n")
    
    return df


if __name__ == "__main__":
    """
    Automatic tuning for all instance sizes
    No command-line arguments needed - just run: python tune_with_lhs.py
    """
    
    # Configuration for each size
    tuning_config = {
        50:   {'samples': 100, 'instances': 8},   # Small, can afford more samples
        100:  {'samples': 100, 'instances': 8},
        200:  {'samples': 80,  'instances': 6},
        500:  {'samples': 60,  'instances': 5},
        1000: {'samples': 50,  'instances': 4},
        2000: {'samples': 40,  'instances': 3},   # Larger, fewer samples
        5000: {'samples': 30,  'instances': 2},
        10000: {'samples': 20, 'instances': 2},
    }
    
    print("="*80)
    print("AUTOMATED PARAMETER TUNING - LATIN HYPERCUBE SAMPLING")
    print("="*80)
    print("\nConfiguration:")
    for size, config in tuning_config.items():
        print(f"  Size {size:5d}: {config['samples']:3d} samples, {config['instances']} training instances")
    print("="*80)
    
    # Store all results
    all_results = {}
    
    # Run tuning for each size
    for size, config in tuning_config.items():
        # Check if instances exist
        train_dir = f"./instances/{size}/train"
        if not os.path.exists(train_dir):
            print(f"\n⚠ Skipping size {size}: directory {train_dir} not found")
            continue
        
        try:
            results_df = tune_parameters_lhs(
                instance_size=size,
                n_samples=config['samples'],
                n_train_instances=config['instances']
            )
            
            all_results[size] = results_df
            
            print(f"\n✓ Completed tuning for size {size}")
            print("Top 3 configurations:")
            print(results_df.head(3)[['pop_size', 'generations', 'crossover_rate', 
                                      'mutation_rate', 'elite_size', 'avg_objective']])
            
        except Exception as e:
            print(f"\n✗ Error tuning size {size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary of best parameters for each size
    print("\n" + "="*80)
    print("SUMMARY: BEST PARAMETERS PER SIZE")
    print("="*80)
    
    summary_data = []
    for size, df in all_results.items():
        best = df.iloc[0]
        summary_data.append({
            'size': size,
            'pop_size': int(best['pop_size']),
            'generations': int(best['generations']),
            'crossover_rate': round(best['crossover_rate'], 4),
            'mutation_rate': round(best['mutation_rate'], 4),
            'elite_size': int(best['elite_size']),
            'tournament_size': int(best['tournament_size']),
            'early_stop_gens': int(best['early_stop_gens']),
            'avg_objective': round(best['avg_objective'], 2),
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_df.to_csv('./tuning/tuning_summary.csv', index=False)
    print("\n" + summary_df.to_string(index=False))
    print(f"\n✓ Summary saved to ./tuning/tuning_summary.csv")
    print("="*80)