"""
Simple EA Runner - Runs all instances with ALL fairness measures
Writes results incrementally with fairness type in folder structure

Usage:
    python run_ea_simple.py
"""

import os
import sys
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from SCFDP import (
    SCFPDPInstance, Solution, 
    write_solution, get_instance_name
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# EA Parameters
# ============================================================================
# CONFIGURATION - DYNAMIC BASED ON SIZE
# ============================================================================

def get_ea_config(instance_size: int) -> Dict:
    """
    Return EA configuration based on instance size.
    Includes convergence threshold for tracking.
    
    50-200: early_stop_gens=200 (your original setting)
    500-1000: early_stop_gens=50 (your updated setting)
    2000+: Reduced settings for speed
    """
    if instance_size <= 200:
        # ORIGINAL SETTINGS (50, 100, 200)
        return {
            'pop_size': 100,
            'generations': 500,
            'crossover_rate': 0.8,
            'mutation_rate': 0.3,
            'elite_size': 5,
            'tournament_size': 3,
            'max_time_seconds': 900,  # 15 min
            'early_stop_gens': 200,   # ORIGINAL threshold
        }
    elif instance_size <= 1000:
        # UPDATED SETTINGS (500, 1000)
        return {
            'pop_size': 100,
            'generations': 500,
            'crossover_rate': 0.8,
            'mutation_rate': 0.3,
            'elite_size': 5,
            'tournament_size': 3,
            'max_time_seconds': 900,  # 15 min
            'early_stop_gens': 50,    # UPDATED threshold
        }
    elif instance_size == 2000:
        # Medium reduction for 2000
        return {
            'pop_size': 75,
            'generations': 400,
            'crossover_rate': 0.8,
            'mutation_rate': 0.3,
            'elite_size': 4,
            'tournament_size': 3,
            'max_time_seconds': 600,  # 10 min
            'early_stop_gens': 40,
        }
    elif instance_size == 5000:
        # Larger reduction for 5000
        return {
            'pop_size': 50,
            'generations': 300,
            'crossover_rate': 0.8,
            'mutation_rate': 0.3,
            'elite_size': 3,
            'tournament_size': 3,
            'max_time_seconds': 450,  # 7.5 min
            'early_stop_gens': 30,
        }
    else:  # 10000
        # Maximum reduction for 10000
        return {
            'pop_size': 40,
            'generations': 200,
            'crossover_rate': 0.8,
            'mutation_rate': 0.3,
            'elite_size': 2,
            'tournament_size': 3,
            'max_time_seconds': 300,  # 5 min
            'early_stop_gens': 25,
        }

# Instance sampling: How many per size
INSTANCES_CONFIG = {
    '50': None,      # None = all (30)
    '100': None,     # None = all (30)
    '200': None,     # None = all (30)
    '500': None,     # None = all (30)
    '1000': None,    # None = all (30)
    '2000': 15,      # 15 instances
    '5000': 10,      # 10 instances
    '10000': 5,      # 5 instances
}

# Fairness measures to test
FAIRNESS_MEASURES = ['jain', 'maxmin', 'gini']

# Instances to test
INSTANCES_ROOT = "./instances"
INSTANCES_PER_SIZE = None  # None = all instances

# Output
OUTPUT_DIR = "./ea_results"
SAVE_SOLUTIONS = True
SAVE_CONVERGENCE = True

# ============================================================================
# INCREMENTAL CSV WRITER
# ============================================================================

class IncrementalCSVWriter:
    """Writes results to CSV incrementally after each instance"""
    
    def __init__(self, filename: str, fieldnames: List[str]):
        self.filename = filename
        self.fieldnames = fieldnames
        self.is_new_file = not os.path.exists(filename)
        
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        if self.is_new_file:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def append_row(self, row: Dict):
        formatted = row.copy()
        for key in ['objective', 'total_duration', 'fairness', 'runtime',
                   'avg_duration', 'min_duration', 'max_duration', 'avg_stops_per_route']:
            if key in formatted and isinstance(formatted[key], float):
                if formatted[key] == float('inf'):
                    formatted[key] = 'INF'
                else:
                    formatted[key] = f"{formatted[key]:.4f}"
        
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(formatted)
        
        print(f"  ✓ Result appended to: {self.filename}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_completed_instances(results_file: str) -> set:
    completed = set()
    if not os.path.exists(results_file):
        return completed
    
    with open(results_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if 'instance' not in reader.fieldnames:
            return completed
        for row in reader:
            inst = row.get('instance')
            if inst:
                completed.add(inst)
    return completed

# ============================================================================
# EA WITH FAIRNESS PARAMETER
# ============================================================================

def run_ea_with_tracking(inst: SCFPDPInstance, fairness_type: str = 'jain', **params) -> Tuple[Solution, List[Dict]]:
    """Run EA with specified fairness measure"""
    from evolutionary_alg import (
        Individual, tournament_selection, crossover_solutions,
        mutate_solution
    )
    from SCFDP import create_random_solution
    import random
    import time
    
    pop_size = params.get('pop_size', 100)
    max_generations = params.get('generations', 10000)
    crossover_rate = params.get('crossover_rate', 0.8)
    mutation_rate = params.get('mutation_rate', 0.3)
    elite_size = params.get('elite_size', 5)
    tournament_size = params.get('tournament_size', 3)
    max_time_seconds = params.get('max_time_seconds', 900)
    
    start_time = time.time()
    
    # Initialize with specified fairness type
    population = [Individual(create_random_solution(inst, min_requests=inst.gamma, fairness_type=fairness_type))
                  for _ in range(pop_size)]
    for ind in population:
        ind.evaluate()
    population.sort(key=lambda ind: (ind.feasible, ind.fitness), reverse=True)
    
    best_ever = population[0].copy()
    no_improvement = 0
    best_per_generation = []
    found_feasible = best_ever.feasible
    gen = 0
    
    while True:
        gen += 1
        elapsed = time.time() - start_time
        if elapsed > max_time_seconds:
            print(f"  ⚠ Time limit reached at generation {gen}")
            break
        
        current_best = population[0]
        if current_best.feasible:
            stats = current_best.solution.get_statistics()
            total_stops = sum(len(route.stops) for route in current_best.solution.routes)
            best_per_generation.append({
                'generation': gen,
                'objective': stats['objective'],
                'total_duration': stats['total_duration'],
                'fairness': stats['fairness'],
                'num_served': stats['num_served'],
                'num_active_routes': stats['num_active_routes'],
                'total_stops': total_stops,
                'feasible': True
            })
        else:
            best_per_generation.append({
                'generation': gen,
                'objective': float('inf'),
                'total_duration': float('inf'),
                'fairness': 0.0,
                'num_served': current_best.solution.num_served_requests(),
                'num_active_routes': 0,
                'total_stops': 0,
                'feasible': False
            })
        
        # Evolution
        offspring = []
        elites = population[:elite_size]
        while len(offspring) < pop_size - elite_size:
            parents = tournament_selection(population, num_select=2, tournament_size=tournament_size)
            if random.random() < crossover_rate and len(parents) >= 2:
                child1_sol, child2_sol = crossover_solutions(parents[0].solution, parents[1].solution)
                offspring.extend([Individual(child1_sol), Individual(child2_sol)])
            else:
                offspring.extend([parents[0].copy(), parents[1].copy() if len(parents) > 1 else parents[0].copy()])
        
        for ind in offspring:
            if random.random() < mutation_rate:
                ind.solution = mutate_solution(ind.solution, mutation_rate=mutation_rate)
        for ind in offspring:
            ind.evaluate()
        
        population = elites + offspring[:pop_size - elite_size]
        population.sort(key=lambda ind: (ind.feasible, ind.fitness), reverse=True)
        
        # Update best feasible
        improved = False
        for ind in population:
            if ind.feasible:
                if (not found_feasible) or (not best_ever.feasible) or (ind.fitness > best_ever.fitness):
                    best_ever = ind.copy()
                    found_feasible = True
                    no_improvement = 0
                    improved = True

        if improved and found_feasible:
            print(f"  ✓ FEASIBLE at gen {gen}! Obj={-best_ever.fitness:.2f}, Time={elapsed:.1f}s")
        else:
            no_improvement += 1
        
        # Progress
        if gen % 50 == 0:
            feasible_count = sum(1 for ind in population if ind.feasible)
            if found_feasible:
                print(f"  Gen {gen}: Best={-best_ever.fitness:.2f}, Feas={feasible_count}/{pop_size}")
            else:
                print(f"  Gen {gen}: Searching... Feas={feasible_count}/{pop_size}, Time={elapsed:.1f}s")
        
        # Stopping
        if found_feasible and no_improvement > 50:
            print(f"  Stopping: No improvement for 50 generations")
            break
        if gen >= max_generations:
            break
    
    tracking_info = {
        'total_generations': gen,
        'best_per_generation': best_per_generation,
        'found_feasible': found_feasible
    }
    
    return best_ever.solution, tracking_info

# ============================================================================
# SELECT INSTANCES
# ============================================================================

def select_instances(instances_root: str) -> Dict[str, List[str]]:  # ← Changed return type
    """Select instances based on INSTANCES_CONFIG"""
    instances_by_size = {}  # ← Returns dict instead of list
    
    for size, limit in INSTANCES_CONFIG.items():  # ← Uses INSTANCES_CONFIG
        train_dir = os.path.join(instances_root, size, "train")
        if not os.path.exists(train_dir):
            print(f"  ⚠ Not found: {train_dir}")
            continue
        
        instance_files = sorted(
            os.path.join(train_dir, f) 
            for f in os.listdir(train_dir) 
            if f.endswith('.txt')
        )
        
        if limit is None:  # ← Uses limit from INSTANCES_CONFIG
            instances_by_size[size] = instance_files
            print(f"  ✓ Size {size}: ALL ({len(instance_files)}) instances")
        else:
            instances_by_size[size] = instance_files[:limit]
            print(f"  ✓ Size {size}: {limit} instances (out of {len(instance_files)})")
    
    return instances_by_size  # ← Returns dict, not list

# ============================================================================
# SAVE CONVERGENCE
# ============================================================================

def save_convergence_data(convergence_data: List[Dict], instance_name: str, output_dir: str):
    conv_file = os.path.join(output_dir, f"{instance_name}_convergence.csv")
    if not convergence_data:
        return
    fieldnames = ['generation', 'objective', 'total_duration', 'fairness', 
                  'num_served', 'num_active_routes', 'total_stops', 'feasible']
    with open(conv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in convergence_data:
            formatted = row.copy()
            for key in ['objective', 'total_duration', 'fairness']:
                if key in formatted and isinstance(formatted[key], float):
                    formatted[key] = 'INF' if formatted[key] == float('inf') else f"{formatted[key]:.4f}"
            writer.writerow(formatted)

# ============================================================================
# RUN SINGLE INSTANCE WITH FAIRNESS
# ============================================================================

def run_instance(instance_file: str, fairness_type: str, config: Dict, 
                output_dir: str, results_writer: IncrementalCSVWriter) -> Tuple[Dict, Solution, List[Dict]]:
    inst_name = get_instance_name(instance_file)
    print(f"\nProcessing: {inst_name} [{fairness_type.upper()}]")
    
    instance = SCFPDPInstance(instance_file)
    print(f"  {instance.n} req, {instance.n_K} veh, cap={instance.C}, gamma={instance.gamma}")
    
    start_time = time.time()
    solution, tracking_info = run_ea_with_tracking(instance, fairness_type=fairness_type, **config)
    runtime = time.time() - start_time
    
    convergence_data = tracking_info['best_per_generation']
    stats = solution.get_statistics()
    total_stops = sum(len(route.stops) for route in solution.routes)
    feasible, _ = solution.is_feasible()
    
    result = {
        'instance': inst_name,
        'fairness_measure': fairness_type,  # NEW FIELD
        'instance_size': instance.n,
        'num_vehicles': instance.n_K,
        'capacity': instance.C,
        'min_requests': instance.gamma,
        'rho': instance.rho,
        'feasible': feasible,
        'objective': stats['objective'] if feasible else float('inf'),
        'total_duration': stats['total_duration'],
        'fairness': stats['fairness'],
        'num_served': stats['num_served'],
        'num_active_routes': stats['num_active_routes'],
        'total_stops': total_stops,
        'avg_stops_per_route': total_stops / instance.n_K if instance.n_K > 0 else 0,
        'min_duration': stats['min_duration'],
        'max_duration': stats['max_duration'],
        'avg_duration': stats['avg_duration'],
        'runtime': runtime,
        'total_generations': len(convergence_data),
    }
    
    print(f"  Result: {'FEASIBLE' if feasible else 'INFEASIBLE'}, "
          f"Obj={'%.2f' % result['objective'] if feasible else 'INF'}, "
          f"Fairness={result['fairness']:.4f}, Runtime={runtime:.1f}s")
    
    results_writer.append_row(result)
    
    # Save solution
    if SAVE_SOLUTIONS and feasible:
        sol_file = os.path.join(output_dir, "solutions", f"{inst_name}_solution.txt")
        write_solution(solution, sol_file, inst_name)
    
    # Save convergence
    if SAVE_CONVERGENCE:
        conv_dir = os.path.join(output_dir, "convergence")
        save_convergence_data(convergence_data, inst_name, conv_dir)
    
    return result, solution, convergence_data

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("EVOLUTIONARY ALGORITHM - ALL FAIRNESS MEASURES")
    print("=" * 80)
    print(f"Fairness measures: {', '.join(FAIRNESS_MEASURES)}")
    print("=" * 80)
    
    # NOW RETURNS DICT, NOT LIST
    instances_by_size = select_instances(INSTANCES_ROOT)
    
    if not instances_by_size:
        print("✗ No instances found!")
        return
    
    result_fieldnames = [
        'instance', 'fairness_measure', 'instance_size', 'num_vehicles', 
        'capacity', 'min_requests', 'rho', 'feasible', 'objective', 
        'total_duration', 'fairness', 'num_served', 'num_active_routes', 
        'total_stops', 'avg_stops_per_route', 'min_duration', 'max_duration', 
        'avg_duration', 'runtime', 'total_generations', 'stopped_reason',
        'config_pop_size', 'config_max_gens', 'config_convergence_threshold'
    ]
    
    # Process each size and fairness measure
    for size in sorted(instances_by_size.keys(), key=int):
        instance_size = int(size)
        ea_config = get_ea_config(instance_size)
        
        for fairness_type in FAIRNESS_MEASURES:
            print(f"\n{'='*80}")
            print(f"SIZE: {size} | FAIRNESS: {fairness_type.upper()}")
            print(f"{'='*80}")
            
            size_fair_dir = os.path.join(OUTPUT_DIR, size, fairness_type)
            os.makedirs(size_fair_dir, exist_ok=True)
            os.makedirs(os.path.join(size_fair_dir, "solutions"), exist_ok=True)
            os.makedirs(os.path.join(size_fair_dir, "convergence"), exist_ok=True)
            
            results_file = os.path.join(size_fair_dir, "results.csv")
            
            # ADD THIS DEBUG CODE:
            print(f"  Looking for: {results_file}")
            print(f"  File exists: {os.path.exists(results_file)}")
            
            completed = load_completed_instances(results_file)
            
            # ADD THIS DEBUG CODE:
            print(f"  Completed instances found: {len(completed)}")
            if len(completed) > 0 and len(completed) <= 5:
                print(f"  Sample: {list(completed)[:5]}")
            results_writer = IncrementalCSVWriter(results_file, result_fieldnames)
            
            size_instances = instances_by_size[size]
            print(f"Instances: {len(size_instances)}, Completed: {len(completed)}\n")
            
            for i, inst_file in enumerate(size_instances, 1):
                inst_name = get_instance_name(inst_file)
                
                if inst_name in completed:  # ← SKIPS IF ALREADY DONE
                    print(f"[{i}/{len(size_instances)}] SKIP: {inst_name}")
                    continue
                
                print(f"[{i}/{len(size_instances)}]", end=" ")
                
                try:
                    run_instance(inst_file, fairness_type, ea_config, size_fair_dir, results_writer)
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

