"""
Evolutionary Algorithm for SCF-PDP
Uses precedence-preserving genetic operators to maintain pickup-before-dropoff constraints.
"""

import random
from typing import List, Tuple
from SCFDP import (
    SCFPDPInstance, Solution, Route, PrecedenceRoute,
    create_empty_solution, repair_solution, repair_precedence,
    create_random_solution, write_solution, get_instance_name, repair_capacity_violations, repair_gamma
)


# ============================================================================
# INDIVIDUAL CLASS
# ============================================================================

class Individual:
    """Represents a solution in the population"""
    
    def __init__(self, solution: Solution):
        self.solution = solution
        self.fitness = None
        self.feasible = False
        self.penalty = 0.0
    
    def evaluate(self):
        """
        Calculate fitness with SMART penalties for infeasibility.
        This guides evolution toward feasibility.
        """
        feasible, msg = self.solution.is_feasible()
        self.feasible = feasible
        
        if feasible:
            # Feasible: fitness = negative objective (for maximization)
            obj_value = self.solution.objective_value()
            self.fitness = -obj_value
            self.penalty = 0.0
        else:
            # Infeasible: calculate penalties
            inst = self.solution.inst
            
            # Penalty components
            capacity_penalty = 0.0
            precedence_penalty = 0.0
            gamma_penalty = 0.0
            
            # 1. Capacity violations
            for route in self.solution.routes:
                if not route.is_empty():
                    try:
                        loads = route.calculate_load_at_each_stop(inst)
                        for load in loads:
                            if load > inst.C:
                                capacity_penalty += (load - inst.C) * 100
                            if load < 0:
                                capacity_penalty += abs(load) * 100
                    except:
                        capacity_penalty += 10000
            
            # 2. Precedence violations
            for route in self.solution.routes:
                requests = route.get_requests(inst)
                for req in requests:
                    pickup_idx = inst.pickup_idx(req)
                    dropoff_idx = inst.dropoff_idx(req)
                    
                    if pickup_idx in route.stops and dropoff_idx in route.stops:
                        pickup_pos = route.stops.index(pickup_idx)
                        dropoff_pos = route.stops.index(dropoff_idx)
                        
                        if pickup_pos >= dropoff_pos:
                            precedence_penalty += 1000
                    else:
                        precedence_penalty += 500  # Incomplete request
            
            # 3. Gamma constraint violation
            num_served = self.solution.num_served_requests()
            if num_served < inst.gamma:
                gamma_penalty = (inst.gamma - num_served) * 500
            
            # Total penalty
            self.penalty = capacity_penalty + precedence_penalty + gamma_penalty
            
            # Fitness = small base value - penalties
            # This way, less-infeasible solutions have higher fitness
            self.fitness = -10000 - self.penalty
        
        return self.fitness
    
    def copy(self) -> 'Individual':
        """Create a copy"""
        ind = Individual(self.solution.copy())
        ind.fitness = self.fitness
        ind.feasible = self.feasible
        ind.penalty = self.penalty
        return ind
    
# ============================================================================
# CROSSOVER OPERATORS
# ============================================================================

def ppox_crossover(parent1: PrecedenceRoute, parent2: PrecedenceRoute) -> Tuple[PrecedenceRoute, PrecedenceRoute]:
    """
    Precedence-Preserving Order Crossover (PPOX).
    Maintains pickup-before-dropoff constraints during crossover.
    Allows pickups and dropoffs to be positioned anywhere (not adjacent).
    """
    if len(parent1.sequence) == 0 or len(parent2.sequence) == 0:
        return parent1.copy(), parent2.copy()
    
    size1 = len(parent1.sequence)
    size2 = len(parent2.sequence)
    
    # Choose crossover points for parent1
    cx1 = random.randint(0, max(0, size1 - 1))
    cx2 = random.randint(cx1, size1)
    
    # Create child1: copy segment from parent1
    child1_seq = parent1.sequence[cx1:cx2].copy()
    child1_items = set(child1_seq)
    child1_picked = set(req for req, typ in child1_seq if typ == 'P')
    
    # Add items from parent2 that aren't already in child1
    for req, typ in parent2.sequence:
        if (req, typ) not in child1_items:
            # For dropoffs, only add if pickup is already in child
            if typ == 'D':
                if req in child1_picked or (req, 'P') in [(r, t) for r, t in child1_seq]:
                    child1_seq.append((req, typ))
            else:  # Pickup
                child1_seq.append((req, typ))
                child1_picked.add(req)
    
    # Create child2: same process with swapped parents
    child2_seq = parent2.sequence[cx1:min(cx2, size2)].copy()
    child2_items = set(child2_seq)
    child2_picked = set(req for req, typ in child2_seq if typ == 'P')
    
    for req, typ in parent1.sequence:
        if (req, typ) not in child2_items:
            if typ == 'D':
                if req in child2_picked or (req, 'P') in [(r, t) for r, t in child2_seq]:
                    child2_seq.append((req, typ))
            else:
                child2_seq.append((req, typ))
                child2_picked.add(req)
    
    child1 = PrecedenceRoute(child1_seq)
    child2 = PrecedenceRoute(child2_seq)
    
    # Repair if needed
    if not child1.is_valid():
        child1 = repair_precedence(child1)
    if not child2.is_valid():
        child2 = repair_precedence(child2)
    
    return child1, child2


def crossover_solutions(parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
    """
    Solution-level crossover with route-by-route PPOX.
    Applies repair after crossover to maintain feasibility.
    """
    inst = parent1.inst
    child1 = create_empty_solution(inst)
    child2 = create_empty_solution(inst)
    
    # Apply crossover to each route pair
    for k in range(inst.n_K):
        p1_prec = PrecedenceRoute.from_route(parent1.routes[k], inst)
        p2_prec = PrecedenceRoute.from_route(parent2.routes[k], inst)
        
        c1_prec, c2_prec = ppox_crossover(p1_prec, p2_prec)
        
        child1.routes[k] = c1_prec.to_route(inst)
        child2.routes[k] = c2_prec.to_route(inst)
    
    # Repair duplicates across routes
    child1 = repair_solution(child1)
    child2 = repair_solution(child2)
    
    # CRITICAL: Repair capacity violations
    child1 = repair_capacity_violations(child1)
    child2 = repair_capacity_violations(child2)

    child1 = repair_gamma(child1)
    child2 = repair_gamma(child2)
    
    return child1, child2

def crossover_inter_route(solution: Solution) -> Solution:
    """
    Crossover that exchanges requests between different vehicles within the same solution.
    """
    mutated = solution.copy()
    inst = solution.inst
    
    # Pick two random routes
    if inst.n_K < 2:
        return mutated
    
    route1_idx, route2_idx = random.sample(range(inst.n_K), 2)
    route1 = mutated.routes[route1_idx]
    route2 = mutated.routes[route2_idx]
    
    # Get requests from each route
    requests1 = route1.get_requests(inst)
    requests2 = route2.get_requests(inst)
    
    if not requests1 or not requests2:
        return mutated
    
    # Pick random requests to swap
    req1 = random.choice(list(requests1))
    req2 = random.choice(list(requests2))
    
    # Remove both requests from their routes
    route1.remove_request(inst, req1)
    route2.remove_request(inst, req2)
    
    # Try to add them to the other route (check capacity)
    # Convert to precedence routes for safe insertion
    prec1 = PrecedenceRoute.from_route(route1, inst)
    prec2 = PrecedenceRoute.from_route(route2, inst)
    
    # Add req2 to route1
    if len(prec1.sequence) == 0:
        prec1.sequence = [(req2, 'P'), (req2, 'D')]
    else:
        pos = random.randint(0, len(prec1.sequence))
        prec1.sequence.insert(pos, (req2, 'P'))
        prec1.sequence.insert(pos + 1, (req2, 'D'))
    
    # Add req1 to route2
    if len(prec2.sequence) == 0:
        prec2.sequence = [(req1, 'P'), (req1, 'D')]
    else:
        pos = random.randint(0, len(prec2.sequence))
        prec2.sequence.insert(pos, (req1, 'P'))
        prec2.sequence.insert(pos + 1, (req1, 'D'))
    
    # Convert back
    mutated.routes[route1_idx] = prec1.to_route(inst)
    mutated.routes[route2_idx] = prec2.to_route(inst)
    
    return mutated

# ============================================================================
# MUTATION OPERATORS
# ============================================================================

def mutate_swap_safe(route: PrecedenceRoute) -> PrecedenceRoute:
    """
    Swap two positions only if it doesn't violate precedence.
    Tries multiple times to find a valid swap.
    """
    if len(route.sequence) < 2:
        return route.copy()
    
    mutated = route.copy()
    
    # Try up to 10 times to find a valid swap
    for _ in range(10):
        i, j = random.sample(range(len(mutated.sequence)), 2)
        
        # Try swap
        temp_seq = mutated.sequence.copy()
        temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
        temp_route = PrecedenceRoute(temp_seq)
        
        if temp_route.is_valid():
            return temp_route
    
    # No valid swap found, return original
    return mutated


def mutate_insert_safe(route: PrecedenceRoute) -> PrecedenceRoute:
    """
    Remove an item and reinsert it at a valid position.
    Maintains precedence constraints while allowing separation of pickup/dropoff.
    """
    if len(route.sequence) < 2:
        return route.copy()
    
    mutated = route.sequence.copy()
    
    # Remove a random item
    i = random.randint(0, len(mutated) - 1)
    item = mutated.pop(i)
    req, typ = item
    
    # Find valid insertion positions
    valid_positions = []
    
    if typ == 'P':
        # Pickup: must be before its dropoff (if exists)
        dropoff_pos = None
        for idx, (r, t) in enumerate(mutated):
            if r == req and t == 'D':
                dropoff_pos = idx
                break
        
        if dropoff_pos is not None:
            valid_positions = list(range(dropoff_pos + 1))
        else:
            valid_positions = list(range(len(mutated) + 1))
    
    else:  # typ == 'D'
        # Dropoff: must be after its pickup (if exists)
        pickup_pos = None
        for idx, (r, t) in enumerate(mutated):
            if r == req and t == 'P':
                pickup_pos = idx
                break
        
        if pickup_pos is not None:
            valid_positions = list(range(pickup_pos + 1, len(mutated) + 1))
        else:
            valid_positions = [len(mutated)]
    
    if valid_positions:
        new_pos = random.choice(valid_positions)
        mutated.insert(new_pos, item)
    else:
        mutated.insert(i, item)
    
    return PrecedenceRoute(mutated)


def mutate_inversion(route: PrecedenceRoute) -> PrecedenceRoute:
    """
    Reverse a subsequence, then repair if needed.
    """
    if len(route.sequence) < 2:
        return route.copy()
    
    mutated = route.sequence.copy()
    
    # Choose subsequence to reverse
    i = random.randint(0, len(mutated) - 2)
    j = random.randint(i + 1, len(mutated))
    
    # Reverse
    mutated[i:j] = reversed(mutated[i:j])
    
    result = PrecedenceRoute(mutated)
    
    # Repair if invalid
    if not result.is_valid():
        result = repair_precedence(result)
    
    return result


def mutate_solution(solution: Solution, mutation_rate: float = 0.3) -> Solution:
    """
    Mutate solution with capacity-aware operators.
    """
    mutated = solution.copy()
    
    for k in range(solution.inst.n_K):
        if random.random() < mutation_rate:
            # Convert to precedence representation
            prec_route = PrecedenceRoute.from_route(mutated.routes[k], solution.inst)
            
            if len(prec_route.sequence) == 0:
                continue
            
            # Choose mutation type
            mutation_type = random.choice(['swap', 'insert', 'inversion', 'dropoff_shift'])
            
            if mutation_type == 'swap':
                prec_route = mutate_swap_safe(prec_route)
            elif mutation_type == 'insert':
                prec_route = mutate_insert_safe(prec_route)
            elif mutation_type == 'inversion':
                prec_route = mutate_inversion(prec_route)
            else:  # 'dropoff_shift' - NEW: helps with capacity
                prec_route = mutate_shift_dropoff(prec_route)
            
            # Convert back
            mutated.routes[k] = prec_route.to_route(solution.inst)
    
    # Repair any cross-route issues
    mutated = repair_solution(mutated)
    
    # CRITICAL: Repair capacity
    mutated = repair_capacity_violations(mutated)

    mutated = repair_gamma(mutated)
    
    return mutated


def mutate_shift_dropoff(route: PrecedenceRoute) -> PrecedenceRoute:
    """
    NEW MUTATION: Shift a dropoff earlier (closer to its pickup).
    This helps reduce peak capacity usage.
    """
    if len(route.sequence) < 3:
        return route.copy()
    
    mutated = route.sequence.copy()
    
    # Find all dropoffs that are far from their pickups
    candidates = []
    for i, (req, typ) in enumerate(mutated):
        if typ == 'D':
            # Find its pickup
            pickup_pos = None
            for j, (r, t) in enumerate(mutated):
                if r == req and t == 'P':
                    pickup_pos = j
                    break
            
            if pickup_pos is not None and i - pickup_pos > 2:
                candidates.append((i, req, pickup_pos))
    
    if candidates:
        # Pick random dropoff to shift
        dropoff_pos, req, pickup_pos = random.choice(candidates)
        
        # Remove dropoff
        mutated.pop(dropoff_pos)
        
        # Insert closer to pickup (but still after it)
        new_pos = random.randint(pickup_pos + 1, min(pickup_pos + 3, len(mutated) + 1))
        mutated.insert(new_pos, (req, 'D'))
    
    return PrecedenceRoute(mutated)


# ============================================================================
# SELECTION
# ============================================================================

def tournament_selection(population: List[Individual], num_select: int, tournament_size: int = 3) -> List[Individual]:
    """
    Select individuals using tournament selection.
    """
    selected = []
    for _ in range(num_select):
        tournament = random.sample(population, min(tournament_size, len(population)))
        winner = max(tournament, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))
        selected.append(winner.copy())
    return selected


# ============================================================================
# MAIN EVOLUTIONARY ALGORITHM
# ============================================================================

def evolutionary_algorithm(
    inst: SCFPDPInstance,
    pop_size: int = 100,
    generations: int = 1000,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.3,
    elite_size: int = 5,
    tournament_size: int = 3,
    verbose: bool = True,
    max_time_seconds: float = 900  # 15 minutes safety limit
) -> Solution:
    """
    Evolutionary Algorithm that RUNS UNTIL A FEASIBLE SOLUTION IS FOUND.
    
    Stopping criteria:
    1. Feasible solution found + no improvement for 200 generations, OR
    2. Maximum time limit reached (safety)
    """
    import time
    
    start_time = time.time()
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    if verbose:
        print(f"Initializing population of {pop_size} individuals (RANDOM)...")
        print(f"Will run until feasible solution found (max {max_time_seconds}s)")
    
    population = []
    for i in range(pop_size):
        sol = create_random_solution(inst, min_requests=inst.gamma, fairness_type='jain')
        sol = repair_gamma(sol)

        ind = Individual(sol)
        ind.evaluate()
        population.append(ind)
        
        if verbose and (i + 1) % 20 == 0:
            feasible = "✓" if ind.feasible else "✗"
            penalty = f"penalty={ind.penalty:.0f}" if not ind.feasible else ""
            print(f"  Created {i + 1}/{pop_size}: {feasible} {penalty}")
    
    # Sort by fitness
    population.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=True)
    
    feasible_count = sum(1 for ind in population if ind.feasible)
    
    if verbose:
        best = population[0]
        print(f"\n  Initial population: {feasible_count}/{pop_size} feasible")
        
        if best.feasible:
            print(f"  ✓ Initial best objective: {-best.fitness:.2f}")
        else:
            print(f"  Initial best penalty: {best.penalty:.0f}")
            print(f"  Will evolve until feasible solution found...")
    
    # ========================================================================
    # EVOLUTION LOOP - Modified stopping criteria
    # ========================================================================
    
    best_ever = population[0].copy()
    no_improvement = 0
    gen = 0
    found_feasible = best_ever.feasible
    
    # KEEP GOING UNTIL FEASIBLE SOLUTION FOUND
    while True:
        gen += 1
        
        # Safety check: time limit
        elapsed = time.time() - start_time
        if elapsed > max_time_seconds:
            if verbose:
                print(f"\n⚠ Time limit ({max_time_seconds}s) reached at generation {gen}")
                if not found_feasible:
                    print(f"  WARNING: No feasible solution found!")
                    print(f"  Best penalty: {best_ever.penalty:.0f}")
            break
        
        # Generate offspring
        offspring = []
        elites = population[:elite_size]
        
        while len(offspring) < pop_size - elite_size:
            # Selection
            parents = tournament_selection(population, num_select=2, tournament_size=tournament_size)
            
            # Crossover
            if random.random() < crossover_rate and len(parents) >= 2:
                child1_sol, child2_sol = crossover_solutions(parents[0].solution, parents[1].solution)
                offspring.extend([Individual(child1_sol), Individual(child2_sol)])
            else:
                offspring.extend([parents[0].copy(), parents[1].copy() if len(parents) > 1 else parents[0].copy()])
        
        # Mutation
        for ind in offspring:
            if random.random() < mutation_rate:
                ind.solution = mutate_solution(ind.solution, mutation_rate=mutation_rate)
        
        # Evaluate offspring
        for ind in offspring:
            ind.evaluate()
        
        # Replacement
        population = elites + offspring[:pop_size - elite_size]
        population.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=True)
        
        # Update best
        current_best = population[0]
        
        # Track if we found feasible for the first time
        if current_best.feasible and not found_feasible:
            found_feasible = True
            best_ever = current_best.copy()
            no_improvement = 0
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"✓✓✓ FEASIBLE SOLUTION FOUND at generation {gen}! ✓✓✓")
                print(f"{'='*60}")
                print(f"  Objective: {-best_ever.fitness:.2f}")
                print(f"  Time: {elapsed:.1f}s")
                print(f"  Now optimizing for {200} more generations...")
                print(f"{'='*60}\n")
        
        # Update best if improved
        elif current_best.feasible and (not best_ever.feasible or current_best.fitness > best_ever.fitness):
            improvement = current_best.fitness - best_ever.fitness
            best_ever = current_best.copy()
            no_improvement = 0
            
            if verbose and gen % 50 == 0:
                print(f"  Gen {gen}: NEW BEST! Obj={-best_ever.fitness:.2f} (improved by {-improvement:.2f})")
        
        else:
            no_improvement += 1
        
        # Print progress
        if verbose and gen % 50 == 0:
            feasible_count = sum(1 for ind in population if ind.feasible)
            
            if best_ever.feasible:
                print(f"  Gen {gen}: Best={-best_ever.fitness:.2f}, "
                      f"Current={-current_best.fitness:.2f if current_best.feasible else 'INFEAS'}, "
                      f"Feasible={feasible_count}/{pop_size}, NoImpr={no_improvement}, "
                      f"Time={elapsed:.1f}s")
            else:
                print(f"  Gen {gen}: Searching for feasibility... "
                      f"BestPenalty={best_ever.penalty:.0f}, "
                      f"Feasible={feasible_count}/{pop_size}, "
                      f"Time={elapsed:.1f}s")
        
        # STOPPING CRITERIA
        # Only stop if:
        # 1. We have a feasible solution, AND
        # 2. No improvement for 200 generations, OR
        # 3. Reached max generations
        
        if found_feasible and no_improvement > 200:
            if verbose:
                print(f"\n  Stopping: No improvement for 200 generations")
            break
        
        if gen >= generations:
            if verbose:
                if found_feasible:
                    print(f"\n  Stopping: Reached maximum generations ({generations})")
                else:
                    print(f"\n  ⚠ Stopping: Reached max generations WITHOUT finding feasible solution")
            break
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    
    if verbose:
        print("\n" + "=" * 60)
        print("EVOLUTION COMPLETE")
        print("=" * 60)
        print(f"Total generations: {gen}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        
        if best_ever.feasible:
            print(f"✓ FEASIBLE SOLUTION FOUND")
            best_ever.solution.print_statistics()
        else:
            print(f"✗ NO FEASIBLE SOLUTION FOUND")
            print(f"Best penalty: {best_ever.penalty:.0f}")
            print(f"Requests served: {best_ever.solution.num_served_requests()}/{inst.gamma}")
            print("\nBest infeasible solution statistics:")
            best_ever.solution.print_statistics()
        
        print("=" * 60)
    
    return best_ever.solution

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evolutionary_algorithm.py <instance_file> [output_file]")
        print("\nExample: python evolutionary_algorithm.py instances/instance1.txt solution_ea.txt")
        sys.exit(1)
    
    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"solution_{get_instance_name(instance_file)}_ea.txt"
    
    # Load instance
    print(f"Loading instance: {instance_file}")
    inst = SCFPDPInstance(instance_file)
    
    print(f"\nInstance details:")
    print(f"  Requests: {inst.n}")
    print(f"  Vehicles: {inst.n_K}")
    print(f"  Capacity: {inst.C}")
    print(f"  Minimum requests to serve: {inst.gamma}")
    print(f"  Fairness weight (rho): {inst.rho}")
    
    # Run evolutionary algorithm
    print("\n" + "=" * 60)
    print("Running Evolutionary Algorithm")
    print("=" * 60)
    
    best_solution = evolutionary_algorithm(
        inst,
        pop_size=100,
        generations=1000,
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_size=5,
        tournament_size=3,
        verbose=True
    )
    
    # Save solution
    write_solution(best_solution, output_file, get_instance_name(instance_file))
    print(f"\nSolution saved to: {output_file}")
    
    # Validate
    print(f"\nValidating solution...")
    feasible, msg = best_solution.is_feasible()
    if feasible:
        print(f"✓ Solution is FEASIBLE")
    else:
        print(f"✗ Solution is INFEASIBLE: {msg}")