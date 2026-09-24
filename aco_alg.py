"""
Ant Colony Optimization (ACO) for SCF-PDP
- Builds solutions with precedence + capacity integrated during construction
- Uses SCFDP repair functions (precedence/capacity/gamma) for robustness
- Tracks convergence (best per iteration)
- OPTIONAL greedy initialization for large instances:
    (1) Greedy seed solution to quickly reach near-feasibility
    (2) Seed pheromone matrix using greedy edges
    (3) Bias request subset selection by depot distance
"""

import time
import random
from typing import Dict, List, Tuple, Optional

from SCFDP import (
    SCFPDPInstance, Solution, Route,
    repair_solution, repair_capacity_violations, repair_gamma,
    create_empty_solution, PrecedenceRoute
)

# ============================================================
# Helpers
# ============================================================

def _roulette_select(items: List[int], weights: List[float]) -> int:
    """Roulette wheel selection (weights must be >= 0)."""
    s = sum(weights)
    if s <= 0:
        return random.choice(items)
    r = random.random() * s
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if acc >= r:
            return it
    return items[-1]


def _objective_safe(sol: Solution) -> float:
    """Objective, guarded against unexpected errors."""
    try:
        return sol.objective_value()
    except Exception:
        return float("inf")


def _edges_of_solution(inst: SCFPDPInstance, sol: Solution) -> List[Tuple[int, int]]:
    """Return all directed edges used (including depot start/end)."""
    edges = []
    for r in sol.routes:
        if r.is_empty():
            continue
        prev = 0
        for s in r.stops:
            edges.append((prev, s))
            prev = s
        edges.append((prev, 0))
    return edges


# ============================================================
# Size-dependent config (as in EA)
# ============================================================

def get_aco_config(instance_size: int) -> Dict:
    """
    Size-dependent ACO configuration.
    Includes greedy seeding for large instances.
    """
    if instance_size <= 200:
        return {
            "n_ants": 40,
            "max_iters": 400,
            "alpha": 1.0,
            "beta": 3.0,
            "evap": 0.20,
            "Q": 200.0,
            "q0": 0.10,
            "subset_extra_frac": 0.10,
            "elite_deposit": True,
            "local_search_rate": 0.20,
            "early_stop_iters": 120,
            "max_time_seconds": 900,

            # Greedy controls
            "use_greedy_seed": False,
            "greedy_seed_strength": 4.0,   # ignored if use_greedy_seed False
            "large_instance_threshold": 2000,
            "subset_bias_power": 0.0,      # 0 => uniform
        }
    elif instance_size <= 1000:
        return {
            "n_ants": 30,
            "max_iters": 250,
            "alpha": 1.0,
            "beta": 3.5,
            "evap": 0.25,
            "Q": 200.0,
            "q0": 0.15,
            "subset_extra_frac": 0.10,
            "elite_deposit": True,
            "local_search_rate": 0.15,
            "early_stop_iters": 60,
            "max_time_seconds": 900,

            "use_greedy_seed": False,
            "greedy_seed_strength": 4.0,
            "large_instance_threshold": 2000,
            "subset_bias_power": 0.0,
        }
    elif instance_size == 2000:
        return {
            "n_ants": 18,
            "max_iters": 160,
            "alpha": 1.0,
            "beta": 4.0,
            "evap": 0.30,
            "Q": 250.0,
            "q0": 0.20,
            "subset_extra_frac": 0.06,
            "elite_deposit": True,
            "local_search_rate": 0.10,
            "early_stop_iters": 45,
            "max_time_seconds": 600,

            # Greedy on
            "use_greedy_seed": True,
            "greedy_seed_strength": 6.0,
            "large_instance_threshold": 2000,
            "subset_bias_power": 1.0,   # bias to depot-near requests
        }
    elif instance_size == 5000:
        return {
            "n_ants": 12,
            "max_iters": 120,
            "alpha": 1.0,
            "beta": 4.5,
            "evap": 0.35,
            "Q": 300.0,
            "q0": 0.25,
            "subset_extra_frac": 0.04,
            "elite_deposit": True,
            "local_search_rate": 0.08,
            "early_stop_iters": 35,
            "max_time_seconds": 450,

            "use_greedy_seed": True,
            "greedy_seed_strength": 7.0,
            "large_instance_threshold": 2000,
            "subset_bias_power": 1.2,
        }
    else:  # 10000
        return {
            "n_ants": 8,
            "max_iters": 90,
            "alpha": 1.0,
            "beta": 5.0,
            "evap": 0.40,
            "Q": 350.0,
            "q0": 0.30,
            "subset_extra_frac": 0.03,
            "elite_deposit": True,
            "local_search_rate": 0.05,
            "early_stop_iters": 25,
            "max_time_seconds": 300,

            "use_greedy_seed": True,
            "greedy_seed_strength": 8.0,
            "large_instance_threshold": 2000,
            "subset_bias_power": 1.5,
        }


# ============================================================
# Greedy seed solution (fast feasibility helper)
# ============================================================

def greedy_seed_solution(inst: SCFPDPInstance, fairness_type: str = "jain") -> Solution:
    """
    Fast greedy construction that tries to serve at least gamma requests quickly.
    Strategy:
      - Sort requests by depot distance (pickup distance)
      - Round-robin assign to vehicles if capacity allows (using aggregated load approximation)
      - Append pickup then dropoff (keeps precedence)
    Then apply SCFDP repairs for capacity/gamma completeness.
    """
    sol = create_empty_solution(inst, fairness_type=fairness_type)

    # Sort by distance depot->pickup (ascending)
    reqs = list(range(1, inst.n + 1))
    reqs.sort(key=lambda r: inst.get_distance(0, inst.pickup_idx(r)))

    # Track simple per-route "current_load" approximation (since we append P then D, load peaks at +demand then returns)
    # For append-P-then-D pairs, peak load per request is demand, but overlaps if we insert multiple pickups before dropoffs.
    # Here we append P then D immediately to keep peak low; load returns to 0 each time in this simplified pattern.
    # So capacity check is trivial: demand <= C.
    served_count = 0
    k = 0

    for r in reqs:
        if served_count >= inst.gamma:
            break
        d = inst.demands[r - 1]
        if d > inst.C:
            continue

        # round-robin: find next vehicle that can take it
        tried = 0
        placed = False
        while tried < inst.n_K:
            route = sol.routes[k]
            # since we append P then D immediately, capacity respected if demand <= C
            route.stops.append(inst.pickup_idx(r))
            route.stops.append(inst.dropoff_idx(r))
            served_count += 1
            placed = True
            k = (k + 1) % inst.n_K
            break

        if not placed:
            k = (k + 1) % inst.n_K

    # Repairs (same pipeline)
    sol = repair_solution(sol)
    sol = repair_capacity_violations(sol)
    sol = repair_gamma(sol)

    sol.fairness_type = fairness_type
    return sol


# ============================================================
# Local Search (lightweight, precedence-safe)
# ============================================================

def local_search_light(sol: Solution, tries: int = 30) -> Solution:
    """
    Lightweight improvement:
    - pick a random route
    - precedence-safe reinsert of one (req,type) item
    - accept if objective improves (only if feasible)
    """
    inst = sol.inst
    best = sol.copy()
    best_obj = _objective_safe(best)

    for _ in range(tries):
        cand = best.copy()
        k = random.randrange(inst.n_K)
        route = cand.routes[k]
        if route.is_empty() or len(route.stops) < 4:
            continue

        pr = PrecedenceRoute.from_route(route, inst)
        if len(pr.sequence) < 4:
            continue

        seq = pr.sequence.copy()
        i = random.randrange(len(seq))
        item = seq.pop(i)  # (req, 'P'/'D')
        req, typ = item

        # valid insertion positions
        if typ == "P":
            drop_pos = None
            for idx, (r, t) in enumerate(seq):
                if r == req and t == "D":
                    drop_pos = idx
                    break
            valid_positions = list(range((drop_pos + 1) if drop_pos is not None else (len(seq) + 1)))
        else:
            pick_pos = None
            for idx, (r, t) in enumerate(seq):
                if r == req and t == "P":
                    pick_pos = idx
                    break
            valid_positions = list(range((pick_pos + 1) if pick_pos is not None else len(seq), len(seq) + 1))

        if not valid_positions:
            continue

        seq.insert(random.choice(valid_positions), item)
        pr2 = PrecedenceRoute(seq)
        cand.routes[k] = pr2.to_route(inst)

        cand = repair_solution(cand)
        cand = repair_capacity_violations(cand)
        cand = repair_gamma(cand)

        feas, _ = cand.is_feasible()
        if not feas:
            continue

        obj = _objective_safe(cand)
        if obj < best_obj:
            best = cand
            best_obj = obj

    return best


# ============================================================
# ACO Construction
# ============================================================

def _init_pheromone(inst: SCFPDPInstance, tau0: float = 1.0) -> List[List[float]]:
    nloc = 1 + 2 * inst.n
    return [[tau0 for _ in range(nloc)] for __ in range(nloc)]


def _select_subset_requests(
    inst: SCFPDPInstance,
    subset_extra_frac: float,
    bias_power: float = 0.0
) -> List[int]:
    """
    Select which requests the ant will try to serve:
      - choose target size m in [gamma, min(n, gamma + extra)]
      - pick m requests
    If bias_power > 0:
      - bias selection toward requests closer to depot (pickup distance)
      - still randomized via weighted sampling without replacement
    """
    extra = int(inst.n * subset_extra_frac)
    target = random.randint(inst.gamma, min(inst.n, inst.gamma + max(0, extra)))

    reqs = list(range(1, inst.n + 1))
    if bias_power <= 0.0:
        random.shuffle(reqs)
        return reqs[:target]

    # weights: higher for closer-to-depot
    # w_r = (1/(dist+1))^bias_power
    weights = []
    for r in reqs:
        dist = inst.get_distance(0, inst.pickup_idx(r))
        w = (1.0 / (dist + 1.0)) ** bias_power
        weights.append(w)

    # weighted sampling without replacement (simple O(n*target))
    selected = []
    available = reqs[:]
    w_avail = weights[:]
    for _ in range(target):
        if not available:
            break
        idx = _roulette_select(list(range(len(available))), w_avail)
        selected.append(available.pop(idx))
        w_avail.pop(idx)

    return selected


def _feasible_candidates(
    inst: SCFPDPInstance,
    current_loc: int,
    load: int,
    unpicked: set,
    open_picked: set
) -> List[int]:
    """Candidate next locations respecting precedence and capacity."""
    cand = []

    # dropoffs help reduce load (good for capacity)
    for req in open_picked:
        cand.append(inst.dropoff_idx(req))

    # pickups only if capacity allows
    for req in unpicked:
        d = inst.demands[req - 1]
        if load + d <= inst.C:
            cand.append(inst.pickup_idx(req))

    return cand


def _apply_move(
    inst: SCFPDPInstance,
    loc: int,
    load: int,
    unpicked: set,
    open_picked: set,
    served: set
) -> int:
    """Update load and sets after visiting loc. Returns new load."""
    if loc == 0:
        return load

    if loc <= inst.n:
        req = loc
        d = inst.demands[req - 1]
        load += d
        if req in unpicked:
            unpicked.remove(req)
        open_picked.add(req)
        served.add(req)
        return load

    req = loc - inst.n
    d = inst.demands[req - 1]
    load -= d
    if req in open_picked:
        open_picked.remove(req)
    return load


def construct_ant_solution(
    inst: SCFPDPInstance,
    tau: List[List[float]],
    alpha: float,
    beta: float,
    q0: float,
    subset_extra_frac: float,
    fairness_type: str = "jain",
    max_steps_factor: float = 2.5,
    subset_bias_power: float = 0.0
) -> Solution:
    """
    Construct a solution via multi-vehicle PDP walk (round-robin over vehicles).
    """
    sol = create_empty_solution(inst, fairness_type=fairness_type)

    subset = _select_subset_requests(
        inst,
        subset_extra_frac=subset_extra_frac,
        bias_power=subset_bias_power
    )
    unpicked_global = set(subset)

    current = [0] * inst.n_K
    load = [0] * inst.n_K
    open_picked = [set() for _ in range(inst.n_K)]
    served = set()

    max_steps = int(max_steps_factor * 2 * len(subset) + 10)

    for _ in range(max_steps):
        if not unpicked_global and all(len(s) == 0 for s in open_picked):
            break

        for k in range(inst.n_K):
            cur = current[k]
            cand = _feasible_candidates(inst, cur, load[k], unpicked_global, open_picked[k])

            if not cand:
                if open_picked[k]:
                    cand = [inst.dropoff_idx(req) for req in open_picked[k]]
                else:
                    continue

            desir = []
            for nxt in cand:
                dist = inst.get_distance(cur, nxt)
                eta = 1.0 / (dist + 1.0)
                desir.append((tau[cur][nxt] ** alpha) * (eta ** beta))

            if random.random() < q0:
                nxt = cand[max(range(len(cand)), key=lambda i: desir[i])]
            else:
                nxt = _roulette_select(cand, desir)

            sol.routes[k].stops.append(nxt)
            load[k] = _apply_move(inst, nxt, load[k], unpicked_global, open_picked[k], served)
            current[k] = nxt

    # close open pickups (deliver)
    for k in range(inst.n_K):
        leftovers = list(open_picked[k])
        for req in leftovers:
            sol.routes[k].stops.append(inst.dropoff_idx(req))
            load[k] = _apply_move(inst, inst.dropoff_idx(req), load[k], unpicked_global, open_picked[k], served)

    sol = repair_solution(sol)
    sol = repair_capacity_violations(sol)
    sol = repair_gamma(sol)
    sol.fairness_type = fairness_type
    return sol


# ============================================================
# ACO main loop with greedy seeding
# ============================================================

def run_aco_with_tracking(
    inst: SCFPDPInstance,
    fairness_type: str = "jain",
    **params
) -> Tuple[Solution, Dict]:
    """
    Runs ACO and returns (best_solution, tracking_info).
    tracking_info keys:
      - total_iters
      - best_per_iteration (list[dict])
      - found_feasible
    """

    n_ants = int(params.get("n_ants", 30))
    max_iters = int(params.get("max_iters", 200))
    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", 3.0))
    evap = float(params.get("evap", 0.25))
    Q = float(params.get("Q", 200.0))
    q0 = float(params.get("q0", 0.15))
    subset_extra_frac = float(params.get("subset_extra_frac", 0.10))
    elite_deposit = bool(params.get("elite_deposit", True))
    local_search_rate = float(params.get("local_search_rate", 0.15))
    early_stop_iters = int(params.get("early_stop_iters", 60))
    max_time_seconds = float(params.get("max_time_seconds", 900))

    # greedy-related
    use_greedy_seed = bool(params.get("use_greedy_seed", False))
    greedy_seed_strength = float(params.get("greedy_seed_strength", 5.0))
    large_instance_threshold = int(params.get("large_instance_threshold", 2000))
    subset_bias_power = float(params.get("subset_bias_power", 0.0))

    tau = _init_pheromone(inst, tau0=1.0)

    # --- Greedy seeding: only if enabled and instance is large enough
    seed_solution = None
    if use_greedy_seed and inst.n >= large_instance_threshold:
        seed_solution = greedy_seed_solution(inst, fairness_type=fairness_type)
        feas, _ = seed_solution.is_feasible()
        if feas:
            seed_obj = _objective_safe(seed_solution)
            if seed_obj < float("inf") and seed_obj > 0:
                # Deposit strong initial pheromone along greedy edges
                base_delta = (greedy_seed_strength * Q) / seed_obj
                for (a, b) in _edges_of_solution(inst, seed_solution):
                    tau[a][b] += base_delta
        # Bias subset selection in large instances (if configured)
        # (subset_bias_power already set by config)

    start = time.time()
    best_global = None
    best_obj = float("inf")
    found_feasible = False
    no_impr = 0
    best_per_it = []

    # If greedy found feasible, initialize best_global with it
    if seed_solution is not None:
        feas, _ = seed_solution.is_feasible()
        if feas:
            best_global = seed_solution.copy()
            best_obj = _objective_safe(best_global)
            found_feasible = True
            no_impr = 0

    for it in range(1, max_iters + 1):
        if (time.time() - start) > max_time_seconds:
            print(f"  ⚠ Time limit reached at iter {it}")
            break

        ant_solutions = []
        for _ in range(n_ants):
            s = construct_ant_solution(
                inst, tau,
                alpha=alpha, beta=beta, q0=q0,
                subset_extra_frac=subset_extra_frac,
                fairness_type=fairness_type,
                subset_bias_power=subset_bias_power if inst.n >= large_instance_threshold else 0.0
            )

            if random.random() < local_search_rate:
                s = local_search_light(s, tries=20)

            ant_solutions.append(s)

        iter_best = None
        iter_best_obj = float("inf")
        iter_feasible_count = 0

        for s in ant_solutions:
            feas, _ = s.is_feasible()
            if feas:
                iter_feasible_count += 1
                obj = _objective_safe(s)
                if obj < iter_best_obj:
                    iter_best_obj = obj
                    iter_best = s

        improved = False
        if iter_best is not None:
            found_feasible = True
            if iter_best_obj < best_obj:
                best_obj = iter_best_obj
                best_global = iter_best.copy()
                improved = True
                no_impr = 0
            else:
                no_impr += 1
        else:
            no_impr += 1

        # tracking
        if best_global is not None:
            stats = best_global.get_statistics()
            total_stops = sum(len(r.stops) for r in best_global.routes)
            best_per_it.append({
                "iteration": it,
                "objective": stats["objective"],
                "total_duration": stats["total_duration"],
                "fairness": stats["fairness"],
                "num_served": stats["num_served"],
                "num_active_routes": stats["num_active_routes"],
                "total_stops": total_stops,
                "feasible": True,
                "iter_feasible_count": iter_feasible_count,
            })
        else:
            best_per_it.append({
                "iteration": it,
                "objective": float("inf"),
                "total_duration": float("inf"),
                "fairness": 0.0,
                "num_served": 0,
                "num_active_routes": 0,
                "total_stops": 0,
                "feasible": False,
                "iter_feasible_count": iter_feasible_count,
            })

        # evaporate
        nloc = 1 + 2 * inst.n
        for i in range(nloc):
            row = tau[i]
            for j in range(nloc):
                row[j] *= (1.0 - evap)
                if row[j] < 1e-8:
                    row[j] = 1e-8

        # deposit
        deposit_sol = best_global if (elite_deposit and best_global is not None) else iter_best
        if deposit_sol is not None:
            obj = _objective_safe(deposit_sol)
            if obj > 0 and obj < float("inf"):
                delta = Q / obj
                for (a, b) in _edges_of_solution(inst, deposit_sol):
                    tau[a][b] += delta

        # progress
        if it % 25 == 0:
            elapsed = time.time() - start
            if best_global is not None:
                print(f"  Iter {it}: Best={best_obj:.2f}, FeasAnts={iter_feasible_count}/{n_ants}, NoImpr={no_impr}, Time={elapsed:.1f}s")
            else:
                print(f"  Iter {it}: Searching feasibility... FeasAnts={iter_feasible_count}/{n_ants}")

        # early stop
        if found_feasible and no_impr >= early_stop_iters:
            print(f"  Stopping: No improvement for {early_stop_iters} iterations")
            break

    if best_global is None:
        # fallback: return best among last ants even if infeasible
        best_global = ant_solutions[0].copy()

    tracking = {
        "total_iters": len(best_per_it),
        "best_per_iteration": best_per_it,
        "found_feasible": found_feasible,
    }
    return best_global, tracking
