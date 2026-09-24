# VND.py
"""
Variable Neighborhood Descent (VND) for the SCF-PDP
Time-Limited implementation with adaptive parameters based on instance size.
(A2-aligned: uses SCFDP framework + fixed PDP construction)
"""

import time
from typing import Callable, Iterable, Tuple, Dict, List

from SCFDP import Solution, SCFPDPInstance
from a1_constructors_fixed import greedy_construction_fixed

from local_search import local_search
from neighborhoods import (
    generate_intra_route_reloc,     # N1
    generate_inter_route_reloc,     # N2
    generate_swap_requests        # N3
)

DEFAULT_TIME_LIMIT = 880.0
DEFAULT_MAX_LS_ITERS = 1000

NeighborhoodFunc = Callable[[Solution], Iterable[Tuple[Solution, float, Dict]]]


def vnd_time_limited(
    inst: SCFPDPInstance,
    neighborhoods: List[NeighborhoodFunc],
    step_function: str = "best",
    time_limit: float = DEFAULT_TIME_LIMIT,
    fairness_type: str = "jain",
) -> Solution:
    start_time = time.time()
    n = inst.n

    if n >= 5000:
        max_neighbors = 50
    elif n >= 2000:
        max_neighbors = 150
    elif n >= 1000:
        max_neighbors = 400
    elif n >= 500:
        max_neighbors = 800
    else:
        max_neighbors = None

    # Initial solution (deterministic) - FIXED constructor
    current = greedy_construction_fixed(inst, fairness_type=fairness_type)
    best_obj = current.objective_value()

    k = 0
    k_max = len(neighborhoods)

    while k < k_max:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break

        neighborhood_func = neighborhoods[k]

        improved_sol = local_search(
            initial_solution=current,
            neighborhood=neighborhood_func,
            step_function=step_function,
            max_iterations=DEFAULT_MAX_LS_ITERS,
            max_neighbors_per_iter=max_neighbors,
            time_limit=(time_limit - elapsed)
        )

        new_obj = improved_sol.objective_value()

        if new_obj < best_obj - 1e-4:
            current = improved_sol
            best_obj = new_obj
            k = 0
        else:
            k += 1

    return current


def vnd_standard_best(inst: SCFPDPInstance) -> Solution:
    neighborhoods = [generate_intra_route_reloc, generate_inter_route_reloc, generate_swap_requests]
    return vnd_time_limited(inst, neighborhoods, step_function="best")

def vnd_standard_first(inst: SCFPDPInstance) -> Solution:
    neighborhoods = [generate_intra_route_reloc, generate_inter_route_reloc, generate_swap_requests]
    return vnd_time_limited(inst, neighborhoods, step_function="first")

def vnd_reverse_first(inst: SCFPDPInstance) -> Solution:
    neighborhoods = [generate_swap_requests, generate_inter_route_reloc, generate_intra_route_reloc]
    return vnd_time_limited(inst, neighborhoods, step_function="first")

def vnd_light_first(inst: SCFPDPInstance) -> Solution:
    neighborhoods = [generate_intra_route_reloc, generate_inter_route_reloc]
    return vnd_time_limited(inst, neighborhoods, step_function="first")
