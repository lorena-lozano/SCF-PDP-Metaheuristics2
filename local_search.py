# local_search.py
"""
Local Search framework for the SCF-PDP (A2-aligned)
Supports:
  - Different neighborhood structures (N1..N3)
  - Step functions: first-improvement / best-improvement
  - Delta evaluation provided by neighborhoods.py

"""

from typing import Callable, Iterable, Tuple, Dict, Optional
import time

from SCFDP import Solution, SCFPDPInstance
from a1_constructors_fixed import greedy_construction_fixed

# Neighborhoods
from neighborhoods import (
    generate_intra_route_reloc,     # N1
    generate_inter_route_reloc,     # N2
    generate_swap_requests,         # N3
)

NeighborhoodFunc = Callable[[Solution], Iterable[Tuple[Solution, float, Dict]]]


def local_search(
    initial_solution: Solution,
    neighborhood: NeighborhoodFunc,
    step_function: str = "best",
    max_iterations: int = 200,
    time_limit: Optional[float] = None,
    max_neighbors_per_iter: Optional[int] = None
) -> Solution:
    """
    Local search for a neighborhood structure.
      - max_iterations: maximum number of iterations
      - time_limit: time limit in seconds (None = no limit)
      - max_neighbors_per_iter: limit number of neighbors explored per iteration
    """
    current = initial_solution
    iteration = 0
    start_time = time.time()

    while iteration < max_iterations:
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break

        iteration += 1
        improved = False

        if step_function == "first":
            neighbor_count = 0
            for neighbor, delta, move in neighborhood(current):
                neighbor_count += 1
                if max_neighbors_per_iter is not None and neighbor_count > max_neighbors_per_iter:
                    break
                if delta < 0:
                    current = neighbor
                    improved = True
                    break

        elif step_function == "best":
            best_delta = 0.0
            best_neighbor = None
            neighbor_count = 0

            for neighbor, delta, move in neighborhood(current):
                neighbor_count += 1
                if max_neighbors_per_iter is not None and neighbor_count > max_neighbors_per_iter:
                    break
                if delta < best_delta:
                    best_delta = delta
                    best_neighbor = neighbor

            if best_neighbor is not None:
                current = best_neighbor
                improved = True

        else:
            raise ValueError("Unknown step function. Use 'first' or 'best'.")

        if not improved:
            break

        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break

    return current


# ----------------------------
# Convenience wrappers (N1..N3)
# ----------------------------
LS_MAX_ITERS = 400
LS_TIME_LIMIT = 60.0
LS_MAX_NEIGHBORS = 5000


def ls_intra_route(initial_solution: Solution, step_function="best", max_iterations: int = LS_MAX_ITERS) -> Solution:
    return local_search(
        initial_solution,
        generate_intra_route_reloc,
        step_function=step_function,
        max_iterations=max_iterations,
        time_limit=LS_TIME_LIMIT,
        max_neighbors_per_iter=LS_MAX_NEIGHBORS,
    )


def ls_inter_route(initial_solution: Solution, step_function="best", max_iterations: int = LS_MAX_ITERS) -> Solution:
    return local_search(
        initial_solution,
        generate_inter_route_reloc,
        step_function=step_function,
        max_iterations=max_iterations,
        time_limit=LS_TIME_LIMIT,
        max_neighbors_per_iter=LS_MAX_NEIGHBORS,
    )


def ls_swap(initial_solution: Solution, step_function="best", max_iterations: int = LS_MAX_ITERS) -> Solution:
    return local_search(
        initial_solution,
        generate_swap_requests,
        step_function=step_function,
        max_iterations=max_iterations,
        time_limit=LS_TIME_LIMIT,
        max_neighbors_per_iter=LS_MAX_NEIGHBORS,
    )


# Run-from-instance wrappers (use FIXED constructor, not A1 greedy)
def ls_N1_best(inst: SCFPDPInstance, fairness_type: str = "jain") -> Solution:
    initial = greedy_construction_fixed(inst, fairness_type=fairness_type)
    return ls_intra_route(initial, step_function="best")


def ls_N1_first(inst: SCFPDPInstance, fairness_type: str = "jain") -> Solution:
    initial = greedy_construction_fixed(inst, fairness_type=fairness_type)
    return ls_intra_route(initial, step_function="first")


def ls_N2_best(inst: SCFPDPInstance, fairness_type: str = "jain") -> Solution:
    initial = greedy_construction_fixed(inst, fairness_type=fairness_type)
    return ls_inter_route(initial, step_function="best")


def ls_N2_first(inst: SCFPDPInstance, fairness_type: str = "jain") -> Solution:
    initial = greedy_construction_fixed(inst, fairness_type=fairness_type)
    return ls_inter_route(initial, step_function="first")


def ls_N3_best(inst: SCFPDPInstance, fairness_type: str = "jain") -> Solution:
    initial = greedy_construction_fixed(inst, fairness_type=fairness_type)
    return ls_swap(initial, step_function="best")


def ls_N3_first(inst: SCFPDPInstance, fairness_type: str = "jain") -> Solution:
    initial = greedy_construction_fixed(inst, fairness_type=fairness_type)
    return ls_swap(initial, step_function="first")
