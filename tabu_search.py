# tabu_search.py
from typing import Callable, Iterable, Tuple, Dict
import time

from SCFDP import Solution, SCFPDPInstance

from a1_constructors_fixed import greedy_construction_fixed, randomized_greedy_fixed
from neighborhoods import (
    generate_intra_route_reloc,     # N1
    generate_inter_route_reloc,     # N2
    generate_swap_requests,         # N3
)

NeighborhoodFunc = Callable[[Solution], Iterable[Tuple[Solution, float, Dict]]]


def tabu_search_core(
    inst: SCFPDPInstance,
    neighborhoods: list,
    construction_func: Callable,
    max_iterations: int = None,
    tabu_tenure: int = 10,
    step_function: str = "best",
    max_neighbors_per_iter: int = None,
    time_limit: float = 840.0
) -> Solution:

    construction_start = time.time()
    current = construction_func(inst)
    best_solution = current.copy()
    construction_time = time.time() - construction_start

    remaining_time = time_limit - construction_time - 10.0
    tabu_list: list[Dict] = []
    start_time = time.time()

    n = inst.n

    if n >= 8000:
        estimated_time_per_iter = 120.0
        max_neighbors_per_iter = 50
        tabu_tenure = 3
    elif n >= 5000:
        estimated_time_per_iter = 60.0
        max_neighbors_per_iter = 150
        tabu_tenure = 5
    elif n >= 2000:
        estimated_time_per_iter = 20.0
        max_neighbors_per_iter = 300
        tabu_tenure = 7
    elif n >= 1000:
        estimated_time_per_iter = 8.0
        max_neighbors_per_iter = 500
        tabu_tenure = 8
    elif n >= 500:
        estimated_time_per_iter = 3.0
        max_neighbors_per_iter = 1000
        tabu_tenure = 10
    elif n >= 200:
        estimated_time_per_iter = 1.5
        max_neighbors_per_iter = 2000
        tabu_tenure = 10
    else:
        estimated_time_per_iter = 0.5
        max_neighbors_per_iter = 5000
        tabu_tenure = 15

    max_iterations = max(5, int(remaining_time / estimated_time_per_iter))
    max_iterations = min(max_iterations, 300)

    iteration = 0
    iterations_without_improvement = 0

    while iteration < max_iterations:
        iteration += 1

        elapsed = time.time() - start_time
        total_elapsed = construction_time + elapsed
        if total_elapsed >= time_limit:
            break

        time_pressure = total_elapsed / time_limit

        if iterations_without_improvement > 10:
            convergence_factor = 0.5
        elif iterations_without_improvement > 5:
            convergence_factor = 0.7
        else:
            convergence_factor = 1.0

        if time_pressure > 0.85:
            current_max_neighbors = max(50, int(max_neighbors_per_iter * 0.2 * convergence_factor))
        elif time_pressure > 0.7:
            current_max_neighbors = max(100, int(max_neighbors_per_iter * 0.4 * convergence_factor))
        elif time_pressure > 0.5:
            current_max_neighbors = max(200, int(max_neighbors_per_iter * 0.6 * convergence_factor))
        else:
            current_max_neighbors = int(max_neighbors_per_iter * convergence_factor)

        if iteration > 1:
            avg_time_per_iter = elapsed / iteration
            estimated_next_iter_time = total_elapsed + avg_time_per_iter
            if estimated_next_iter_time > time_limit - 30:
                current_max_neighbors = min(current_max_neighbors, 100)
            if estimated_next_iter_time > time_limit - 10:
                break

        best_delta = float("inf")
        best_neighbor = None
        best_move_info = None
        neighbors_checked = 0

        if step_function == "first":
            found_improving = False
            for neighborhood in neighborhoods:
                if found_improving:
                    break
                for neighbor, delta, move_info in neighborhood(current):
                    neighbors_checked += 1

                    if _is_tabu(move_info, tabu_list) and neighbor.objective_value() >= best_solution.objective_value():
                        continue

                    if delta < 0:
                        best_neighbor = neighbor
                        best_move_info = move_info
                        found_improving = True
                        break

                    if neighbors_checked >= current_max_neighbors:
                        break
        else:
            for neighborhood in neighborhoods:
                if neighbors_checked >= current_max_neighbors:
                    break

                for neighbor, delta, move_info in neighborhood(current):
                    neighbors_checked += 1

                    if _is_tabu(move_info, tabu_list) and neighbor.objective_value() >= best_solution.objective_value():
                        continue

                    if delta < best_delta:
                        best_delta = delta
                        best_neighbor = neighbor
                        best_move_info = move_info

                    if best_delta < -50:
                        break

                    if neighbors_checked >= current_max_neighbors:
                        break

        if best_neighbor is None:
            break

        current = best_neighbor

        tabu_list.append(best_move_info)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        if current.objective_value() < best_solution.objective_value():
            best_solution = current.copy()
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        if iterations_without_improvement > 30 and iteration > max_iterations * 0.5:
            break

    return best_solution


def _is_tabu(move_info: Dict, tabu_list: list) -> bool:
    if not tabu_list:
        return False

    move_type = move_info.get("type")
    for tabu_move in tabu_list:
        if tabu_move.get("type") != move_type:
            continue

        if move_type == "intra_reloc":
            if (tabu_move.get("route") == move_info.get("route") and
                tabu_move.get("req") == move_info.get("req")):
                return True

        elif move_type == "inter_reloc":
            if (tabu_move.get("req") == move_info.get("req") and
                tabu_move.get("from") == move_info.get("from") and
                tabu_move.get("to") == move_info.get("to")):
                return True

        elif move_type == "swap":
            if (tabu_move.get("req1") == move_info.get("req1") and
                tabu_move.get("req2") == move_info.get("req2")):
                return True

    return False


# ============== Greedy Construction Variants ==============

def greedy_tabu_best(inst: SCFPDPInstance) -> Solution:
    neighborhoods = [generate_intra_route_reloc, generate_inter_route_reloc, generate_swap_requests]
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=lambda i: greedy_construction_fixed(i, fairness_type="jain"),
        step_function="best",
        max_iterations=200,
    )

def greedy_tabu_first(inst: SCFPDPInstance) -> Solution:
    neighborhoods = [generate_intra_route_reloc, generate_inter_route_reloc, generate_swap_requests]
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=lambda i: greedy_construction_fixed(i, fairness_type="jain"),
        step_function="first",
        max_iterations=200,
    )


# ============== Randomized Construction Variants ==============

def random_tabu_best(inst: SCFPDPInstance) -> Solution:
    neighborhoods = [generate_intra_route_reloc, generate_inter_route_reloc, generate_swap_requests]
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=lambda i: randomized_greedy_fixed(i, fairness_type="jain", alpha=0.3, seed=1234),
        step_function="best",
        max_iterations=200,
    )

def random_tabu_first(inst: SCFPDPInstance) -> Solution:
    neighborhoods = [generate_intra_route_reloc, generate_inter_route_reloc, generate_swap_requests]
    return tabu_search_core(
        inst=inst,
        neighborhoods=neighborhoods,
        construction_func=lambda i: randomized_greedy_fixed(i, fairness_type="jain", alpha=0.3, seed=1234),
        step_function="first",
        max_iterations=200,
    )
