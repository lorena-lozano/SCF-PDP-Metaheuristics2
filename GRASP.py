# GRASP.py
"""
GRASP metaheuristic for the SCF-PDP
Time-Limited implementation to handle large instances within 15 min.
(A2-aligned: uses SCFDP framework + fixed PDP construction)
"""

import time
from math import inf
from typing import Callable, Optional

from SCFDP import SCFPDPInstance, Solution
from a1_constructors_fixed import randomized_greedy_fixed

from local_search import (
    ls_intra_route,   # N1
    ls_inter_route,   # N2
    ls_swap          # N3
)

DEFAULT_TIME_LIMIT = 870.0  # 14.5 min
DEFAULT_LS_ITERS = 300

ImproveFunc = Callable[[Solution], Solution]


def grasp_time_limited(
    inst: SCFPDPInstance,
    improve_func: ImproveFunc,
    max_iterations: int = 50,
    alpha: float = 0.3,
    base_seed: Optional[int] = None,
    time_limit: float = DEFAULT_TIME_LIMIT,
    fairness_type: str = "jain",
) -> Solution:
    start_time = time.time()
    best_solution: Optional[Solution] = None
    best_obj: float = inf

    n = inst.n
    if n >= 5000:
        target_iters = 5
    elif n >= 2000:
        target_iters = 15
    elif n >= 500:
        target_iters = 30
    else:
        target_iters = max_iterations

    for i in range(target_iters):
        elapsed = time.time() - start_time
        if elapsed >= (time_limit - 30):
            break

        if i > 0:
            avg_time = elapsed / i
            if elapsed + avg_time > time_limit:
                break

        iter_seed = (base_seed + i) if base_seed is not None else None
        current_alpha = 0.0 if i == 0 else alpha

        sol = randomized_greedy_fixed(
            inst,
            fairness_type=fairness_type,
            alpha=current_alpha,
            seed=iter_seed
        )

        sol = improve_func(sol)

        obj = sol.objective_value()
        if obj < best_obj:
            best_obj = obj
            best_solution = sol

    if best_solution is None:
        best_solution = randomized_greedy_fixed(inst, fairness_type=fairness_type, alpha=0.0)

    return best_solution


# Wrappers
def grasp_N1_best(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_intra_route(s, step_function="best", max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )

def grasp_N1_first(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_intra_route(s, step_function="first", max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )

def grasp_N2_best(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_inter_route(s, step_function="best", max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )

def grasp_N2_first(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_inter_route(s, step_function="first", max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )

def grasp_N3_best(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_swap(s, step_function="best", max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )

def grasp_N3_first(inst: SCFPDPInstance) -> Solution:
    return grasp_time_limited(
        inst,
        improve_func=lambda s: ls_swap(s, step_function="first", max_iterations=DEFAULT_LS_ITERS),
        base_seed=1234
    )
