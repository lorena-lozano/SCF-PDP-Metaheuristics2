"""
ACO tuning helper (called by LHS tuning)
"""

import random
import numpy as np

from SCFDP import SCFPDPInstance
from aco_alg import run_aco_with_tracking

def run_aco_for_tuning(instance_file: str, seed: int, fairness_type: str = "jain", **config) -> float:
    random.seed(seed)
    np.random.seed(seed)

    inst = SCFPDPInstance(instance_file)
    sol, _ = run_aco_with_tracking(inst, fairness_type=fairness_type, **config)

    feasible, _ = sol.is_feasible()
    if not feasible:
        return float("inf")
    return sol.objective_value()
