# a1_constructors_fixed.py
from typing import Optional, Tuple
import random

from SCFDP import SCFPDPInstance, Solution, Route


def _route_feasible_capacity(route: Route, inst: SCFPDPInstance) -> bool:
    loads = route.calculate_load_at_each_stop(inst)
    if not loads:
        return True
    return (min(loads) >= 0) and (max(loads) <= inst.C)


def _route_duration(stops, inst: SCFPDPInstance) -> int:
    if not stops:
        return 0
    total = inst.dist[0][stops[0]]
    for i in range(len(stops) - 1):
        total += inst.dist[stops[i]][stops[i + 1]]
    total += inst.dist[stops[-1]][0]
    return total


def best_insertion_positions(route: Route, inst: SCFPDPInstance, req: int) -> Optional[Tuple[int, int, int]]:
    """
    Best insertion for request req into `route`:
      - insert pickup at position i
      - insert dropoff at position j > i
    Returns (i, j, new_duration) or None if no capacity-feasible insertion exists.
    """
    p = inst.pickup_idx(req)
    d = inst.dropoff_idx(req)

    base = route.stops
    L = len(base)

    best = None
    best_dur = None

    for i in range(L + 1):
        for j in range(i + 1, L + 2):
            new_stops = base.copy()
            new_stops.insert(i, p)
            new_stops.insert(j, d)

            test_route = Route(stops=new_stops)
            if not _route_feasible_capacity(test_route, inst):
                continue

            dur = _route_duration(new_stops, inst)
            if best_dur is None or dur < best_dur:
                best_dur = dur
                best = (i, j, dur)

    return best


def greedy_construction_fixed(inst: SCFPDPInstance, fairness_type: str = "jain") -> Solution:
    """
    Deterministic greedy:
    insert requests one-by-one using best feasible insertion positions (pickup/dropoff separated).
    """
    sol = Solution(inst, fairness_type=fairness_type)

    requests = list(range(1, inst.n + 1))
    requests.sort(key=lambda r: inst.demands[r - 1], reverse=True)

    served = 0
    for req in requests:
        best_move = None  # (route_idx, i, j, dur)

        for k, route in enumerate(sol.routes):
            ins = best_insertion_positions(route, inst, req)
            if ins is None:
                continue
            i, j, dur = ins
            if best_move is None or dur < best_move[3]:
                best_move = (k, i, j, dur)

        if best_move is None:
            continue

        k, i, j, _ = best_move
        p = inst.pickup_idx(req)
        d = inst.dropoff_idx(req)
        sol.routes[k].stops.insert(i, p)
        sol.routes[k].stops.insert(j, d)

        served += 1
        if served >= inst.gamma:
            break

    return sol


def randomized_greedy_fixed(
    inst: SCFPDPInstance,
    fairness_type: str = "jain",
    alpha: float = 0.3,
    seed: Optional[int] = None
) -> Solution:
    """
    GRASP-style randomized greedy:
    for each request, build a list of candidate insertions and choose from an RCL.
    """
    if seed is not None:
        random.seed(seed)

    sol = Solution(inst, fairness_type=fairness_type)
    requests = list(range(1, inst.n + 1))
    random.shuffle(requests)

    served = 0
    for req in requests:
        moves = []  # (dur, route_idx, i, j)
        for k, route in enumerate(sol.routes):
            ins = best_insertion_positions(route, inst, req)
            if ins is None:
                continue
            i, j, dur = ins
            moves.append((dur, k, i, j))

        if not moves:
            continue

        moves.sort(key=lambda x: x[0])
        rcl_len = max(1, int(round(alpha * len(moves))))
        dur, k, i, j = random.choice(moves[:rcl_len])

        p = inst.pickup_idx(req)
        d = inst.dropoff_idx(req)
        sol.routes[k].stops.insert(i, p)
        sol.routes[k].stops.insert(j, d)

        served += 1
        if served >= inst.gamma:
            break

    return sol
