# neighborhoods.py
"""
Neighborhood structures (A2-aligned, SCFDP framework)

Each generator takes a Solution and yields tuples:
    (neighbor_solution, delta_obj, move_info)

Key change vs A1:
- We DO NOT force dropoff immediately after pickup.
- When inserting a request, we choose pickup position i and dropoff position j>i.

Included neighborhoods:
- N1: intra-route relocate (remove a request and reinsert elsewhere in same route)
- N2: inter-route relocate (move a request from one route to another)
- N3: swap requests between routes (swap req1 and req2, reinserting with flexible positions)

"""

from typing import List, Set, Optional, Tuple, Dict, Iterable
from SCFDP import Solution, Route, SCFPDPInstance


# -----------------------------
# Helpers
# -----------------------------

def _remove_request_from_route_stops(stops: List[int], inst: SCFPDPInstance, req: int) -> List[int]:
    p = inst.pickup_idx(req)
    d = inst.dropoff_idx(req)
    return [s for s in stops if s not in (p, d)]


def _insert_request_general(stops: List[int], inst: SCFPDPInstance, req: int, pickup_pos: int, dropoff_pos: int) -> List[int]:
    """
    Insert pickup at pickup_pos and dropoff at dropoff_pos (dropoff_pos must be > pickup_pos)
    Returns a NEW list.
    """
    assert dropoff_pos > pickup_pos
    p = inst.pickup_idx(req)
    d = inst.dropoff_idx(req)

    new_stops = stops.copy()
    new_stops.insert(pickup_pos, p)
    new_stops.insert(dropoff_pos, d)
    return new_stops


def _route_feasible_capacity(stops: List[int], inst: SCFPDPInstance) -> bool:
    """
    Fast capacity check (assumes precedence is already respected by construction).
    """
    load = 0
    for stop in stops:
        if stop <= inst.n:  # pickup node id in [1..n]
            load += inst.demands[stop - 1]
        else:               # dropoff node id in [n+1..2n]
            req = stop - inst.n
            load -= inst.demands[req - 1]

        if load < 0 or load > inst.C:
            return False
    return True


def _route_duration_from_stops(stops: List[int], inst: SCFPDPInstance) -> float:
    if not stops:
        return 0.0
    total = inst.dist[0][stops[0]]
    for i in range(len(stops) - 1):
        total += inst.dist[stops[i]][stops[i + 1]]
    total += inst.dist[stops[-1]][0]
    return total


def _best_insertion_in_route(
    base_stops: List[int],
    inst: SCFPDPInstance,
    req: int,
    max_pos_samples: Optional[int] = None
) -> Optional[Tuple[List[int], float, int, int]]:
    """
    Find best (lowest route duration) insertion of req into base_stops
    among all pickup_pos i and dropoff_pos j>i.

    max_pos_samples:
      - None => full enumeration (can be expensive)
      - integer => limits the number of (i,j) pairs checked by sampling positions
    Returns: (new_stops, new_duration, pickup_pos, dropoff_pos) or None
    """
    L = len(base_stops)
    best = None
    best_dur = None

    # full enumeration can be O(L^2). For big routes you may want to cap.
    if max_pos_samples is None or L <= 60:
        i_positions = range(L + 1)
        for i in i_positions:
            for j in range(i + 1, L + 2):
                new_stops = _insert_request_general(base_stops, inst, req, i, j)
                if not _route_feasible_capacity(new_stops, inst):
                    continue
                dur = _route_duration_from_stops(new_stops, inst)
                if best_dur is None or dur < best_dur:
                    best_dur = dur
                    best = (new_stops, dur, i, j)
        return best

    # sampled positions mode (fast approximation)
    import random
    candidates = []
    for _ in range(max_pos_samples):
        i = random.randint(0, L)
        j = random.randint(i + 1, L + 1)
        candidates.append((i, j))

    for i, j in candidates:
        new_stops = _insert_request_general(base_stops, inst, req, i, j)
        if not _route_feasible_capacity(new_stops, inst):
            continue
        dur = _route_duration_from_stops(new_stops, inst)
        if best_dur is None or dur < best_dur:
            best_dur = dur
            best = (new_stops, dur, i, j)

    return best


# -----------------------------
# N1: Intra-route relocate
# -----------------------------

def generate_intra_route_reloc(solution: Solution, max_pos_samples: Optional[int] = None):
    """
    Neighborhood N1: relocate a complete request (pickup+dropoff) within the same route.
    Reinsertion uses flexible positions i<j (NOT consecutive).

    max_pos_samples:
      - None => full (can be expensive)
      - e.g. 200 => sampled insertions (faster)
    """
    inst = solution.inst
    current_durations = [r.duration(inst) for r in solution.routes]
    current_obj = solution.objective_value_from_durations(current_durations)

    for k, route in enumerate(solution.routes):
        route_requests = route.get_requests(inst)

        for req in route_requests:
            base_stops = _remove_request_from_route_stops(route.stops, inst, req)

            ins = _best_insertion_in_route(base_stops, inst, req, max_pos_samples=max_pos_samples)
            if ins is None:
                continue

            new_stops, new_dur, i, j = ins
            if new_stops == route.stops:
                continue

            new_durations = current_durations.copy()
            new_durations[k] = new_dur

            new_obj = solution.objective_value_from_durations(new_durations)
            delta = new_obj - current_obj

            neighbor = solution.copy()
            neighbor.routes[k].stops = new_stops

            move_info = {"type": "intra_reloc", "route": k, "req": req, "pickup_pos": i, "dropoff_pos": j}
            yield neighbor, delta, move_info


# -----------------------------
# N2: Inter-route relocate
# -----------------------------

def generate_inter_route_reloc(solution: Solution, max_pos_samples: Optional[int] = None):
    """
    Neighborhood N2: move a request from one route to another with flexible insertion i<j.
    """
    inst = solution.inst
    current_durations = [r.duration(inst) for r in solution.routes]
    current_obj = solution.objective_value_from_durations(current_durations)
    n_routes = len(solution.routes)

    for k_from in range(n_routes):
        route_from = solution.routes[k_from]
        from_reqs = route_from.get_requests(inst)

        for req in from_reqs:
            from_without = _remove_request_from_route_stops(route_from.stops, inst, req)
            from_dur = _route_duration_from_stops(from_without, inst)

            for k_to in range(n_routes):
                if k_to == k_from:
                    continue

                route_to = solution.routes[k_to]
                ins = _best_insertion_in_route(route_to.stops, inst, req, max_pos_samples=max_pos_samples)
                if ins is None:
                    continue

                new_to_stops, to_dur, i, j = ins

                new_durations = current_durations.copy()
                new_durations[k_from] = from_dur
                new_durations[k_to] = to_dur

                new_obj = solution.objective_value_from_durations(new_durations)
                delta = new_obj - current_obj

                neighbor = solution.copy()
                neighbor.routes[k_from].stops = from_without
                neighbor.routes[k_to].stops = new_to_stops

                move_info = {"type": "inter_reloc", "from": k_from, "to": k_to, "req": req, "pickup_pos": i, "dropoff_pos": j}
                yield neighbor, delta, move_info


# -----------------------------
# N3: Swap between routes
# -----------------------------

def generate_swap_requests(solution: Solution, max_pos_samples: Optional[int] = None):
    """
    Neighborhood N3: swap req1 and req2 between two routes.
    We remove both requests and reinsert them flexibly (best insertion) into opposite routes.
    """
    inst = solution.inst
    current_durations = [r.duration(inst) for r in solution.routes]
    current_obj = solution.objective_value_from_durations(current_durations)
    n_routes = len(solution.routes)

    for k1 in range(n_routes):
        r1 = solution.routes[k1]
        reqs1 = list(r1.get_requests(inst))

        for k2 in range(k1 + 1, n_routes):
            r2 = solution.routes[k2]
            reqs2 = list(r2.get_requests(inst))

            for req1 in reqs1:
                r1_wo = _remove_request_from_route_stops(r1.stops, inst, req1)

                for req2 in reqs2:
                    r2_wo = _remove_request_from_route_stops(r2.stops, inst, req2)

                    ins1 = _best_insertion_in_route(r1_wo, inst, req2, max_pos_samples=max_pos_samples)
                    if ins1 is None:
                        continue
                    r1_new, r1_dur, i1, j1 = ins1

                    ins2 = _best_insertion_in_route(r2_wo, inst, req1, max_pos_samples=max_pos_samples)
                    if ins2 is None:
                        continue
                    r2_new, r2_dur, i2, j2 = ins2

                    # (capacity already checked inside _best_insertion_in_route)
                    new_durations = current_durations.copy()
                    new_durations[k1] = r1_dur
                    new_durations[k2] = r2_dur

                    new_obj = solution.objective_value_from_durations(new_durations)
                    delta = new_obj - current_obj

                    neighbor = solution.copy()
                    neighbor.routes[k1].stops = r1_new
                    neighbor.routes[k2].stops = r2_new

                    move_info = {
                        "type": "swap",
                        "route1": k1, "route2": k2,
                        "req1": req1, "req2": req2,
                        "pickup_pos1": i1, "dropoff_pos1": j1,
                        "pickup_pos2": i2, "dropoff_pos2": j2,
                    }
                    yield neighbor, delta, move_info
