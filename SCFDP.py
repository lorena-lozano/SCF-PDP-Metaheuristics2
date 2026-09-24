"""
SCF-PDP Framework: Reusable components for any algorithm
Import this module to access instance parsing, solution representation,
validation, evaluation, and I/O utilities.
"""

import math
import os
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass, field
from copy import deepcopy
import time
import random



class SCFPDPInstance:
    """Problem instance with all data and distance calculations"""

    def __init__(self, filename: str):
        """Parse instance file"""
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Parse header
        header = lines[0].split()
        self.n = int(header[0])  # number of requests
        self.n_K = int(header[1])  # number of vehicles
        self.C = int(header[2])  # vehicle capacity
        self.gamma = int(header[3])  # minimum requests to serve
        self.rho = float(header[4])  # fairness weight

        # Parse demands
        demands_idx = lines.index('# demands') + 1
        self.demands = list(map(int, lines[demands_idx].split()))

        # Parse locations
        loc_idx = lines.index('# request locations') + 1
        locations = []
        for i in range(loc_idx, len(lines)):
            coords = list(map(float, lines[i].split()))
            locations.extend(
                [(coords[j], coords[j + 1]) for j in range(0, len(coords), 2)])

        self.depot = locations[0]
        self.pickups = locations[1:self.n + 1]
        self.dropoffs = locations[self.n + 1:2 * self.n + 1]

        # Precompute distance matrix
        import torch
    
        all_locs = [self.depot] + self.pickups + self.dropoffs
        self.n_locs = len(all_locs)
        
        print(f"[INSTANCE] Computing distance matrix with PyTorch...")
        start = time.time()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        locs_tensor = torch.tensor(all_locs, dtype=torch.float32, device=device)
        
        # Compute pairwise distances
        diff = locs_tensor.unsqueeze(1) - locs_tensor.unsqueeze(0)
        distances = torch.sqrt((diff ** 2).sum(dim=2))
        self.dist = torch.ceil(distances).cpu().numpy().astype(int).tolist()
        
        
    def pickup_idx(self, req: int) -> int:
        """Get location index for pickup of request req (1-indexed)"""
        return req

    def dropoff_idx(self, req: int) -> int:
        """Get location index for dropoff of request req (1-indexed)"""
        return self.n + req

    def get_distance(self, from_idx: int, to_idx: int) -> float:
        """Get distance between two location indices (0=depot, 1..n=pickups, n+1..2n=dropoffs)"""
        return self.dist[from_idx][to_idx]

    def request_from_location(self, loc_idx: int) -> Optional[int]:
        """Get request number from location index (returns None for depot)"""
        if loc_idx == 0:
            return None
        elif loc_idx <= self.n:
            return loc_idx
        else:
            return loc_idx - self.n
        
    def get_all_requests(self) -> Set[int]:
        """Get set of all request numbers"""
        return set(range(1, self.n + 1))


@dataclass
class Route:
    """Represents a single vehicle route"""
    stops: List[int] = field(
        default_factory=list)  # sequence of location indices

    def copy(self) -> 'Route':
        """Create a deep copy of the route"""
        return Route(stops=self.stops.copy())

    def is_empty(self) -> bool:
        """Check if route has no stops"""
        return len(self.stops) == 0

    def get_requests(self, inst: SCFPDPInstance) -> Set[int]:
        """Get set of all requests served in this route"""
        requests = set()
        for stop in self.stops:
            req = inst.request_from_location(stop)
            if req:
                requests.add(req)
        return requests

    def duration(self, inst: SCFPDPInstance) -> float:
        """Calculate total route duration including depot trips"""
        if not self.stops:
            return 0.0

        total = inst.dist[0][self.stops[0]]  # depot to first stop
        for i in range(len(self.stops) - 1):
            total += inst.dist[self.stops[i]][self.stops[i + 1]]
        total += inst.dist[self.stops[-1]][0]  # last stop to depot
        return total

    def calculate_load_at_each_stop(self, inst: SCFPDPInstance) -> List[int]:
        """Calculate vehicle load after each stop"""
        loads = []
        current_load = 0

        for stop in self.stops:
            if stop <= inst.n:  # pickup
                req = stop
                current_load += inst.demands[req - 1]
            else:  # dropoff
                req = stop - inst.n
                current_load -= inst.demands[req - 1]
            loads.append(current_load)

        return loads
    
    def remove_request(self, inst: 'SCFPDPInstance', req: int):
        """Remove both pickup and dropoff for a request from the route"""
        pickup_idx = inst.pickup_idx(req)
        dropoff_idx = inst.dropoff_idx(req)
        
        self.stops = [s for s in self.stops if s not in [pickup_idx, dropoff_idx]]
    
    def has_request(self, inst: 'SCFPDPInstance', req: int) -> bool:
        """Check if route contains a specific request"""
        return req in self.get_requests(inst)
    
    def is_valid_precedence(self, inst: 'SCFPDPInstance') -> bool:
        """Check if all pickups come before their dropoffs"""
        requests = self.get_requests(inst)
        for req in requests:
            pickup_idx = inst.pickup_idx(req)
            dropoff_idx = inst.dropoff_idx(req)
            
            try:
                pickup_pos = self.stops.index(pickup_idx)
                dropoff_pos = self.stops.index(dropoff_idx)
                
                if pickup_pos >= dropoff_pos:
                    return False
            except ValueError:
                return False  # Missing pickup or dropoff
        
        return True


class Solution:
    """Complete solution with all vehicle routes"""

    def __init__(self, inst: SCFPDPInstance,
                 routes: Optional[List[Route]] = None,
                 fairness_type: str = 'jain'):  # NEW PARAMETER
        self.inst = inst
        self.fairness_type = fairness_type  # Store fairness type
        if routes is None:
            self.routes = [Route() for _ in range(inst.n_K)]
        else:
            self.routes = routes

    def copy(self) -> 'Solution':
        """Create a deep copy of the solution"""
        return Solution(self.inst, [r.copy() for r in self.routes], self.fairness_type)


    def get_all_served_requests(self) -> Set[int]:
        """Get set of all served requests across all routes"""
        served = set()
        for route in self.routes:
            served.update(route.get_requests(self.inst))
        return served

    def num_served_requests(self) -> int:
        """Count total number of served requests"""
        return len(self.get_all_served_requests())

    def is_feasible(self) -> Tuple[bool, str]:
        """
        Check if solution is feasible.
        Returns (is_feasible, error_message)
        """
        served = set()

        for k, route in enumerate(self.routes):
            # Check capacity constraints
            loads = route.calculate_load_at_each_stop(self.inst)
            for i, load in enumerate(loads):
                if load > self.inst.C:
                    return False, f"Route {k} exceeds capacity at stop {i}: load={load}, capacity={self.inst.C}"
                if load < 0:
                    return False, f"Route {k} has negative load at stop {i}: load={load}"

            # Check pickup before dropoff for each request
            route_requests = route.get_requests(self.inst)
            for req in route_requests:
                pickup_idx = self.inst.pickup_idx(req)
                dropoff_idx = self.inst.dropoff_idx(req)

                try:
                    pickup_pos = route.stops.index(pickup_idx)
                    dropoff_pos = route.stops.index(dropoff_idx)

                    if pickup_pos >= dropoff_pos:
                        return False, f"Route {k}: dropoff before pickup for request {req}"
                except ValueError:
                    return False, f"Route {k}: incomplete request {req} (missing pickup or dropoff)"

            # Check for duplicate requests
            for req in route_requests:
                if req in served:
                    return False, f"Request {req} served multiple times"
                served.add(req)

        # Check minimum requests constraint
        if len(served) < self.inst.gamma:
            return False, f"Insufficient requests served: {len(served)} < {self.inst.gamma}"

        return True, "Feasible"
    
    def get_route_for_request(self, req: int) -> Optional[int]:
        """
        Find which route (index) contains the given request.
        Returns None if request is not served.
        """
        for k, route in enumerate(self.routes):
            if route.has_request(self.inst, req):
                return k
        return None
    
    def remove_request(self, req: int):
        """Remove a request from whichever route contains it"""
        route_idx = self.get_route_for_request(req)
        if route_idx is not None:
            self.routes[route_idx].remove_request(self.inst, req)

    def objective_value(self) -> float:
        """Calculate objective function value with specified fairness measure"""
        durations = [r.duration(self.inst) for r in self.routes]
        total_duration = sum(durations)
        fairness = self.calculate_fairness_value(durations)
        return total_duration + self.inst.rho * (1 - fairness)

    def objective_value_from_durations(self, durations: List[float]) -> float:
        """Compute the objective value given a list of route durations"""
        total_duration = sum(durations)
        fairness = self.calculate_fairness_value(durations)
        return total_duration + self.inst.rho * (1 - fairness)

    @staticmethod
    def jain_fairness(durations: List[float]) -> float:
        """Calculate Jain fairness index"""
        n_k = len(durations)
        if n_k == 0:
            return 1.0  # no routes: trivially "perfect"

        sum_d = sum(durations)
        sum_d2 = sum(d * d for d in durations)

        if sum_d2 == 0:
            # all durations are zero: perfectly equal
            return 1.0

        return (sum_d * sum_d) / (n_k * sum_d2)

    def fairness_maxmin(route_durations: List[float]) -> float:
        """
        Max-min fairness measure.
        M = min_k d(R_k) / max_k d(R_k)
        
        Returns value between 0 (unfair) and 1 (perfectly fair).
        Higher is better (more fair).
        """
        if not route_durations or all(d == 0 for d in route_durations):
            return 1.0
        
        # Filter out zero durations (empty routes)
        active_durations = [d for d in route_durations if d > 0]
        
        if not active_durations:
            return 1.0
        
        min_duration = min(active_durations)
        max_duration = max(active_durations)
        
        if max_duration == 0:
            return 1.0
        
        return min_duration / max_duration


    def fairness_gini(route_durations: List[float]) -> float:
        """
        Inverted Gini coefficient.
        G = 1 - (sum_k' sum_k'' |d(R_k') - d(R_k'')|) / (2 * n_K * sum_k' d(R_k'))
        
        Returns value between 0 (unfair) and 1 (perfectly fair).
        Higher is better (more fair).
        """
        if not route_durations or all(d == 0 for d in route_durations):
            return 1.0
        
        n = len(route_durations)
        sum_d = sum(route_durations)
        
        if sum_d == 0:
            return 1.0
        
        # Calculate sum of absolute differences
        sum_abs_diff = 0.0
        for d1 in route_durations:
            for d2 in route_durations:
                sum_abs_diff += abs(d1 - d2)
        
        denominator = 2 * n * sum_d
        
        if denominator == 0:
            return 1.0
        
        gini = 1.0 - (sum_abs_diff / denominator)
        
        return gini


    @staticmethod
    def jain_fairness(durations: List[float]) -> float:
        """Calculate Jain fairness index"""
        n_k = len(durations)
        if n_k == 0:
            return 1.0

        sum_d = sum(durations)
        sum_d2 = sum(d * d for d in durations)

        if sum_d2 == 0:
            return 1.0

        return (sum_d * sum_d) / (n_k * sum_d2)

    @staticmethod
    def fairness_maxmin(route_durations: List[float]) -> float:
        """Max-min fairness measure"""
        if not route_durations or all(d == 0 for d in route_durations):
            return 1.0
        
        active_durations = [d for d in route_durations if d > 0]
        if not active_durations:
            return 1.0
        
        min_duration = min(active_durations)
        max_duration = max(active_durations)
        
        if max_duration == 0:
            return 1.0
        
        return min_duration / max_duration

    @staticmethod
    def fairness_gini(route_durations: List[float]) -> float:
        """Inverted Gini coefficient"""
        if not route_durations or all(d == 0 for d in route_durations):
            return 1.0
        
        n = len(route_durations)
        sum_d = sum(route_durations)
        
        if sum_d == 0:
            return 1.0
        
        sum_abs_diff = 0.0
        for d1 in route_durations:
            for d2 in route_durations:
                sum_abs_diff += abs(d1 - d2)
        
        denominator = 2 * n * sum_d
        
        if denominator == 0:
            return 1.0
        
        gini = 1.0 - (sum_abs_diff / denominator)
        
        return gini

    def get_statistics(self) -> dict:
        """Get comprehensive solution statistics"""
        durations = [r.duration(self.inst) for r in self.routes]
        active_routes = sum(1 for d in durations if d > 0)
        fairness = self.calculate_fairness_value(durations)

        return {
            'objective': self.objective_value(),
            'total_duration': sum(durations),
            'fairness': fairness,
            'fairness_type': self.fairness_type,  # Include type in stats
            'num_served': self.num_served_requests(),
            'num_active_routes': active_routes,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'route_durations': durations
        }
    
    def calculate_fairness_value(self, durations: List[float]) -> float:
        """Calculate fairness using the specified measure"""
        if self.fairness_type == 'jain':
            return self.jain_fairness(durations)
        elif self.fairness_type == 'maxmin':
            return self.fairness_maxmin(durations)
        elif self.fairness_type == 'gini':
            return self.fairness_gini(durations)
        else:
            raise ValueError(f"Unknown fairness type: {self.fairness_type}")


    def calculate_fairness(self):
        """Return the fairness value used in the objective"""
        try:
            durations = [r.duration(self.inst) for r in self.routes if not r.is_empty()]
            if len(durations) == 0:
                return 1.0
            return self.calculate_fairness_value(durations)
        except Exception as e:
            print(f"Fairness calculation error: {e}")
            return 0.0


    def print_statistics(self):
        """Print solution statistics in a readable format"""
        stats = self.get_statistics()
        feasible, msg = self.is_feasible()

        print("=" * 60)
        print("SOLUTION STATISTICS")
        print("=" * 60)
        print(f"Feasible: {feasible} - {msg}")
        print(
            f"Requests served: {stats['num_served']}/{self.inst.n} (required: {self.inst.gamma})")
        print(f"Active routes: {stats['num_active_routes']}/{self.inst.n_K}")
        print(f"\nObjective value: {stats['objective']:.2f}")
        print(f"  Total duration: {stats['total_duration']:.2f}")
        print(f"  Fairness: {stats['fairness']:.4f}")
        print(
            f"  Fairness penalty: {self.inst.rho * (1 - stats['fairness']):.2f}")
        print(f"\nRoute durations:")
        for k, dur in enumerate(stats['route_durations']):
            print(f"  Route {k + 1}: {dur:.2f}")
        print(
            f"  Min: {stats['min_duration']:.2f}, Max: {stats['max_duration']:.2f}, Avg: {stats['avg_duration']:.2f}")
        print("=" * 60)


class PrecedenceRoute:
    """
    Route encoded as sequence of (request, type) pairs.
    This maintains pickup-before-dropoff constraint explicitly.
    Pickups and dropoffs can be positioned anywhere in the route.
    """
    def __init__(self, sequence: List[Tuple[int, str]]):
        """
        Args:
            sequence: List of (request_id, type) where type is 'P' (pickup) or 'D' (dropoff)
                     Example: [(3, 'P'), (1, 'P'), (3, 'D'), (1, 'D'), (2, 'P'), (2, 'D')]
                     This means: pickup 3, pickup 1, dropoff 3, dropoff 1, pickup 2, dropoff 2
        """
        self.sequence = sequence
    
    def to_stops(self, inst: SCFPDPInstance) -> List[int]:
        """Convert precedence sequence to location indices"""
        stops = []
        for req, typ in self.sequence:
            if typ == 'P':
                stops.append(inst.pickup_idx(req))
            else:  # 'D'
                stops.append(inst.dropoff_idx(req))
        return stops
    
    def to_route(self, inst: SCFPDPInstance) -> Route:
        """Convert to Route object"""
        return Route(stops=self.to_stops(inst))
    
    @staticmethod
    def from_route(route: Route, inst: SCFPDPInstance) -> 'PrecedenceRoute':
        """Convert Route to PrecedenceRoute"""
        sequence = []
        for stop in route.stops:
            req = inst.request_from_location(stop)
            if req is not None:
                if stop <= inst.n:  # pickup
                    sequence.append((req, 'P'))
                else:  # dropoff
                    sequence.append((req, 'D'))
        return PrecedenceRoute(sequence)
    
    def is_valid(self) -> bool:
        """Check if precedence constraints are satisfied (pickup before dropoff)"""
        picked_up = set()
        for req, typ in self.sequence:
            if typ == 'P':
                picked_up.add(req)
            else:  # 'D'
                if req not in picked_up:
                    return False  # Dropoff before pickup
        return True
    
    def get_requests(self) -> Set[int]:
        """Get set of all requests in this route"""
        return set(req for req, _ in self.sequence)
    
    def copy(self) -> 'PrecedenceRoute':
        """Create a deep copy"""
        return PrecedenceRoute(self.sequence.copy())
    
    def __len__(self):
        """Length of the sequence"""
        return len(self.sequence)
    
    def __repr__(self):
        """String representation"""
        return f"PrecedenceRoute({self.sequence})"


# ============================================================================
# STANDALONE REPAIR FUNCTIONS
# ============================================================================

def repair_precedence(route: PrecedenceRoute) -> PrecedenceRoute:
    """
    Repair precedence violations by ensuring pickups come before dropoffs.
    Removes orphaned dropoffs (dropoffs without pickups).
    """
    repaired = []
    pickup_seen = set()
    
    # First pass: add all pickups and valid dropoffs
    for req, typ in route.sequence:
        if typ == 'P':
            repaired.append((req, 'P'))
            pickup_seen.add(req)
        else:  # 'D'
            if req in pickup_seen:
                # Pickup already seen, can add dropoff
                repaired.append((req, 'D'))
            # else: skip this dropoff (orphaned - no pickup)
    
    return PrecedenceRoute(repaired)


def repair_solution(solution: Solution) -> Solution:
    """
    Repair a solution to ensure:
    1. Each request appears at most once across all routes
    2. Pickup comes before dropoff in each route
    3. No partial requests (both pickup and dropoff present or neither)
    """
    inst = solution.inst
    repaired = solution.copy()
    
    # Track which requests are served across all routes
    served_requests = set()
    
    for k, route in enumerate(repaired.routes):
        # Convert to precedence representation
        prec_route = PrecedenceRoute.from_route(route, inst)
        
        # Remove duplicates (requests already served in previous routes)
        cleaned_sequence = []
        for req, typ in prec_route.sequence:
            if req not in served_requests:
                cleaned_sequence.append((req, typ))
        
        prec_route.sequence = cleaned_sequence
        
        # Repair precedence violations
        prec_route = repair_precedence(prec_route)
        
        # Remove incomplete requests (pickup without dropoff or vice versa)
        request_types = {}
        for req, typ in prec_route.sequence:
            if req not in request_types:
                request_types[req] = set()
            request_types[req].add(typ)
        
        # Keep only complete requests
        complete_sequence = []
        for req, typ in prec_route.sequence:
            if 'P' in request_types[req] and 'D' in request_types[req]:
                complete_sequence.append((req, typ))
        
        prec_route.sequence = complete_sequence
        
        # Update served requests
        served_requests.update(prec_route.get_requests())
        
        # Convert back to Route
        repaired.routes[k] = prec_route.to_route(inst)
    
    return repaired

def repair_capacity_violations(solution: Solution) -> Solution:
    """
    Repair capacity violations by:
    1. Reordering stops to minimize peak load
    2. Moving requests between routes
    3. Dropping requests if necessary
    
    This maintains randomness while ensuring feasibility.
    """
    repaired = solution.copy()
    inst = solution.inst
    max_repair_iterations = 50
    
    for iteration in range(max_repair_iterations):
        violations_fixed = 0
        
        for route_idx, route in enumerate(repaired.routes):
            if route.is_empty():
                continue
            
            # Check capacity
            loads = route.calculate_load_at_each_stop(inst)
            
            if not loads or (max(loads) <= inst.C and min(loads) >= 0):
                continue  # Route is feasible
            
            # VIOLATION DETECTED - Try to fix
            
            # Strategy 1: Reorder dropoffs earlier (move dropoffs toward their pickups)
            prec_route = PrecedenceRoute.from_route(route, inst)
            requests_in_route = list(route.get_requests(inst))
            
            # Rebuild sequence with dropoffs immediately after pickups
            new_sequence = []
            for req, typ in prec_route.sequence:
                if typ == 'P':
                    new_sequence.append((req, 'P'))
                    # Check if we should add dropoff right after
                    if random.random() < 0.5:  # 50% chance for immediate dropoff
                        new_sequence.append((req, 'D'))
            
            # Add remaining dropoffs
            picked = set(req for req, typ in new_sequence if typ == 'P')
            dropped = set(req for req, typ in new_sequence if typ == 'D')
            
            for req in picked:
                if req not in dropped:
                    new_sequence.append((req, 'D'))
            
            test_route = PrecedenceRoute(new_sequence).to_route(inst)
            test_loads = test_route.calculate_load_at_each_stop(inst)
            
            if test_loads and max(test_loads) <= inst.C and min(test_loads) >= 0:
                # Fixed by reordering!
                repaired.routes[route_idx] = test_route
                violations_fixed += 1
                continue
            
            # Strategy 2: Move most demanding request to another route
            if requests_in_route:
                # Find request with largest demand
                req_demands = [(req, inst.demands[req - 1]) for req in requests_in_route]
                req_demands.sort(key=lambda x: x[1], reverse=True)
                
                for req_to_move, _ in req_demands[:3]:  # Try top 3 heaviest
                    # Remove from this route
                    temp_route = route.copy()
                    temp_route.remove_request(inst, req_to_move)
                    
                    # Check if this fixes the violation
                    temp_loads = temp_route.calculate_load_at_each_stop(inst)
                    
                    if temp_loads and max(temp_loads) <= inst.C and min(temp_loads) >= 0:
                        # Try to insert into another route
                        other_route_idx = (route_idx + 1) % inst.n_K
                        other_route = repaired.routes[other_route_idx]
                        
                        pickup_idx = inst.pickup_idx(req_to_move)
                        dropoff_idx = inst.dropoff_idx(req_to_move)
                        
                        if other_route.is_empty():
                            other_route.stops = [pickup_idx, dropoff_idx]
                        else:
                            other_route.stops.append(pickup_idx)
                            other_route.stops.append(dropoff_idx)
                        
                        # Update the original route
                        repaired.routes[route_idx] = temp_route
                        violations_fixed += 1
                        break
            
        
        # If no violations fixed in this iteration, we're done
        if violations_fixed == 0:
            break
    
    return repaired

def create_random_solution(inst: SCFPDPInstance, min_requests: Optional[int] = None, fairness_type: str = 'jain') -> Solution:
    """
    Create a RANDOM solution with capacity awareness.
    Still random (for diversity), but uses intelligent insertion to avoid obvious violations.
    """
    import random
    
    if min_requests is None:
        min_requests = inst.gamma
    
    solution = create_empty_solution(inst)
    
    # Randomly decide how many requests (gamma to n)
    num_to_serve = random.randint(min_requests, inst.n)
    
    # RANDOM selection of requests (not greedy)
    all_requests = list(range(1, inst.n + 1))
    random.shuffle(all_requests)
    requests_to_serve = all_requests[:num_to_serve]
    
    # RANDOM assignment to routes
    for req in requests_to_serve:
        # Pick a RANDOM route
        route_idx = random.randint(0, inst.n_K - 1)
        route = solution.routes[route_idx]
        
        pickup_idx = inst.pickup_idx(req)
        dropoff_idx = inst.dropoff_idx(req)
        
        if route.is_empty():
            route.stops = [pickup_idx, dropoff_idx]
        else:
            # RANDOM positions, but maintain pickup-before-dropoff
            max_pos = len(route.stops)
            pickup_pos = random.randint(0, max_pos)
            dropoff_pos = random.randint(pickup_pos + 1, max_pos + 1)
            
            route.stops.insert(pickup_pos, pickup_idx)
            route.stops.insert(dropoff_pos, dropoff_idx)
    
    # CRITICAL: Repair capacity violations
    solution = repair_capacity_violations(solution)
    
    return solution

def read_solution(filename: str, inst: SCFPDPInstance) -> Solution:
    """Read solution from file"""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Skip first line (instance name)
    routes = []
    for line in lines[1:]:
        if line:
            stops = list(map(int, line.split()))
            routes.append(Route(stops=stops))
        else:
            routes.append(Route())

    # Ensure we have exactly n_K routes
    while len(routes) < inst.n_K:
        routes.append(Route())

    return Solution(inst, routes[:inst.n_K])


def write_solution(solution: Solution, filename: str, instance_name: str):
    """Write solution to file"""
    with open(filename, 'w') as f:
        f.write(f"{instance_name}\n")
        for route in solution.routes:
            if route.stops:
                f.write(" ".join(map(str, route.stops)) + "\n")
            else:
                f.write("\n")


def validate_solution_file(instance_file: str, solution_file: str) -> bool:
    """
    Validate a solution file against an instance.
    Returns True if valid, False otherwise.
    Prints detailed error messages.
    """
    try:
        inst = SCFPDPInstance(instance_file)
        sol = read_solution(solution_file, inst)

        feasible, msg = sol.is_feasible()
        if not feasible:
            print(f"INFEASIBLE: {msg}")
            return False

        print("Solution is FEASIBLE")
        sol.print_statistics()
        return True

    except Exception as e:
        print(f"ERROR validating solution: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_instance_name(filepath: str) -> str:
    """Extract instance name from file path (without extension)"""
    return os.path.splitext(os.path.basename(filepath))[0]


# Utility functions for algorithms

def create_empty_solution(inst: SCFPDPInstance, fairness_type: str = 'jain') -> Solution:
    """Create an empty solution"""
    return Solution(inst)


def get_unserved_requests(solution: Solution) -> Set[int]:
    """Get set of unserved requests"""
    all_requests = set(range(1, solution.inst.n + 1))
    served = solution.get_all_served_requests()
    return all_requests - served


def route_has_capacity(route: Route, inst: SCFPDPInstance,
                       additional_demand: int) -> bool:
    """Check if route can accommodate additional demand"""
    if route.is_empty():
        return additional_demand <= inst.C

    loads = route.calculate_load_at_each_stop(inst)
    max_load = max(loads) if loads else 0
    return max_load + additional_demand <= inst.C

def repair_gamma(solution: Solution) -> Solution:
    inst = solution.inst
    repaired = solution.copy()

    served = repaired.get_all_served_requests()
    missing = list(set(range(1, inst.n + 1)) - served)
    random.shuffle(missing)

    while repaired.num_served_requests() < inst.gamma and missing:
        req = missing.pop()

        # try all routes
        for route in repaired.routes:
            demand = inst.demands[req - 1]

            if route_has_capacity(route, inst, demand):
                pickup = inst.pickup_idx(req)
                dropoff = inst.dropoff_idx(req)

                route.stops.append(pickup)
                route.stops.append(dropoff)
                break

    return repaired