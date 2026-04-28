"""
hybrid_astar_planner.py
=======================
Hybrid A* planner with Dubins path primitives.

State space: (x, y, θ) instead of just (x, y)
Expansion: 8 heading angles × Dubins curves

Key innovation:
- Every edge is a Dubins curve → respects minimum turn radius
- Terrain suitability checked along curve (not just at endpoints)
- Flight constraints enforced for each primitive
"""

from __future__ import annotations

import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import bisect

from sector_search import (
    SectorParams,
    distance,
    compute_turn_angle_deg,
    is_segment_feasible_by_flight_constraints,
)
from tan_suitability import build_suitability_map
from dubins_primitives import DubinsPathPlanner, DubinsSegment
from path_planner import PathPlanningResult
from sector_search import TANPoint


@dataclass
class HybridState:
    """Hybrid A* state: (x, y, theta)."""
    x: float
    y: float
    theta: float  # Heading in radians
    
    def __hash__(self):
        # Discretize for hashing (0.5m, 0.5m, 5° resolution)
        xi = int(np.round(self.x / 0.5))
        yi = int(np.round(self.y / 0.5))
        ti = int(np.round(np.degrees(self.theta) / 5)) % 72
        return hash((xi, yi, ti))
    
    def __eq__(self, other):
        if not isinstance(other, HybridState):
            return False
        return (
            abs(self.x - other.x) < 0.5 and
            abs(self.y - other.y) < 0.5 and
            abs(self.theta - other.theta) < np.radians(5)
        )


@dataclass
class _HybridOpenItem:
    """Item in open list."""
    f: float
    g: float
    state: HybridState
    path_segments: List[DubinsSegment]
    
    def __lt__(self, other: _HybridOpenItem) -> bool:
        return self.f < other.f


class HybridAStarPlanner:
    """
    Hybrid A* planner for non-holonomic vehicle path planning.
    
    Combines:
    - Discrete A* search in (x, y, θ) space
    - Dubins curve primitives for feasible paths
    - Terrain constraints validation
    - Flight dynamics enforcement
    """
    
    def __init__(
        self,
        terrain: np.ndarray,
        params: SectorParams,
        turn_radius: float = 300.0,
        entropy_threshold: Optional[float] = None,
        heading_resolution_deg: float = 10.0,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        terrain : np.ndarray
            Prior terrain map.
        params : SectorParams
            Flight/sector parameters.
        turn_radius : float
            Minimum turn radius R (m).
        entropy_threshold : float or None
            For suitability map.
        heading_resolution_deg : float
            Discretization of heading (degrees).
        verbose : bool
            Print progress.
        """
        self.terrain = terrain
        self.params = params
        self.turn_radius = turn_radius
        self.verbose = verbose
        self.heading_res_deg = heading_resolution_deg
        self.heading_res_rad = np.radians(heading_resolution_deg)
        
        # Build suitability map
        self.entropy_map, self.suitability_map, self.threshold = build_suitability_map(
            terrain,
            block_size=params.N,
            entropy_threshold=entropy_threshold,
        )
        
        # Dubins planner
        self.dubins_planner = DubinsPathPlanner(R=turn_radius)
        
        if verbose:
            print(f"[HybridAStar] Terrain: {terrain.shape}, "
                  f"Turn radius: {turn_radius}m, "
                  f"Heading resolution: {heading_resolution_deg}°")
    
    # ─────────────────────────────────────────────────────────────
    # Heuristic & Cost
    # ─────────────────────────────────────────────────────────────
    
    def _heuristic(self, state: HybridState, target_xy: Tuple[float, float]) -> float:
        """
        Heuristic: Euclidean distance to target.
        (Admissible since Dubins paths ≥ Euclidean)
        """
        dx = target_xy[0] - state.x
        dy = target_xy[1] - state.y
        return np.sqrt(dx**2 + dy**2)
    
    def _edge_cost(self, segments: List[DubinsSegment]) -> float:
        """Cost = total arc length of Dubins path."""
        return sum(seg.arc_length for seg in segments)
    
    # ─────────────────────────────────────────────────────────────
    # Constraint checking
    # ─────────────────────────────────────────────────────────────
    
    def _is_path_feasible(
        self,
        segments: List[DubinsSegment],
        prev_state: Optional[HybridState],
        next_state: HybridState,
    ) -> bool:
        """
        Check if Dubins path satisfies all constraints:
        1. Terrain suitability along path
        2. Flight dynamics (turn angle, distance)
        3. Not too long
        """
        # Sample points along path
        all_points = []
        for seg in segments:
            pts = seg.sample_points(num_samples=30)
            all_points.extend(pts)
        
        if len(all_points) < 2:
            return False
        
        # Check terrain suitability
        if not self._check_terrain_along_path(np.array(all_points)):
            return False
        
        # Check flight constraints
        path_length = self._edge_cost(segments)
        if path_length > self.params.L_max:
            if self.verbose:
                print(f"  Path too long: {path_length:.1f}m > L_max={self.params.L_max}m")
            return False
        
        if path_length < self.params.L_min:
            if self.verbose:
                print(f"  Path too short: {path_length:.1f}m < L_min={self.params.L_min}m")
            return False
        
        # Turn angle constraint
        if prev_state is not None:
            prev_theta = prev_state.theta
            curr_theta = segments[0].theta0
            next_theta = next_state.theta
            
            # Angle changes
            a_in_deg = np.degrees(abs(curr_theta - prev_theta))
            a_out_deg = np.degrees(abs(next_theta - curr_theta))
            
            if a_in_deg > self.params.a_max_deg or a_out_deg > self.params.a_max_deg:
                if self.verbose:
                    print(f"  Turn angle violated: in={a_in_deg:.1f}°, out={a_out_deg:.1f}°")
                return False
        
        return True
    
    def _check_terrain_along_path(self, points: np.ndarray) -> bool:
        """
        Check that path passes through suitable terrain.
        
        Requirement: ≥80% of sample points must be in suitable blocks.
        """
        block_size = self.params.N
        n_y, n_x = self.suitability_map.shape
        
        suitable_count = 0
        for pt in points:
            x, y = pt[0], pt[1]
            bx = int(np.clip(x // block_size, 0, n_x - 1))
            by = int(np.clip(y // block_size, 0, n_y - 1))
            
            if self.suitability_map[by, bx]:
                suitable_count += 1
        
        suitability_ratio = suitable_count / len(points) if len(points) > 0 else 0
        return suitability_ratio >= self.params.suitability_min_ratio
    
    # ─────────────────────────────────────────────────────────────
    # Neighbor expansion
    # ─────────────────────────────────────────────────────────────
    
    def _get_successor_states(
        self,
        current_state: HybridState,
        target_xy: Tuple[float, float],
    ) -> List[Tuple[HybridState, List[DubinsSegment]]]:
        """
        Generate successor states by trying Dubins curves to 8 heading angles.
        
        Returns: List of (next_state, dubins_segments) pairs
        """
        successors = []
        
        # Try all 8 heading angles (45° increments)
        n_headings = int(360 / self.heading_res_deg)
        
        for i in range(n_headings):
            next_heading = np.radians(i * self.heading_res_deg)
            
            # Compute a target position a fixed distance away
            distance_to_next = self.turn_radius * 2  # Heuristic
            
            target_x = current_state.x + distance_to_next * np.cos(next_heading)
            target_y = current_state.y + distance_to_next * np.sin(next_heading)
            
            # Compute Dubins path
            dubins_result = self.dubins_planner.compute_dubins_path(
                current_state.x, current_state.y, current_state.theta,
                target_x, target_y, next_heading,
            )
            
            if dubins_result is None:
                continue
            
            segments, path_length = dubins_result
            
            # Check feasibility
            next_state = HybridState(target_x, target_y, next_heading)
            
            if not self._is_path_feasible(segments, current_state, next_state):
                continue
            
            successors.append((next_state, segments))
        
        return successors
    
    # ─────────────────────────────────────────────────────────────
    # Main A* search
    # ─────────────────────────────────────────────────────────────
    
    def plan_path(
        self,
        start_x: float, start_y: float, start_theta: float,
        target_x: float, target_y: float,
        max_iterations: int = 10000,
    ) -> PathPlanningResult:
        """
        Plan path using Hybrid A*.
        
        Parameters
        ----------
        start_x, start_y : float
            Start position.
        start_theta : float
            Start heading (radians, or degrees if > 2π).
        target_x, target_y : float
            Target position.
        max_iterations : int
            Max search iterations.
        
        Returns
        -------
        result : PathPlanningResult
        """
        # Normalize start heading
        if start_theta > 2 * np.pi:
            start_theta = np.radians(start_theta)
        
        target_xy = (target_x, target_y)
        start_state = HybridState(start_x, start_y, start_theta)
        
        open_heap: List[_HybridOpenItem] = [
            _HybridOpenItem(
                f=self._heuristic(start_state, target_xy),
                g=0.0,
                state=start_state,
                path_segments=[],
            )
        ]
        
        g_score: Dict[HybridState, float] = {start_state: 0.0}
        closed: Set[HybridState] = set()
        parent: Dict[HybridState, Tuple[HybridState, List[DubinsSegment]]] = {}
        
        iterations = 0
        final_state = None
        
        while open_heap and iterations < max_iterations:
            iterations += 1
            
            current_item = heapq.heappop(open_heap)
            curr_state = current_item.state
            
            if curr_state in closed:
                continue
            
            closed.add(curr_state)
            
            # Goal check: within ~100m of target
            dist_to_target = distance((curr_state.x, curr_state.y), target_xy)
            if dist_to_target < 100:
                if self.verbose:
                    print(f"[HybridAStar] Goal reached! Iterations: {iterations}")
                final_state = curr_state
                break
            
            # Expand neighbors
            for next_state, segments in self._get_successor_states(curr_state, target_xy):
                if next_state in closed:
                    continue
                
                tentative_g = g_score[curr_state] + self._edge_cost(segments)
                
                if tentative_g < g_score.get(next_state, float('inf')):
                    g_score[next_state] = tentative_g
                    parent[next_state] = (curr_state, segments)
                    
                    h = self._heuristic(next_state, target_xy)
                    heapq.heappush(open_heap, _HybridOpenItem(
                        f=tentative_g + h,
                        g=tentative_g,
                        state=next_state,
                        path_segments=segments,
                    ))
            
            if iterations % 100 == 0 and self.verbose:
                print(f"[HybridAStar] Iteration {iterations}, "
                      f"open_size={len(open_heap)}, "
                      f"closed_size={len(closed)}")
        
        # Reconstruct path
        result = PathPlanningResult(
            start_point=(start_x, start_y),
            target_point=(target_x, target_y),
            params=self.params,
        )
        
        if final_state is None:
            if self.verbose:
                print(f"[HybridAStar] No path found in {iterations} iterations")
            return result
        
        # Trace back path
        waypoints: List[TANPoint] = []
        total_distance = 0.0
        
        current = final_state
        path_states = [current]
        
        while current in parent:
            prev_state, segments = parent[current]
            path_states.append(prev_state)
            current = prev_state
        
        path_states.reverse()
        
        # Convert states to waypoints
        for state in path_states[1:]:  # Skip start
            # Sample final point of Dubins curves
            if state in parent:
                _, segments = parent[state]
                cost = self._edge_cost(segments)
                total_distance += cost
                
                # Get entropy at this location
                block_x = int(np.clip(state.x // self.params.N, 0,
                                    self.entropy_map.shape[1] - 1))
                block_y = int(np.clip(state.y // self.params.N, 0,
                                    self.entropy_map.shape[0] - 1))
                entropy = float(self.entropy_map[block_y, block_x])
                
                waypoints.append(TANPoint(
                    x=state.x,
                    y=state.y,
                    entropy=entropy,
                    tan_location_error=0.0,
                ))
        
        # Add final approach to target
        dist_to_target = distance((current.x, current.y), target_xy)
        total_distance += dist_to_target
        
        result.waypoints = waypoints
        result.total_distance = total_distance
        result.max_tan_error = 0.0
        result.mean_tan_error = 0.0
        
        if self.verbose:
            print(f"[HybridAStar] Path found: {len(waypoints)} waypoints, "
                  f"{total_distance:.1f}m distance")
            result.print_summary()
        
        return result
