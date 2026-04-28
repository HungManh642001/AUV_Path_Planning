"""
astar_planner.py
================
A* path planner for terrain-aided navigation with flight constraints.

This planner searches globally on TAN-suitable blocks instead of selecting
waypoints greedily/sequentially from a local sector.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sector_search import (
    TANPoint,
    SectorParams,
    compute_turn_angle_deg,
    distance,
    is_segment_feasible_by_flight_constraints,
)
from tan_suitability import build_suitability_map
from path_planner import PathPlanningResult


GridNode = Tuple[int, int]
State = Tuple[GridNode, Optional[GridNode]]    # (current_idx, prev_idx)


@dataclass
class _OpenItem:
    f: float
    g: float
    state: State

    def __lt__(self, other: "_OpenItem") -> bool:
        return self.f < other.f


class AStarTerrainPlanner:
    """
    Global A* planner on suitable terrain blocks.
    
    Notes
    -----
    - Nodes are block centers in suitability map.
    - Edge validity enforces flight constraints during expansion.
    - Goal test  uses final-leg interval constraint to target.
    """

    def __init__(
        self,
        terrain: np.ndarray,
        params: SectorParams,
        noise_std: float = 0.3,
        entropy_threshold: Optional[float] = None,
        verbose: bool = True
    ):
        self.terrain = terrain
        self.params = params
        self.noise_std = noise_std
        self.verbose = verbose
        
        # Build suitability map
        self.entropy_map, self.suitability_map, self.threshold = build_suitability_map(
            terrain,
            block_size=params.N,
            entropy_threshold=entropy_threshold,
        )
    
    # ---------------------------
    # Geometry/index helpers
    # ---------------------------
    def _to_block_idx(self, x: float, y: float) -> GridNode:
        n_y, n_x = self.suitability_map.shape
        ix = int(np.clip(x // self.params.N, 0, n_x - 1))
        iy = int(np.clip(y // self.params.N, 0, n_y - 1))
        return ix, iy
    
    def _to_coord(self, idx: GridNode) -> Tuple[float, float]:
        ix, iy = idx
        return (
            ix * self.params.N + self.params.N // 2.0,
            iy * self.params.N + self.params.N // 2.0,
        )
    
    def _iter_candidate_neighbors(self, curr_idx: GridNode, mode: str) -> List[GridNode]:
        """Return feasible geometric neighbors by range [L_min, L_max]"""
        n_y, n_x = self.suitability_map.shape
        cx, cy = self._to_coord(curr_idx)

        neighbors: List[GridNode] = []
        for iy in range(n_y):
            for ix in range(n_x):
                if (ix, iy) == curr_idx:
                    continue
                if not self.suitability_map[iy, ix]:
                    continue
                nx, ny = self._to_coord((ix, iy))
                d = distance((cx, cy), (nx, ny))
                if mode == 'first':
                    L_R = min(self.params.L_max, self.params.L_R) + self.params.l_min
                    L_r = self.params.l + self.params.l_min
                else:
                    L_R = min(self.params.L_max, self.params.L_R)
                    L_r = self.params.l
                if d < L_r or d > L_R:
                    continue
                neighbors.append((ix, iy))
            
        return neighbors
    
    def _heuristic(self, idx: GridNode, target_xy: Tuple[float, float]) -> float:
        x, y = self._to_coord(idx)
        return distance((x, y), target_xy)
    
    def _edge_cost(self, curr_idx: GridNode, next_idx: GridNode) -> float:
        cx, cy = self._to_coord(curr_idx)
        nx, ny = self._to_coord(next_idx)
        d = distance((cx, cy), (nx, ny))
        entropy = float(self.entropy_map[next_idx[1], next_idx[0]])
        entropy_penalty = 1.0 / (entropy + 1e-6)
        return d 
    
    def _is_transition_feasible(
        self,
        prev_idx: Optional[GridNode],
        curr_idx: GridNode,
        next_idx: GridNode,
        target_xy: Tuple[float, float],
    ) -> bool:
        """Check first/middle flight constraints for curr->next edge."""
        curr_xy = self._to_coord(curr_idx)
        next_xy = self._to_coord(next_idx)
        seg_dist = distance(curr_xy, next_xy)

        if prev_idx is None:
            a_in = 0.0
            mode = "first"
        else:
            prev_xy = self._to_coord(prev_idx)
            a_in = compute_turn_angle_deg(prev_xy, curr_xy, next_xy)
            mode = "middle"

        # Estimate outgoing angle from next toward final target
        a_out = self.params.a_max_deg

        return is_segment_feasible_by_flight_constraints(
            seg_dist=seg_dist,
            params=self.params,
            a_in_deg=a_in,
            a_out_deg=a_out,
            mode=mode,
        )

    
    def _is_goal_reachable_from(
        self, 
        prev_idx: Optional[GridNode],
        curr_idx: GridNode,
        target_xy: Tuple[float, float],
    ) -> bool:
        """Check final-leg interval constraint from current node to target."""
        curr_xy = self._to_coord(curr_idx)
        seg_dist = distance(curr_xy, target_xy)

        if prev_idx is None:
            a_last = 0.0
        else:
            prev_xy = self._to_coord(prev_idx)
            a_last = compute_turn_angle_deg(prev_xy, curr_xy, target_xy)
        
        return is_segment_feasible_by_flight_constraints(
            seg_dist=seg_dist,
            params=self.params,
            a_in_deg=a_last,
            a_out_deg=0.0,
            mode="last"
        )
    
    def _reconstruct_path(
        self,
        parent: Dict[State, Optional[State]],
        last_state: State,
    ) -> list[GridNode]:
        path_states: List[State] = []
        s = last_state
        while s is not None:
            path_states.append(s)
            s = parent[s]
        path_states.reverse()

        # Extract current_idx of each state (state = (curr, prev))
        return [st[0] for st in path_states]
    
    def plan_path(
        self, 
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
        max_iterations: int = 50,
    ) -> PathPlanningResult:
        """Plan path using A* and return PathPlanningResult-compatible output."""
        target_xy = (target_x, target_y)
        start_idx = self._to_block_idx(start_x, start_y)

        # If start block is unsuitable, choose nearest suitable block.
        if not self.suitability_map[start_idx[1], start_idx[0]]:
            suitable = np.argwhere(self.suitability_map)
            if suitable.size == 0:
                return PathPlanningResult(
                    start_point=(start_x, start_y),
                    target_point=(target_x, target_y),
                    params=self.params,
                    waypoints=[],
                    total_distance=0.0,
                    max_tan_error=0.0,
                    mean_tan_error=0.0,
                )
            sx, sy = start_x, start_y
            best = min(
                suitable,
                key=lambda rc: (rc[1] * self.params.N + self.params.N / 2 - sx) ** 2
                + (rc[0] * self.params.N + self.params.N / 2 - sy) ** 2,
            )
            start_idx = (int(best[1]), int(best[0]))
        
        start_state: State = (start_idx, None)
        open_heap: List[_OpenItem] = [
            _OpenItem(
                f=self._heuristic(start_idx, target_xy),
                g=0.0,
                state=start_state,
            )
        ]

        g_score: Dict[State, float] = {start_state: 0.0}
        parent: Dict[State, Optional[State]] = {start_state: None}

        iterations = 0
        best_terminal: Optional[State] = None

        while open_heap and iterations < max_iterations:
            iterations += 1
            current_item = heapq.heappop(open_heap)
            curr_state = current_item.state
            curr_idx, prev_idx = curr_state
            mode = 'first' if prev_idx is None else 'middle'

            # Goal condition: from current node we can legally fly final leg to target.
            if self._is_goal_reachable_from(prev_idx, curr_idx, target_xy):
                best_terminal = curr_state
                break

            for next_idx in self._iter_candidate_neighbors(curr_idx, mode):
                if not self._is_transition_feasible(prev_idx, curr_idx, next_idx, target_xy):
                    continue

                next_state: State = (next_idx, curr_idx)
                tentative_g = g_score[curr_state] + self._edge_cost(curr_idx, next_idx)

                if tentative_g < g_score.get(next_state, float('inf')):
                    g_score[next_state] = tentative_g
                    parent[next_state] = curr_state
                    h = self._heuristic(next_idx, target_xy)
                    heapq.heappush(open_heap, _OpenItem(f=tentative_g + h, g=tentative_g, state=next_state))
        
        result = PathPlanningResult(
            start_point=(start_x, start_y),
            target_point=(target_x, target_y),
            params=self.params,
        )

        if best_terminal is None:
            if self.verbose:
                print(f"[AStarPlanner] No feasible A* path found under constraints in {iterations} iterations.")
            return result
        
        idx_path = self._reconstruct_path(parent, best_terminal)
        coord_path = [self._to_coord(idx) for idx in idx_path]

        # Build waypoints (exclude first start-like node), add target-aided as final node before target.
        waypoints: List[TANPoint] = []
        for x, y in coord_path[1:]:
            e = float(self.entropy_map[int(y // self.params.N), int(x // self.params.N)])
            waypoints.append(TANPoint(x=x, y=y, entropy=e, tan_location_error=0.0))

        if waypoints:
            waypoints[-1].is_target_aided = True
            result.target_aided_point = waypoints[-1]
        
        # Compute path distance including start->first and last->target
        pts = [(start_x, start_y)] + [(wp.x, wp.y) for wp in waypoints] + [(target_x, target_y)]
        total_dist = 0.0
        for i in range(len(pts) - 1):
            total_dist += distance(pts[i], pts[i+1])
        
        result.waypoints = waypoints
        result.total_distance = total_dist
        result.max_tan_error = 0.0
        result.mean_tan_error = 0.0

        if self.verbose:
            print(f"[AStarPlanner] Feasible path found in {iterations} iterations.")
            result.print_summary()
        
        return result
