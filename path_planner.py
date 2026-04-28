"""
path_planner.py
===============
Main TAN Path Planning Algorithm for AUV.

This module implements the complete path planning algorithm described in the paper:
"AUV Path Planning Algorithm for Terrain Aided Navigation"
Zhang et al., J. Mar. Sci. Eng. 2022, 10, 1393.

Algorithm flow (Figure 2 in the paper):
1. Start at starting point
2. Search for target-aided point near the final target
3. Iteratively search for TAN-suitable areas in sector:
   a. Draw sector centered at current AUV position, oriented toward target-aided point
   b. Find best TAN-suitable block in sector
   c. Move AUV to that block, perform TAN fix
   d. Update AUV position using TAN result
   e. Check if target-aided point is in current sector → if yes, go to step 4
4. Navigate to target-aided point, perform TAN fix
5. Navigate to final target

The algorithm generates a sequence of TAN waypoints forming the path.
"""

import numpy as np
import copy
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from sector_search import (
    TANPoint, SectorParams,
    search_tan_suitable_in_sector,
    search_target_aided_point,
    angle_between, distance, is_point_in_sector,
    search_dynamic_target_aided_point,
    compute_turn_angle_deg,
    is_segment_feasible_by_flight_constraints,
    print_sector_analysis,
)
from tan_suitability import (
    build_suitability_map,
    get_entropy_at_coord,
    get_block_index_from_coord,
)
from tan_simulator import (
    ParticleFilterTAN,
    INSSimulator,
    simulate_tan_at_waypoint,
)


# ---------------------------------------------------------------------------
# Path planning result
# ---------------------------------------------------------------------------

@dataclass
class PathPlanningResult:
    """Result of the TAN path planning algorithm."""
    start_point: Tuple[float, float]
    target_point: Tuple[float, float]
    waypoints: List[TANPoint] = field(default_factory=list)
    target_aided_point: Optional[TANPoint] = None
    params: Optional[SectorParams] = None
    total_distance: float = 0.0
    max_tan_error: float = 0.0
    mean_tan_error: float = 0.0

    def print_summary(self):
        """Print a formatted summary of the path planning result."""
        print(f"\n{'='*65}")
        print(f"  TAN Path Planning Result")
        print(f"{'='*65}")
        print(f"  Start:  ({self.start_point[0]:.0f}, {self.start_point[1]:.0f})")
        print(f"  Target: ({self.target_point[0]:.0f}, {self.target_point[1]:.0f})")
        print(f"  Parameters: alpha={self.params.alpha}°, beta={self.params.beta}°, "
              f"L_max={self.params.L_max}m")
        print(f"{'─'*65}")
        print(f"  {'No.':<5} {'x':>8} {'y':>8} {'Entropy':>10} {'TAN Error':>12} {'Type'}")
        print(f"  {'─'*60}")
        print(f"  {'S':<5} {self.start_point[0]:>8.0f} {self.start_point[1]:>8.0f} "
              f"{'─':>10} {'─':>12}")
        for i, wp in enumerate(self.waypoints):
            tag = " ← TARGET-AIDED" if wp.is_target_aided else ""
            print(f"  {i+1:<5} {wp.x:>8.1f} {wp.y:>8.1f} "
                  f"{wp.entropy:>10.4f} {wp.tan_location_error:>10.2f} m{tag}")
        print(f"  {'T':<5} {self.target_point[0]:>8.0f} {self.target_point[1]:>8.0f} "
              f"{'─':>10} {'─':>12}")
        print(f"{'─'*65}")
        print(f"  Total waypoints: {len(self.waypoints)}")
        print(f"  Total distance:  {self.total_distance:.1f} m")
        print(f"  Max TAN error:   {self.max_tan_error:.2f} m")
        print(f"  Mean TAN error:  {self.mean_tan_error:.2f} m")
        print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main path planner class
# ---------------------------------------------------------------------------

class TANPathPlanner:
    """
    TAN Path Planner using sector search method.

    Implements the complete algorithm from the paper including:
    - Target-aided point search (Section 2.4)
    - Iterative sector search for TAN waypoints (Section 2.1-2.3)
    - TAN simulation at each waypoint
    """

    def __init__(
        self,
        terrain: np.ndarray,
        params: SectorParams,
        noise_std: float = 0.3,
        entropy_threshold: Optional[float] = None,
        n_pf_particles: int = 300,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        terrain : np.ndarray
            Prior terrain map (500×500).
        params : SectorParams
            Algorithm parameters.
        noise_std : float
            MBES noise standard deviation (0.3m or 0.5m).
        entropy_threshold : float or None
            Entropy threshold for TAN suitability. If None, use median.
        n_pf_particles : int
            Number of particles for PF-TAN.
        verbose : bool
            Print progress information.
        """
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

        # Particle filter for TAN simulation
        self.pf = ParticleFilterTAN(
            terrain,
            n_particles=n_pf_particles,
            scan_size=20,
            noise_std=noise_std,
        )

        if verbose:
            print(f"[PathPlanner] Terrain: {terrain.shape}, "
                  f"Block size: {params.N}m, "
                  f"Entropy threshold: {self.threshold:.4f}")
            print(f"[PathPlanner] Suitable blocks: "
                  f"{self.suitability_map.sum()}/{self.suitability_map.size}")

    def _is_target_in_sector(
        self,
        auv_x: float, auv_y: float,
        target_x: float, target_y: float,
        aided_x: float, aided_y: float,
    ) -> bool:
        """
        Check if the target-aided point is within the current sector.

        The sector is centered at AUV position, oriented toward the
        target-aided point.
        """
        center_angle = angle_between((auv_x, auv_y), (aided_x, aided_y))
        L_R = min(self.params.L_R, self.params.L_max)
        L_R = max(L_R, self.params.L_min)

        return is_point_in_sector(
            aided_x, aided_y,
            auv_x, auv_y,
            center_angle,
            self.params.alpha,
            self.params.l,
            L_R,
        )

    def _simulate_tan(
        self,
        true_x: float, true_y: float,
        ins_x: float, ins_y: float,
        dist_since_last_fix: float,
    ) -> Tuple[float, float, float]:
        """
        Run TAN simulation at a waypoint.

        Returns (est_x, est_y, tan_error).
        """
        ins_error_std = self.params.p * dist_since_last_fix
        ins_error_std = max(ins_error_std, 2.0)  # minimum 2m std

        est_x, est_y, err = self.pf.estimate_position(
            ins_x, ins_y, true_x, true_y,
            ins_error_std=ins_error_std,
        )
        return est_x, est_y, err
    
    def _is_final_leg_feasible(
        self,
        prev_x: float,
        prev_y: float,
        auv_x: float,
        auv_y: float,
        aided_x: float,
        aided_y: float,
        target_x: float,
        target_y: float,
    ) -> bool:
        """
        Check final-segment constraint: 
        """
        seg_dist = distance((auv_x, auv_y), (aided_x, aided_y))
        a_in = compute_turn_angle_deg(
            (prev_x, prev_y), (auv_x, auv_y), (aided_x, aided_y)
        )
        a_out = compute_turn_angle_deg(
            (auv_x, auv_y), (aided_x, aided_y), (target_x, target_y)
        )
        
        return is_segment_feasible_by_flight_constraints(
            seg_dist=seg_dist,
            params=self.params,
            a_in_deg=a_in,
            a_out_deg=a_out,
            mode="middle",
        )  

    def plan_path(
        self,
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
        max_iterations: int = 50,
    ) -> PathPlanningResult:
        """
        Plan TAN path from start to target.

        Parameters
        ----------
        start_x, start_y : float
            Starting point coordinates.
        target_x, target_y : float
            Target point coordinates.
        max_iterations : int
            Maximum number of sector search iterations.

        Returns
        -------
        result : PathPlanningResult
        """
        result = PathPlanningResult(
            start_point=(start_x, start_y),
            target_point=(target_x, target_y),
            params=self.params,
        )

        if self.verbose:
            print(f"\n[PathPlanner] Planning path: "
                  f"({start_x:.0f},{start_y:.0f}) → ({target_x:.0f},{target_y:.0f})")

        # ── Step 1: Find target-aided point ──────────────────────────────────
        aided_point = None

        # ── Step 2: Iterative sector search ──────────────────────────────────
        # Current AUV true position (starts at start point)
        auv_x, auv_y = start_x, start_y
        prev_x, prev_y = None, None
        ins_x, ins_y = start_x, start_y
        dist_since_last_fix = 0.0
        total_dist = 0.0

        # Use final target for orientation; aided point is searched dynamically.
        current_target_x = target_x
        current_target_y = target_y

        waypoints = []
        iteration = 0

        dynamic_aided = search_dynamic_target_aided_point(
            current_x=auv_x,
            current_y=auv_y,
            target_x=target_x,
            target_y=target_y,
            start_x=start_x, 
            start_y=start_y,
            entropy_map=self.entropy_map,
            suitability_map=self.suitability_map,
            params=self.params,
        )

        if dynamic_aided is not None:
            aided_point = dynamic_aided
            result.target_aided_point = aided_point
            current_target_x = aided_point.x
            current_target_y = aided_point.y

        while iteration < max_iterations:
            iteration += 1
            
            # Check if aided point is within current sector
            if aided_point is not None and self._is_target_in_sector(
                auv_x, auv_y, target_x, target_y,
                current_target_x, current_target_y
            ):
                # Only stop early if final leg to target also meets missile constraints.         
                if prev_x is not None and not self._is_final_leg_feasible(
                    prev_x, prev_y,
                    auv_x, auv_y,
                    current_target_x, current_target_y,
                    target_x, target_y
                ):
                    if self.verbose:
                        print(f"[PathPlanner] Iter {iteration}: "
                            f"Target-aided point is in sector but final leg violates flight constraints; continue searching.")
                else:
                    if self.verbose:
                        print(f"[PathPlanner] Iter {iteration}: "
                            f"Target-aided point is in sector → navigating to it")
                    break

            # Check if we're close enough to the aided point
            dist_to_aided = distance((auv_x, auv_y),
                                     (current_target_x, current_target_y))
            if aided_point is not None and dist_to_aided < self.params.N:
                if self.verbose:
                    print(f"[PathPlanner] Iter {iteration}: "
                          f"Close to aided point (dist={dist_to_aided:.1f}m) → stopping")
                break

            # Apply limit lines when close to target
            dist_to_target = distance((auv_x, auv_y), (target_x, target_y))
            apply_limits = dist_to_target < 2 * self.params.L_max

            # Search for next TAN-suitable area in sector
            next_wp = search_tan_suitable_in_sector(
                auv_x, auv_y,
                current_target_x, current_target_y,
                self.entropy_map, self.suitability_map,
                self.params,
                apply_limit_lines=apply_limits,
                start_x=start_x, start_y=start_y,
                prev_x=prev_x, prev_y=prev_y,
                next_ref_x=None, next_ref_y=None,
                enforce_flight_constraints=True,
                segment_mode="first" if prev_x is None else "middle",
            )

            if next_wp is None:
                if self.verbose:
                    print(f"[PathPlanner] Iter {iteration}: "
                          f"No TAN-suitable area found in sector. "
                          f"Expanding search...L_R {self.params.L_R}")
                # Try with relaxed parameters (larger sector)
                relaxed_params = copy.deepcopy(self.params)
                relaxed_params.k = self.params.k * 1.5
                relaxed_params.alpha=(min(self.params.alpha * 1.2, 60.0))

                next_wp = search_tan_suitable_in_sector(
                    auv_x, auv_y,
                    current_target_x, current_target_y,
                    self.entropy_map, self.suitability_map,
                    relaxed_params,
                    prev_x=prev_x, prev_y=prev_y,
                    next_ref_x=None, next_ref_y=None,
                    enforce_flight_constraints=True,
                    segment_mode="first" if prev_x is None else "middle",
                )
                if next_wp is None:
                    if self.verbose:
                        print(f"[PathPlanner] Iter {iteration}: "
                              f"Still no suitable area. Stopping search. L_R {relaxed_params.L_R}")
                    break

            # Move AUV to next waypoint
            step_dist = distance((auv_x, auv_y), (next_wp.x, next_wp.y))
            total_dist += step_dist
            dist_since_last_fix += step_dist

            # Simulate INS error accumulation
            ins_error = self.params.p * dist_since_last_fix
            angle_err = np.random.uniform(0, 2 * np.pi)
            ins_x = next_wp.x + ins_error * np.cos(angle_err)
            ins_y = next_wp.y + ins_error * np.sin(angle_err)

            # Simulate TAN at this waypoint
            est_x, est_y, tan_err = self._simulate_tan(
                next_wp.x, next_wp.y,
                ins_x, ins_y,
                dist_since_last_fix,
            )

            next_wp.tan_location_error = tan_err

            # Update AUV position using TAN fix
            prev_x, prev_y = auv_x, auv_y
            auv_x, auv_y = next_wp.x, next_wp.y
            ins_x, ins_y = est_x, est_y
            dist_since_last_fix = 0.0  # reset after TAN fix

            waypoints.append(next_wp)

            if self.verbose:
                print(f"[PathPlanner] Iter {iteration}: "
                      f"WP ({next_wp.x:.1f},{next_wp.y:.1f}), "
                      f"entropy={next_wp.entropy:.4f}, "
                      f"TAN_err={tan_err:.2f}m")

        # ── Step 3: Navigate to target-aided point ────────────────────────────
        if aided_point is None:
            if self.verbose:
                print("[PathPlanner] WARNING: No dynamic target-aided point found. Using target directly.")
            aided_point = TANPoint(
                x=target_x, y=target_y, entropy=0.0, is_target_aided=True
            )
            result.target_aided_point = aided_point

        step_dist = distance((auv_x, auv_y),
                             (aided_point.x, aided_point.y))
        total_dist += step_dist
        dist_since_last_fix += step_dist

        ins_error = self.params.p * dist_since_last_fix
        angle_err = np.random.uniform(0, 2 * np.pi)
        ins_x = aided_point.x + ins_error * np.cos(angle_err)
        ins_y = aided_point.y + ins_error * np.sin(angle_err)

        est_x, est_y, tan_err = self._simulate_tan(
            aided_point.x, aided_point.y,
            ins_x, ins_y,
            dist_since_last_fix,
        )
        aided_point.tan_location_error = tan_err
        aided_point.is_target_aided = True
        waypoints.append(aided_point)

        if self.verbose:
            print(f"[PathPlanner] Target-aided point: "
                  f"({aided_point.x:.1f},{aided_point.y:.1f}), "
                  f"TAN_err={tan_err:.2f}m")

        # ── Step 4: Navigate to final target ─────────────────────────────────
        step_dist = distance((aided_point.x, aided_point.y),
                             (target_x, target_y))
        total_dist += step_dist

        # Final TAN error at target (using aided point fix)
        ins_error_final = self.params.p * step_dist
        angle_err = np.random.uniform(0, 2 * np.pi)
        ins_x_final = target_x + ins_error_final * np.cos(angle_err)
        ins_y_final = target_y + ins_error_final * np.sin(angle_err)

        # TAN error at target is approximately the INS error from aided point
        target_tan_err = float(np.sqrt(
            (ins_x_final - target_x)**2 + (ins_y_final - target_y)**2
        ))

        # Compile results
        result.waypoints = waypoints
        result.total_distance = total_dist

        errors = [wp.tan_location_error for wp in waypoints if wp.tan_location_error > 0]
        errors.append(target_tan_err)
        if errors:
            result.max_tan_error = max(errors)
            result.mean_tan_error = float(np.mean(errors))

        if self.verbose:
            print(f"[PathPlanner] Final target TAN error: {target_tan_err:.2f}m")
            result.print_summary()

        return result


# ---------------------------------------------------------------------------
# Convenience function for running a single case study
# ---------------------------------------------------------------------------

def run_case_study(
    terrain: np.ndarray,
    start_point: Tuple[float, float],
    target_point: Tuple[float, float],
    params: SectorParams,
    noise_std: float = 0.3,
    entropy_threshold: Optional[float] = None,
    case_name: str = "Case Study",
    verbose: bool = True,
) -> PathPlanningResult:
    """
    Run a single case study of TAN path planning.

    Parameters
    ----------
    terrain : np.ndarray
        Prior terrain map.
    start_point : tuple (x, y)
        Starting point.
    target_point : tuple (x, y)
        Target point.
    params : SectorParams
        Algorithm parameters.
    noise_std : float
        MBES noise standard deviation.
    entropy_threshold : float or None
        Entropy threshold. If None, use median.
    case_name : str
        Name for display.
    verbose : bool
        Print progress.

    Returns
    -------
    result : PathPlanningResult
    """
    if verbose:
        print(f"\n{'#'*65}")
        print(f"  {case_name}")
        print(f"{'#'*65}")
        print_sector_analysis(params)

    planner = TANPathPlanner(
        terrain=terrain,
        params=params,
        noise_std=noise_std,
        entropy_threshold=entropy_threshold,
        verbose=verbose,
    )

    result = planner.plan_path(
        start_x=start_point[0],
        start_y=start_point[1],
        target_x=target_point[0],
        target_y=target_point[1],
    )

    return result


if __name__ == "__main__":
    from terrain_map import generate_synthetic_terrain

    np.random.seed(42)
    terrain = generate_synthetic_terrain(size=500, seed=42, noise_coefficient=0.3)

    # Case study 3 parameters (Table 6 in paper)
    params = SectorParams(
        N=50, k=2.0, alpha=45.0, beta=60.0,
        L_max=100.0, L_min=40.0, l=10.0, p=0.05,
        # terrain_size=500,
    )

    result = run_case_study(
        terrain=terrain,
        start_point=(60, 490),
        target_point=(430, 10),
        params=params,
        noise_std=0.3,
        case_name="Case Study 3 (L_max=100m) - Group 1",
    )
