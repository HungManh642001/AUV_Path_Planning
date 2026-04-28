"""
sector_search.py
================
Core sector search algorithm for AUV TAN path planning.

Implements all geometric computations from the paper:
  - Section 2.2: Central angle calculation (alpha_min, alpha_max)
  - Section 2.3: Limit line at target point (beta constraint)
  - Section 2.4: Target-aided point selection
  - Full sector search for TAN-suitable areas

Key equations from the paper:
  Eq.(3):  alpha_min = arctan( N / sqrt(4*L_R^2 - N^2) - 2N )
  Eq.(5):  alpha_min = arccot( sqrt(4k^2 - 1) - 2 )   where L_R = k*N
  Eq.(8):  alpha_max = arccos( (2k^2 - 3*sqrt(2)*k + 2 - 2*sqrt(2)) /
                               (2*(k-sqrt(2))*sqrt(k^2 - sqrt(2)*k + 1 - 2*sqrt(2))) ) + 45°
  Eq.(11): arccot(sqrt(4k^2-1)-2) < alpha <= 45°
  Eq.(13): 45° <= beta <= 90°
  Eq.(14): |TA| = l + sigma + N/2
  Eq.(15): |TB| = l + sigma + N
  Eq.(16): l + sigma + N/2 <= L <= l + sigma + N
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SectorParams:
    """Parameters for the sector search algorithm."""
    N: int = 50              # TAN-suitable area block size (m)
    k: float = 2.0           # Sector radius coefficient: L_R = k * N
    alpha: float = 45.0      # Half central angle of sector (degrees)
    beta: float = 60.0       # Limit angle at target point (degrees)
    L_max: float = 100.0     # Maximum single matching voyage (m)
    L_min: float = 40.0      # Minimum single matching voyage (m)
    l: float = 10.0          # AUV turning circle (m)
    p: float = 0.05          # INS error ratio (5% of distance)
    terrain_size: int = 500  # Terrain map size (m)
    
    # Flight-constraint parameters
    l_min: float = 20.0
    a_max_deg: float = 60.0
    R: float = 30.0
    l_n: float = 20.0
    d_ss: float = 25.0

    nav_sigma_min: float = 5
    suitability_min_ratio: float = 0.7  # Minimum expected suitability coverage along curved segment
    curve_samples: int = 15             # Number of curve samples for suitability check

    @property
    def L_R(self) -> float:
        """Sector radius."""
        return self.k * self.N

    @property
    def sigma(self) -> float:
        """Maximum INS error after traveling L_R distance."""
        return self.p * self.L_R

    @property
    def TA_min(self) -> float:
        """Minimum distance from target-aided point to target (Eq.14)."""
        return self.l + self.sigma + self.N / 2

    @property
    def TA_max(self) -> float:
        """Maximum distance from target-aided point to target (Eq.15)."""
        return self.l + self.sigma + self.N

    def alpha_min_deg(self) -> float:
        """Minimum half sector angle in degrees (Eq.5)."""
        val = np.sqrt(4 * self.k**2 - 1) - 2
        if val <= 0:
            return 0.0
        return float(np.degrees(np.arctan(1.0 / val)))  # arccot(x) = arctan(1/x)

    def alpha_max_deg(self) -> float:
        """Maximum half sector angle in degrees (Eq.8, limit 45°)."""
        return 45.0
    

@dataclass
class TANPoint:
    """Represents a TAN-suitable waypoint."""
    x: float
    y: float
    entropy: float
    is_target_aided: bool = False
    tan_location_error: float = 0.0
    params: Tuple[float, float, float] = None

    def __repr__(self):
        tag = " [TARGET-AIDED]" if self.is_target_aided else ""
        return (f"TANPoint(x={self.x:.1f}, y={self.y:.1f}, "
                f"entropy={self.entropy:.4f}, err={self.tan_location_error:.2f}m){tag}")


# ---------------------------------------------------------------------------
# Geometric helper functions
# ---------------------------------------------------------------------------

def angle_between(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute the angle (in degrees) of the vector from p1 to p2,
    measured counter-clockwise from the positive X axis.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return float(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))


def is_point_in_sector(
    px: float, py: float,
    cx: float, cy: float,
    center_angle_deg: float,
    half_angle_deg: float,
    r_min: float,
    r_max: float,
) -> bool:
    """
    Check if point (px, py) lies within a sector.

    The sector is centered at (cx, cy), oriented along center_angle_deg,
    with half-opening angle half_angle_deg, and radial range [r_min, r_max].

    Parameters
    ----------
    px, py : float
        Point to test.
    cx, cy : float
        Sector center (AUV position).
    center_angle_deg : float
        Direction from AUV to target (degrees, CCW from +X).
    half_angle_deg : float
        Half of the sector's central angle (degrees).
    r_min : float
        Minimum radius (inner boundary, e.g., turning circle L_r).
    r_max : float
        Maximum radius (outer boundary L_R).

    Returns
    -------
    bool
    """
    dx = px - cx
    dy = py - cy
    r = np.sqrt(dx**2 + dy**2)

    if r < r_min or r > r_max:
        return False

    point_angle = np.degrees(np.arctan2(dy, dx))
    # Angular difference (normalized to [-180, 180])
    diff = (point_angle - center_angle_deg + 180) % 360 - 180
    return abs(diff) <= half_angle_deg


def is_point_within_limit_lines(
    px: float, py: float,
    target_x: float, target_y: float,
    start_x: float, start_y: float,
    beta_deg: float,
) -> bool:
    """
    Check if point (px, py) is within the limit lines at the target point.

    The limit lines pass through the target point, making angle beta with
    the line from start to target, opening toward the start side.

    A point is valid if it is on the start-side of the target (i.e., the
    projection onto the start->target direction is <= 0 from target),
    AND within the angular cone defined by beta.

    Parameters
    ----------
    px, py : float
        Point to test.
    target_x, target_y : float
        Target point coordinates.
    start_x, start_y : float
        Starting point coordinates (defines the centerline direction).
    beta_deg : float
        Limit angle (45° <= beta <= 90°).

    Returns
    -------
    bool
    """
    # Direction from target toward start (the "backward" direction)
    dx_ts = start_x - target_x
    dy_ts = start_y - target_y
    backward_angle = np.degrees(np.arctan2(dy_ts, dx_ts))

    # Direction from target to candidate point
    dx_tp = px - target_x
    dy_tp = py - target_y
    dist_tp = np.sqrt(dx_tp**2 + dy_tp**2)

    if dist_tp < 1e-6:
        return True  # At target itself

    point_angle = np.degrees(np.arctan2(dy_tp, dx_tp))
    diff = (point_angle - backward_angle + 180) % 360 - 180
    return abs(diff) <= beta_deg


# ---------------------------------------------------------------------------
# Sector search for TAN-suitable areas
# ---------------------------------------------------------------------------

def search_tan_suitable_in_sector(
    auv_x: float,
    auv_y: float,
    target_x: float,
    target_y: float,
    entropy_map: np.ndarray,
    suitability_map: np.ndarray,
    params: SectorParams,
    apply_limit_lines: bool = False,
    start_x: float = None,
    start_y: float = None,
    prev_x: float = None,
    prev_y: float = None,
    next_ref_x: float = None,
    next_ref_y: float = None,
    enforce_flight_constraints: bool = True,
    segment_mode: str = "middle",
) -> Optional[TANPoint]:
    """
    Search for the best TAN-suitable area within the sector.

    The sector is centered at (auv_x, auv_y), oriented toward (target_x, target_y),
    with half-angle alpha and radius L_R.

    Selection criterion: Among all TAN-suitable blocks in the sector,
    select the one with the HIGHEST entropy (most terrain information),
    and that is closest to the AUV-target centerline (to minimize lateral deviation).

    Parameters
    ----------
    auv_x, auv_y : float
        Current AUV position.
    target_x, target_y : float
        Target point.
    entropy_map : np.ndarray
        Block entropy map.
    suitability_map : np.ndarray
        Block suitability map (bool).
    params : SectorParams
        Algorithm parameters.
    apply_limit_lines : bool
        Whether to apply limit line constraints (used near target).
    start_x, start_y : float or None
        Original start point (needed for limit line direction).
    prev_x, prev_y: float or None
        Previous position before current AUV position (for incoming turn angle).
    next_ref_x, next_ref_y: float or None
        Expected next_reference target after candidate point (for outgoing angle).
    enforce_flight_constraints: bool
        Whether to enforce missile segment constraints while selecting candidates.
    segment_mode: str
        One of {"first", "middle", "last"}.

    Returns
    -------
    best_point : TANPoint or None
        Best TAN-suitable point found, or None if none found.
    """
    block_size = params.N
    n_y, n_x = suitability_map.shape
    terrain_size = params.terrain_size

    center_angle = angle_between((auv_x, auv_y), (target_x, target_y))
    L_R = params.L_R
    L_r = params.l  # minimum radius = turning circle

    # Constrain L_R to not exceed L_max and also ensure L_R >= L_min
    L_R = min(L_R, params.L_max)
    L_R = max(L_R, params.L_min)

    if segment_mode == 'first':
        L_R = L_R + params.l_min
        L_r = L_r + params.l_min
        

    alpha = params.alpha  # half sector angle in degrees

    candidates = []

    for iy in range(n_y):
        for ix in range(n_x):
            if not suitability_map[iy, ix]:
                continue

            # Block center coordinates
            cx = ix * block_size + block_size / 2
            cy = iy * block_size + block_size / 2

            # Check if block center is within sector
            if not is_point_in_sector(cx, cy, auv_x, auv_y,
                                       center_angle, alpha, L_r, L_R):
                continue

            # Apply limit line constraint if near target
            if apply_limit_lines and start_x is not None:
                if not is_point_within_limit_lines(
                    cx, cy, target_x, target_y, start_x, start_y, params.beta
                ):
                    continue
            
            # Enforce flight constraints during planning stage
            if enforce_flight_constraints:
                seg_dist = distance((auv_x, auv_y), (cx, cy))
                a_in = (
                    compute_turn_angle_deg((prev_x, prev_y), (auv_x, auv_y), (cx, cy))
                    if prev_x is not None and prev_y is not None
                    else 0.0
                )
                a_out = (
                    compute_turn_angle_deg((auv_x, auv_y), (cx, cy), (next_ref_x, next_ref_y))
                    if next_ref_x is not None and next_ref_y is not None
                    else params.a_max_deg
                )

                if not is_segment_feasible_by_flight_constraints(
                    seg_dist=seg_dist,
                    params=params,
                    a_in_deg=a_in,
                    a_out_deg=a_out,
                    mode=segment_mode
                ):
                    continue

            entropy = entropy_map[iy, ix]
            dist_to_center_line = _distance_to_line(
                cx, cy, auv_x, auv_y, target_x, target_y
            )
            dist_to_auv = distance((cx, cy), (auv_x, auv_y))
            candidates.append((cx, cy, entropy, dist_to_center_line, dist_to_auv))

    if not candidates:
        return None

    # Selection: maximize entropy, then minimize lateral deviation
    # Score = entropy - weight * normalized_lateral_deviation
    entropies = np.array([c[2] for c in candidates])
    laterals = np.array([c[3] for c in candidates])

    # Normalize
    e_range = entropies.max() - entropies.min() + 1e-10
    l_range = laterals.max() - laterals.min() + 1e-10

    e_norm = (entropies - entropies.min()) / e_range
    l_norm = (laterals - laterals.min()) / l_range

    # Score: high entropy + low lateral deviation
    scores = e_norm - 0.8 * l_norm
    best_idx = int(np.argmax(scores))

    bx, by, bent, _, _ = candidates[best_idx]
    return TANPoint(x=bx, y=by, entropy=bent, params=(L_r, L_R, alpha))


def _distance_to_line(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """
    Perpendicular distance from point (px, py) to line through (x1,y1)-(x2,y2).
    """
    dx = x2 - x1
    dy = y2 - y1
    denom = np.sqrt(dx**2 + dy**2)
    if denom < 1e-10:
        return distance((px, py), (x1, y1))
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / denom


def compute_turn_angle_deg(
    p_prev: Tuple[float, float],
    p_curr: Tuple[float, float],
    p_next: Tuple[float, float],
) -> float:
    """Absolute heading change at waypoint p_curr."""
    v1 = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]], dtype=np.float64)
    v2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]], dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_a = float(np.dot(v1, v2) / (n1 * n2))
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def segment_constraint_min_distance(
    params: SectorParams,
    a_in_deg: float,
    a_out_deg: float,
    mode: str,
) -> float:
    """
    Compute minimum required segment distance by flight constraints.
    
    mode:
      - "first": d_1 = l_1 + R * tan(a_1 / 2), with l_1 >= l_min
      - "middle": d_{i+1} = R * (tan(a_i / 2) + tan(a_{i+1}/2)) + l_{i+1}
      - "last": not used for min-only form; see interval check in is_segment_feasible_by_flight_constrants()
    """
    a_in = np.radians(max(a_in_deg, 0.0))
    a_out = np.radians(max(a_out_deg, 0.0))
    if mode == 'first':
        return params.l_min + params.R * np.tan(a_out / 2.0)
    else:
        return params.R * (np.tan(a_in / 2.0) + np.tan(a_out / 2.0))


def is_segment_feasible_by_flight_constraints(
    seg_dist: float,
    params: SectorParams,
    a_in_deg: float,
    a_out_deg: float,
    mode: str = "middle",
) -> bool:
    """Check turn-angle and distance constraints for a segment."""
    if a_in_deg > params.a_max_deg or a_out_deg > params.a_max_deg:
        return False
    if mode == "last":
        offset = params.R * np.tan(np.radians(max(a_in_deg, 0.0)) / 2.0)
        d_lower = offset + params.TA_min
        d_upper = offset + params.TA_max
        return d_lower <= seg_dist <= d_upper
    required = segment_constraint_min_distance(params, a_in_deg, a_out_deg, mode)
    return seg_dist >= required


# ---------------------------------------------------------------------------
# Target-aided point search (Section 2.4)
# ---------------------------------------------------------------------------

def search_target_aided_point(
    target_x: float,
    target_y: float,
    start_x: float,
    start_y: float,
    entropy_map: np.ndarray,
    suitability_map: np.ndarray,
    params: SectorParams,
) -> Optional[TANPoint]:
    """
    Search for the target-aided point T_A near the target.

    The target-aided point is a TAN-suitable area within the annular region
    [|TA|, |TB|] from the target, within the limit angle beta from the
    start->target centerline (on the start side).

    From Eq.(14-16):
        |TA| = l + sigma + N/2
        |TB| = l + sigma + N
        |TA| <= L <= |TB|

    Parameters
    ----------
    target_x, target_y : float
        Target point coordinates.
    start_x, start_y : float
        Starting point coordinates.
    entropy_map : np.ndarray
        Block entropy map.
    suitability_map : np.ndarray
        Block suitability map.
    params : SectorParams
        Algorithm parameters.

    Returns
    -------
    aided_point : TANPoint or None
    """
    block_size = params.N
    n_y, n_x = suitability_map.shape

    TA_min = params.TA_min
    TA_max = params.TA_max

    candidates = []

    for iy in range(n_y):
        for ix in range(n_x):
            if not suitability_map[iy, ix]:
                continue

            cx = ix * block_size + block_size / 2
            cy = iy * block_size + block_size / 2

            # Distance from target
            dist = distance((cx, cy), (target_x, target_y))
            if dist < TA_min or dist > TA_max:
                continue

            # Must be within limit angle beta from backward direction
            if not is_point_within_limit_lines(
                cx, cy, target_x, target_y, start_x, start_y, params.beta
            ): 
                continue

            entropy = entropy_map[iy, ix]
            candidates.append((cx, cy, entropy, dist))

    if not candidates:
        # Relax distance constraint and try again with wider search
        return _search_target_aided_relaxed(
            target_x, target_y, start_x, start_y,
            entropy_map, suitability_map, params
        )

    # Select the candidate with highest entropy
    best = max(candidates, key=lambda c: c[2])
    return TANPoint(x=best[0], y=best[1], entropy=best[2], is_target_aided=True)


def _search_target_aided_relaxed(
    target_x: float,
    target_y: float,
    start_x: float,
    start_y: float,
    entropy_map: np.ndarray,
    suitability_map: np.ndarray,
    params: SectorParams,
) -> Optional[TANPoint]:
    """
    Relaxed search for target-aided point with expanded distance range.
    """
    block_size = params.N
    n_y, n_x = suitability_map.shape

    # Expand search range
    TA_min = params.l + params.N / 4
    TA_max = params.l + 2 * params.N

    dx_ts = start_x - target_x
    dy_ts = start_y - target_y
    backward_angle = np.degrees(np.arctan2(dy_ts, dx_ts))

    candidates = []
    for iy in range(n_y):
        for ix in range(n_x):
            if not suitability_map[iy, ix]:
                continue
            cx = ix * block_size + block_size / 2
            cy = iy * block_size + block_size / 2
            dist = distance((cx, cy), (target_x, target_y))
            if dist < TA_min or dist > TA_max:
                continue
            dx_tp = cx - target_x
            dy_tp = cy - target_y
            point_angle = np.degrees(np.arctan2(dy_tp, dx_tp))
            diff = (point_angle - backward_angle + 180) % 360 - 180
            if abs(diff) > 90:  # relaxed angle
                continue
            entropy = entropy_map[iy, ix]
            candidates.append((cx, cy, entropy, dist))

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c[2])
    return TANPoint(x=best[0], y=best[1], entropy=best[2], is_target_aided=True)


def search_dynamic_target_aided_point(
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
    start_x: float,
    start_y: float,
    entropy_map: np.ndarray,
    suitability_map: np.ndarray,
    params: SectorParams,
) -> Optional[TANPoint]:
    """
    The target-aided point is a TAN-suitable area within the annular region
    [|TA|, |TB|] from the target, within the limit angle beta from the
    start->target centerline (on the start side).

    Dynamic target-aided point search using constraint:
        d_n is distance from target-aided point to target, and

        R * tan(a_{n-1}/2) + |TA| <= d_n <= R * tan(a_{n-1}/2) + |TB|
    
    Here a_{n-1} is computed at candidate aided point with geometry: 
        current -> aided -> target

    Parameters
    ----------
    current_x, current_y: float
        Current point coordinates.
    target_x, target_y : float
        Target point coordinates.
    start_x, start_y : float
        Starting point coordinates.
    entropy_map : np.ndarray
        Block entropy map.
    suitability_map : np.ndarray
        Block suitability map.
    params : SectorParams
        Algorithm parameters.

    Returns
    -------
    aided_point : TANPoint or None
    """
    block_size = params.N
    n_y, n_x = suitability_map.shape

    TA_min = params.TA_min
    TA_max = params.TA_max

    candidates = []

    for iy in range(n_y):
        for ix in range(n_x):
            if not suitability_map[iy, ix]:
                continue

            cx = ix * block_size + block_size / 2
            cy = iy * block_size + block_size / 2

            # Distance from target
            a_last_deg = compute_turn_angle_deg(
                (current_x, current_y), (cx, cy), (target_x, target_y)
            )
            if a_last_deg >= params.a_max_deg:
                continue

            d_n = distance((cx, cy), (target_x, target_y))

            if not is_segment_feasible_by_flight_constraints(
                seg_dist=d_n, params=params, 
                a_in_deg=a_last_deg,
                a_out_deg=0.0,
                mode="last"
            ):
                continue

            # Must be within limit angle beta from backward direction
            if not is_point_within_limit_lines(
                cx, cy, target_x, target_y, start_x, start_y, params.beta
            ): 
                continue

            entropy = entropy_map[iy, ix]
            candidates.append((cx, cy, entropy, d_n))

    if not candidates:
        # Relax distance constraint and try again with wider search
        return None
    
    entropies = np.array([c[2] for c in candidates])
    laterals = np.array([c[3] for c in candidates])

    # Normalize
    e_range = entropies.max() - entropies.min() + 1e-10
    l_range = laterals.max() - laterals.min() + 1e-10

    e_norm = (entropies - entropies.min()) / e_range
    l_norm = (laterals - laterals.min()) / l_range

    # Score: high entropy + low lateral deviation
    scores = e_norm - 0.1 * l_norm
    best_idx = int(np.argmax(scores))

    bx, by, bent, _ = candidates[best_idx]

    return TANPoint(x=bx, y=by, entropy=bent, params=(params.l, params.L_R, params.alpha), is_target_aided=True)



# ---------------------------------------------------------------------------
# Sector angle analysis (Section 2.2)
# ---------------------------------------------------------------------------

def compute_alpha_min(k: float) -> float:
    """
    Compute minimum half sector angle alpha_min in degrees.
    Equation (5): alpha_min = arccot(sqrt(4k^2 - 1) - 2)

    Parameters
    ----------
    k : float
        Sector radius coefficient (L_R = k * N).

    Returns
    -------
    alpha_min : float (degrees)
    """
    if k < 0.25:
        return 90.0  # degenerate case
    val = np.sqrt(4 * k**2 - 1) - 2
    if val <= 0:
        return 90.0
    return float(np.degrees(np.arctan(1.0 / val)))  # arccot(x) = arctan(1/x)


def compute_alpha_max(k: float) -> float:
    """
    Compute maximum half sector angle alpha_max in degrees.
    Equation (8):
        alpha_max = arccos( (2k^2 - 3*sqrt(2)*k + 2 - 2*sqrt(2)) /
                            (2*(k-sqrt(2))*sqrt(k^2 - sqrt(2)*k + 1 - 2*sqrt(2))) ) + 45°

    Valid only when k > sqrt(2)/2 + sqrt(4*sqrt(2) - 1/2) ≈ 2.9780

    Parameters
    ----------
    k : float
        Sector radius coefficient.

    Returns
    -------
    alpha_max : float (degrees), capped at 45° for large k.
    """
    k_min = np.sqrt(2) / 2 + np.sqrt(4 * np.sqrt(2) - 0.5)  # ≈ 2.9780

    if k >= 100:
        return 45.0  # limit as k -> infinity

    if k < k_min:
        # For small k, alpha_max is not well-defined; use 45° as safe upper bound
        return 45.0

    sqrt2 = np.sqrt(2)
    numerator = 2 * k**2 - 3 * sqrt2 * k + 2 - 2 * sqrt2
    inner = k**2 - sqrt2 * k + 1 - 2 * sqrt2

    if inner <= 0:
        return 45.0

    denominator = 2 * (k - sqrt2) * np.sqrt(inner)

    if abs(denominator) < 1e-10:
        return 45.0

    cos_val = numerator / denominator
    cos_val = np.clip(cos_val, -1.0, 1.0)
    alpha_max = float(np.degrees(np.arccos(cos_val))) + 45.0
    return min(alpha_max, 80.7)  # paper states 45° < alpha_max < 80.7°


def print_sector_analysis(params: SectorParams):
    """Print sector angle analysis for given parameters."""
    alpha_min = compute_alpha_min(params.k)
    alpha_max = compute_alpha_max(params.k)
    print(f"\n{'='*50}")
    print(f"Sector Analysis (k={params.k}, N={params.N}m)")
    print(f"{'='*50}")
    print(f"  L_R = k × N = {params.k} × {params.N} = {params.L_R:.1f} m")
    print(f"  alpha_min = {alpha_min:.2f}°")
    print(f"  alpha_max = {alpha_max:.2f}°  (limit: 45°)")
    print(f"  Selected alpha = {params.alpha:.1f}°")
    print(f"  beta range: 45° <= {params.beta}° <= 90°")
    print(f"  sigma (INS error at L_R) = {params.sigma:.2f} m")
    print(f"  |TA| = {params.TA_min:.2f} m")
    print(f"  |TB| = {params.TA_max:.2f} m")
    print(f"  L_max = {params.L_max:.1f} m, L_min = {params.L_min:.1f} m")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Demonstrate sector angle analysis for various k values
    print("Sector angle analysis for various k values:")
    print(f"{'k':>6} | {'alpha_min':>10} | {'alpha_max':>10} | {'L_R (N=50)':>12}")
    print("-" * 50)
    for k in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]:
        a_min = compute_alpha_min(k)
        a_max = compute_alpha_max(k)
        L_R = k * 50
        print(f"{k:>6.1f} | {a_min:>9.2f}° | {a_max:>9.2f}° | {L_R:>10.1f} m")

    # Test with paper parameters
    params = SectorParams(N=50, k=2.0, alpha=45.0, beta=60.0,
                          L_max=100.0, L_min=40.0, l=10.0, p=0.05)
    print_sector_analysis(params)
