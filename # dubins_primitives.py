"""
dubins_primitives.py
====================
Dubins path primitives for non-holonomic vehicle.

A Dubins path is the shortest path for a vehicle with:
- Minimum turning radius R
- Bounded turn angle per unit time
- Can only go forward (not backward)

Three primitives: RSL (Right-Straight-Left), etc.
Simplified: compute via lookup table for 8 directions.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class DubinsSegment:
    """Represents one segment of a Dubins path."""
    x0: float
    y0: float
    theta0: float
    turn_direction: str  # 'L', 'S', 'R' (Left, Straight, Right)
    arc_length: float    # For curves, actual arc length
    
    def sample_points(self, num_samples: int = 50) -> np.ndarray:
        """Sample points along the segment."""
        if self.turn_direction == 'S':
            # Straight line
            xs = np.linspace(0, self.arc_length, num_samples)
            ys = np.zeros_like(xs)
        else:
            # Circular arc
            sign = 1 if self.turn_direction == 'L' else -1
            R = 300.0  # Turn radius (from flight constraints)
            theta_sweep = self.arc_length / R  # angle in radians
            
            # Arc in local frame, then rotate
            angles = np.linspace(0, theta_sweep, num_samples)
            xs = R * np.sin(angles)
            ys = R * (1 - np.cos(angles)) * sign
        
        # Rotate to global frame
        cos_theta = np.cos(self.theta0)
        sin_theta = np.sin(self.theta0)
        
        points_global = np.array([
            self.x0 + cos_theta * xs - sin_theta * ys,
            self.y0 + sin_theta * xs + cos_theta * ys,
        ]).T
        
        return points_global


class DubinsPathPlanner:
    """Generates Dubins paths between two states."""
    
    def __init__(self, R: float = 300.0):
        """
        Parameters
        ----------
        R : float
            Minimum turn radius (meters).
        """
        self.R = R
        
    def compute_dubins_path(
        self,
        x0: float, y0: float, theta0: float,
        x1: float, y1: float, theta1: float,
    ) -> Optional[Tuple[List[DubinsSegment], float]]:
        """
        Compute Dubins path from (x0,y0,theta0) to (x1,y1,theta1).
        
        Returns:
            (segments, total_length) or None if no feasible path
        """
        # Normalize angles to [-π, π]
        theta0 = self._normalize_angle(theta0)
        theta1 = self._normalize_angle(theta1)
        
        # Distance and relative angle
        dx = x1 - x0
        dy = y1 - y0
        D = np.sqrt(dx**2 + dy**2)
        
        if D < 0.01:  # Too close
            return None
        
        # Angle to target
        phi = np.arctan2(dy, dx)
        
        # Try all 6 standard Dubins primitives: LSL, LSR, RSL, RSR, RLR, LRL
        best_length = float('inf')
        best_path = None
        
        for primitive in ['LSL', 'LSR', 'RSL', 'RSR', 'RLR', 'LRL']:
            result = self._compute_primitive(
                D, theta0, theta1, phi, primitive
            )
            if result and result < best_length:
                best_length = result[1]
                best_path = result
        
        return best_path
    
    def _compute_primitive(
        self,
        D: float, theta0: float, theta1: float,
        phi: float, primitive: str
    ) -> Optional[Tuple[List[DubinsSegment], float]]:
        """Compute one Dubins primitive (LSL, RSL, etc)."""
        # Simplified implementation: scale-free Dubins
        # In practice, use proper Dubins equations from literature
        
        # This is a placeholder - full implementation would use
        # Dubins' original formulation
        if D < self.R:
            return None
        
        # Rough estimate for now
        total_length = D + self.R * abs(theta1 - theta0)
        
        if total_length > 1e6:  # Unrealistic
            return None
        
        # For simplicity, return synthetic segments
        segments = [
            DubinsSegment(
                x0=0, y0=0, theta0=theta0,
                turn_direction='L' if primitive[0] == 'L' else 'R',
                arc_length=self.R * abs(theta0) / 2,
            ),
            DubinsSegment(
                x0=0, y0=0, theta0=theta0 + np.pi/4,
                turn_direction='S',
                arc_length=D,
            ),
        ]
        
        return segments, total_length
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
