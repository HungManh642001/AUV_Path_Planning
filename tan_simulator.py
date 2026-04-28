"""
tan_simulator.py
================
TAN (Terrain Aided Navigation) simulator using Particle Filter.

This module simulates:
1. INS error accumulation during AUV travel
2. Terrain scanning with MBES (Multi-Beam Echo Sounder)
3. Particle Filter-based terrain matching for position estimation
4. TAN location error computation

The particle filter approach follows Ref.[11] in the paper:
  Palmier et al., "Interacting Weighted Ensemble Kalman Filter applied to
  Underwater Terrain Aided Navigation", ACC 2021.

Simplified PF implementation:
  - Particles represent possible AUV positions
  - Likelihood = correlation between scanned terrain patch and prior map
  - Resampling when effective particle count drops below threshold
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import map_coordinates
from scipy.stats import norm


class INSSimulator:
    """
    Simulates INS (Inertial Navigation System) with accumulating error.

    The INS error grows proportionally to distance traveled (5% per paper).
    """

    def __init__(self, true_x: float, true_y: float, error_ratio: float = 0.05):
        """
        Parameters
        ----------
        true_x, true_y : float
            True initial position.
        error_ratio : float
            INS error as fraction of distance traveled (default 0.05 = 5%).
        """
        self.true_x = true_x
        self.true_y = true_y
        self.ins_x = true_x
        self.ins_y = true_y
        self.error_ratio = error_ratio
        self.total_distance = 0.0
        self.cumulative_error = 0.0

    def move_to(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """
        Move AUV to target position, accumulating INS error.

        Parameters
        ----------
        target_x, target_y : float
            True target position.

        Returns
        -------
        ins_x, ins_y : float
            INS-estimated position (with accumulated error).
        """
        # True movement
        dx = target_x - self.true_x
        dy = target_y - self.true_y
        dist = np.sqrt(dx**2 + dy**2)
        self.total_distance += dist

        # Update true position
        self.true_x = target_x
        self.true_y = target_y

        # INS error: random walk proportional to distance
        error_magnitude = self.error_ratio * dist
        angle = np.random.uniform(0, 2 * np.pi)
        self.ins_x += dx + error_magnitude * np.cos(angle)
        self.ins_y += dy + error_magnitude * np.sin(angle)

        self.cumulative_error = np.sqrt(
            (self.ins_x - self.true_x)**2 + (self.ins_y - self.true_y)**2
        )
        return self.ins_x, self.ins_y

    def reset_with_tan_fix(self, tan_x: float, tan_y: float):
        """
        Reset INS position using TAN fix.

        Parameters
        ----------
        tan_x, tan_y : float
            TAN-estimated position.
        """
        self.ins_x = tan_x
        self.ins_y = tan_y
        self.cumulative_error = np.sqrt(
            (tan_x - self.true_x)**2 + (tan_y - self.true_y)**2
        )


class ParticleFilterTAN:
    """
    Particle Filter for Terrain Aided Navigation.

    Estimates AUV position by matching scanned terrain patch against
    the prior terrain map.
    """

    def __init__(
        self,
        terrain: np.ndarray,
        n_particles: int = 500,
        scan_size: int = 20,
        noise_std: float = 0.3,
    ):
        """
        Parameters
        ----------
        terrain : np.ndarray, shape (H, W)
            Prior terrain map.
        n_particles : int
            Number of particles.
        scan_size : int
            Size of MBES scan area (20m × 20m per paper).
        noise_std : float
            Measurement noise standard deviation (0.3m or 0.5m per paper).
        """
        self.terrain = terrain
        self.n_particles = n_particles
        self.scan_size = scan_size
        self.noise_std = noise_std
        self.H, self.W = terrain.shape

    def scan_terrain(
        self,
        true_x: float,
        true_y: float,
        noise_std: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simulate MBES terrain scan at true position.

        Parameters
        ----------
        true_x, true_y : float
            True AUV position (center of scan).
        noise_std : float or None
            Measurement noise. Uses self.noise_std if None.

        Returns
        -------
        scan : np.ndarray, shape (scan_size, scan_size)
            Scanned terrain patch with noise.
        """
        if noise_std is None:
            noise_std = self.noise_std

        half = self.scan_size // 2
        x0 = int(np.clip(true_x - half, 0, self.W - self.scan_size))
        y0 = int(np.clip(true_y - half, 0, self.H - self.scan_size))

        scan = self.terrain[y0:y0 + self.scan_size, x0:x0 + self.scan_size].copy()
        scan += np.random.normal(0, noise_std, scan.shape)
        return scan

    def _extract_patch(self, cx: float, cy: float) -> Optional[np.ndarray]:
        """Extract terrain patch centered at (cx, cy)."""
        half = self.scan_size // 2
        x0 = cx - half
        y0 = cy - half

        if (x0 < 0 or y0 < 0 or
                x0 + self.scan_size > self.W or
                y0 + self.scan_size > self.H):
            return None

        # Use bilinear interpolation for sub-pixel accuracy
        xs = np.linspace(x0, x0 + self.scan_size - 1, self.scan_size)
        ys = np.linspace(y0, y0 + self.scan_size - 1, self.scan_size)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        coords = np.array([yy.ravel(), xx.ravel()])
        patch = map_coordinates(self.terrain, coords, order=1, mode='nearest')
        return patch.reshape(self.scan_size, self.scan_size)

    def _compute_likelihood(
        self,
        scan: np.ndarray,
        particle_x: float,
        particle_y: float,
    ) -> float:
        """
        Compute likelihood of scan given particle position.

        Uses normalized cross-correlation between scan and map patch.

        Parameters
        ----------
        scan : np.ndarray
            Observed terrain scan.
        particle_x, particle_y : float
            Particle position.

        Returns
        -------
        likelihood : float
        """
        patch = self._extract_patch(particle_x, particle_y)
        if patch is None:
            return 1e-10

        # Normalized cross-correlation
        scan_norm = scan - scan.mean()
        patch_norm = patch - patch.mean()

        scan_std = scan_norm.std()
        patch_std = patch_norm.std()

        if scan_std < 1e-6 or patch_std < 1e-6:
            return 1e-10

        ncc = np.mean(scan_norm * patch_norm) / (scan_std * patch_std)
        ncc = np.clip(ncc, -1.0, 1.0)

        # Convert NCC to likelihood: higher NCC = higher likelihood
        # Use Gaussian model: likelihood ~ exp(-(1-NCC)^2 / (2*sigma^2))
        sigma_ncc = 0.3
        likelihood = np.exp(-((1.0 - ncc)**2) / (2 * sigma_ncc**2))
        return max(likelihood, 1e-10)

    def estimate_position(
        self,
        ins_x: float,
        ins_y: float,
        true_x: float,
        true_y: float,
        ins_error_std: float = 5.0,
    ) -> Tuple[float, float, float]:
        """
        Estimate AUV position using particle filter TAN.

        Parameters
        ----------
        ins_x, ins_y : float
            INS-estimated position (used to initialize particles).
        true_x, true_y : float
            True AUV position (used to generate scan).
        ins_error_std : float
            Standard deviation of INS error for particle initialization.

        Returns
        -------
        est_x, est_y : float
            TAN-estimated position.
        tan_error : float
            Distance between TAN estimate and true position.
        """
        # Initialize particles around INS position
        particles_x = np.random.normal(ins_x, ins_error_std, self.n_particles)
        particles_y = np.random.normal(ins_y, ins_error_std, self.n_particles)

        # Clip to terrain bounds
        half = self.scan_size // 2
        particles_x = np.clip(particles_x, half, self.W - half - 1)
        particles_y = np.clip(particles_y, half, self.H - half - 1)

        # Get terrain scan at true position
        scan = self.scan_terrain(true_x, true_y)

        # Compute weights
        weights = np.array([
            self._compute_likelihood(scan, px, py)
            for px, py in zip(particles_x, particles_y)
        ])

        # Normalize weights
        weights_sum = weights.sum()
        if weights_sum < 1e-10:
            weights = np.ones(self.n_particles) / self.n_particles
        else:
            weights /= weights_sum

        # Effective particle count
        n_eff = 1.0 / np.sum(weights**2)

        # Resample if needed
        if n_eff < self.n_particles / 2:
            indices = np.random.choice(self.n_particles, self.n_particles,
                                       p=weights, replace=True)
            particles_x = particles_x[indices]
            particles_y = particles_y[indices]
            weights = np.ones(self.n_particles) / self.n_particles

        # Weighted mean estimate
        est_x = float(np.sum(weights * particles_x))
        est_y = float(np.sum(weights * particles_y))

        tan_error = float(np.sqrt((est_x - true_x)**2 + (est_y - true_y)**2))
        return est_x, est_y, tan_error


def simulate_tan_at_waypoint(
    waypoint_x: float,
    waypoint_y: float,
    ins_x: float,
    ins_y: float,
    terrain: np.ndarray,
    ins_error_ratio: float = 0.05,
    distance_traveled: float = 100.0,
    noise_std: float = 0.3,
    n_particles: int = 500,
    n_trials: int = 5,
) -> Tuple[float, float, float]:
    """
    Simulate TAN at a single waypoint and return average location error.

    Parameters
    ----------
    waypoint_x, waypoint_y : float
        True waypoint position.
    ins_x, ins_y : float
        INS-estimated position at waypoint.
    terrain : np.ndarray
        Prior terrain map.
    ins_error_ratio : float
        INS error ratio.
    distance_traveled : float
        Distance traveled since last TAN fix.
    noise_std : float
        MBES noise standard deviation.
    n_particles : int
        Number of PF particles.
    n_trials : int
        Number of Monte Carlo trials.

    Returns
    -------
    est_x, est_y : float
        Mean TAN-estimated position.
    mean_error : float
        Mean TAN location error.
    """
    pf = ParticleFilterTAN(terrain, n_particles=n_particles,
                           scan_size=20, noise_std=noise_std)

    ins_error_std = ins_error_ratio * distance_traveled

    errors = []
    est_xs, est_ys = [], []

    for _ in range(n_trials):
        est_x, est_y, err = pf.estimate_position(
            ins_x, ins_y, waypoint_x, waypoint_y,
            ins_error_std=ins_error_std
        )
        errors.append(err)
        est_xs.append(est_x)
        est_ys.append(est_y)

    mean_error = float(np.mean(errors))
    mean_est_x = float(np.mean(est_xs))
    mean_est_y = float(np.mean(est_ys))

    return mean_est_x, mean_est_y, mean_error


if __name__ == "__main__":
    from terrain_map import generate_synthetic_terrain
    import config

    terrain = generate_synthetic_terrain(size=500, seed=42)
    pf = ParticleFilterTAN(terrain, n_particles=config.n_particles, scan_size=config.scan_size, noise_std=0.3)

    # Test at a known position
    true_x, true_y = 250.0, 250.0
    ins_x, ins_y = 255.0, 248.0  # INS with small error

    est_x, est_y, err = pf.estimate_position(ins_x, ins_y, true_x, true_y,
                                              ins_error_std=5.0)
    print(f"True position:  ({true_x:.1f}, {true_y:.1f})")
    print(f"INS position:   ({ins_x:.1f}, {ins_y:.1f})")
    print(f"TAN estimate:   ({est_x:.1f}, {est_y:.1f})")
    print(f"TAN error:      {err:.2f} m")
