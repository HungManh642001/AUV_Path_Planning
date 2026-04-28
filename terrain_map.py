"""
terrain_map.py
==============
Synthetic bathymetric terrain map generator for AUV TAN path planning simulation.

The paper uses real seafloor terrain from Dalian, China (500m x 500m, 1m resolution).
Since we don't have the actual data, we generate a realistic synthetic terrain using
superimposed Gaussian hills/valleys and Perlin-like noise to mimic complex seabed topology.

The terrain is designed so that:
- Most of the map has complex terrain (high entropy) suitable for TAN
- Some flat areas exist (low entropy) that the path planner should avoid
- The terrain matches the visual appearance of Figure 13 in the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter


def generate_synthetic_terrain(
    size: int = 500,
    seed: int = 42,
    noise_coefficient: float = 0.3,
) -> np.ndarray:
    """
    Generate a synthetic 500x500 bathymetric terrain map (depth in meters).

    Parameters
    ----------
    size : int
        Grid size (size x size), default 500.
    seed : int
        Random seed for reproducibility.
    noise_coefficient : float
        Noise level added to terrain (0.3 m or 0.5 m as in the paper).

    Returns
    -------
    terrain : np.ndarray, shape (size, size)
        Terrain depth values (positive = deeper).
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)

    # Base terrain: large-scale undulation
    terrain = np.zeros((size, size))

    # Add several Gaussian hills/ridges to create complex topology
    hills = [
        # (cx, cy, amplitude, sigma_x, sigma_y, angle)
        (0.15, 0.85, 80,  0.12, 0.08, 30),
        (0.35, 0.70, 60,  0.10, 0.15, -20),
        (0.55, 0.55, 90,  0.14, 0.10, 45),
        (0.75, 0.40, 70,  0.09, 0.12, 10),
        (0.20, 0.40, 50,  0.08, 0.08, 0),
        (0.60, 0.20, 65,  0.11, 0.09, -30),
        (0.80, 0.75, 55,  0.10, 0.13, 60),
        (0.40, 0.25, 45,  0.07, 0.10, 15),
        (0.10, 0.60, 40,  0.06, 0.08, -45),
        (0.90, 0.15, 75,  0.12, 0.07, 25),
        (0.50, 0.90, 35,  0.08, 0.06, 0),
        (0.70, 0.60, 50,  0.09, 0.11, -15),
    ]

    for cx, cy, amp, sx, sy, angle_deg in hills:
        angle = np.radians(angle_deg)
        Xr = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
        Yr = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)
        terrain += amp * np.exp(-(Xr**2 / (2 * sx**2) + Yr**2 / (2 * sy**2)))

    # Add ridges (elongated features)
    ridges = [
        (0.3, 0.6, 40, 0.03, 0.25, 60),
        (0.7, 0.3, 35, 0.03, 0.20, -45),
        (0.5, 0.5, 30, 0.02, 0.30, 0),
    ]
    for cx, cy, amp, sx, sy, angle_deg in ridges:
        angle = np.radians(angle_deg)
        Xr = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
        Yr = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)
        terrain += amp * np.exp(-(Xr**2 / (2 * sx**2) + Yr**2 / (2 * sy**2)))

    # Add multi-scale random noise (simulates small-scale seabed roughness)
    for scale, amplitude in [(0.05, 15), (0.02, 8), (0.01, 4)]:
        noise = rng.standard_normal((size, size))
        sigma = scale * size
        noise = gaussian_filter(noise, sigma=sigma)
        noise = noise / noise.std() * amplitude
        terrain += noise

    # Add measurement noise
    terrain += rng.normal(0, noise_coefficient, (size, size))

    # Normalize to realistic depth range (e.g., 50-250 m depth)
    terrain = terrain - terrain.min()
    terrain = terrain / terrain.max() * 200 + 50

    return terrain.astype(np.float64)


def plot_terrain(terrain: np.ndarray, title: str = "A Priori Terrain Map",
                 start_point=None, target_point=None, ax=None, show=True):
    """
    Plot the terrain as a contour map similar to Figure 13 in the paper.

    Parameters
    ----------
    terrain : np.ndarray
        2D terrain depth array.
    title : str
        Plot title.
    start_point : tuple (x, y) or None
        Starting point to mark on the map.
    target_point : tuple (x, y) or None
        Target point to mark on the map.
    ax : matplotlib Axes or None
        Axes to plot on. If None, creates new figure.
    show : bool
        Whether to call plt.show().
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    # Contour filled plot
    levels = np.linspace(terrain.min(), terrain.max(), 30)
    cf = ax.contourf(terrain, levels=levels, cmap='terrain', alpha=0.85)
    cs = ax.contour(terrain, levels=levels[::3], colors='k', linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label='Depth (m)')

    if start_point is not None:
        ax.plot(start_point[0], start_point[1], 'g^', markersize=10,
                label=f'Start ({start_point[0]},{start_point[1]})', zorder=5)
    if target_point is not None:
        ax.plot(target_point[0], target_point[1], 'r*', markersize=12,
                label=f'Target ({target_point[0]},{target_point[1]})', zorder=5)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(0, terrain.shape[1])
    ax.set_ylim(0, terrain.shape[0])
    if start_point or target_point:
        ax.legend(loc='upper right')

    if show:
        plt.tight_layout()
        plt.show()

    return ax


if __name__ == "__main__":
    # terrain = generate_synthetic_terrain(size=500, seed=42, noise_coefficient=0.3)
    from dem_loader import load_dem
    dem = load_dem("SRTM")
    terrain = dem.array
    print(f"Terrain shape: {terrain.shape}")
    print(f"Depth range: {terrain.min():.2f} m ~ {terrain.max():.2f} m")
    plot_terrain(terrain, title="Synthetic Bathymetric Terrain")
