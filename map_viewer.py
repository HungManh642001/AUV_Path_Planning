"""
map_viewer.py
=============
Visualize DEM data in 2D and 3D.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Reuse functions written in dem_loader.py
from dem_loader import load_dem, merge_dems, DEMData

def plot_dem_2d(
    dem: DEMData, 
    cmap: str = "terrain",
    title: str = "2D Terrain Map",
    start_point: tuple[float, float] | None = None, 
    target_point: tuple[float, float] | None = None, 
    ax=None, 
    show: bool = True,
    use_pixels: bool = True
):
    """
    Plot the terrain as a contour map.

    Parameters:
    - use_pixels: If True, axes will be matrix row/col indices.
                  If False, axes will be geographic bounds.
    """
    if ax is None:
        shape = dem.shape
        figsize = (round(shape[1] / min(shape) * 10), round(shape[0] / min(shape) * 10))
        fig, ax = plt.subplots(figsize=figsize)

    lon0, lat0, lon1, lat1 = dem.bounds
    res_x, res_y = dem.resolution
    terrain = dem.array
    rows, cols = dem.shape

    # 1. Create coordinate grids based on mode
    if use_pixels:
        x = np.arange(cols)
        y = np.arange(rows)
        x_label = 'Columns (X)'
        y_label = 'Rows (Y)'
    else:
        x = np.linspace(lon0, lon1, cols)
        y = np.linspace(lat1, lat0, rows)
        x_label = 'Longitude'
        y_label = 'Latitude'

    x_mesh, y_mesh = np.meshgrid(x, y)

    # 2. Calculate levels safely
    min_val = np.nanmin(terrain)
    max_val = np.nanmax(terrain)
    levels = np.linspace(min_val, max_val, 30)

    # 3. Plot contours
    cf = ax.contourf(x_mesh, y_mesh, terrain, levels=levels, cmap=cmap, alpha=0.85)
    cs = ax.contour(x_mesh, y_mesh, terrain, levels=levels[::3], colors='k', linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label='Elevation (m)')

    # 4. Plot points
    if start_point is not None:
        ax.plot(
            start_point[0], start_point[1], 'g^', markersize=10,
            label=f'Start ({start_point[0]:.2f}, {start_point[1]:.2f})', zorder=5
        )
    if target_point is not None:
        ax.plot(
            target_point[0], target_point[1], 'r*', markersize=12,
            label=f'Target ({target_point[0]:.2f}, {target_point[1]:.2f})', zorder=5
        )

    # 5. Set labels and axes limits
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if use_pixels:
        ax.set_xlim(0, cols - 1)
        ax.set_ylim(rows - 1, 0)  # Invert Y axis to match image/matrix coordinates
    else:
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)

    if start_point or target_point:
        ax.legend(loc='upper right')
      
    def format_coord(x_val, y_val):
        if use_pixels:
            col, row = int(x_val), int(y_val)
            x_disp, y_disp = f"{int(x_val)}", f"{int(y_val)}"
        else:
            col = int((x_val - lon0) / res_x)
            row = int((lat1 - y_val) / res_y)
            x_disp, y_disp = f"{x_val:.5f}", f"{y_val:.5f}"

        if 0 <= row < rows and 0 <= col < cols:
            z = terrain[row, col]
            return f"X: {x_disp} | Y: {y_disp} | Elev: {z:.1f} m"
        return f"X: {x_disp} | Y: {y_disp}"

    ax.format_coord = format_coord

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_dem_3d(dem: DEMData, stride: int = 10, cmap: str = "terrain", title: str = "3D Terrain Map", ax=None, show=True, use_pixels: bool = True):
    """
    Plot DEM map in 3D.
    
    Parameters:
    - stride: Step size for downsampling.
    - use_pixels: If True, uses matrix row/column indices for X/Y axes.
    """
    if ax is None:
        shape = dem.shape
        figsize = (round(shape[1] / min(shape) * 10), round(shape[0] / min(shape) * 10))
        fig = plt.figure(figsize=(10, 20))
        ax = fig.add_subplot(111, projection='3d')
    
    rows, cols = dem.shape
    
    # 1. Create X and Y arrays based on mode
    if use_pixels:
        x = np.arange(cols)
        y = np.arange(rows)
        x_label = 'Columns (X)'
        y_label = 'Rows (Y)'
    else:
        lon0, lat0, lon1, lat1 = dem.bounds
        x = np.linspace(lon0, lon1, cols)
        y = np.linspace(lat1, lat0, rows)
        x_label = 'Longitude'
        y_label = 'Latitude'
        
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    # 2. Downsample data
    x_sub = x_mesh[::stride, ::stride]
    y_sub = y_mesh[::stride, ::stride]
    z_sub = dem.array[::stride, ::stride]
    
    # 3. Plot 3D surface
    surf = ax.plot_surface(
        x_sub, y_sub, z_sub, 
        cmap=cmap, 
        linewidth=0, 
        antialiased=False, 
        alpha=0.9
    )
    
    # 4. Set labels
    fig.colorbar(surf, ax=ax, label='Elevation (m)', shrink=0.5, aspect=10)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Elevation (m)')
    
    # If using pixels, invert Y axis to match matrix orientation
    if use_pixels:
        ax.set_ylim(rows - 1, 0)

    if show:
        ax.view_init(elev=45, azim=-45)
        plt.show()
      
    return ax
