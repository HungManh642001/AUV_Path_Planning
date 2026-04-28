"""
map_viewer.py
=============
Visualize DEM data in 2D and 3D.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from dem_loader import load_dem, merge_dems, DEMData
from terrain_map import generate_synthetic_terrain


def plot_dem_2d(dem: DEMData, cmap: str = "terrain", title: str = "2D Terrain Map",
                start_point=None, target_point=None, ax=None, show=True, use_pixels=False):
    """Plot DEM map in 2D (top-down view)."""
    if ax is None:
        shape = dem.shape
        figsize = (round(shape[1] / min(shape) * 10), round(shape[0] / min(shape) * 10))
        fig, ax = plt.subplots(figsize=(7, 14))

    lon0, lat0, lon1, lat1 = dem.bounds
    res_x, res_y = dem.resolution
    rows, cols = dem.shape
    terrain = dem.array

    if use_pixels:
        extent = [0, cols, 0, rows]
        x = np.arange(cols)
        y = np.arange(rows)
        x_label = "Columns (X)"
        y_label = "Rows (Y)"
    else:
        lon0, lat0, lon1, lat1 = dem.bounds
        extent = [lon0, lon1, lat0, lat1]
        x = np.linspace(lon0, lon1, cols)
        y = np.linspace(lat1, lat0, rows)
        x_label = 'Longitude'
        y_label = 'Latitude'
    
    x_mesh, y_mesh = np.meshgrid(x, y)

    # land_masked = np.ma.masked_where(terrain <=0, terrain)
    # sea_masked = np.ma.masked_where(terrain > 0, terrain)
    # ax.imshow(sea_masked, cmap="terrain", extent=extent, origin='upper', vmin=-50, vmax=0)

    # land_vmin = max(0.1, np.nanmin(land_masked))
    # land_vmax = np.nanmax(land_masked)
    # img_land = ax.imshow(land_masked, cmap='YlOrBr_r', extent=extent, origin='upper', vmin=land_vmin, vmax=land_vmax)
    # Contour filled plot
    levels = np.linspace(terrain.min(), terrain.max(), 30)
    cf = ax.contourf(x_mesh, y_mesh, terrain, origin="lower", levels=levels, cmap="terrain", extent=extent, alpha=0.85)
    ax.contour(x_mesh, y_mesh, terrain, origin="lower", levels=levels[::3], extent=extent, colors='k', linewidths=0.4, alpha=0.5)
    ax.contour(x_mesh, y_mesh, terrain, origin="lower", levels=[0], extent=extent, colors='lightblue', linewidths=1, linestyles='solid')
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, label='Elevation (m)')

    if start_point is not None:
        ax.plot(start_point[0], start_point[1], 'g^', markersize=10,
                label=f'Start ({start_point[0]},{start_point[1]})', zorder=5)
    if target_point is not None:
        ax.plot(target_point[0], target_point[1], 'r*', markersize=12,
                label=f'Target ({target_point[0]},{target_point[1]})', zorder=5)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if use_pixels:
        ax.set_ylim(terrain.shape[0] - 1, 0)
    else:
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)
    if start_point or target_point:
        ax.legend(loc='upper right')

    def format_coord(x, y):
        if use_pixels:
            col, row = int(x), int(y)
            x_disp, y_disp = f"{int(x)}", f"{int(y)}"
        else:
            col = int((x - lon0) / res_x)
            row = int((lat1 - y) / res_y)
            x_disp, y_disp = f"{x:.5f}", f"{y:.5f}"

        if 0 <= row < rows and 0 <= col <cols:
            z = dem.array[row, col]
            return f"X: {x_disp} Y: {y_disp} Elev: {z:.1f} m"
        return f"X: {x_disp} Y: {y_disp}"
    
    ax.format_coord = format_coord

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_dem_3d(dem: DEMData, stride: int = 1, cmap: str = "terrain", title: str = "3D Terrain Map",
                start_point=None, target_point=None, ax=None, show=True, use_pixels=True):
    """
    Plot DEM map in 3D.
    
    Parameters:
    - stride: Step size for downsampling. stride=10 means 
              taking 1 point for every 10 pixels along each axis.
    """

    if ax is None:
        shape = dem.shape
        figsize = (round(shape[1] / min(shape) * 10), round(shape[0] / min(shape) * 10))
        fig = plt.figure(figsize=(20, 30))
        ax = fig.add_subplot(111, projection="3d")

    rows, cols = dem.shape

    if use_pixels:
        x = np.arange(cols)
        y = np.arange(rows)
        x_label = "Columns (X)"
        y_label = "Rows (Y)"
    else:
        lon0, lat0, lon1, lat1 = dem.bounds
        x = np.linspace(lon0, lon1, cols)
        y = np.linspace(lat1, lat0, rows)
        x_label = 'Longitude'
        y_label = 'Latitude'

    x_mesh, y_mesh = np.meshgrid(x, y)

    x_sub = x_mesh[::stride, ::stride]
    y_sub = y_mesh[::stride, ::stride]
    z_sub = dem.array[::stride, ::stride]

    surf = ax.plot_surface(
        x_sub, y_sub, z_sub,
        cmap=cmap,
        linewidth=0,
        edgecolor='black',
        antialiased=True,
        alpha=0.9
    )

    fig.colorbar(surf, ax=ax, label="Elevation (m)", shrink=0.5, aspect=10)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Elevation (m)')

    if use_pixels:
        ax.set_ylim(rows - 1, 0)
    else:
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)

    # Set the bounding box aspect ratio of the 3D plot
    extent = (cols, rows, max(cols, rows) * 0.3)
    ax.set_box_aspect(extent)

    if show:
        # Adjust initial viewing angle (elevation and azimuth)
        ax.view_init(elev=45, azim=-45)
        plt.show()

    return ax


def plot_dem_3d_with_suitability(dem: DEMData, suitability_map: np.ndarray, block_size: int = 50, stride: int = 1, cmap: str = "jet", title: str = "3D Terrain Map",
                start_point=None, target_point=None, ax=None, show=True, use_pixels=True):
    """
    Plot DEM map in 3D.
    
    Parameters:
    - stride: Step size for downsampling. stride=10 means 
              taking 1 point for every 10 pixels along each axis.
    """

    if ax is None:
        shape = dem.shape
        figsize = (round(shape[1] / min(shape) * 10), round(shape[0] / min(shape) * 10))
        fig = plt.figure(figsize=(20, 30))
        ax = fig.add_subplot(111, projection="3d")

    rows, cols = dem.shape
    lon0, lat0, lon1, lat1 = dem.bounds

    if use_pixels:
        x = np.arange(cols)
        y = np.arange(rows)
        x_label = "Columns (X)"
        y_label = "Rows (Y)"
    else:
        x = np.linspace(lon0, lon1, cols)
        y = np.linspace(lat1, lat0, rows)
        x_label = 'Longitude'
        y_label = 'Latitude'

    x_mesh, y_mesh = np.meshgrid(x, y)

    x_sub = x_mesh[::stride, ::stride]
    y_sub = y_mesh[::stride, ::stride]
    z_sub = dem.array[::stride, ::stride]

    if use_pixels:
        cols_sub = x_sub
        rows_sub = y_sub
    else:
        res_x, res_y = dem.resolution
        cols_sub = (x_sub - lon0) / res_x
        rows_sub = (lat1 - y_sub) / res_y
    
    n_y, n_x = suitability_map.shape
    ix_mat = (cols_sub // block_size).astype(int)
    iy_mat = (rows_sub // block_size).astype(int)
    ix_mat = np.clip(ix_mat, 0, n_x - 1)
    iy_mat = np.clip(iy_mat, 0, n_y - 1)

    suit_sub = suitability_map[iy_mat, ix_mat]

    face_colors = np.empty(z_sub.shape + (4,), dtype=float)

    color_true = mcolors.to_rgba('limegreen', alpha=0.9)
    color_false = mcolors.to_rgba('crimson', alpha=0.9)

    face_colors[suit_sub] = color_true
    face_colors[~suit_sub] = color_false

    surf = ax.plot_surface(
        x_sub, y_sub, z_sub,
        facecolors=face_colors,
        linewidth=0,
        edgecolor='black',
        antialiased=True,
        alpha=0.9
    )

    legend_elements = [
        Patch(facecolor='limegreen', edgecolor='black', label='Suitable (True)'),
        Patch(facecolor='crimson', edgecolor='black', label='Unsuitable (False)')
    ]

    ax.legend(handles=legend_elements, loc='upper right', title='TAN Suitability')

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Elevation (m)')

    if use_pixels:
        ax.set_ylim(rows - 1, 0)
    else:
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)

    # Set the bounding box aspect ratio of the 3D plot
    extent = (cols, rows, max(cols, rows) * 0.3)
    ax.set_box_aspect(extent)

    if show:
        # Adjust initial viewing angle (elevation and azimuth)
        ax.view_init(elev=45, azim=-45)
        plt.show()

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View DEM map in 2D or 3D.")
    parser.add_argument("--dem-path", type=str, help="Path to DEM folder (.hgt)")
    parser.add_argument("--3d", action="store_true", help="Display in 3D instead 2D", dest="show_3d")
    args = parser.parse_args()

    dem = load_dem(args.dem_path)
    import config
    from tan_suitability import build_suitability_map
    _, suitability_map, _ = build_suitability_map(
        dem.array, 
        block_size=config.N,
        entropy_threshold=config.entropy_threshold
    )
    if args.show_3d:
        plot_dem_3d_with_suitability(dem, suitability_map, block_size=config.N, stride=1)
    else:
        plot_dem_2d(dem)
