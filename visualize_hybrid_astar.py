"""
Visualize Hybrid A* paths with Dubins curves.
"""

import matplotlib.pyplot as plt
import numpy as np
from hybrid_astar_planner import HybridAStarPlanner
from sector_search import SectorParams
from terrain_map import generate_synthetic_terrain


def plot_hybrid_path(result, terrain, entropy_map, suitability_map, figsize=(16, 6)):
    """Plot terrain, suitability, and Hybrid A* path."""
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    H, W = terrain.shape
    
    # --- Plot 1: Terrain ---
    im0 = axes[0].imshow(terrain, cmap='terrain', origin='lower')
    
    # Plot waypoints
    if result.waypoints:
        wp_xs = [wp.x for wp in result.waypoints]
        wp_ys = [wp.y for wp in result.waypoints]
        axes[0].plot(wp_xs, wp_ys, 'r.-', markersize=8, label='Waypoints', linewidth=2)
    
    axes[0].plot(*result.start_point, 'go', markersize=12, label='Start')
    axes[0].plot(*result.target_point, 'r*', markersize=20, label='Target')
    
    axes[0].set_title('Terrain Elevation with Hybrid A* Path')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].legend()
    axes[0].set_ylim(H-1, 0)
    fig.colorbar(im0, ax=axes[0], label='Elevation (m)')
    
    # --- Plot 2: Entropy Map ---
    im1 = axes[1].imshow(entropy_map, cmap='RdYlGn', origin='lower')
    
    if result.waypoints:
        wp_xs = [wp.x for wp in result.waypoints]
        wp_ys = [wp.y for wp in result.waypoints]
        axes[1].plot(wp_xs, wp_ys, 'b.-', markersize=8, label='Hybrid A* Path', linewidth=2)
    
    axes[1].set_title('Block Entropy (Terrain Characterization)')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].legend()
    axes[1].set_ylim(entropy_map.shape[0]-1, 0)
    fig.colorbar(im1, ax=axes[1], label='Entropy')
    
    # --- Plot 3: Suitability Map ---
    im2 = axes[2].imshow(
        suitability_map.astype(int),
        cmap='RdYlGn',
        origin='lower',
        vmin=0, vmax=1
    )
    
    if result.waypoints:
        wp_xs = [wp.x for wp in result.waypoints]
        wp_ys = [wp.y for wp in result.waypoints]
        axes[2].plot(wp_xs, wp_ys, 'b.-', markersize=8, label='Hybrid A* Path', linewidth=2)
    
    axes[2].set_title('TAN Suitability (Green = Suitable)')
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Y (m)')
    axes[2].legend()
    axes[2].set_ylim(suitability_map.shape[0]-1, 0)
    fig.colorbar(im2, ax=axes[2], label='Suitable (1=Yes, 0=No)')
    
    plt.tight_layout()
    return fig


# Main
if __name__ == "__main__":
    from run_hybrid_astar import hybrid_planner, result, terrain
    from tan_suitability import build_suitability_map
    
    _, suitability_map, _ = build_suitability_map(
        terrain, block_size=100
    )
    
    fig = plot_hybrid_path(
        result,
        terrain,
        hybrid_planner.entropy_map,
        suitability_map,
    )
    
    plt.savefig('hybrid_astar_path.png', dpi=150, bbox_inches='tight')
    print("Saved: hybrid_astar_path.png")
    plt.show()
