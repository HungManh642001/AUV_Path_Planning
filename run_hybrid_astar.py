"""
Demo: Run Hybrid A* path planner with Dubins curves.
"""

import numpy as np
from sector_search import SectorParams
from hybrid_astar_planner import HybridAStarPlanner
from terrain_map import generate_synthetic_terrain
from dem_loader import load_dem

# Setup
np.random.seed(42)

# Load terrain
try:
    dem = load_dem("SRTM")
    terrain = dem.array
except:
    terrain = generate_synthetic_terrain(size=500, seed=42, noise_coefficient=0.3)

print(f"Terrain shape: {terrain.shape}")

# Parameters (from Table 6 in paper)
params = SectorParams(
    N=100,
    k=5.0,
    alpha=45.0,
    beta=60.0,
    L_max=1000.0,
    L_min=400.0,
    l=300.0,
    p=0.01,
    a_max_deg=90.0,
    R=300.0,
    d_ss=300.0,
)

# Create Hybrid A* planner
hybrid_planner = HybridAStarPlanner(
    terrain=terrain,
    params=params,
    turn_radius=300.0,
    heading_resolution_deg=10.0,  # 36 heading angles
    verbose=True,
)

# Plan path
print("\n" + "="*60)
print("Hybrid A* Path Planning")
print("="*60)

result = hybrid_planner.plan_path(
    start_x=60.0,
    start_y=490.0,
    start_theta=0.0,  # North
    target_x=430.0,
    target_y=10.0,
    max_iterations=10000,
)

print("\n" + "="*60)
print("Result Summary")
print("="*60)
print(f"Total waypoints: {len(result.waypoints)}")
print(f"Total distance: {result.total_distance:.1f} m")
print(f"Number of Dubins curves: {len(result.waypoints)}")

# Compare with standard A*
print("\n" + "="*60)
print("Comparison: Standard A* vs Hybrid A*")
print("="*60)

from astar_planner import AStarTerrainPlanner

standard_planner = AStarTerrainPlanner(
    terrain=terrain,
    params=params,
    verbose=True,
)

standard_result = standard_planner.plan_path(
    start_x=60.0,
    start_y=490.0,
    target_x=430.0,
    target_y=10.0,
)

print(f"\nStandard A*:")
print(f"  Waypoints: {len(standard_result.waypoints)}")
print(f"  Distance: {standard_result.total_distance:.1f} m")

print(f"\nHybrid A* (with Dubins):")
print(f"  Waypoints: {len(result.waypoints)}")
print(f"  Distance: {result.total_distance:.1f} m")
print(f"  Advantage: Guaranteed continuous, non-holonomic feasible paths!")
