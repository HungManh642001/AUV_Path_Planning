"""
tan_suitability.py
==================
Terrain suitability evaluation for TAN (Terrain Aided Navigation).

This module implements:
1. Terrain Elevation Entropy (HM_c) calculation - Equation (1) in the paper
2. Terrain Standard Deviation (TSD) calculation
3. Block-wise suitability map generation
4. Threshold determination (median of all block entropies)

From the paper:
    P(i,j) = h(i,j) / sum_all(h(i,j))
    HM_c   = -sum_all( P(i,j) * log(P(i,j)) )

    A TAN-suitable area must have HM_c > threshold (median of all block entropies).
    Also TSD > 0.08702 (from Ref [19] in the paper).

The paper uses:
    - Block size N×N = 50m × 50m
    - Critical entropy for noise=0.3m: 7.8055
    - Critical entropy for noise=0.5m: 7.7246
"""

import numpy as np
from typing import Tuple, List, Optional


def compute_block_entropy(block: np.ndarray) -> float:
    """
    Compute terrain elevation entropy HM_c for a single N×N block.

    Equation (1) from the paper:
        P(i,j) = h(i,j) / sum(h(i,j))
        HM_c   = -sum( P(i,j) * log(P(i,j)) )

    Note: h(i,j) must be positive (depth values). We shift if needed.

    Parameters
    ----------
    block : np.ndarray, shape (N, N)
        Terrain depth values in the block.

    Returns
    -------
    entropy : float
        Terrain elevation entropy HM_c.
    """
    h = block.flatten().astype(np.float64)

    # Ensure all values are positive (shift if necessary)
    if h.min() <= 0:
        h = h - h.min() + 1e-10

    total = h.sum()
    if total <= 0:
        return 0.0

    P = h / total
    # Avoid log(0)
    P = P[P > 0]
    entropy = -np.sum(P * np.log(P))
    return float(entropy)


def compute_block_tsd(block: np.ndarray) -> float:
    """
    Compute Terrain Standard Deviation (TSD) for a block.

    Parameters
    ----------
    block : np.ndarray
        Terrain depth values.

    Returns
    -------
    tsd : float
        Standard deviation of terrain depths.
    """
    return float(np.std(block))


def build_suitability_map(
    terrain: np.ndarray,
    block_size: int = 50,
    entropy_threshold: Optional[float] = None,
    tsd_threshold: float = 0.08702,
    use_median_threshold: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build a block-wise TAN suitability map over the entire terrain.

    The terrain is divided into non-overlapping N×N blocks. For each block,
    the terrain elevation entropy HM_c is computed. A block is TAN-suitable
    if HM_c > threshold (median of all block entropies, as stated in the paper).

    Parameters
    ----------
    terrain : np.ndarray, shape (H, W)
        Full terrain depth map.
    block_size : int
        Size of each block N (default 50m).
    entropy_threshold : float or None
        If provided, use this as the entropy threshold. Otherwise use median.
    tsd_threshold : float
        Minimum TSD for TAN suitability (default 0.08702 from paper).
    use_median_threshold : bool
        If True, use median of all block entropies as threshold.

    Returns
    -------
    entropy_map : np.ndarray, shape (n_blocks_y, n_blocks_x)
        Entropy value for each block.
    suitability_map : np.ndarray, shape (n_blocks_y, n_blocks_x), dtype bool
        True where the block is TAN-suitable.
    threshold : float
        The entropy threshold used.
    """
    H, W = terrain.shape
    n_y = H // block_size
    n_x = W // block_size

    entropy_map = np.zeros((n_y, n_x), dtype=np.float64)
    tsd_map = np.zeros((n_y, n_x), dtype=np.float64)

    for iy in range(n_y):
        for ix in range(n_x):
            block = terrain[iy * block_size:(iy + 1) * block_size,
                            ix * block_size:(ix + 1) * block_size]
            entropy_map[iy, ix] = compute_block_entropy(block)
            tsd_map[iy, ix] = compute_block_tsd(block)

    # Determine threshold
    if entropy_threshold is not None:
        threshold = entropy_threshold
    elif use_median_threshold:
        threshold = float(np.median(entropy_map))
    else:
        threshold = 0.0

    # A block is TAN-suitable if entropy > threshold AND TSD > tsd_threshold
    suitability_map = (entropy_map > threshold) & (tsd_map > tsd_threshold)

    return entropy_map, suitability_map, threshold


def get_block_center_coords(
    block_idx_x: int,
    block_idx_y: int,
    block_size: int = 50,
) -> Tuple[float, float]:
    """
    Get the pixel/meter coordinates of the center of a block.

    Parameters
    ----------
    block_idx_x : int
        Block column index.
    block_idx_y : int
        Block row index.
    block_size : int
        Block size in meters/pixels.

    Returns
    -------
    cx, cy : float, float
        Center coordinates (x, y) in the terrain grid.
    """
    cx = block_idx_x * block_size + block_size / 2
    cy = block_idx_y * block_size + block_size / 2
    return cx, cy


def get_block_index_from_coord(
    x: float,
    y: float,
    block_size: int = 50,
    terrain_size: int = 500,
) -> Tuple[int, int]:
    """
    Get block indices (ix, iy) from continuous coordinates (x, y).

    Parameters
    ----------
    x, y : float
        Coordinates in the terrain grid.
    block_size : int
        Block size.
    terrain_size : int
        Total terrain size.

    Returns
    -------
    ix, iy : int, int
        Block column and row indices.
    """
    n_blocks = terrain_size // block_size
    ix = int(np.clip(x // block_size, 0, n_blocks - 1))
    iy = int(np.clip(y // block_size, 0, n_blocks - 1))
    return ix, iy


def get_entropy_at_coord(
    x: float,
    y: float,
    entropy_map: np.ndarray,
    block_size: int = 50,
    terrain_size: int = 500,
) -> float:
    """
    Get the entropy value at a given coordinate.

    Parameters
    ----------
    x, y : float
        Coordinates in the terrain grid.
    entropy_map : np.ndarray
        Block entropy map.
    block_size : int
        Block size.
    terrain_size : int
        Total terrain size.

    Returns
    -------
    entropy : float
    """
    ix, iy = get_block_index_from_coord(x, y, block_size, terrain_size)
    return float(entropy_map[iy, ix])


def find_all_suitable_blocks(
    suitability_map: np.ndarray,
    entropy_map: np.ndarray,
    block_size: int = 50,
) -> List[Tuple[float, float, float]]:
    """
    Return list of (cx, cy, entropy) for all TAN-suitable blocks.

    Parameters
    ----------
    suitability_map : np.ndarray, shape (n_y, n_x), dtype bool
    entropy_map : np.ndarray, shape (n_y, n_x)
    block_size : int

    Returns
    -------
    suitable_blocks : list of (cx, cy, entropy)
    """
    suitable_blocks = []
    n_y, n_x = suitability_map.shape
    for iy in range(n_y):
        for ix in range(n_x):
            if suitability_map[iy, ix]:
                cx, cy = get_block_center_coords(ix, iy, block_size)
                suitable_blocks.append((cx, cy, entropy_map[iy, ix]))
    return suitable_blocks


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from terrain_map import generate_synthetic_terrain
    from dem_loader import load_dem 

    # terrain = generate_synthetic_terrain(size=500, seed=42, noise_coefficient=0.3)
    dem_data = load_dem("data")
    terrain = dem_data.array
    entropy_map, suitability_map, threshold = build_suitability_map(
        terrain, block_size=50
    )

    print(f"Entropy map shape: {entropy_map.shape}")
    print(f"Entropy range: {entropy_map.min():.4f} ~ {entropy_map.max():.4f}")
    print(f"Threshold (median): {threshold:.4f}")
    print(f"Number of suitable blocks: {suitability_map.sum()} / {suitability_map.size}")

    W, H = terrain.shape
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    # Entropy map
    im = axes[0].imshow(entropy_map, origin='lower', cmap='RdYlGn',
                        extent=[0, H, 0, W])
    plt.colorbar(im, ax=axes[0], label='Terrain Elevation Entropy')
    axes[0].set_title(f'Block Entropy Map (threshold={threshold:.4f})')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')

    # Suitability map
    im2 = axes[1].imshow(suitability_map.astype(int), origin='lower',
                         cmap='RdYlGn', extent=[0, H, 0, W], vmin=0, vmax=1)
    plt.colorbar(im2, ax=axes[1], label='TAN Suitable (1=Yes, 0=No)')
    axes[1].set_title('TAN Suitability Map')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')

    plt.tight_layout()
    plt.savefig('suitability_map.png', dpi=150)
    plt.show()
    print("Saved suitability_map.png")
