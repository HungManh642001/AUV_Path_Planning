"""
dem_loader.py
=============
Load real DEM data for TERCOM/UAV workflows.

Supported formats:
- GeoTIFF (.tif, .tiff)
- SRTM HGT (.hgt)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class DEMData:
    """Container for loaded DEM raster and metadata."""

    array: np.ndarray
    transform: Optional[Affine]
    crs: Optional[str]
    nodata: Optional[float]
    bounds: Optional[Tuple[float, float, float, float]]
    resolution: Optional[Tuple[float, float]]
    source_path: str
    source_format: str

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape

    @property
    def min_elev(self) -> float:
        valid = self.valid_array()
        return float(np.min(valid)) if valid.size else float("nan")

    @property
    def max_elev(self) -> float:
        valid = self.valid_array()
        return float(np.max(valid)) if valid.size else float("nan")

    def valid_array(self) -> np.ndarray:
        arr = self.array
        if self.nodata is None:
            return arr[np.isfinite(arr)]
        return arr[np.isfinite(arr) & (arr != self.nodata)]

    def summary(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_format": self.source_format,
            "shape": self.shape,
            "crs": self.crs,
            "nodata": self.nodata,
            "bounds": self.bounds,
            "resolution": self.resolution,
            "min_elev_m": self.min_elev,
            "max_elev_m": self.max_elev,
        }


def _infer_hgt_grid_size(file_size_bytes: int) -> int:
    """Infer square HGT grid side from file size (2 bytes/sample)."""
    n_samples = file_size_bytes // 2
    side = int(np.sqrt(n_samples))
    if side * side != n_samples:
        raise ValueError(
            f"Invalid HGT file size {file_size_bytes} bytes (not square int16 grid)."
        )
    return side


def _parse_hgt_tile_latlon(stem: str) -> Tuple[float, float]:
    """
    Parse SRTM tile naming like N37W122.hgt -> origin lat/lon of SW corner.
    """
    if len(stem) < 7:
        raise ValueError(
            f"Invalid HGT tile name '{stem}'. Expected format like N37W122."
        )

    lat_hem = stem[0].upper()
    lon_hem = stem[3].upper()
    lat_deg = int(stem[1:3])
    lon_deg = int(stem[4:7])

    lat = float(lat_deg if lat_hem == "N" else -lat_deg)
    lon = float(lon_deg if lon_hem == "E" else -lon_deg)
    return lat, lon


def load_hgt(path: str | Path) -> DEMData:
    """
    Load SRTM .hgt file (big-endian int16).

    Notes
    -----
    - SRTM .hgt stores elevations in meters.
    - NoData value in SRTM is usually -32768.
    - Tile is usually 1201x1201 (3 arc-sec) or 3601x3601 (1 arc-sec).
    """
    p = Path(path)
    file_size = p.stat().st_size
    side = _infer_hgt_grid_size(file_size)

    arr = np.fromfile(p, dtype=">i2").reshape((side, side)).astype(np.float32)
    nodata = -32768.0

    # Georeference from tile naming (best effort)
    lat0, lon0 = _parse_hgt_tile_latlon(p.stem)
    # Pixel size in degree: tile spans 1° with (side-1) intervals
    step = 1.0 / (side - 1)
    transform = Affine.translation(lon0, lat0 + 1.0) * Affine.scale(step, -step)
    bounds = (lon0, lat0, lon0 + 1.0, lat0 + 1.0)

    return DEMData(
        array=arr,
        transform=transform,
        crs="EPSG:4326",
        nodata=nodata,
        bounds=bounds,
        resolution=(step, step),
        source_path=str(p),
        source_format="hgt",
    )


def load_geotiff(path: str | Path, band: int = 1) -> DEMData:
    """Load DEM from GeoTIFF (.tif/.tiff)."""
    p = Path(path)
    with rasterio.open(p) as ds:
        arr = ds.read(band).astype(np.float32)
        transform = ds.transform
        crs = ds.crs.to_string() if ds.crs else None
        nodata = float(ds.nodata) if ds.nodata is not None else None
        bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
        res = ds.res

    return DEMData(
        array=arr,
        transform=transform,
        crs=crs,
        nodata=nodata,
        bounds=bounds,
        resolution=(float(res[0]), float(res[1])),
        source_path=str(p),
        source_format="geotiff",
    )


def merge_dems(dem_list: list[DEMData], source_path: str | Path) -> DEMData:
    """
    Merge multiple DEMData objects into a single DEMData map.
    """
    if not dem_list:
        raise ValueError("DEM list is empty. Please provide at least 1 map tile.")

    # 1. Determine global bounding box
    min_left = min(d.bounds[0] for d in dem_list)
    min_bottom = min(d.bounds[1] for d in dem_list)
    max_right = max(d.bounds[2] for d in dem_list)
    max_top = max(d.bounds[3] for d in dem_list)

    # Use metadata from the first tile as reference
    base_dem = dem_list[0]
    res_x, res_y = base_dem.resolution
    nodata = base_dem.nodata
    crs = base_dem.crs

    # 2. Calculate total NumPy array dimensions
    # Use round() to avoid floating-point errors when dividing coordinates
    total_cols = int(round((max_right - min_left) / res_x)) + 1
    total_rows = int(round((max_top - min_bottom) / res_y)) + 1

    # 3. Initialize the merged array with nodata values
    merged_arr = np.full((total_rows, total_cols), nodata, dtype=np.float32)

    # 4. Calculate positions and paste each tile into the merged array
    for dem in dem_list:
        lon0, lat0, lon1, lat1 = dem.bounds
        side_y, side_x = dem.shape

        # Calculate starting column/row index for this tile on the merged array
        col_start = int(round((lon0 - min_left) / res_x))
        row_start = int(round((max_top - lat1) / res_y))

        # Paste the sub-array into the large array.
        # Edges of adjacent tiles will share the same index and overwrite safely.
        merged_arr[row_start : row_start + side_y, col_start : col_start + side_x] = dem.array

    # 5. Create new Transform matrix and Bounds
    merged_transform = Affine.translation(min_left, max_top) * Affine.scale(res_x, -res_y)
    merged_bounds = (min_left, min_bottom, max_right, max_top)

    return DEMData(
        array=merged_arr,
        transform=merged_transform,
        crs=crs,
        nodata=nodata,
        bounds=merged_bounds,
        resolution=(res_x, res_y),
        source_path=str(source_path),
        source_format="merged",
    )


def load_dem(path: str | Path) -> DEMData:
    """Load DEM automatically based on extension (.tif/.tiff/.hgt)."""
    p = Path(path)
    dem_files = list(p.rglob("*.hgt"))
    if not dem_files:
        print("No map file found to process.")
        exit(1)

    dem_list = []
    for file_path in dem_files:
        dem_list.append(load_hgt(file_path))

    if len(dem_list) > 1:
        final_dem = merge_dems(dem_list, source_path=p)
    else:
        final_dem = dem_list[0]

    return final_dem


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Load DEM (.tiff/.hgt) and print metadata summary.")
    parser.add_argument("dem_path", type=str, help="Path to DEM folder (.hgt)")
    args = parser.parse_args()

    dem = load_dem(args.dem_path)
    print(json.dumps(dem.summary(), indent=2))
