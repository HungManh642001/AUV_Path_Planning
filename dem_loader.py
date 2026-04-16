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


def load_dem(path: str | Path) -> DEMData:
    """Load DEM automatically based on extension (.tif/.tiff/.hgt)."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        return load_geotiff(p)
    if suffix == ".hgt":
        return load_hgt(p)

    raise ValueError(
        f"Unsupported DEM format '{suffix}'. Supported: .tif, .tiff, .hgt"
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Load DEM (.tiff/.hgt) and print metadata summary.")
    parser.add_argument("dem_path", type=str, help="Path to DEM file (.tif/.tiff/.hgt)")
    args = parser.parse_args()

    dem = load_dem(args.dem_path)
    print(json.dumps(dem.summary(), indent=2))
