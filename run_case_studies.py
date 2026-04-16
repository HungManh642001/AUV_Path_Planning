"""
run_case_studies.py
===================
Run 4 TAN path-planning case studies, generate visualizations, and
export metrics/verification reports.

Supports two terrain sources:
- Synthetic terrain (legacy): generated in-code
- Real DEM (.tif/.tiff/.hgt): loaded via dem_loader.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from dem_loader import DEMData, load_dem
from path_planner import PathPlanningResult, TANPathPlanner
from sector_search import SectorParams
from terrain_map import generate_synthetic_terrain


@dataclass
class CaseStudyConfig:
    """Configuration for one case study."""

    name: str
    start_point: Tuple[float, float]
    target_point: Tuple[float, float]
    params: SectorParams
    noise_std: float = 0.3


@dataclass
class CaseStudyMetrics:
    """Serializable metrics for one case study output."""

    name: str
    start_point: Tuple[float, float]
    target_point: Tuple[float, float]
    total_waypoints: int
    total_distance: float
    max_tan_error: float
    mean_tan_error: float
    target_aided_point: Optional[Tuple[float, float]]


@dataclass
class TerrainSourceMeta:
    """Metadata for terrain source used in case-study run."""

    source_type: str
    source_path: Optional[str]
    original_shape: Tuple[int, int]
    used_shape: Tuple[int, int]


# NOTE: Replace with paper-exact cases once finalized.
def get_default_case_studies(terrain_size: int) -> List[CaseStudyConfig]:
    """Return 4 representative case studies for current terrain size."""
    # Keep same normalized layout as legacy 500x500 examples
    scale = terrain_size / 500.0

    def s(x: float, y: float) -> Tuple[float, float]:
        return (x * scale, y * scale)

    base_params = dict(
        N=max(20, int(round(50 * scale))),
        k=2.0,
        alpha=45.0,
        beta=60.0,
        L_max=100.0 * scale,
        L_min=40.0 * scale,
        l=10.0 * scale,
        p=0.05,
        terrain_size=terrain_size,
    )
    return [
        CaseStudyConfig(
            name="Case Study 1",
            start_point=s(60, 490),
            target_point=s(430, 10),
            params=SectorParams(**base_params),
            noise_std=0.3,
        ),
        CaseStudyConfig(
            name="Case Study 2",
            start_point=s(35, 460),
            target_point=s(460, 35),
            params=SectorParams(**{**base_params, "L_max": 90.0 * scale}),
            noise_std=0.3,
        ),
        CaseStudyConfig(
            name="Case Study 3",
            start_point=s(80, 420),
            target_point=s(420, 65),
            params=SectorParams(**{**base_params, "L_max": 110.0 * scale}),
            noise_std=0.3,
        ),
        CaseStudyConfig(
            name="Case Study 4",
            start_point=s(40, 440),
            target_point=s(450, 50),
            params=SectorParams(**{**base_params, "L_max": 120.0 * scale}),
            noise_std=0.5,
        ),
    ]


def _clean_dem_array(dem: DEMData) -> np.ndarray:
    """Replace nodata/NaN in DEM by median of valid pixels."""
    arr = dem.array.astype(np.float64).copy()
    invalid = ~np.isfinite(arr)
    if dem.nodata is not None:
        invalid |= arr == dem.nodata

    valid = arr[~invalid]
    fill_value = float(np.median(valid)) if valid.size else 0.0
    arr[invalid] = fill_value
    return arr


def _resample_to_square(arr: np.ndarray, target_size: int) -> np.ndarray:
    """Resample terrain to target_size x target_size for current planner grid logic."""
    if arr.shape == (target_size, target_size):
        return arr

    zy = target_size / arr.shape[0]
    zx = target_size / arr.shape[1]
    return zoom(arr, (zy, zx), order=1)


def _normalize_positive(arr: np.ndarray, out_min: float = 50.0, out_max: float = 250.0) -> np.ndarray:
    """Normalize terrain to positive range for entropy/likelihood stability."""
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if abs(amax - amin) < 1e-12:
        return np.full_like(arr, (out_min + out_max) / 2.0)

    norm = (arr - amin) / (amax - amin)
    return norm * (out_max - out_min) + out_min


def load_terrain_for_case_studies(
    dem_path: Optional[str],
    terrain_size: int,
    terrain_seed: int,
    noise_coefficient: float,
) -> Tuple[np.ndarray, TerrainSourceMeta]:
    """
    Load terrain from real DEM when --dem-path is provided; otherwise synthetic.
    """
    if dem_path:
        dem = load_dem(dem_path)
        cleaned = _clean_dem_array(dem)
        resampled = _resample_to_square(cleaned, terrain_size)
        terrain = _normalize_positive(resampled)

        meta = TerrainSourceMeta(
            source_type="dem",
            source_path=dem_path,
            original_shape=tuple(dem.array.shape),
            used_shape=tuple(terrain.shape),
        )
        return terrain.astype(np.float64), meta

    terrain = generate_synthetic_terrain(
        size=terrain_size,
        seed=terrain_seed,
        noise_coefficient=noise_coefficient,
    )
    meta = TerrainSourceMeta(
        source_type="synthetic",
        source_path=None,
        original_shape=tuple(terrain.shape),
        used_shape=tuple(terrain.shape),
    )
    return terrain.astype(np.float64), meta


def _draw_path(ax: plt.Axes, result: PathPlanningResult, color: str = "white") -> None:
    """Draw planned route including start, waypoints, and target."""
    xs = [result.start_point[0]] + [wp.x for wp in result.waypoints] + [result.target_point[0]]
    ys = [result.start_point[1]] + [wp.y for wp in result.waypoints] + [result.target_point[1]]

    ax.plot(xs, ys, "-o", color=color, linewidth=2.0, markersize=3.5, label="Planned path", zorder=8)
    ax.scatter(result.start_point[0], result.start_point[1], marker="^", s=120, c="lime", label="Start", zorder=9)
    ax.scatter(result.target_point[0], result.target_point[1], marker="*", s=170, c="red", label="Target", zorder=9)

    if result.target_aided_point is not None:
        ax.scatter(
            result.target_aided_point.x,
            result.target_aided_point.y,
            marker="D",
            s=80,
            c="cyan",
            label="Target-aided",
            zorder=9,
        )


def plot_case_result(
    terrain: np.ndarray,
    planner: TANPathPlanner,
    result: PathPlanningResult,
    output_path: Path,
    title: str,
) -> None:
    """Create a 2-panel figure: terrain+path and entropy suitability map."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Left panel: terrain + path
    levels = np.linspace(terrain.min(), terrain.max(), 26)
    cf = axes[0].contourf(terrain, levels=levels, cmap="terrain", alpha=0.88)
    axes[0].contour(terrain, levels=levels[::2], colors="k", linewidths=0.25, alpha=0.35)
    _draw_path(axes[0], result)
    axes[0].set_title(f"{title}\nDEM Terrain & Planned TAN Path")
    axes[0].set_xlabel("X (grid)")
    axes[0].set_ylabel("Y (grid)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.colorbar(cf, ax=axes[0], fraction=0.046, pad=0.03, label="Elevation-like value")

    # Right panel: entropy map + suitability + path
    ent = planner.entropy_map
    blk = planner.params.N
    ext = (0, ent.shape[1] * blk, 0, ent.shape[0] * blk)
    im = axes[1].imshow(
        ent,
        origin="lower",
        extent=ext,
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].contour(
        planner.suitability_map.astype(float),
        levels=[0.5],
        origin="lower",
        extent=ext,
        colors="white",
        linewidths=1.2,
    )
    _draw_path(axes[1], result, color="orange")
    axes[1].set_title(
        f"{title}\nEntropy blocks + suitable regions\nThreshold={planner.threshold:.4f}"
    )
    axes[1].set_xlabel("X (grid)")
    axes[1].set_ylabel("Y (grid)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03, label="Block entropy")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_case(
    terrain: np.ndarray,
    case: CaseStudyConfig,
    seed: int,
    out_dir: Path,
    verbose: bool,
) -> Tuple[PathPlanningResult, TANPathPlanner, CaseStudyMetrics]:
    """Run one case study and return result + planner + serializable metrics."""
    np.random.seed(seed)
    planner = TANPathPlanner(
        terrain=terrain,
        params=case.params,
        noise_std=case.noise_std,
        entropy_threshold=None,
        n_pf_particles=300,
        verbose=verbose,
    )

    result = planner.plan_path(
        start_x=case.start_point[0],
        start_y=case.start_point[1],
        target_x=case.target_point[0],
        target_y=case.target_point[1],
        max_iterations=50,
    )

    metrics = CaseStudyMetrics(
        name=case.name,
        start_point=case.start_point,
        target_point=case.target_point,
        total_waypoints=len(result.waypoints),
        total_distance=float(result.total_distance),
        max_tan_error=float(result.max_tan_error),
        mean_tan_error=float(result.mean_tan_error),
        target_aided_point=(result.target_aided_point.x, result.target_aided_point.y)
        if result.target_aided_point is not None
        else None,
    )

    fig_path = out_dir / f"{case.name.lower().replace(' ', '_')}.png"
    plot_case_result(terrain, planner, result, fig_path, title=case.name)

    return result, planner, metrics


def verify_results(metrics: List[CaseStudyMetrics]) -> Dict[str, Dict[str, str]]:
    """Rule-based verification (sanity checks + placeholders for paper values)."""
    checks: Dict[str, Dict[str, str]] = {}
    for m in metrics:
        case_checks = {
            "path_has_waypoint": "PASS" if m.total_waypoints >= 1 else "FAIL",
            "tan_error_reasonable": "PASS" if m.max_tan_error <= 80 else "WARN",
            "path_distance_nonzero": "PASS" if m.total_distance > 0 else "FAIL",
        }
        checks[m.name] = case_checks
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4 TAN path-planning case studies + plotting.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for plots and JSON outputs.")
    parser.add_argument("--dem-path", type=str, default=None, help="Path to DEM file (.tif/.tiff/.hgt).")
    parser.add_argument("--terrain-size", type=int, default=500, help="Terrain size used by planner (default 500).")
    parser.add_argument("--terrain-seed", type=int, default=42, help="Seed for synthetic terrain generation.")
    parser.add_argument("--noise-coef", type=float, default=0.3, help="Synthetic terrain noise coefficient.")
    parser.add_argument("--planner-seed", type=int, default=2026, help="Base seed for planner stochastic simulation.")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose planner logs.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    terrain, terrain_meta = load_terrain_for_case_studies(
        dem_path=args.dem_path,
        terrain_size=args.terrain_size,
        terrain_seed=args.terrain_seed,
        noise_coefficient=args.noise_coef,
    )

    print(
        "Terrain source:",
        terrain_meta.source_type,
        "| original_shape=", terrain_meta.original_shape,
        "| used_shape=", terrain_meta.used_shape,
        "| source_path=", terrain_meta.source_path,
    )

    cases = get_default_case_studies(terrain_size=args.terrain_size)
    all_metrics: List[CaseStudyMetrics] = []

    for i, case in enumerate(cases):
        seed = args.planner_seed + i
        print(f"\n{'=' * 72}")
        print(f"Running {case.name} | start={case.start_point} -> target={case.target_point} | seed={seed}")
        print(f"{'=' * 72}")

        _, _, metrics = run_case(
            terrain=terrain,
            case=case,
            seed=seed,
            out_dir=out_dir,
            verbose=not args.quiet,
        )
        all_metrics.append(metrics)

        print(
            f"{case.name}: waypoints={metrics.total_waypoints}, "
            f"distance={metrics.total_distance:.2f}m, "
            f"max_err={metrics.max_tan_error:.2f}m, mean_err={metrics.mean_tan_error:.2f}m"
        )

    checks = verify_results(all_metrics)

    metrics_json = out_dir / "case_study_metrics.json"
    checks_json = out_dir / "verification_report.json"
    config_json = out_dir / "case_study_configs.json"
    terrain_json = out_dir / "terrain_source.json"

    metrics_json.write_text(json.dumps([asdict(m) for m in all_metrics], indent=2), encoding="utf-8")
    checks_json.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    config_json.write_text(
        json.dumps(
            [
                {
                    "name": c.name,
                    "start_point": c.start_point,
                    "target_point": c.target_point,
                    "noise_std": c.noise_std,
                    "params": asdict(c.params),
                }
                for c in cases
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    terrain_json.write_text(json.dumps(asdict(terrain_meta), indent=2), encoding="utf-8")

    print(f"\nSaved metrics: {metrics_json}")
    print(f"Saved verification report: {checks_json}")
    print(f"Saved case study configs: {config_json}")
    print(f"Saved terrain source info: {terrain_json}")


if __name__ == "__main__":
    main()
