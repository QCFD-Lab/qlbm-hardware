"""Velocity-profile analysis from velocity-resolved QLBM counts."""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from qlbm.lattice.spacetime.properties_base import LatticeDiscretizationProperties
from .count_decoding import decode_grid_velocity_count, grid_shape


def velocity_vectors_and_masses(lattice) -> tuple[np.ndarray, np.ndarray]:
    """Return the velocity vectors and channel masses for the lattice discretization."""

    if not hasattr(lattice, "discretization"):
        raise ValueError("Velocity analysis requires a lattice with discretization.")

    velocity_vectors = np.asarray(
        LatticeDiscretizationProperties.get_velocity_vectors(lattice.discretization),
        dtype=float,
    )
    channel_masses = np.asarray(
        LatticeDiscretizationProperties.get_channel_masses(lattice.discretization),
        dtype=float,
    )
    if velocity_vectors.ndim != 2 or velocity_vectors.shape[1] != 2:
        raise ValueError("Velocity analysis currently expects 2D velocity vectors.")
    if channel_masses.shape != (velocity_vectors.shape[0],):
        raise ValueError(
            "Channel masses must have one entry per velocity vector: "
            f"velocities={velocity_vectors.shape[0]}, masses={channel_masses.shape}."
        )
    return velocity_vectors, channel_masses


def counts_to_velocity_fields(
    counts: Mapping[str, int | float],
    lattice,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Decode ABGridMeasurement(..., measure_velocity_qubits=True) counts."""

    width, height = grid_shape(lattice)
    velocity_vectors, channel_masses = velocity_vectors_and_masses(lattice)
    num_velocity_channels = velocity_vectors.shape[0]

    populations = np.zeros((width, height, num_velocity_channels), dtype=float)
    invalid_velocity_xy = np.zeros((width, height), dtype=float)
    valid_raw_mass = 0.0
    invalid_velocity_raw_mass = 0.0

    for count_key, count_value in counts.items():
        decoded = decode_grid_velocity_count(count_key, lattice)
        if decoded.x >= width or decoded.y >= height:
            continue
        if decoded.velocity < num_velocity_channels:
            populations[decoded.x, decoded.y, decoded.velocity] += float(count_value)
            valid_raw_mass += float(count_value)
        else:
            invalid_velocity_xy[decoded.x, decoded.y] += float(count_value)
            invalid_velocity_raw_mass += float(count_value)

    if normalize:
        total = float(sum(counts.values()))
        if total > 0:
            populations /= total
            invalid_velocity_xy /= total

    rho_xy = (populations * channel_masses).sum(axis=2)
    momentum_x = (populations * velocity_vectors[:, 0]).sum(axis=2)
    momentum_y = (populations * velocity_vectors[:, 1]).sum(axis=2)
    ux_xy = np.divide(
        momentum_x, rho_xy, out=np.zeros_like(momentum_x), where=rho_xy > 0
    )
    uy_xy = np.divide(
        momentum_y, rho_xy, out=np.zeros_like(momentum_y), where=rho_xy > 0
    )

    return {
        "populations": populations,
        "rho_xy": rho_xy,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
        "ux_xy": ux_xy,
        "uy_xy": uy_xy,
        "invalid_velocity_xy": invalid_velocity_xy,
        "valid_velocity_raw_mass": np.array(valid_raw_mass),
        "invalid_velocity_raw_mass": np.array(invalid_velocity_raw_mass),
        "total_raw_mass": np.array(float(sum(counts.values()))),
        "velocity_vectors": velocity_vectors,
        "channel_masses": channel_masses,
    }


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def velocity_profiles(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute velocity profiles along x and y for both velocity components."""

    ux_xy = fields["ux_xy"]
    uy_xy = fields["uy_xy"]
    rho_xy = fields["rho_xy"]
    momentum_x = fields["momentum_x"]
    momentum_y = fields["momentum_y"]

    rho_x = rho_xy.sum(axis=1)
    rho_y = rho_xy.sum(axis=0)
    x_mid = ux_xy.shape[0] // 2
    y_mid = ux_xy.shape[1] // 2
    return {
        "ux_x_density_weighted": _safe_divide(momentum_x.sum(axis=1), rho_x),
        "uy_x_density_weighted": _safe_divide(momentum_y.sum(axis=1), rho_x),
        "ux_y_density_weighted": _safe_divide(momentum_x.sum(axis=0), rho_y),
        "uy_y_density_weighted": _safe_divide(momentum_y.sum(axis=0), rho_y),
        "ux_x_mean": ux_xy.mean(axis=1),
        "uy_x_mean": uy_xy.mean(axis=1),
        "ux_y_mean": ux_xy.mean(axis=0),
        "uy_y_mean": uy_xy.mean(axis=0),
        "ux_centerline_x": ux_xy[:, y_mid],
        "uy_centerline_x": uy_xy[:, y_mid],
        "ux_centerline_y": ux_xy[x_mid, :],
        "uy_centerline_y": uy_xy[x_mid, :],
    }


def compare_velocity_fields(
    baseline_counts: Mapping[str, int | float],
    noisy_counts: Mapping[str, int | float],
    lattice,
) -> dict[str, Any]:
    """Compare velocity fields and profiles against a noiseless baseline."""

    baseline_fields = counts_to_velocity_fields(baseline_counts, lattice)
    noisy_fields = counts_to_velocity_fields(noisy_counts, lattice)
    baseline_profiles = velocity_profiles(baseline_fields)
    noisy_profiles = velocity_profiles(noisy_fields)

    ux_diff = noisy_fields["ux_xy"] - baseline_fields["ux_xy"]
    uy_diff = noisy_fields["uy_xy"] - baseline_fields["uy_xy"]
    rho_diff = noisy_fields["rho_xy"] - baseline_fields["rho_xy"]
    profile_metrics = {}
    for profile_name, baseline_profile in baseline_profiles.items():
        noisy_profile = noisy_profiles[profile_name]
        difference = noisy_profile - baseline_profile
        baseline_norm = float(np.linalg.norm(baseline_profile, ord=2))
        difference_norm = float(np.linalg.norm(difference, ord=2))
        profile_metrics[f"{profile_name}_l2_error"] = difference_norm
        profile_metrics[f"{profile_name}_relative_l2_error"] = (
            difference_norm / baseline_norm if baseline_norm > 0 else 0.0
        )
        profile_metrics[f"{profile_name}_linf_error"] = float(
            np.max(np.abs(difference))
        )

    baseline_total_raw_mass = float(baseline_fields["total_raw_mass"])
    noisy_total_raw_mass = float(noisy_fields["total_raw_mass"])
    baseline_invalid_velocity_raw_mass = float(
        baseline_fields["invalid_velocity_raw_mass"]
    )
    noisy_invalid_velocity_raw_mass = float(noisy_fields["invalid_velocity_raw_mass"])

    return {
        "baseline_fields": baseline_fields,
        "noisy_fields": noisy_fields,
        "baseline_profiles": baseline_profiles,
        "noisy_profiles": noisy_profiles,
        "metrics": {
            "baseline_raw_mass": float(sum(baseline_counts.values())),
            "noisy_raw_mass": float(sum(noisy_counts.values())),
            "baseline_valid_velocity_raw_mass": float(
                baseline_fields["valid_velocity_raw_mass"]
            ),
            "noisy_valid_velocity_raw_mass": float(
                noisy_fields["valid_velocity_raw_mass"]
            ),
            "baseline_invalid_velocity_raw_mass": baseline_invalid_velocity_raw_mass,
            "noisy_invalid_velocity_raw_mass": noisy_invalid_velocity_raw_mass,
            "baseline_invalid_velocity_fraction": (
                baseline_invalid_velocity_raw_mass / baseline_total_raw_mass
                if baseline_total_raw_mass > 0
                else 0.0
            ),
            "noisy_invalid_velocity_fraction": (
                noisy_invalid_velocity_raw_mass / noisy_total_raw_mass
                if noisy_total_raw_mass > 0
                else 0.0
            ),
            "invalid_velocity_fraction_difference": (
                (noisy_invalid_velocity_raw_mass / noisy_total_raw_mass)
                if noisy_total_raw_mass > 0
                else 0.0
            )
            - (
                (baseline_invalid_velocity_raw_mass / baseline_total_raw_mass)
                if baseline_total_raw_mass > 0
                else 0.0
            ),
            "rho_field_l2_error": float(np.linalg.norm(rho_diff.ravel(), ord=2)),
            "rho_field_rmse": float(np.sqrt(np.mean(rho_diff**2))),
            "rho_field_linf_error": float(np.max(np.abs(rho_diff))),
            "ux_field_l2_error": float(np.linalg.norm(ux_diff.ravel(), ord=2)),
            "uy_field_l2_error": float(np.linalg.norm(uy_diff.ravel(), ord=2)),
            "ux_field_rmse": float(np.sqrt(np.mean(ux_diff**2))),
            "uy_field_rmse": float(np.sqrt(np.mean(uy_diff**2))),
            "ux_field_linf_error": float(np.max(np.abs(ux_diff))),
            "uy_field_linf_error": float(np.max(np.abs(uy_diff))),
            **profile_metrics,
        },
    }


def analyze_velocity_counts(
    baseline_counts: Mapping[str, int | float],
    noisy_counts: Mapping[str, int | float],
    lattice,
    output_dir: Path,
    step: int,
    baseline_label: str = "noiseless",
    noisy_label: str = "noisy",
) -> dict[str, float]:
    """Analyze and plot velocity profiles directly from velocity-resolved counts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = compare_velocity_fields(
        baseline_counts,
        noisy_counts,
        lattice,
    )
    _save_velocity_arrays(output_dir, comparison)
    _save_velocity_profile_csv(output_dir, comparison, step)
    _save_velocity_metrics_json(
        output_dir,
        comparison["metrics"],
        {
            "step": step,
            "source": "velocity_resolved_counts",
        },
    )
    _save_velocity_plots(
        output_dir,
        comparison,
        step=step,
        baseline_label=baseline_label,
        noisy_label=noisy_label,
    )
    return comparison["metrics"]


def _save_velocity_arrays(output_dir: Path, comparison: dict[str, Any]) -> None:
    for label in ("baseline", "noisy"):
        fields = comparison[f"{label}_fields"]
        np.savetxt(output_dir / f"{label}_ux_xy.csv", fields["ux_xy"], delimiter=",")
        np.savetxt(output_dir / f"{label}_uy_xy.csv", fields["uy_xy"], delimiter=",")
        np.savetxt(output_dir / f"{label}_rho_xy.csv", fields["rho_xy"], delimiter=",")
        np.savetxt(
            output_dir / f"{label}_invalid_velocity_xy.csv",
            fields["invalid_velocity_xy"],
            delimiter=",",
        )
    np.savetxt(
        output_dir / "ux_xy_difference.csv",
        comparison["noisy_fields"]["ux_xy"] - comparison["baseline_fields"]["ux_xy"],
        delimiter=",",
    )
    np.savetxt(
        output_dir / "uy_xy_difference.csv",
        comparison["noisy_fields"]["uy_xy"] - comparison["baseline_fields"]["uy_xy"],
        delimiter=",",
    )


def _save_velocity_profile_csv(
    output_dir: Path,
    comparison: dict[str, Any],
    step: int,
) -> None:
    baseline = comparison["baseline_profiles"]
    noisy = comparison["noisy_profiles"]
    with (output_dir / f"velocity_x_profiles_step_{step}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "x",
                "baseline_ux_density_weighted",
                "noisy_ux_density_weighted",
                "baseline_uy_density_weighted",
                "noisy_uy_density_weighted",
                "baseline_ux_arithmetic_mean",
                "noisy_ux_arithmetic_mean",
                "baseline_uy_arithmetic_mean",
                "noisy_uy_arithmetic_mean",
            ]
        )
        for x_index in range(len(baseline["ux_x_density_weighted"])):
            writer.writerow(
                [
                    x_index,
                    baseline["ux_x_density_weighted"][x_index],
                    noisy["ux_x_density_weighted"][x_index],
                    baseline["uy_x_density_weighted"][x_index],
                    noisy["uy_x_density_weighted"][x_index],
                    baseline["ux_x_mean"][x_index],
                    noisy["ux_x_mean"][x_index],
                    baseline["uy_x_mean"][x_index],
                    noisy["uy_x_mean"][x_index],
                ]
            )

    with (output_dir / f"velocity_y_profiles_step_{step}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "y",
                "baseline_ux_density_weighted",
                "noisy_ux_density_weighted",
                "baseline_uy_density_weighted",
                "noisy_uy_density_weighted",
                "baseline_ux_arithmetic_mean",
                "noisy_ux_arithmetic_mean",
                "baseline_uy_arithmetic_mean",
                "noisy_uy_arithmetic_mean",
            ]
        )
        for y_index in range(len(baseline["ux_y_density_weighted"])):
            writer.writerow(
                [
                    y_index,
                    baseline["ux_y_density_weighted"][y_index],
                    noisy["ux_y_density_weighted"][y_index],
                    baseline["uy_y_density_weighted"][y_index],
                    noisy["uy_y_density_weighted"][y_index],
                    baseline["ux_y_mean"][y_index],
                    noisy["ux_y_mean"][y_index],
                    baseline["uy_y_mean"][y_index],
                    noisy["uy_y_mean"][y_index],
                ]
            )


def _save_velocity_metrics_json(
    output_dir: Path,
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    with (output_dir / "velocity_metrics.json").open("w", encoding="utf-8") as file:
        json.dump({**metadata, "metrics": metrics}, file, indent=2, sort_keys=True)


def _save_velocity_plots(
    output_dir: Path,
    comparison: dict[str, Any],
    step: int,
    baseline_label: str,
    noisy_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = comparison["baseline_profiles"]
    noisy = comparison["noisy_profiles"]

    x_values = np.arange(len(baseline["ux_x_density_weighted"]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(
        x_values, baseline["ux_x_density_weighted"], marker="o", label=baseline_label
    )
    axes[0].plot(
        x_values, noisy["ux_x_density_weighted"], marker="s", label=noisy_label
    )
    axes[0].set_ylabel("density-weighted u_x")
    axes[0].set_title(f"Velocity profiles along x at timestep {step}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        x_values, baseline["uy_x_density_weighted"], marker="o", label=baseline_label
    )
    axes[1].plot(
        x_values, noisy["uy_x_density_weighted"], marker="s", label=noisy_label
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("density-weighted u_y")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"velocity_profiles_x_step_{step}.png", dpi=180)
    plt.close(fig)

    y_values = np.arange(len(baseline["ux_y_density_weighted"]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(
        y_values, baseline["ux_y_density_weighted"], marker="o", label=baseline_label
    )
    axes[0].plot(
        y_values, noisy["ux_y_density_weighted"], marker="s", label=noisy_label
    )
    axes[0].set_ylabel("density-weighted u_x")
    axes[0].set_title(f"Velocity profiles along y at timestep {step}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        y_values, baseline["uy_y_density_weighted"], marker="o", label=baseline_label
    )
    axes[1].plot(
        y_values, noisy["uy_y_density_weighted"], marker="s", label=noisy_label
    )
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("density-weighted u_y")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"velocity_profiles_y_step_{step}.png", dpi=180)
    plt.close(fig)
