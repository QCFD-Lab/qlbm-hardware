"""Velocity-profile analysis from velocity-resolved QLBM counts."""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from qlbm.lattice.spacetime.properties_base import LatticeDiscretizationProperties
from .count_decoding import decode_grid_velocity_count, grid_shape

VELOCITY_PROFILE_DENSITY_FLOOR = 1e-3
VELOCITY_COMPONENTS = ("ux", "uy")
PROFILE_AXES = ("x", "y")


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
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )

def _profile_support_masks(
    baseline_fields: dict[str, np.ndarray],
    noisy_fields: dict[str, np.ndarray],
    density_floor: float,
) -> dict[str, np.ndarray]:
    """Return profile bins with enough density support for stable velocities."""

    baseline_rho_xy = baseline_fields["rho_xy"]
    noisy_rho_xy = noisy_fields["rho_xy"]
    baseline_rho_x = baseline_rho_xy.sum(axis=1)
    noisy_rho_x = noisy_rho_xy.sum(axis=1)
    baseline_rho_y = baseline_rho_xy.sum(axis=0)
    noisy_rho_y = noisy_rho_xy.sum(axis=0)

    return {
        "x": np.maximum(baseline_rho_x, noisy_rho_x) >= density_floor,
        "y": np.maximum(baseline_rho_y, noisy_rho_y) >= density_floor,
        "baseline_x": baseline_rho_x >= density_floor,
        "noisy_x": noisy_rho_x >= density_floor,
        "baseline_y": baseline_rho_y >= density_floor,
        "noisy_y": noisy_rho_y >= density_floor,
        "baseline_rho_x": baseline_rho_x,
        "noisy_rho_x": noisy_rho_x,
        "baseline_rho_y": baseline_rho_y,
        "noisy_rho_y": noisy_rho_y,
    }


def _profile_axis(profile_name: str) -> str:
    if "_x_" in profile_name or profile_name.endswith("_x"):
        return "x"
    if "_y_" in profile_name or profile_name.endswith("_y"):
        return "y"
    raise ValueError(f"Cannot infer profile axis from {profile_name!r}.")

def _zero_low_density_profile(values: np.ndarray, density_mask: np.ndarray) -> np.ndarray:
    filtered = values.astype(float, copy=True)
    filtered[~density_mask] = 0.0
    return filtered

def _profile_density_mask(
    support_masks: dict[str, np.ndarray],
    profile_name: str,
    label: str,
) -> np.ndarray:
    return support_masks[f"{label}_{_profile_axis(profile_name)}"]

def _profile_key(component: str, axis: str, suffix: str) -> str:
    return f"{component}_{axis}_{suffix}"

def _masked_profile(
    profiles: dict[str, np.ndarray],
    support_masks: dict[str, np.ndarray],
    label: str,
    profile_name: str,
) -> np.ndarray:
    return _zero_low_density_profile(
        profiles[profile_name],
        _profile_density_mask(support_masks, profile_name, label),
    )

def _velocity_profile_error_metrics(
    baseline_profiles: dict[str, np.ndarray],
    noisy_profiles: dict[str, np.ndarray],
    support_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    metrics = {}
    for profile_name in baseline_profiles:
        baseline_profile = _masked_profile(
            baseline_profiles,
            support_masks,
            "baseline",
            profile_name,
        )
        noisy_profile = _masked_profile(
            noisy_profiles,
            support_masks,
            "noisy",
            profile_name,
        )
        difference = noisy_profile - baseline_profile
        baseline_norm = float(np.linalg.norm(baseline_profile, ord=2))
        difference_norm = float(np.linalg.norm(difference, ord=2))
        metrics[f"{profile_name}_l2_error"] = difference_norm
        metrics[f"{profile_name}_relative_l2_error"] = (
            difference_norm / baseline_norm if baseline_norm > 0 else 0.0
        )
        metrics[f"{profile_name}_linf_error"] = (
            float(np.max(np.abs(difference))) if difference.size else 0.0
        )
    return metrics


def _fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


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
    density_floor: float = VELOCITY_PROFILE_DENSITY_FLOOR,
) -> dict[str, Any]:
    """Compare velocity fields and profiles against a noiseless baseline."""

    baseline_fields = counts_to_velocity_fields(baseline_counts, lattice)
    noisy_fields = counts_to_velocity_fields(noisy_counts, lattice)
    baseline_profiles = velocity_profiles(baseline_fields)
    noisy_profiles = velocity_profiles(noisy_fields)
    support_masks = _profile_support_masks(
        baseline_fields,
        noisy_fields,
        density_floor=density_floor,
    )

    ux_diff = noisy_fields["ux_xy"] - baseline_fields["ux_xy"]
    uy_diff = noisy_fields["uy_xy"] - baseline_fields["uy_xy"]
    rho_diff = noisy_fields["rho_xy"] - baseline_fields["rho_xy"]
    profile_metrics = _velocity_profile_error_metrics(
        baseline_profiles,
        noisy_profiles,
        support_masks,
    )

    baseline_total_raw_mass = float(baseline_fields["total_raw_mass"])
    noisy_total_raw_mass = float(noisy_fields["total_raw_mass"])
    baseline_invalid_velocity_raw_mass = float(
        baseline_fields["invalid_velocity_raw_mass"]
    )
    noisy_invalid_velocity_raw_mass = float(noisy_fields["invalid_velocity_raw_mass"])
    baseline_invalid_fraction = baseline_invalid_velocity_raw_mass / baseline_total_raw_mass \
        if baseline_total_raw_mass > 0 else 0.0
    noisy_invalid_fraction = noisy_invalid_velocity_raw_mass / noisy_total_raw_mass \
        if noisy_total_raw_mass > 0 else 0.0

    return {
        "baseline_fields": baseline_fields,
        "noisy_fields": noisy_fields,
        "baseline_profiles": baseline_profiles,
        "noisy_profiles": noisy_profiles,
        "profile_support_masks": support_masks,
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
            "baseline_invalid_velocity_fraction": baseline_invalid_fraction,
            "noisy_invalid_velocity_fraction": noisy_invalid_fraction,
            "invalid_velocity_fraction_difference": (
                noisy_invalid_fraction - baseline_invalid_fraction
            ),
            "profile_density_floor": density_floor,
            "x_profile_supported_bins": int(np.count_nonzero(support_masks["x"])),
            "y_profile_supported_bins": int(np.count_nonzero(support_masks["y"])),
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
    for axis in PROFILE_AXES:
        _save_velocity_axis_profile_csv(output_dir, comparison, step, axis)

def _save_velocity_axis_profile_csv(
    output_dir: Path,
    comparison: dict[str, Any],
    step: int,
    axis: str,
) -> None:
    baseline = comparison["baseline_profiles"]
    noisy = comparison["noisy_profiles"]
    support_masks = comparison["profile_support_masks"]
    masked_density_profiles = {
        (label, component): _masked_profile(
            profiles,
            support_masks,
            label,
            _profile_key(component, axis, "density_weighted"),
        )
        for label, profiles in (("baseline", baseline), ("noisy", noisy))
        for component in VELOCITY_COMPONENTS
    }
    density_profile_key = _profile_key("ux", axis, "density_weighted")

    with (output_dir / f"velocity_{axis}_profiles_step_{step}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                axis,
                "profile_supported",
                f"baseline_rho_{axis}",
                f"noisy_rho_{axis}",
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
        for index in range(len(baseline[density_profile_key])):
            writer.writerow(
                [
                    index,
                    int(bool(support_masks[axis][index])),
                    support_masks[f"baseline_rho_{axis}"][index],
                    support_masks[f"noisy_rho_{axis}"][index],
                    masked_density_profiles[("baseline", "ux")][index],
                    masked_density_profiles[("noisy", "ux")][index],
                    masked_density_profiles[("baseline", "uy")][index],
                    masked_density_profiles[("noisy", "uy")][index],
                    baseline[_profile_key("ux", axis, "mean")][index],
                    noisy[_profile_key("ux", axis, "mean")][index],
                    baseline[_profile_key("uy", axis, "mean")][index],
                    noisy[_profile_key("uy", axis, "mean")][index],
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

    for axis in PROFILE_AXES:
        _save_velocity_axis_plot(
            output_dir,
            comparison,
            step=step,
            axis=axis,
            baseline_label=baseline_label,
            noisy_label=noisy_label,
            plt=plt,
        )

def _save_velocity_axis_plot(
    output_dir: Path,
    comparison: dict[str, Any],
    step: int,
    axis: str,
    baseline_label: str,
    noisy_label: str,
    plt,
) -> None:
    baseline = comparison["baseline_profiles"]
    noisy = comparison["noisy_profiles"]
    support_masks = comparison["profile_support_masks"]

    values = np.arange(len(baseline[_profile_key("ux", axis, "density_weighted")]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for subplot, component in zip(axes, VELOCITY_COMPONENTS, strict=True):
        profile_name = _profile_key(component, axis, "density_weighted")
        subplot.plot(
            values,
            _masked_profile(baseline, support_masks, "baseline", profile_name),
            marker="o",
            label=baseline_label,
        )
        subplot.plot(
            values,
            _masked_profile(noisy, support_masks, "noisy", profile_name),
            marker="s",
            label=noisy_label,
        )
        subplot.set_ylabel(f"density-weighted u_{component[-1]}")
        subplot.grid(True, alpha=0.25)
        subplot.legend()

    axes[0].set_title(f"Velocity profiles along {axis} at timestep {step}")
    axes[1].set_xlabel(axis)
    fig.tight_layout()
    fig.savefig(output_dir / f"velocity_profiles_{axis}_step_{step}.png", dpi=180)
    plt.close(fig)
