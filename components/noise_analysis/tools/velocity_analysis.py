"""Velocity-profile analysis from velocity-resolved QLBM counts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

D2Q9_VELOCITIES = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=float,
)


def counts_to_velocity_fields(
    counts: Mapping[str, int | float],
    lattice,
    *,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Decode ``ABGridMeasurement(..., measure_velocity_qubits=True)`` counts."""

    if lattice.num_dims != 2:
        raise ValueError("Velocity analysis currently expects a 2D lattice.")

    x_bits = lattice.num_gridpoints[0].bit_length()
    y_bits = lattice.num_gridpoints[1].bit_length()
    velocity_bits = lattice.num_velocity_qubits
    width = lattice.num_gridpoints[0] + 1
    height = lattice.num_gridpoints[1] + 1

    populations = np.zeros((width, height, len(D2Q9_VELOCITIES)), dtype=float)
    required_bits = x_bits + y_bits + velocity_bits

    for count_key, count_value in counts.items():
        bitstring = count_key.replace(" ", "")
        if len(bitstring) < required_bits:
            raise ValueError(
                f"Count key {count_key!r} is too short for {required_bits} measured bits."
            )

        x = int(bitstring[:x_bits], 2)
        y = int(bitstring[x_bits : x_bits + y_bits], 2)
        velocity = int(bitstring[x_bits + y_bits : required_bits], 2)
        if x < width and y < height and velocity < len(D2Q9_VELOCITIES):
            populations[x, y, velocity] += float(count_value)

    if normalize:
        total = float(sum(counts.values()))
        if total > 0:
            populations /= total

    rho_xy = populations.sum(axis=2)
    momentum_x = (populations * D2Q9_VELOCITIES[:, 0]).sum(axis=2)
    momentum_y = (populations * D2Q9_VELOCITIES[:, 1]).sum(axis=2)
    ux_xy = np.divide(momentum_x, rho_xy, out=np.zeros_like(momentum_x), where=rho_xy > 0)
    uy_xy = np.divide(momentum_y, rho_xy, out=np.zeros_like(momentum_y), where=rho_xy > 0)

    return {
        "populations": populations,
        "rho_xy": rho_xy,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
        "ux_xy": ux_xy,
        "uy_xy": uy_xy,
    }


def velocity_profiles(
    fields: dict[str, np.ndarray],
    *,
    exclude_y_boundary: bool = False,
    exclude_x_boundary: bool = False,
) -> dict[str, np.ndarray]:
    """Compute velocity profiles along x and y for both velocity components."""

    ux_xy = fields["ux_xy"]
    uy_xy = fields["uy_xy"]
    y_slice = slice(1, -1) if exclude_y_boundary and ux_xy.shape[1] > 2 else slice(None)
    x_slice = slice(1, -1) if exclude_x_boundary and ux_xy.shape[0] > 2 else slice(None)

    x_mid = ux_xy.shape[0] // 2
    y_mid = ux_xy.shape[1] // 2
    return {
        "ux_x_mean": ux_xy[:, y_slice].mean(axis=1),
        "uy_x_mean": uy_xy[:, y_slice].mean(axis=1),
        "ux_y_mean": ux_xy[x_slice, :].mean(axis=0),
        "uy_y_mean": uy_xy[x_slice, :].mean(axis=0),
        "ux_centerline_x": ux_xy[:, y_mid],
        "uy_centerline_x": uy_xy[:, y_mid],
        "ux_centerline_y": ux_xy[x_mid, :],
        "uy_centerline_y": uy_xy[x_mid, :],
    }


def compare_velocity_fields(
    baseline_counts: Mapping[str, int | float],
    noisy_counts: Mapping[str, int | float],
    lattice,
    *,
    exclude_y_boundary: bool = False,
    exclude_x_boundary: bool = False,
) -> dict[str, Any]:
    """Compare velocity fields and profiles against a noiseless baseline."""

    baseline_fields = counts_to_velocity_fields(baseline_counts, lattice)
    noisy_fields = counts_to_velocity_fields(noisy_counts, lattice)
    baseline_profiles = velocity_profiles(
        baseline_fields,
        exclude_y_boundary=exclude_y_boundary,
        exclude_x_boundary=exclude_x_boundary,
    )
    noisy_profiles = velocity_profiles(
        noisy_fields,
        exclude_y_boundary=exclude_y_boundary,
        exclude_x_boundary=exclude_x_boundary,
    )

    ux_diff = noisy_fields["ux_xy"] - baseline_fields["ux_xy"]
    uy_diff = noisy_fields["uy_xy"] - baseline_fields["uy_xy"]
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
        profile_metrics[f"{profile_name}_linf_error"] = float(np.max(np.abs(difference)))

    return {
        "baseline_fields": baseline_fields,
        "noisy_fields": noisy_fields,
        "baseline_profiles": baseline_profiles,
        "noisy_profiles": noisy_profiles,
        "metrics": {
            "baseline_raw_mass": float(sum(baseline_counts.values())),
            "noisy_raw_mass": float(sum(noisy_counts.values())),
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
    *,
    step: int,
    baseline_label: str = "noiseless",
    noisy_label: str = "noisy",
    exclude_y_boundary: bool = False,
    exclude_x_boundary: bool = False,
) -> dict[str, float]:
    """Analyze and plot velocity profiles directly from velocity-resolved counts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = compare_velocity_fields(
        baseline_counts,
        noisy_counts,
        lattice,
        exclude_y_boundary=exclude_y_boundary,
        exclude_x_boundary=exclude_x_boundary,
    )
    _save_velocity_arrays(output_dir, comparison)
    _save_velocity_profile_csv(output_dir, comparison, step)
    _save_velocity_metrics_json(
        output_dir,
        comparison["metrics"],
        {
            "step": step,
            "source": "velocity_resolved_counts",
            "exclude_y_boundary": exclude_y_boundary,
            "exclude_x_boundary": exclude_x_boundary,
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
        writer.writerow(["x", "baseline_ux", "noisy_ux", "baseline_uy", "noisy_uy"])
        for x_index in range(len(baseline["ux_x_mean"])):
            writer.writerow(
                [
                    x_index,
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
        writer.writerow(["y", "baseline_ux", "noisy_ux", "baseline_uy", "noisy_uy"])
        for y_index in range(len(baseline["ux_y_mean"])):
            writer.writerow(
                [
                    y_index,
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
    *,
    step: int,
    baseline_label: str,
    noisy_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = comparison["baseline_profiles"]
    noisy = comparison["noisy_profiles"]

    x_values = np.arange(len(baseline["ux_x_mean"]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(x_values, baseline["ux_x_mean"], marker="o", label=baseline_label)
    axes[0].plot(x_values, noisy["ux_x_mean"], marker="s", label=noisy_label)
    axes[0].set_ylabel("mean u_x")
    axes[0].set_title(f"Velocity profiles along x at timestep {step}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(x_values, baseline["uy_x_mean"], marker="o", label=baseline_label)
    axes[1].plot(x_values, noisy["uy_x_mean"], marker="s", label=noisy_label)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("mean u_y")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"velocity_profiles_x_step_{step}.png", dpi=180)
    plt.close(fig)

    y_values = np.arange(len(baseline["ux_y_mean"]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(y_values, baseline["ux_y_mean"], marker="o", label=baseline_label)
    axes[0].plot(y_values, noisy["ux_y_mean"], marker="s", label=noisy_label)
    axes[0].set_ylabel("mean u_x")
    axes[0].set_title(f"Velocity profiles along y at timestep {step}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(y_values, baseline["uy_y_mean"], marker="o", label=baseline_label)
    axes[1].plot(y_values, noisy["uy_y_mean"], marker="s", label=noisy_label)
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("mean u_y")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"velocity_profiles_y_step_{step}.png", dpi=180)
    plt.close(fig)
