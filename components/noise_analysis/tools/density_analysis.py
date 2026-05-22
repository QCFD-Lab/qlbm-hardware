"""Density-profile analysis and plotting for noise-comparison runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def counts_to_density_field(
    counts: Mapping[str, int | float],
    lattice,
) -> np.ndarray:
    """Decode density-only grid measurement counts into a 2D count field."""

    if lattice.num_dims != 2:
        raise ValueError("Density analysis currently expects a 2D lattice.")

    x_bits = lattice.num_gridpoints[0].bit_length()
    y_bits = lattice.num_gridpoints[1].bit_length()
    width = lattice.num_gridpoints[0] + 1
    height = lattice.num_gridpoints[1] + 1
    density = np.zeros((width, height), dtype=float)

    for count_key, count_value in counts.items():
        bitstring = _clean_count_key(count_key)
        if len(bitstring) < x_bits + y_bits:
            raise ValueError(
                f"Count key {count_key!r} is too short for {x_bits + y_bits} grid bits."
            )

        x = int(bitstring[:x_bits], 2)
        y = int(bitstring[x_bits : x_bits + y_bits], 2)
        if x < width and y < height:
            density[x, y] += float(count_value)

    return density


def _clean_count_key(count_key: str) -> str:
    return count_key.replace(" ", "")


def normalized_density(field: np.ndarray) -> np.ndarray:
    """Normalize a density/count field by its total mass."""

    total = float(np.sum(field))
    if total <= 0:
        return np.zeros_like(field, dtype=float)
    return field.astype(float) / total


def density_profiles(
    field: np.ndarray,
    *,
    normalize: bool = True,
    exclude_y_boundary: bool = False,
) -> dict[str, np.ndarray]:
    """Compute x-directed density profiles from a 2D density/count field."""

    density = normalized_density(field) if normalize else field.astype(float)
    if density.ndim != 2:
        raise ValueError(f"Expected a 2D density field, received shape {density.shape}.")

    y_slice = slice(1, -1) if exclude_y_boundary and density.shape[1] > 2 else slice(None)
    profile_region = density[:, y_slice]
    y_center = density.shape[1] // 2

    return {
        "rho_x_mean": profile_region.mean(axis=1),
        "rho_x_sum": profile_region.sum(axis=1),
        "rho_x_centerline": density[:, y_center],
    }


def compare_density_fields(
    baseline_field: np.ndarray,
    noisy_field: np.ndarray,
    *,
    normalize: bool = True,
    exclude_y_boundary: bool = False,
) -> dict[str, Any]:
    """Compare noiseless and noisy density fields with scalar error metrics."""

    baseline_density = (
        normalized_density(baseline_field) if normalize else baseline_field.astype(float)
    )
    noisy_density = normalized_density(noisy_field) if normalize else noisy_field.astype(float)
    difference = noisy_density - baseline_density

    baseline_profiles = density_profiles(
        baseline_field,
        normalize=normalize,
        exclude_y_boundary=exclude_y_boundary,
    )
    noisy_profiles = density_profiles(
        noisy_field,
        normalize=normalize,
        exclude_y_boundary=exclude_y_boundary,
    )
    profile_difference = noisy_profiles["rho_x_mean"] - baseline_profiles["rho_x_mean"]

    l2 = float(np.linalg.norm(difference.ravel(), ord=2))
    baseline_l2 = float(np.linalg.norm(baseline_density.ravel(), ord=2))
    profile_l2 = float(np.linalg.norm(profile_difference, ord=2))
    baseline_profile_l2 = float(np.linalg.norm(baseline_profiles["rho_x_mean"], ord=2))

    return {
        "baseline_density": baseline_density,
        "noisy_density": noisy_density,
        "density_difference": difference,
        "baseline_profiles": baseline_profiles,
        "noisy_profiles": noisy_profiles,
        "metrics": {
            "baseline_raw_mass": float(np.sum(baseline_field)),
            "noisy_raw_mass": float(np.sum(noisy_field)),
            "raw_mass_difference": float(np.sum(noisy_field) - np.sum(baseline_field)),
            "field_l1_error": float(np.sum(np.abs(difference))),
            "field_total_variation_distance": float(0.5 * np.sum(np.abs(difference))),
            "field_l2_error": l2,
            "field_relative_l2_error": l2 / baseline_l2 if baseline_l2 > 0 else 0.0,
            "field_rmse": float(np.sqrt(np.mean(difference**2))),
            "field_linf_error": float(np.max(np.abs(difference))),
            "rho_x_mean_l1_error": float(np.sum(np.abs(profile_difference))),
            "rho_x_mean_l2_error": profile_l2,
            "rho_x_mean_relative_l2_error": (
                profile_l2 / baseline_profile_l2 if baseline_profile_l2 > 0 else 0.0
            ),
            "rho_x_mean_linf_error": float(np.max(np.abs(profile_difference))),
            "rho_x_sum_total_variation_distance": float(
                0.5
                * np.sum(
                    np.abs(
                        noisy_profiles["rho_x_sum"] - baseline_profiles["rho_x_sum"]
                    )
                )
            ),
        },
    }


def analyze_density_counts(
    baseline_counts: Mapping[str, int | float],
    noisy_counts: Mapping[str, int | float],
    lattice,
    output_dir: Path,
    *,
    step: int,
    baseline_label: str = "noiseless",
    noisy_label: str = "noisy",
    normalize: bool = True,
    exclude_y_boundary: bool = False,
) -> dict[str, Any]:
    """Analyze density differences directly from sampled grid-measurement counts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_field = counts_to_density_field(baseline_counts, lattice)
    noisy_field = counts_to_density_field(noisy_counts, lattice)
    comparison = compare_density_fields(
        baseline_field,
        noisy_field,
        normalize=normalize,
        exclude_y_boundary=exclude_y_boundary,
    )

    _save_density_arrays(output_dir, comparison)
    _save_profile_csv(output_dir, comparison, step)
    _save_metrics_json(
        output_dir,
        comparison["metrics"],
        {
            "step": step,
            "source": "counts",
            "normalize": normalize,
            "exclude_y_boundary": exclude_y_boundary,
        },
    )
    _save_density_plots(
        output_dir,
        comparison,
        step=step,
        baseline_label=baseline_label,
        noisy_label=noisy_label,
    )
    return comparison["metrics"]


def save_density_error_growth(
    output_dir: Path,
    step_metrics: list[dict[str, Any]],
    *,
    noisy_label: str,
) -> None:
    """Save CSV and plots of density-error metrics versus timestep."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "density_error_vs_timestep.csv"
    fieldnames = [
        "timestep",
        "field_total_variation_distance",
        "field_relative_l2_error",
        "field_rmse",
        "rho_x_sum_total_variation_distance",
        "rho_x_mean_relative_l2_error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in step_metrics:
            metrics = item["metrics"]
            writer.writerow(
                {
                    "timestep": item["timestep"],
                    **{name: metrics[name] for name in fieldnames if name != "timestep"},
                }
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timesteps = [item["timestep"] for item in step_metrics]
    field_tvd = [
        item["metrics"]["field_total_variation_distance"] for item in step_metrics
    ]
    field_l2 = [item["metrics"]["field_relative_l2_error"] for item in step_metrics]
    profile_tvd = [
        item["metrics"]["rho_x_sum_total_variation_distance"] for item in step_metrics
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(timesteps, field_tvd, marker="o", linewidth=1.8, label="rho(x,y) TVD")
    ax.plot(timesteps, profile_tvd, marker="s", linewidth=1.8, label="rho_x TVD")
    ax.plot(timesteps, field_l2, marker="^", linewidth=1.8, label="rho(x,y) relative L2")
    ax.set_xlabel("timestep")
    ax.set_ylabel("density error vs noiseless")
    ax.set_title(f"Density error growth: {noisy_label}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "density_error_vs_timestep.png", dpi=180)
    plt.close(fig)


def save_depolarizing_probability_sweep(
    output_dir: Path,
    sweep_metrics: list[dict[str, Any]],
) -> None:
    """Save CSV and plots for a depolarizing-probability density sweep."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "two_qubit_probability",
        "single_qubit_probability",
        "field_total_variation_distance",
        "field_relative_l2_error",
        "field_rmse",
        "rho_x_sum_total_variation_distance",
        "rho_x_mean_relative_l2_error",
    ]

    with (output_dir / "depolarizing_probability_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in sweep_metrics:
            metrics = item["metrics"]
            writer.writerow(
                {
                    "two_qubit_probability": item["two_qubit_probability"],
                    "single_qubit_probability": item["single_qubit_probability"],
                    **{
                        name: metrics[name]
                        for name in fieldnames
                        if name
                        not in {
                            "two_qubit_probability",
                            "single_qubit_probability",
                        }
                    },
                }
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probabilities = [item["two_qubit_probability"] for item in sweep_metrics]
    tvd = [item["metrics"]["field_total_variation_distance"] for item in sweep_metrics]
    relative_l2 = [item["metrics"]["field_relative_l2_error"] for item in sweep_metrics]
    profile_tvd = [
        item["metrics"]["rho_x_sum_total_variation_distance"]
        for item in sweep_metrics
    ]

    nonzero_probabilities = [p for p in probabilities if p > 0]
    if nonzero_probabilities:
        positive_min = min(nonzero_probabilities)
        x_values = [p if p > 0 else positive_min / 3 for p in probabilities]
    else:
        x_values = probabilities

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_values, tvd, marker="o", linewidth=1.8)
    if nonzero_probabilities:
        ax.set_xscale("log")
    ax.set_xlabel("two-qubit depolarizing probability")
    ax.set_ylabel("density total variation distance")
    ax.set_title("Density sensitivity after one QLBM timestep")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "depolarizing_probability_vs_density_tvd.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_values, tvd, marker="o", linewidth=1.8, label="rho(x,y) TVD")
    ax.plot(
        x_values,
        profile_tvd,
        marker="s",
        linewidth=1.8,
        label="rho_x TVD",
    )
    ax.plot(
        x_values,
        relative_l2,
        marker="^",
        linewidth=1.8,
        label="rho(x,y) relative L2",
    )
    if nonzero_probabilities:
        ax.set_xscale("log")
    ax.set_xlabel("two-qubit depolarizing probability")
    ax.set_ylabel("density error vs noiseless")
    ax.set_title("Depolarizing sweep after one QLBM timestep")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "depolarizing_probability_density_errors.png", dpi=180)
    plt.close(fig)


def _save_density_arrays(output_dir: Path, comparison: dict[str, Any]) -> None:
    np.savetxt(
        output_dir / "baseline_rho_xy_normalized.csv",
        comparison["baseline_density"],
        delimiter=",",
    )
    np.savetxt(
        output_dir / "noisy_rho_xy_normalized.csv",
        comparison["noisy_density"],
        delimiter=",",
    )
    np.savetxt(
        output_dir / "rho_xy_difference.csv",
        comparison["density_difference"],
        delimiter=",",
    )


def _save_profile_csv(output_dir: Path, comparison: dict[str, Any], step: int) -> None:
    baseline_profiles = comparison["baseline_profiles"]
    noisy_profiles = comparison["noisy_profiles"]
    baseline_profile = baseline_profiles["rho_x_mean"]
    noisy_profile = noisy_profiles["rho_x_mean"]

    with (output_dir / f"rho_x_comparison_step_{step}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "x",
                "baseline_rho_x_mean",
                "noisy_rho_x_mean",
                "signed_error",
                "absolute_error",
                "baseline_rho_x_sum",
                "noisy_rho_x_sum",
                "baseline_rho_x_centerline",
                "noisy_rho_x_centerline",
            ]
        )
        for x_index in range(len(baseline_profile)):
            signed_error = noisy_profile[x_index] - baseline_profile[x_index]
            writer.writerow(
                [
                    x_index,
                    baseline_profile[x_index],
                    noisy_profile[x_index],
                    signed_error,
                    abs(signed_error),
                    baseline_profiles["rho_x_sum"][x_index],
                    noisy_profiles["rho_x_sum"][x_index],
                    baseline_profiles["rho_x_centerline"][x_index],
                    noisy_profiles["rho_x_centerline"][x_index],
                ]
            )


def _save_metrics_json(
    output_dir: Path,
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    payload = {**metadata, "metrics": metrics}
    with (output_dir / "density_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _save_density_plots(
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

    baseline_profiles = comparison["baseline_profiles"]
    noisy_profiles = comparison["noisy_profiles"]
    baseline_profile = baseline_profiles["rho_x_mean"]
    noisy_profile = noisy_profiles["rho_x_mean"]
    x_values = np.arange(len(baseline_profile))
    difference = noisy_profile - baseline_profile

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_values, baseline_profile, marker="o", linewidth=1.8, label=baseline_label)
    ax.plot(x_values, noisy_profile, marker="s", linewidth=1.8, label=noisy_label)
    ax.set_xlabel("x")
    ax.set_ylabel("mean normalized density rho(x)")
    ax.set_title(f"Density profile at timestep {step}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"rho_x_profile_step_{step}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(x_values, difference, marker="o", linewidth=1.6)
    ax.set_xlabel("x")
    ax.set_ylabel("noisy - noiseless")
    ax.set_title(f"Density profile error at timestep {step}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"rho_x_error_step_{step}.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    image_specs = [
        (comparison["baseline_density"], baseline_label),
        (comparison["noisy_density"], noisy_label),
        (comparison["density_difference"], "difference"),
    ]
    for axis, (field, title) in zip(axes, image_specs, strict=True):
        image = axis.imshow(field.T, origin="lower", aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / f"rho_xy_fields_step_{step}.png", dpi=180)
    plt.close(fig)
