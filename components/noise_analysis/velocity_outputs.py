"""Output writers for multi-hardware velocity profile comparisons."""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from components.noise_analysis.experiment_utils import safe_path_name


def save_multi_hardware_velocity_metrics(analysis_dir: Path, metrics_rows: list[dict[str, Any]]) -> None:
    """Write the per-hardware velocity error metrics table."""

    if not metrics_rows:
        return
    analysis_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "timestep",
        "hardware_name",
        "hardware_label",
        "ux_x_density_weighted_relative_l2_error",
        "uy_x_density_weighted_relative_l2_error",
        "noisy_invalid_velocity_fraction",
        "invalid_velocity_fraction_difference",
    ]
    with (analysis_dir / "velocity_hardware_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics_rows)


def save_multi_hardware_velocity_x_outputs(
    step_dir: Path,
    timestep: int,
    noise_kind: str,
    step_results: list[dict[str, Any]],
) -> None:
    """Write the combined x-profile CSV and plots for one timestep."""

    if not step_results:
        return

    step_dir.mkdir(parents=True, exist_ok=True)
    reference = step_results[0]["comparison"]["baseline_profiles"]
    x_count = len(reference["ux_x_density_weighted"])
    hardware_column_names = _unique_hardware_column_names(step_results)

    with (step_dir / f"multi_hardware_velocity_along_x_step_{timestep}.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = ["x", "noiseless_reference_ux", "noiseless_reference_uy"]
        for result in step_results:
            hardware_name = hardware_column_names[id(result)]
            fieldnames.extend([f"{hardware_name}_noisy_ux", f"{hardware_name}_noisy_uy"])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for x_index in range(x_count):
            row = {
                "x": x_index,
                "noiseless_reference_ux": reference["ux_x_density_weighted"][x_index],
                "noiseless_reference_uy": reference["uy_x_density_weighted"][x_index],
            }
            for result in step_results:
                hardware_name = hardware_column_names[id(result)]
                noisy = result["comparison"]["noisy_profiles"]
                row[f"{hardware_name}_noisy_ux"] = noisy[
                    "ux_x_density_weighted"
                ][x_index]
                row[f"{hardware_name}_noisy_uy"] = noisy[
                    "uy_x_density_weighted"
                ][x_index]
            writer.writerow(row)

    _save_velocity_component_plot(
        step_dir,
        timestep=timestep,
        noise_kind=noise_kind,
        step_results=step_results,
        component="ux",
        reference_values=reference["ux_x_density_weighted"],
        output_name=f"velocity_profile_along_x_step_{timestep}.png",
        title=f"Velocity Profile Along x, timestep {timestep}",
        ylabel="density-weighted u_x",
    )
    _save_velocity_component_plot(
        step_dir,
        timestep=timestep,
        noise_kind=noise_kind,
        step_results=step_results,
        component="uy",
        reference_values=reference["uy_x_density_weighted"],
        output_name=f"transverse_velocity_along_x_step_{timestep}.png",
        title=f"Transverse Velocity Along x, timestep {timestep}",
        ylabel="density-weighted u_y",
    )


def _save_velocity_component_plot(
    step_dir: Path,
    timestep: int,
    noise_kind: str,
    step_results: list[dict[str, Any]],
    component: str,
    reference_values,
    output_name: str,
    title: str,
    ylabel: str,
) -> None:
    x_values = list(range(len(reference_values)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        x_values,
        reference_values,
        color="black",
        linewidth=2.0,
        marker="o",
        label="Noiseless reference",
    )
    markers = ["s", "^", "D", "v"]
    profile_key = f"{component}_x_density_weighted"
    for index, result in enumerate(step_results):
        noisy = result["comparison"]["noisy_profiles"]
        ax.plot(
            x_values,
            noisy[profile_key],
            marker=markers[index % len(markers)],
            label=f"{result['hardware_label']} ({noise_kind})",
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(step_dir / output_name, dpi=180)
    plt.close(fig)


def _unique_hardware_column_names(step_results: list[dict[str, Any]]) -> dict[int, str]:
    """Return stable, unique CSV-safe names derived from full hardware names."""

    used: dict[str, int] = {}
    names_by_result_id: dict[int, str] = {}
    for result in step_results:
        base_name = safe_path_name(str(result["hardware_name"]))
        occurrence = used.get(base_name, 0) + 1
        used[base_name] = occurrence
        if occurrence == 1:
            unique_name = base_name
        else:
            unique_name = f"{base_name}_{occurrence}"
        names_by_result_id[id(result)] = unique_name
    return names_by_result_id
