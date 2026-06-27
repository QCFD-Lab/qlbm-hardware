"""Output writers for multi-hardware density comparisons."""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Any
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from components.noise_analysis.experiment_utils import safe_path_name


def save_multi_hardware_density_outputs(
    analysis_dir: Path,
    metrics_rows: list[dict[str, Any]],
    final_profiles: list[dict[str, Any]],
    reference_final_profile: Any,
    max_timesteps: int,
    noise_kind: str,
    heatmap_step: int | None = None,
    density_heatmap_fields: list[dict[str, Any]] | None = None,
) -> None:
    """Write combined multi-hardware density metrics and plots."""

    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_density_metrics_csv(analysis_dir, metrics_rows)
    _save_density_tvd_plot(analysis_dir, metrics_rows)

    if heatmap_step is not None and density_heatmap_fields:
        save_multi_hardware_density_heatmap(
            analysis_dir,
            timestep=heatmap_step,
            density_heatmap_fields=density_heatmap_fields,
        )

    if reference_final_profile is None or not final_profiles:
        return

    _write_final_density_profile_csv(
        analysis_dir,
        final_profiles=final_profiles,
        reference_final_profile=reference_final_profile,
        max_timesteps=max_timesteps,
    )
    _save_final_density_profile_plot(
        analysis_dir,
        final_profiles=final_profiles,
        reference_final_profile=reference_final_profile,
        max_timesteps=max_timesteps,
        noise_kind=noise_kind,
    )


def save_multi_hardware_density_heatmap(
    analysis_dir: Path,
    timestep: int,
    density_heatmap_fields: list[dict[str, Any]],
) -> None:
    """Write the timestep density heatmap figure and its backing CSV."""

    if not density_heatmap_fields:
        return

    heatmap_dir = analysis_dir / f"step_{timestep:03d}"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    fields = [item["field"] for item in density_heatmap_fields]
    vmin = 0.0
    vmax = max(float(np.max(field)) for field in fields)

    fig, axes = plt.subplots(
        1,
        len(density_heatmap_fields),
        figsize=(4.2 * len(density_heatmap_fields), 3.8),
        constrained_layout=True,
    )
    if len(density_heatmap_fields) == 1:
        axes = [axes]

    image = None
    for axis, item in zip(axes, density_heatmap_fields, strict=True):
        image = axis.imshow(
            item["field"].T,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        axis.set_title(item["label"])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.88, label="normalized density")
    fig.suptitle(f"Density heatmaps after {timestep} QLBM timestep")
    fig.savefig(
        heatmap_dir / f"hardware_depolarizing_density_heatmaps_step_{timestep}.png",
        dpi=180,
    )
    plt.close(fig)

    with (
        heatmap_dir / f"hardware_depolarizing_density_heatmaps_step_{timestep}.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "x", "y", "normalized_density"])
        for item in density_heatmap_fields:
            field = item["field"]
            for x_index in range(field.shape[0]):
                for y_index in range(field.shape[1]):
                    writer.writerow(
                        [item["label"], x_index, y_index, field[x_index, y_index]]
                    )


def _write_density_metrics_csv(
    analysis_dir: Path,
    metrics_rows: list[dict[str, Any]],
) -> None:
    metric_columns = [
        "timestep",
        "hardware_name",
        "hardware_label",
        "field_total_variation_distance",
        "field_relative_l2_error",
        "field_rmse",
        "rho_x_sum_total_variation_distance",
        "rho_x_sum_relative_l2_error",
        "rho_x_mean_relative_l2_error",
    ]
    with (analysis_dir / "hardware_depolarizing_density_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=metric_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics_rows)


def _save_density_tvd_plot(
    analysis_dir: Path,
    metrics_rows: list[dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hardware_labels = []
    for row in metrics_rows:
        if row["hardware_label"] not in hardware_labels:
            hardware_labels.append(row["hardware_label"])
    for hardware_label in hardware_labels:
        hardware_rows = [
            row for row in metrics_rows if row["hardware_label"] == hardware_label
        ]
        ax.plot(
            [row["timestep"] for row in hardware_rows],
            [row["field_total_variation_distance"] for row in hardware_rows],
            marker="o",
            linewidth=1.8,
            label=hardware_label,
        )
    ax.set_xlabel("QLBM timestep")
    ax.set_ylabel("density total variation distance")
    ax.set_title("Density error growth under hardware-derived depolarizing noise")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        analysis_dir / "hardware_depolarizing_density_tvd_vs_timestep.png",
        dpi=180,
    )
    plt.close(fig)


def _write_final_density_profile_csv(
    analysis_dir: Path,
    final_profiles: list[dict[str, Any]],
    reference_final_profile: Any,
    max_timesteps: int,
) -> None:
    hardware_column_names = _unique_hardware_column_names(final_profiles)
    with (
        analysis_dir
        / f"hardware_depolarizing_final_density_profile_step_{max_timesteps}.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["x", "noiseless_reference_rho_x_sum"]
        for profile in final_profiles:
            fieldnames.append(f"{hardware_column_names[id(profile)]}_rho_x_sum")
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for x_index in range(len(reference_final_profile)):
            row = {
                "x": x_index,
                "noiseless_reference_rho_x_sum": reference_final_profile[x_index],
            }
            for profile in final_profiles:
                row[f"{hardware_column_names[id(profile)]}_rho_x_sum"] = profile[
                    "noisy_rho_x_sum"
                ][x_index]
            writer.writerow(row)


def _save_final_density_profile_plot(
    analysis_dir: Path,
    final_profiles: list[dict[str, Any]],
    reference_final_profile: Any,
    max_timesteps: int,
    noise_kind: str,
) -> None:
    x_values = np.arange(len(reference_final_profile))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        x_values,
        reference_final_profile,
        color="black",
        marker="o",
        linewidth=2.0,
        label="Noiseless reference",
    )
    markers = ["s", "^", "D", "v"]
    for index, profile in enumerate(final_profiles):
        ax.plot(
            x_values,
            profile["noisy_rho_x_sum"],
            marker=markers[index % len(markers)],
            linewidth=1.8,
            label=f"{profile['hardware_label']} ({noise_kind})",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("normalized marginal density rho_x(x)")
    ax.set_title(f"Final density profile at timestep {max_timesteps}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        analysis_dir
        / f"hardware_depolarizing_final_density_profile_step_{max_timesteps}.png",
        dpi=180,
    )
    plt.close(fig)


def _unique_hardware_column_names(
    final_profiles: list[dict[str, Any]],
) -> dict[int, str]:
    """Return stable, unique CSV-safe names derived from full hardware names."""

    used: dict[str, int] = {}
    names_by_profile_id: dict[int, str] = {}
    for profile in final_profiles:
        base_name = safe_path_name(str(profile["hardware_name"]))
        occurrence = used.get(base_name, 0) + 1
        used[base_name] = occurrence
        if occurrence == 1:
            unique_name = base_name
        else:
            unique_name = f"{base_name}_{occurrence}"
        names_by_profile_id[id(profile)] = unique_name
    return names_by_profile_id
