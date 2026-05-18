"""Visualize saved D2Q9 tube-analysis outputs."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "output" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"rho_x_step_(\d+)\.csv$")


def discover_steps(run_dir: Path) -> list[int]:
    """Return all timestep indices with saved profile data."""

    steps: list[int] = []
    for path in run_dir.glob("rho_x_step_*.csv"):
        match = STEP_RE.match(path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def load_vector(run_dir: Path, name: str, step: int) -> np.ndarray:
    """Load a 1D saved observable."""

    data = np.loadtxt(run_dir / f"{name}_step_{step}.csv", delimiter=",")
    return np.atleast_1d(data)


def load_matrix(run_dir: Path, name: str, step: int) -> np.ndarray:
    """Load a 2D saved observable."""

    data = np.loadtxt(run_dir / f"{name}_step_{step}.csv", delimiter=",")
    return np.atleast_2d(data)


def plot_step_profiles(run_dir: Path, vis_dir: Path, step: int) -> Path:
    """Plot ``u_x(x)`` and ``rho(x)`` for one timestep."""

    ux_x = load_vector(run_dir, "ux_x", step)
    rho_x = load_vector(run_dir, "rho_x", step)
    x_values = np.arange(len(ux_x))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(x_values, ux_x, marker="o", linewidth=1.6)
    axes[0].set_ylabel("mean u_x")
    axes[0].set_title(f"Tube profiles, step {step}")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x_values, rho_x, marker="o", linewidth=1.6)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("mean rho")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    output = vis_dir / f"profiles_step_{step}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_step_heatmaps(run_dir: Path, vis_dir: Path, step: int) -> Path:
    """Plot ``u_x(x,y)`` and ``rho(x,y)`` heatmaps for one timestep."""

    ux_xy = load_matrix(run_dir, "ux_xy", step)
    rho_xy = load_matrix(run_dir, "rho_xy", step)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    rho_image = axes[0].imshow(
        rho_xy.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    axes[0].set_title(f"rho(x,y), step {step}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(rho_image, ax=axes[0], fraction=0.046, pad=0.04)

    ux_abs = max(1e-12, float(np.nanmax(np.abs(ux_xy))))
    ux_image = axes[1].imshow(
        ux_xy.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-ux_abs,
        vmax=ux_abs,
    )
    axes[1].set_title(f"u_x(x,y), step {step}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(ux_image, ax=axes[1], fraction=0.046, pad=0.04)

    output = vis_dir / f"heatmaps_step_{step}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_profile_evolution(run_dir: Path, vis_dir: Path, steps: list[int]) -> Path:
    """Overlay profile curves for multiple timesteps."""

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for step in steps:
        ux_x = load_vector(run_dir, "ux_x", step)
        rho_x = load_vector(run_dir, "rho_x", step)
        x_values = np.arange(len(ux_x))
        axes[0].plot(x_values, ux_x, marker="o", linewidth=1.4, label=f"step {step}")
        axes[1].plot(x_values, rho_x, marker="o", linewidth=1.4, label=f"step {step}")

    axes[0].set_ylabel("mean u_x")
    axes[0].set_title("Profile evolution")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("mean rho")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    output = vis_dir / "profile_evolution.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def visualize_run(run_dir: Path, steps: list[int] | None = None) -> list[Path]:
    """Generate visualizations for one saved experiment directory."""

    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    selected_steps = discover_steps(run_dir) if steps is None else sorted(steps)
    if not selected_steps:
        raise ValueError(f"No saved step CSV files found in {run_dir}")

    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for step in selected_steps:
        outputs.append(plot_step_profiles(run_dir, vis_dir, step))
        outputs.append(plot_step_heatmaps(run_dir, vis_dir, step))
    if len(selected_steps) > 1:
        outputs.append(plot_profile_evolution(run_dir, vis_dir, selected_steps))
    return outputs


def latest_run_directory(output_root: Path) -> Path:
    """Return the most recently modified tube-analysis run directory."""

    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No experiment directories found in {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    # Edit these variables to choose which experiment and timesteps to visualize.
    output_root = Path(__file__).resolve().parent / "output" / "tube_analysis"
    run_dir = None
    steps = None

    selected_run_dir = latest_run_directory(output_root) if run_dir is None else run_dir
    outputs = visualize_run(selected_run_dir, steps)
    print("Generated visualizations:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
