"""Count decoding and flow observables for D2Q9 tube experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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


@dataclass(frozen=True)
class DecodedMeasurement:
    """A decoded grid-plus-velocity measurement result."""

    x: int
    y: int
    velocity: int


@dataclass(frozen=True)
class FlowFields:
    """Density and streamwise-velocity fields derived from sampled counts."""

    populations: np.ndarray
    rho_xy: np.ndarray
    ux_xy: np.ndarray
    rho_x: np.ndarray
    ux_x: np.ndarray


def _clean_count_key(count_key: str) -> str:
    return count_key.replace(" ", "")


def _read_little_endian_register(bitstring: str, start: int, size: int) -> int:
    value = 0
    for offset in range(size):
        string_index = len(bitstring) - 1 - (start + offset)
        if string_index < 0:
            raise ValueError(f"Count key {bitstring!r} is too short for measured registers.")
        if bitstring[string_index] == "1":
            value |= 1 << offset
    return value


def decode_grid_velocity_count(count_key: str, lattice) -> DecodedMeasurement:
    """Decode a Qiskit count key from ``ABGridMeasurement(..., True)``."""

    bitstring = _clean_count_key(count_key)
    x_bits = len(lattice.grid_index(0))
    y_bits = len(lattice.grid_index(1))
    v_bits = lattice.num_velocity_qubits

    x = _read_little_endian_register(bitstring, 0, x_bits)
    y = _read_little_endian_register(bitstring, x_bits, y_bits)
    velocity = _read_little_endian_register(bitstring, x_bits + y_bits, v_bits)

    return DecodedMeasurement(x=x, y=y, velocity=velocity)


def counts_to_flow_fields(
    counts: Mapping[str, int | float],
    lattice,
    *,
    normalize: bool = True,
) -> FlowFields:
    """Convert velocity-resolved measurement counts into density and velocity fields."""

    width = lattice.num_gridpoints[0] + 1
    height = lattice.num_gridpoints[1] + 1
    populations = np.zeros((width, height, len(D2Q9_VELOCITIES)), dtype=float)

    for count_key, count_value in counts.items():
        decoded = decode_grid_velocity_count(count_key, lattice)
        if decoded.x >= width or decoded.y >= height:
            continue
        if decoded.velocity >= len(D2Q9_VELOCITIES):
            continue
        populations[decoded.x, decoded.y, decoded.velocity] += float(count_value)

    if normalize:
        total = float(sum(counts.values()))
        if total > 0:
            populations /= total

    rho_xy = populations.sum(axis=2)
    x_momentum = (populations * D2Q9_VELOCITIES[:, 0]).sum(axis=2)
    ux_xy = np.divide(
        x_momentum,
        rho_xy,
        out=np.zeros_like(x_momentum),
        where=rho_xy > 0,
    )

    interior = slice(1, height - 1)
    rho_x = rho_xy[:, interior].mean(axis=1)
    interior_rho = rho_xy[:, interior].sum(axis=1)
    interior_x_momentum = x_momentum[:, interior].sum(axis=1)
    ux_x = np.divide(
        interior_x_momentum,
        interior_rho,
        out=np.zeros_like(interior_x_momentum),
        where=interior_rho > 0,
    )

    return FlowFields(
        populations=populations,
        rho_xy=rho_xy,
        ux_xy=ux_xy,
        rho_x=rho_x,
        ux_x=ux_x,
    )


def save_counts(output_dir: Path, counts: Mapping[str, int | float], step: int) -> None:
    """Save raw counts for a timestep."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"counts_step_{step}.json").open("w", encoding="utf-8") as file:
        json.dump(dict(counts), file, indent=2, sort_keys=True)


def save_flow_fields(output_dir: Path, fields: FlowFields, step: int) -> None:
    """Save density and streamwise velocity arrays for a timestep."""

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / f"rho_xy_step_{step}.csv", fields.rho_xy, delimiter=",")
    np.savetxt(output_dir / f"ux_xy_step_{step}.csv", fields.ux_xy, delimiter=",")
    np.savetxt(output_dir / f"rho_x_step_{step}.csv", fields.rho_x, delimiter=",")
    np.savetxt(output_dir / f"ux_x_step_{step}.csv", fields.ux_x, delimiter=",")


def save_profile_plots(output_dir: Path, fields: FlowFields, step: int) -> None:
    """Save simple profile plots for ``rho(x)`` and ``u_x(x)``."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    x_values = np.arange(fields.rho_x.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(x_values, fields.ux_x, marker="o", linewidth=1.5)
    axes[0].set_ylabel("mean u_x")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x_values, fields.rho_x, marker="o", linewidth=1.5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("mean rho")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / f"profiles_step_{step}.png", dpi=160)
    plt.close(fig)
