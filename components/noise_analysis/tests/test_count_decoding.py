"""Focused tests for Qiskit count decoding in noise-analysis tools."""

from __future__ import annotations
import pytest
from components.noise_analysis.tools.count_decoding import (
    decode_grid_count,
    decode_grid_velocity_count,
)
from components.noise_analysis.tools.density_analysis import counts_to_density_field
from components.noise_analysis.tools.velocity_analysis import counts_to_velocity_fields
from qlbm.lattice import ABLattice


class OneDimensionalLattice:
    num_dims = 1
    num_gridpoints = [7]


@pytest.fixture()
def ab_lattice() -> ABLattice:
    return ABLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 4}, "velocities": "D2Q9"},
            "geometry": [],
        }
    )


def _qiskit_count_key(
    lattice: ABLattice,
    *,
    x: int,
    y: int,
    velocity: int | None = None,
) -> str:
    """Encode measured classical bits as Qiskit displays a count key."""

    x_bits = len(lattice.grid_index(0))
    y_bits = len(lattice.grid_index(1))
    velocity_bits = lattice.num_velocity_qubits if velocity is not None else 0
    classical_bits = ["0"] * (x_bits + y_bits + velocity_bits)

    for offset in range(x_bits):
        classical_bits[offset] = "1" if (x >> offset) & 1 else "0"
    for offset in range(y_bits):
        classical_bits[x_bits + offset] = "1" if (y >> offset) & 1 else "0"
    if velocity is not None:
        for offset in range(velocity_bits):
            classical_bits[x_bits + y_bits + offset] = (
                "1" if (velocity >> offset) & 1 else "0"
            )

    return "".join(reversed(classical_bits))


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (1, 0),
        (4, 3),
        (7, 1),
    ],
)
def test_density_count_decoding_uses_classical_bit_order(
    ab_lattice: ABLattice,
    x: int,
    y: int,
) -> None:
    key = _qiskit_count_key(ab_lattice, x=x, y=y)

    decoded = decode_grid_count(key, ab_lattice)
    field = counts_to_density_field({key: 5}, ab_lattice)

    assert decoded.x == x
    assert decoded.y == y
    assert field[x, y] == 5
    assert field.sum() == 5


@pytest.mark.parametrize(
    ("x", "y", "velocity"),
    [
        (1, 0, 1),
        (4, 3, 5),
        (7, 1, 8),
    ],
)
def test_velocity_count_decoding_uses_classical_bit_order(
    ab_lattice: ABLattice,
    x: int,
    y: int,
    velocity: int,
) -> None:
    key = _qiskit_count_key(ab_lattice, x=x, y=y, velocity=velocity)

    decoded = decode_grid_velocity_count(key, ab_lattice)
    fields = counts_to_velocity_fields({key: 7}, ab_lattice, normalize=False)

    assert decoded.x == x
    assert decoded.y == y
    assert decoded.velocity == velocity
    assert fields["populations"][x, y, velocity] == 7
    assert fields["populations"].sum() == 7


def test_density_decoding_ignores_extra_high_classical_bits(
    ab_lattice: ABLattice,
) -> None:
    key = _qiskit_count_key(ab_lattice, x=4, y=3, velocity=5)

    field = counts_to_density_field({key: 11}, ab_lattice)

    assert field[4, 3] == 11
    assert field.sum() == 11


def test_count_tools_require_2d_lattices() -> None:
    lattice = OneDimensionalLattice()

    with pytest.raises(ValueError, match="2D lattice"):
        counts_to_density_field({"001": 1}, lattice)
    with pytest.raises(ValueError, match="2D lattice"):
        counts_to_velocity_fields({"001": 1}, lattice)
