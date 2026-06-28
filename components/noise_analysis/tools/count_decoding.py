"""Shared Qiskit count-key decoding helpers for noise-analysis tools."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DecodedGridMeasurement:
    """Grid coordinates decoded from a Qiskit count key."""
    x: int
    y: int


@dataclass(frozen=True)
class DecodedGridVelocityMeasurement:
    """Grid coordinates and velocity channel decoded from a count key."""
    x: int
    y: int
    velocity: int


def read_little_endian_register(bitstring: str, start: int, size: int) -> int:
    """Read classical bits start:start+size from a Qiskit display bitstring.

    Qiskit count keys are displayed with the highest classical bit on the left.
    The measurement circuits used here write x, y, and velocity into increasing
    classical-bit indices, so register decoding must read from the right.
    """

    if start < 0:
        raise ValueError("start must be non-negative.")
    if size < 0:
        raise ValueError("size must be non-negative.")
    if len(bitstring) < start + size:
        raise ValueError(
            f"Count key {bitstring!r} is too short for register "
            f"starting at {start} with size {size}."
        )

    value = 0
    for offset in range(size):
        string_index = len(bitstring) - 1 - (start + offset)
        if bitstring[string_index] == "1":
            value |= 1 << offset
    return value


def grid_register_sizes(lattice) -> tuple[int, int]:
    """Get the x/y grid-register sizes for a 2D lattice."""
    if lattice.num_dims != 2:
        raise ValueError("Count decoding currently expects a 2D lattice.")
    return (
        lattice.num_gridpoints[0].bit_length(),
        lattice.num_gridpoints[1].bit_length(),
    )


def grid_shape(lattice) -> tuple[int, int]:
    """Get the x/y field shape for a 2D lattice."""

    grid_register_sizes(lattice)
    return lattice.num_gridpoints[0] + 1, lattice.num_gridpoints[1] + 1


def decode_grid_count(count_key: str, lattice) -> DecodedGridMeasurement:
    """Decode x/y grid coordinates from a count key."""
    bitstring = count_key.replace(" ", "")
    x_bits, y_bits = grid_register_sizes(lattice)
    x = read_little_endian_register(bitstring, 0, x_bits)
    y = read_little_endian_register(bitstring, x_bits, y_bits)
    return DecodedGridMeasurement(x=x, y=y)


def decode_grid_velocity_count(count_key: str, lattice) -> DecodedGridVelocityMeasurement:
    """Decode x/y grid coordinates and velocity channel from a count key."""

    bitstring = count_key.replace(" ", "")
    x_bits, y_bits = grid_register_sizes(lattice)
    velocity_bits = lattice.num_velocity_qubits
    x = read_little_endian_register(bitstring, 0, x_bits)
    y = read_little_endian_register(bitstring, x_bits, y_bits)
    velocity = read_little_endian_register(bitstring, x_bits + y_bits, velocity_bits)
    return DecodedGridVelocityMeasurement(x=x, y=y, velocity=velocity)
