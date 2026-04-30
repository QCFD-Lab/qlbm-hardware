"""Tests for the tube-flow noise-analysis scaffold."""

from __future__ import annotations

from qiskit.quantum_info import Statevector

from config import BeamConfig, TubeConfig, build_tube_lattice_config
from initial_conditions import build_single_beam_initial_condition
from observables import counts_to_flow_fields, decode_grid_velocity_count
from qlbm.components.ab import ABQLBM
from qlbm.lattice import ABLattice


def _encode_count_key(x: int, y: int, velocity: int, lattice: ABLattice) -> str:
    x_bits = len(lattice.grid_index(0))
    y_bits = len(lattice.grid_index(1))
    v_bits = lattice.num_velocity_qubits
    num_bits = x_bits + y_bits + v_bits
    classical_bits = ["0"] * num_bits
    for offset in range(x_bits):
        classical_bits[offset] = "1" if (x >> offset) & 1 else "0"
    for offset in range(y_bits):
        classical_bits[x_bits + offset] = "1" if (y >> offset) & 1 else "0"
    for offset in range(v_bits):
        classical_bits[x_bits + y_bits + offset] = "1" if (velocity >> offset) & 1 else "0"
    return "".join(reversed(classical_bits))


def test_tube_lattice_has_two_bounceback_walls() -> None:
    lattice = ABLattice(build_tube_lattice_config(TubeConfig(width=16, height=8)))

    assert lattice.num_dims == 2
    assert lattice.num_gridpoints == [15, 7]
    assert len(lattice.shapes["bounceback"]) == 2
    assert lattice.num_total_qubits > 0
    assert ABQLBM(lattice).circuit.num_qubits == lattice.num_total_qubits


def test_single_beam_initial_condition_prepares_expected_basis_state() -> None:
    tube = TubeConfig(width=16, height=8)
    beam = BeamConfig(velocity_channel=5)
    lattice = ABLattice(build_tube_lattice_config(tube))
    circuit = build_single_beam_initial_condition(lattice, tube, beam)

    state = Statevector.from_instruction(circuit)
    index = int(abs(state.data).argmax())
    expected_x = 0
    expected_y = tube.height // 2
    expected_velocity = 5

    for bit_index, qubit_index in enumerate(lattice.grid_index(0)):
        assert ((index >> qubit_index) & 1) == ((expected_x >> bit_index) & 1)
    for bit_index, qubit_index in enumerate(lattice.grid_index(1)):
        assert ((index >> qubit_index) & 1) == ((expected_y >> bit_index) & 1)
    for bit_index, qubit_index in enumerate(lattice.velocity_index()):
        assert ((index >> qubit_index) & 1) == ((expected_velocity >> bit_index) & 1)


def test_decode_grid_velocity_count_and_observables() -> None:
    lattice = ABLattice(build_tube_lattice_config(TubeConfig(width=8, height=4)))
    key_right = _encode_count_key(2, 1, 1, lattice)
    key_left = _encode_count_key(2, 2, 3, lattice)
    key_wall = _encode_count_key(2, 0, 1, lattice)

    decoded = decode_grid_velocity_count(key_right, lattice)
    assert decoded.x == 2
    assert decoded.y == 1
    assert decoded.velocity == 1

    fields = counts_to_flow_fields({key_right: 3, key_left: 1, key_wall: 10}, lattice)

    assert fields.rho_xy[2, 1] == 3 / 14
    assert fields.rho_xy[2, 2] == 1 / 14
    assert fields.rho_x[2] == (4 / 14) / 2
    assert fields.ux_x[2] == 0.5

