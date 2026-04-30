"""Initial-condition circuits for tube-flow noise-analysis experiments."""

from __future__ import annotations

from qiskit import QuantumCircuit

from config import BeamConfig, TubeConfig


def build_single_beam_initial_condition(
    lattice,
    tube: TubeConfig,
    beam: BeamConfig,
) -> QuantumCircuit:
    """Prepare one D2Q9 particle at the configured tube inlet beam state."""

    tube.validate()
    beam.validate(tube)

    circuit = QuantumCircuit(*lattice.registers)
    grid_position = (beam.x, beam.resolved_y(tube))

    for dim, position in enumerate(grid_position):
        for bit_index, qubit_index in enumerate(lattice.grid_index(dim)):
            if (position >> bit_index) & 1:
                circuit.x(qubit_index)

    for bit_index, qubit_index in enumerate(lattice.velocity_index()):
        if (beam.velocity_channel >> bit_index) & 1:
            circuit.x(qubit_index)

    return circuit

