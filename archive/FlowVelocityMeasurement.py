from qlbm import ABLattice
from qlbm.components.ab.measurement import ABGridMeasurement
from qiskit import ClassicalRegister, QuantumCircuit


class FlowVelocityMeasurement(ABGridMeasurement):
    """
    Extended measurement for both grid position and velocity distribution.
    To track flow velocity profiles across the tube.
    """

    def __init__(
        self,
        lattice: ABLattice,
        logger=None,
    ):
        super().__init__(lattice, measure_velocity_qubits=True, logger=logger)

    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        # Add classical registers for both grid and velocity measurement
        circuit.add_register(
            ClassicalRegister(
                self.lattice.num_grid_qubits + self.lattice.num_velocity_qubits,
                name="measurement"
            )
        )

        # Measure all grid and velocity qubits
        all_qubits = self.lattice.grid_index() + self.lattice.velocity_index()
        circuit.measure(all_qubits, range(len(all_qubits)))

        return circuit

    def __str__(self):
        return f"[Primitive FlowVelocityMeasurement for lattice {self.lattice}]"
