from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
from qlbm.components.ab.initial import ABInitialConditions
from qlbm.components.base import LBMPrimitive
from qlbm.lattice.lattices.ab_lattice import ABLattice

class FlowInitialConditions(LBMPrimitive):
    """
    Custom initial conditions for ABQLBM with x-direction flow.

    Creates a superposition where right-moving velocities (1, 5, 8) have higher amplitude
    than left-moving velocities (3, 6, 7), creating net flow in +x direction.
    """

    def __init__(
        self,
        lattice: ABLattice,
        flow_strength: float = 0.7,  # Weight for right-moving velocities (0.5 = equal)
        logger=None,
    ):
        super().__init__(logger)
        self.lattice = lattice
        self.flow_strength = np.clip(flow_strength, 0.5, 1.0)

        # Normalize so probabilities sum to 1
        right_weight = flow_strength
        left_weight = 2.0 - flow_strength  # Ensure avg = 1

        # Velocity indices
        # Right movingg 1, 5, 8
        # Left moving 3, 6, 7
        # Neutral: 0, 2, 4 (weight = 1)
        amplitudes = np.zeros(9, dtype=complex)
        amplitudes[0] = 1.0  # Rest particle
        amplitudes[1] = right_weight  # Right
        amplitudes[2] = 1.0  # Up
        amplitudes[3] = left_weight  # Left
        amplitudes[4] = 1.0  # Down
        amplitudes[5] = right_weight  # Up-Right
        amplitudes[6] = left_weight  # Up-Left
        amplitudes[7] = left_weight  # Down-Left
        amplitudes[8] = right_weight  # Down-Right

        # Normalize
        norm = np.linalg.norm(amplitudes)
        self.amplitudes = amplitudes / norm

        self.logger.info(f"Creating circuit {str(self)}...")
        self.circuit = self.create_circuit()

    def create_circuit(self) -> QuantumCircuit:
        from qlbm.components.common.primitives import TruncatedQFT

        circuit = QuantumCircuit(*self.lattice.registers)

        # First, create equal superposition with TruncatedQFT (standard approach)
        circuit.compose(
            TruncatedQFT(
                self.lattice.num_velocity_qubits,
                self.lattice.num_velocities_per_point,
                self.logger,
            ).circuit,
            qubits=self.lattice.velocity_index(),
            inplace=True,
        )

        # Then apply amplitude adjustment to create flow bias
        # This is done by applying a diagonal unitary that scales amplitudes
        self._apply_flow_bias(circuit)

        return circuit

    def _apply_flow_bias(self, circuit: QuantumCircuit):
        """Apply diagonal unitary to create velocity bias."""
        from qiskit import QuantumRegister
        from qiskit.quantum_info import Operator

        num_v_qubits = self.lattice.num_velocity_qubits
        v_reg = QuantumRegister(num_v_qubits, name="v")

        # Build diagonal unitary: exp(i * theta_i) for each basis state
        # We want to boost right-moving and suppress left-moving velocities
        diag_phases = np.zeros(9)
        for i in range(9):
            if i in [1, 5, 8]:  # Right-moving
                diag_phases[i] = np.arcsin(self.flow_strength)
            elif i in [3, 6, 7]:  # Left-moving
                diag_phases[i] = -np.arcsin(self.flow_strength)

        # Create diagonal unitary
        diag_matrix = np.diag(np.exp(1j * diag_phases))
        # Pad to full size (2^num_v_qubits)
        full_size = 2 ** num_v_qubits
        U = np.eye(full_size, dtype=complex)
        U[:9, :9] = diag_matrix
        U = Operator(U)

        # Apply to velocity register
        circuit.append(U, self.lattice.velocity_index())

    def __str__(self):
        return f"[Primitive FlowInitialConditions with flow_strength={self.flow_strength}]"