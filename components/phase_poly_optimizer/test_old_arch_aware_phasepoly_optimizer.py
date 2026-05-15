from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.transpiler import CouplingMap

from components.phase_poly_optimizer import architecture_aware_phasepoly_optimizer as aa
from components.phase_poly_optimizer.architecture_aware_phasepoly_optimizer import (
    ArchitectureAwarePhasePolyOptimizer,
    PhasePolynomialBlock,
    find_phase_polynomial_blocks,
    synthesize_linear_transform_architecture_aware,
    unitary_equiv_up_to_global_phase,
)


def _cx_edges(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    edges = []
    for instruction in circuit.data:
        if instruction.operation.name != "cx":
            continue
        control = circuit.find_bit(instruction.qubits[0]).index
        target = circuit.find_bit(instruction.qubits[1]).index
        edges.append((control, target))
    return edges


def _assert_cxs_are_local(circuit: QuantumCircuit, coupling_map: CouplingMap) -> None:
    undirected_edges = {frozenset(edge) for edge in coupling_map.get_edges()}
    for control, target in _cx_edges(circuit):
        assert frozenset((control, target)) in undirected_edges


def test_find_phase_polynomial_blocks_in_mixed_circuit() -> None:
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(0.37, 1)
    circuit.cx(0, 1)
    circuit.x(2)
    circuit.cx(2, 3)
    circuit.rz(-0.21, 3)
    circuit.cx(2, 3)
    circuit.h(3)

    assert find_phase_polynomial_blocks(circuit) == [
        PhasePolynomialBlock(
            start=1,
            end=3,
            active_qubits=[0, 1],
            phase_instruction_indices=[1, 2, 3],
            passthrough_instruction_indices=[],
        ),
        PhasePolynomialBlock(
            start=5,
            end=7,
            active_qubits=[2, 3],
            phase_instruction_indices=[5, 6, 7],
            passthrough_instruction_indices=[],
        ),
    ]


def test_optimize_mixed_circuit_preserves_unitary_and_local_cxs() -> None:
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(0.37, 1)
    circuit.cx(0, 1)
    circuit.x(2)
    circuit.cx(2, 3)
    circuit.rz(-0.21, 3)
    circuit.cx(2, 3)
    circuit.h(3)

    coupling_map = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
    optimized = ArchitectureAwarePhasePolyOptimizer().optimize(circuit, coupling_map)

    assert unitary_equiv_up_to_global_phase(circuit, optimized)
    _assert_cxs_are_local(optimized, coupling_map)
    assert optimized.count_ops().get("h", 0) == 2
    assert optimized.count_ops().get("x", 0) == 1


def test_optimize_remaps_active_block_qubits_to_local_coupling_map() -> None:
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(1, 3)
    circuit.rz(0.25, 3)
    circuit.cx(1, 3)
    circuit.h(2)

    coupling_map = CouplingMap([(0, 1), (1, 0), (1, 3), (3, 1), (2, 3), (3, 2)])
    optimized = ArchitectureAwarePhasePolyOptimizer().optimize(circuit, coupling_map)

    assert find_phase_polynomial_blocks(circuit) == [
        PhasePolynomialBlock(
            start=1,
            end=3,
            active_qubits=[1, 3],
            phase_instruction_indices=[1, 2, 3],
            passthrough_instruction_indices=[],
        )
    ]
    assert unitary_equiv_up_to_global_phase(circuit, optimized)
    _assert_cxs_are_local(optimized, coupling_map)


def test_disconnected_active_block_is_left_unchanged() -> None:
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.rz(0.25, 2)
    circuit.cx(0, 2)
    circuit.h(1)

    coupling_map = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1)])
    optimized = ArchitectureAwarePhasePolyOptimizer().optimize(circuit, coupling_map)

    assert unitary_equiv_up_to_global_phase(circuit, optimized)
    assert _cx_edges(optimized) == [(0, 2), (0, 2)]


def test_mixed_circuit_with_measurements_preserves_classical_structure() -> None:
    circuit = QuantumCircuit(2, 2)
    circuit.cx(0, 1)
    circuit.rz(0.1, 1)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    circuit.x(1)
    circuit.measure(1, 1)

    coupling_map = CouplingMap([(0, 1), (1, 0)])
    optimized = ArchitectureAwarePhasePolyOptimizer().optimize(circuit, coupling_map)

    assert optimized.num_qubits == circuit.num_qubits
    assert optimized.num_clbits == circuit.num_clbits
    assert optimized.count_ops().get("measure", 0) == 2
    assert optimized.count_ops().get("x", 0) == 1
    assert optimized.data[-3].operation.name == "measure"
    assert optimized.data[-2].operation.name == "x"
    assert optimized.data[-1].operation.name == "measure"


def test_mixed_circuit_with_loose_qubits_is_reconstructed_correctly() -> None:
    circuit = QuantumCircuit()
    qubits = [Qubit() for _ in range(3)]
    circuit.add_bits(qubits)
    circuit.h(qubits[0])
    circuit.cx(qubits[0], qubits[1])
    circuit.rz(0.1, qubits[1])
    circuit.cx(qubits[0], qubits[1])
    circuit.x(qubits[2])

    coupling_map = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1)])
    optimized = ArchitectureAwarePhasePolyOptimizer().optimize(circuit, coupling_map)

    assert circuit.qregs == []
    assert optimized.qregs == []
    assert optimized.num_qubits == circuit.num_qubits
    assert unitary_equiv_up_to_global_phase(circuit, optimized)
    _assert_cxs_are_local(optimized, coupling_map)


def test_block_extraction_can_span_disjoint_x_sx_passthroughs() -> None:
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.x(3)
    circuit.rz(0.2, 1)
    circuit.sx(2)
    circuit.cx(0, 1)

    assert find_phase_polynomial_blocks(circuit) == [
        PhasePolynomialBlock(
            start=0,
            end=4,
            active_qubits=[0, 1],
            phase_instruction_indices=[0, 2, 4],
            passthrough_instruction_indices=[1, 3],
        )
    ]


def test_optimize_preserves_disjoint_passthrough_gates_inside_relaxed_block() -> None:
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.x(3)
    circuit.rz(0.2, 1)
    circuit.sx(2)
    circuit.cx(0, 1)

    coupling_map = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
    optimized = ArchitectureAwarePhasePolyOptimizer().optimize(circuit, coupling_map)

    assert unitary_equiv_up_to_global_phase(circuit, optimized)
    assert optimized.count_ops().get("x", 0) == 1
    assert optimized.count_ops().get("sx", 0) == 1
    assert [inst.operation.name for inst in optimized.data] == [inst.operation.name for inst in circuit.data]


def test_residual_synthesis_falls_back_to_local_exact_circuit(monkeypatch) -> None:
    rows = [
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
    ]
    coupling_map = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1)])

    def fail_steiner(*args, **kwargs):
        raise ValueError("forced steiner failure")

    monkeypatch.setattr(aa, "synthesize_linear_transform_steiner_gauss", fail_steiner)

    expected = QuantumCircuit(3)
    expected.cx(0, 1)
    expected.cx(1, 2)
    actual = synthesize_linear_transform_architecture_aware(rows, coupling_map)

    assert unitary_equiv_up_to_global_phase(expected, actual)
    _assert_cxs_are_local(actual, coupling_map)
