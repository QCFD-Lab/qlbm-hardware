from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.transpiler import CouplingMap
import pytest

from components.phase_poly_optimizer import architecture_aware_phasepoly_optimizer as aa
from components.phase_poly_optimizer.architecture_aware_phasepoly_optimizer import (
    ArchitectureAwarePhasePolyOptimizer,
    BitVec,
    PhasePolynomialBlock,
    extract_phase_polynomial,
    find_phase_polynomial_blocks,
    gf2_add_rows,
    gf2_identity,
    parity_to_str,
    residual_linear_transform,
    synthesize_linear_transform_steiner_gauss,
    synthesize_phase_support_architecture_aware,
    synthesize_linear_transform_architecture_aware,
    unitary_equiv_up_to_global_phase,
    _qubit_index_map,
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


def _extract_linear_out_parities_from_cx_circuit(circuit: QuantumCircuit) -> list[BitVec]:
    if any(inst.operation.name not in {"cx", "barrier"} for inst in circuit.data):
        raise ValueError("Circuit must contain only cx and optional barriers.")

    rows = gf2_identity(circuit.num_qubits)
    qmap = _qubit_index_map(circuit)
    for instruction in circuit.data:
        if instruction.operation.name == "barrier":
            continue
        control = qmap[instruction.qubits[0]]
        target = qmap[instruction.qubits[1]]
        rows[target] = gf2_add_rows(rows[control], rows[target])
    return rows


def _linear_transform_matches(matrix_rows: list[BitVec], circuit: QuantumCircuit) -> bool:
    return matrix_rows == _extract_linear_out_parities_from_cx_circuit(circuit)


def _toy_block_two_qubits() -> tuple[QuantumCircuit, CouplingMap]:
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.rz(0.37, 1)
    circuit.cx(0, 1)
    circuit.rz(-0.21, 0)
    coupling_map = CouplingMap([(0, 1), (1, 0)])
    return circuit, coupling_map


def _toy_block_three_qubit_line() -> tuple[QuantumCircuit, CouplingMap]:
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.rz(0.11, 1)
    circuit.cx(1, 2)
    circuit.rz(-0.43, 2)
    circuit.cx(0, 1)
    circuit.rz(0.29, 1)
    circuit.cx(1, 2)
    coupling_map = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1)])
    return circuit, coupling_map


def _toy_block_four_qubit_line() -> tuple[QuantumCircuit, CouplingMap]:
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.rz(0.17, 1)
    circuit.cx(1, 2)
    circuit.rz(-0.31, 2)
    circuit.cx(2, 3)
    circuit.rz(0.52, 3)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    circuit.rz(-0.08, 1)
    circuit.cx(1, 2)
    circuit.rz(0.26, 2)
    coupling_map = CouplingMap(
        [
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
        ]
    )
    return circuit, coupling_map


def _toy_block_three_qubit_star() -> tuple[QuantumCircuit, CouplingMap]:
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.rz(0.5, 1)
    circuit.cx(0, 2)
    circuit.rz(-0.2, 2)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.rz(0.125, 0)
    coupling_map = CouplingMap([(0, 1), (1, 0), (0, 2), (2, 0)])
    return circuit, coupling_map


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


@pytest.mark.parametrize(
    "builder",
    [
        _toy_block_two_qubits,
        _toy_block_three_qubit_line,
        _toy_block_four_qubit_line,
        _toy_block_three_qubit_star,
    ],
)
def test_optimize_pure_phase_polynomial_blocks_preserves_unitary_and_local_cxs(builder) -> None:
    circuit, coupling_map = builder()

    optimized = ArchitectureAwarePhasePolyOptimizer().optimize(circuit, coupling_map)

    assert unitary_equiv_up_to_global_phase(circuit, optimized)
    _assert_cxs_are_local(optimized, coupling_map)


def test_phase_support_and_residual_compose_to_original_block() -> None:
    circuit, coupling_map = _toy_block_three_qubit_line()
    phase_poly = extract_phase_polynomial(circuit)

    phase_circuit, emitted = synthesize_phase_support_architecture_aware(
        phase_poly,
        coupling_map,
    )
    residual_rows = residual_linear_transform(phase_poly.out_parities, emitted)
    residual_circuit = synthesize_linear_transform_architecture_aware(
        residual_rows,
        coupling_map,
    )

    composed = QuantumCircuit(circuit.num_qubits)
    composed.compose(phase_circuit, inplace=True)
    composed.compose(residual_circuit, inplace=True)

    assert unitary_equiv_up_to_global_phase(circuit, composed)
    assert _linear_transform_matches(residual_rows, residual_circuit)
    _assert_cxs_are_local(composed, coupling_map)


@pytest.mark.parametrize(
    ("rows", "coupling_map"),
    [
        (
            [
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1),
            ],
            CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1)]),
        ),
        (
            [
                (1, 0, 0, 0),
                (1, 1, 0, 0),
                (0, 1, 1, 0),
                (1, 1, 1, 1),
            ],
            CouplingMap(
                [
                    (0, 1), (1, 0),
                    (1, 2), (2, 1),
                    (2, 3), (3, 2),
                ]
            ),
        ),
    ],
    ids=["line_3", "line_4"],
)
def test_steiner_gauss_synthesizes_requested_linear_transform(
    rows: list[BitVec],
    coupling_map: CouplingMap,
) -> None:
    circuit = synthesize_linear_transform_steiner_gauss(rows, coupling_map)

    assert _linear_transform_matches(rows, circuit), [
        parity_to_str(row) for row in rows
    ]
    _assert_cxs_are_local(circuit, coupling_map)


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


def test_residual_synthesis_chooses_lowest_cost_reduce_order(monkeypatch) -> None:
    rows = [
        (1, 0),
        (1, 1),
    ]
    coupling_map = CouplingMap([(0, 1), (1, 0)])

    def fake_steiner_gauss(matrix_rows, coupling_map, *, reduce_order=None):
        circuit = QuantumCircuit(2)
        if reduce_order == [1, 0]:
            circuit.cx(0, 1)
            circuit.cx(1, 0)
            return circuit
        if reduce_order == [0, 1]:
            circuit.cx(0, 1)
            return circuit
        raise ValueError("unused candidate order")

    monkeypatch.setattr(
        aa,
        "synthesize_linear_transform_steiner_gauss",
        fake_steiner_gauss,
    )

    result = aa.synthesize_linear_transform_architecture_aware_result(
        rows,
        coupling_map,
    )

    assert result.method == "steiner_gauss:natural"
    assert result.reduce_order == [0, 1]
    assert result.circuit.count_ops().get("cx", 0) == 1
