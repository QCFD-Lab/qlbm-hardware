import math
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler import CouplingMap

from components.phase_poly_optimizer.a2a_phase_poly import (
    A2APhasePoly,
    synthesize_linear_transform_all_to_all,
    synthesize_phase_support_all_to_all,
)
from components.phase_poly_optimizer.tools_for_optimization import (
    BitVec,
    extract_phase_polynomial,
    find_phase_polynomial_blocks,
    gf2_add_rows,
    gf2_identity,
    residual_linear_transform,
    unitary_equiv_up_to_global_phase,
    _qubit_index_map,
)
from components.phase_poly_optimizer.optimizer_benchmark_simulation import (
    optimize_circuit_with_phase_poly,
)


def _extract_linear_out_parities_from_cx_circuit(circuit: QuantumCircuit) -> list[BitVec]:
    rows = gf2_identity(circuit.num_qubits)
    qmap = _qubit_index_map(circuit)
    for instruction in circuit.data:
        if instruction.operation.name == "barrier":
            continue
        if instruction.operation.name != "cx":
            raise ValueError("Circuit must contain only CX gates.")
        control = qmap[instruction.qubits[0]]
        target = qmap[instruction.qubits[1]]
        rows[target] = gf2_add_rows(rows[control], rows[target])
    return rows


def _toy_block() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.rz(0.17, 1)
    circuit.cx(2, 3)
    circuit.rz(-0.31, 3)
    circuit.cx(1, 2)
    circuit.rz(0.52, 2)
    circuit.cx(0, 3)
    circuit.rz(-0.08, 3)
    circuit.cx(1, 2)
    return circuit


def _same_numeric_phase_map(left, right) -> bool:
    if set(left) != set(right):
        return False
    return all(math.isclose(float(left[key]), float(right[key]), abs_tol=1e-9) for key in left)


def test_phase_support_and_residual_compose_to_original_block_all_to_all() -> None:
    circuit = _toy_block()
    phase_poly = extract_phase_polynomial(circuit)

    phase_circuit, emitted = synthesize_phase_support_all_to_all(phase_poly)
    residual_rows = residual_linear_transform(phase_poly.out_parities, emitted)
    residual_circuit = synthesize_linear_transform_all_to_all(residual_rows)

    composed = QuantumCircuit(circuit.num_qubits)
    composed.compose(phase_circuit, inplace=True)
    composed.compose(residual_circuit, inplace=True)

    assert unitary_equiv_up_to_global_phase(circuit, composed)
    assert _extract_linear_out_parities_from_cx_circuit(residual_circuit) == residual_rows


def test_phase_support_synthesizes_same_phase_polynomial_and_tracked_linear_map() -> None:
    circuit = _toy_block()
    phase_poly = extract_phase_polynomial(circuit)

    phase_circuit, emitted = synthesize_phase_support_all_to_all(phase_poly)
    synthesized_phase_poly = extract_phase_polynomial(phase_circuit)

    assert _same_numeric_phase_map(synthesized_phase_poly.zphases, phase_poly.zphases)
    assert synthesized_phase_poly.out_parities == emitted


def test_random_small_blocks_preserve_unitary_with_forced_replacement() -> None:
    rng = random.Random(20260516)
    angles = [0.1, -0.2, 0.37, -0.91, 1.4, -2.2]

    for num_qubits in range(1, 6):
        for _ in range(20):
            circuit = QuantumCircuit(num_qubits)
            for _ in range(rng.randint(1, 18)):
                if num_qubits == 1 or rng.random() < 0.45:
                    circuit.rz(rng.choice(angles), rng.randrange(num_qubits))
                else:
                    control, target = rng.sample(range(num_qubits), 2)
                    circuit.cx(control, target)

            optimized = A2APhasePoly(
                debug=True,
                keep_original_if_worse=False,
            ).optimize(circuit)

            assert unitary_equiv_up_to_global_phase(circuit, optimized)


def test_all_to_all_optimizer_preserves_mixed_circuit_and_reports() -> None:
    circuit = QuantumCircuit(5, 1)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(0.25, 1)
    circuit.cx(2, 3)
    circuit.rz(-0.5, 3)
    circuit.cx(0, 3)
    circuit.x(4)
    circuit.measure(4, 0)

    optimizer = A2APhasePoly(debug=True)
    optimized = optimizer.optimize(circuit, coupling_map=CouplingMap.from_full(5))

    assert unitary_equiv_up_to_global_phase(
        circuit.remove_final_measurements(inplace=False),
        optimized.remove_final_measurements(inplace=False),
    )
    assert optimized.count_ops().get("measure", 0) == 1
    assert optimizer.last_run_report is not None
    assert optimizer.last_run_report.num_blocks == len(find_phase_polynomial_blocks(circuit))
    assert optimizer.last_run_report.block_reports[0].residual_method == "all_to_all_gauss"


def test_all_to_all_optimizer_accepts_old_constructor_style() -> None:
    circuit = _toy_block()

    optimized = A2APhasePoly(circuit, debug=True).optimize()

    assert unitary_equiv_up_to_global_phase(circuit, optimized)


def test_all_to_all_optimizer_preserves_pure_circuit_structure() -> None:
    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qreg, creg, name="phase_block")
    circuit.metadata = {"case": "pure"}
    circuit.global_phase = 0.25
    circuit.cx(qreg[0], qreg[1])
    circuit.rz(0.4, qreg[1])

    optimized = A2APhasePoly(keep_original_if_worse=False).optimize(circuit)

    assert optimized.qregs == circuit.qregs
    assert optimized.cregs == circuit.cregs
    assert optimized.name == circuit.name
    assert optimized.metadata == circuit.metadata
    assert optimized.global_phase == circuit.global_phase


def test_benchmark_optimizer_selector_can_run_all_to_all() -> None:
    circuit = _toy_block()
    optimized, optimizer = optimize_circuit_with_phase_poly(
        circuit,
        CouplingMap.from_full(circuit.num_qubits),
        optimizer_name="all_to_all",
    )

    assert unitary_equiv_up_to_global_phase(circuit, optimized)
    assert isinstance(optimizer, A2APhasePoly)
