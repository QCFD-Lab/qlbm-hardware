from architecture_aware_phasepoly_optimizer import *
from components.phase_poly_optimizer.architecture_aware_phasepoly_optimizer import _qubit_index_map


def unitary_equiv_up_to_global_phase(
    circuit_a: QuantumCircuit,
    circuit_b: QuantumCircuit,
    *,
    atol: float = 1e-9,
) -> bool:
    """
    Compare two small circuits by converting them to dense unitaries and checking
    equality up to global phase.

    This is intended for toy tests only. It uses qiskit's `Operator` class to obtain
    the dense matrix representation, so it should only be used on small circuits.
    """
    u_a = Operator(circuit_a).data
    u_b = Operator(circuit_b).data

    if u_a.shape != u_b.shape:
        return False

    # Choose a stable pivot on the largest-magnitude entry.
    idx = np.unravel_index(np.argmax(np.abs(u_b)), u_b.shape)
    denom = u_b[idx]
    if abs(denom) < atol:
        return np.allclose(u_a, u_b, atol=atol, rtol=0.0)

    phase = u_a[idx] / denom
    if abs(phase) < atol:
        return False
    phase /= abs(phase)
    return np.allclose(u_a, phase * u_b, atol=atol, rtol=0.0)



def _toy_block_two_qubits() -> Tuple[QuantumCircuit, CouplingMap]:
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rz(0.37, 1)
    qc.cx(0, 1)
    qc.rz(-0.21, 0)
    cmap = CouplingMap([(0, 1), (1, 0)])
    return qc, cmap



def _toy_block_three_qubit_line() -> Tuple[QuantumCircuit, CouplingMap]:
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.rz(0.11, 1)
    qc.cx(1, 2)
    qc.rz(-0.43, 2)
    qc.cx(0, 1)
    qc.rz(0.29, 1)
    qc.cx(1, 2)
    cmap = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1)])
    return qc, cmap



def _toy_block_four_qubit_line() -> Tuple[QuantumCircuit, CouplingMap]:
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.rz(0.17, 1)
    qc.cx(1, 2)
    qc.rz(-0.31, 2)
    qc.cx(2, 3)
    qc.rz(0.52, 3)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.rz(-0.08, 1)
    qc.cx(1, 2)
    qc.rz(0.26, 2)
    cmap = CouplingMap(
        [
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
        ]
    )
    return qc, cmap



def _toy_block_three_qubit_star() -> Tuple[QuantumCircuit, CouplingMap]:
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.rz(0.5, 1)
    qc.cx(0, 2)
    qc.rz(-0.2, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.rz(0.125, 0)
    cmap = CouplingMap([(0, 1), (1, 0), (0, 2), (2, 0)])
    return qc, cmap



def run_toy_phasepoly_tests(*, atol: float = 1e-9) -> List[Dict[str, object]]:
    """
    Run a few small end-to-end checks for the current optimizer implementation.

    Each test optimizes a toy {cx, rz} circuit and checks full unitary equivalence
    up to global phase. This is much stronger than checking only the output on
    |0...0>, and it will catch mistakes in the residual linear transform too.
    """
    test_builders = [
        ("two_qubits", _toy_block_two_qubits),
        ("three_qubit_line", _toy_block_three_qubit_line),
        ("four_qubit_line", _toy_block_four_qubit_line),
        ("three_qubit_star", _toy_block_three_qubit_star),
    ]

    results: List[Dict[str, object]] = []
    optimizer = ArchitectureAwarePhasePolyOptimizer()

    for name, builder in test_builders:
        circuit, cmap = builder()
        optimized = optimizer.optimize(circuit, cmap)
        equivalent = unitary_equiv_up_to_global_phase(circuit, optimized, atol=atol)
        results.append(
            {
                "name": name,
                "equivalent": equivalent,
                "input_cx": circuit.count_ops().get("cx", 0),
                "optimized_cx": optimized.count_ops().get("cx", 0),
                "input_depth": circuit.depth(),
                "optimized_depth": optimized.depth(),
                "optimized_circuit": optimized,
            }
        )

    return results


def extract_linear_out_parities_from_cx_circuit(circuit: QuantumCircuit) -> List[BitVec]:
    """
    Extract the forward GF(2) linear transform induced by a CNOT-only circuit.
    Each output row is represented as a parity bit-vector over the input bits.
    """
    if any(inst.operation.name not in {"cx", "barrier"} for inst in circuit.data):
        raise ValueError("Circuit must contain only cx and optional barriers.")
    rows = gf2_identity(circuit.num_qubits)
    qmap = _qubit_index_map(circuit)
    for inst in circuit.data:
        if inst.operation.name == "barrier":
            continue
        c = qmap[inst.qubits[0]]
        t = qmap[inst.qubits[1]]
        rows[t] = gf2_add_rows(rows[c], rows[t])
    return rows



def linear_transform_matches(matrix_rows: Sequence[BitVec], circuit: QuantumCircuit) -> bool:
    actual = extract_linear_out_parities_from_cx_circuit(circuit)
    return list(matrix_rows) == actual



def random_invertible_gf2_matrix(n: int, rng: np.random.Generator) -> List[BitVec]:
    while True:
        mat = rng.integers(0, 2, size=(n, n), endpoint=False).tolist()
        try:
            _ = gf2_inverse(mat)
            return gf2_matrix_to_rows(mat)
        except ValueError:
            continue



def run_linear_synthesis_sanity_tests(
    *,
    max_qubits: int = 5,
    trials_per_qubits: int = 10,
    seed: int = 1234,
) -> List[Dict[str, object]]:
    """
    Directly test the all-to-all linear synthesizer on random invertible GF(2) matrices.
    This remains useful as a debugging baseline for the unconstrained helper.
    """
    rng = np.random.default_rng(seed)
    results: List[Dict[str, object]] = []
    for n in range(1, max_qubits + 1):
        for trial in range(trials_per_qubits):
            rows = random_invertible_gf2_matrix(n, rng)
            qc = synthesize_linear_transform_all_to_all(rows)
            ok = linear_transform_matches(rows, qc)
            results.append(
                {
                    "num_qubits": n,
                    "trial": trial,
                    "ok": ok,
                    "rows": [parity_to_str(r) for r in rows],
                    "cx": qc.count_ops().get("cx", 0),
                    "depth": qc.depth(),
                }
            )
    return results



def run_steiner_gauss_sanity_tests() -> List[Dict[str, object]]:
    """
    Directly test the architecture-aware residual synthesizer on a few small connected
    coupling maps. This isolates the new Steiner-Gauss-style implementation.
    """
    test_cases: List[Tuple[str, List[BitVec], CouplingMap]] = [
        (
            "line_3",
            [
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1),
            ],
            CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1)]),
        ),
        (
            "line_4",
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
    ]

    results: List[Dict[str, object]] = []
    for name, rows, cmap in test_cases:
        qc = synthesize_linear_transform_steiner_gauss(rows, cmap)
        ok = linear_transform_matches(rows, qc)
        results.append(
            {
                "name": name,
                "ok": ok,
                "rows": [parity_to_str(r) for r in rows],
                "cx": qc.count_ops().get("cx", 0),
                "depth": qc.depth(),
                "circuit": qc,
            }
        )
    return results



def diagnose_toy_case(name: str) -> Dict[str, object]:
    builders = {
        "two_qubits": _toy_block_two_qubits,
        "three_qubit_line": _toy_block_three_qubit_line,
        "four_qubit_line": _toy_block_four_qubit_line,
        "three_qubit_star": _toy_block_three_qubit_star,
    }
    if name not in builders:
        raise ValueError(f"Unknown toy case: {name}")

    circuit, cmap = builders[name]()
    pp = extract_phase_polynomial(circuit)

    phase_circuit, emitted = synthesize_phase_support_architecture_aware(pp, cmap)
    residual_rows = residual_linear_transform(pp.out_parities, emitted)
    residual_circuit = synthesize_linear_transform_steiner_gauss(residual_rows, cmap)

    composed = QuantumCircuit(circuit.num_qubits)
    composed.compose(phase_circuit, inplace=True)
    composed.compose(residual_circuit, inplace=True)

    return {
        "name": name,
        "original": circuit,
        "phase_support": phase_circuit,
        "emitted_out_parities": [parity_to_str(p) for p in emitted],
        "desired_out_parities": [parity_to_str(p) for p in pp.out_parities],
        "residual_rows": [parity_to_str(p) for p in residual_rows],
        "residual_circuit": residual_circuit,
        "residual_matches_rows": linear_transform_matches(residual_rows, residual_circuit),
        "equiv_total": unitary_equiv_up_to_global_phase(circuit, composed),
        "composed": composed,
    }


# =========================
# Minimal self-check helper
# =========================



def summarize_phase_poly(block: QuantumCircuit) -> Dict[str, object]:
    """Convenience helper for quick debugging in notebooks/tests."""
    pp = extract_phase_polynomial(block)
    return {
        "num_qubits": pp.num_qubits,
        "support": [parity_to_str(p) for p in pp.support()],
        "zphases": {parity_to_str(k): v for k, v in pp.zphases.items()},
        "out_parities": [parity_to_str(p) for p in pp.out_parities],
        "out_matrix": gf2_matrix_from_rows(pp.out_parities),
    }



def compare_phase_support_only(input_block: QuantumCircuit, coupling_map: CouplingMap) -> Dict[str, object]:
    """
    Debug helper that exposes the intermediate architecture-aware phase-support synthesis.
    """
    pp = extract_phase_polynomial(input_block)
    phase_circuit, emitted = synthesize_phase_support_architecture_aware(pp, coupling_map)
    residual = residual_linear_transform(pp.out_parities, emitted)
    return {
        "phase_support_circuit": phase_circuit,
        "emitted_out_parities": [parity_to_str(p) for p in emitted],
        "desired_out_parities": [parity_to_str(p) for p in pp.out_parities],
        "residual": [parity_to_str(p) for p in residual],
    }




lin_results = run_linear_synthesis_sanity_tests()
bad = [r for r in lin_results if not r["ok"]]
print("bad linear tests:", len(bad))
if bad:
    print(bad[0])

diag = diagnose_toy_case("three_qubit_line")
print("residual_matches_rows:", diag["residual_matches_rows"])
print("equiv_total:", diag["equiv_total"])
print("desired_out_parities:", diag["desired_out_parities"])
print("emitted_out_parities:", diag["emitted_out_parities"])
print("residual_rows:", diag["residual_rows"])


results = run_toy_phasepoly_tests()
for r in results:
    print(r["name"], r["equivalent"], r["input_cx"], r["optimized_cx"])

sg_results = run_steiner_gauss_sanity_tests()
for r in sg_results:
    print(r["name"], r["ok"], r["cx"], r["depth"])