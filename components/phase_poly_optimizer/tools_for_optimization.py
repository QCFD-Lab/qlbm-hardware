from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, Qubit
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap


BitVec = Tuple[int, ...]
Matrix = List[List[int]]


def gf2_identity(n: int) -> List[BitVec]:
    return [tuple(1 if i == j else 0 for j in range(n)) for i in range(n)]


def gf2_add_rows(a: BitVec, b: BitVec) -> BitVec:
    if len(a) != len(b):
        raise ValueError("GF(2) row sizes do not match.")
    return tuple((x ^ y) for x, y in zip(a, b))


def gf2_matrix_from_rows(rows: Sequence[BitVec]) -> Matrix:
    return [list(row) for row in rows]


def gf2_matrix_to_rows(matrix: Matrix) -> List[BitVec]:
    return [tuple(int(x) for x in row) for row in matrix]


def gf2_matmul(a: Matrix, b: Matrix) -> Matrix:
    n = len(a)
    if n == 0:
        return []
    k = len(a[0])
    if any(len(row) != k for row in a):
        raise ValueError("Left matrix is jagged.")
    if len(b) != k:
        raise ValueError("Inner GF(2) matrix dimensions do not match.")
    m = len(b[0])
    if any(len(row) != m for row in b):
        raise ValueError("Right matrix is jagged.")

    out = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            acc = 0
            for t in range(k):
                acc ^= a[i][t] & b[t][j]
            out[i][j] = acc
    return out


def gf2_inverse(matrix: Matrix) -> Matrix:
    n = len(matrix)
    if n == 0:
        return []
    if any(len(row) != n for row in matrix):
        raise ValueError("GF(2) inverse requires a square matrix.")

    identity = gf2_matrix_from_rows(gf2_identity(n))
    aug = [row[:] + ident_row[:] for row, ident_row in zip(matrix, identity)]

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if aug[row][col] == 1:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Matrix is not invertible over GF(2).")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        for row in range(n):
            if row != col and aug[row][col] == 1:
                for j in range(2 * n):
                    aug[row][j] ^= aug[col][j]

    return [row[n:] for row in aug]


def parity_to_str(parity: BitVec) -> str:
    return "".join(str(x) for x in parity)


@dataclass
class PhasePolynomial:
    """
    Qiskit-native representation of a phase-polynomial block.
    """

    zphases: "OrderedDict[BitVec, Any]"
    out_parities: List[BitVec]
    num_qubits: int

    def support(self) -> List[BitVec]:
        return list(self.zphases.keys())

    def remove_zero_phases(self, atol: float = 1e-12) -> None:
        kept = OrderedDict()
        for parity, phase in self.zphases.items():
            try:
                numeric_phase = float(phase)
            except (TypeError, ValueError):
                kept[parity] = phase
                continue
            if abs(numeric_phase) > atol:
                kept[parity] = numeric_phase
        self.zphases = kept


@dataclass(frozen=True)
class PhasePolynomialBlock:
    start: int
    end: int
    active_qubits: List[int]
    phase_instruction_indices: List[int]
    passthrough_instruction_indices: List[int]


@dataclass
class BlockOptimizationReport:
    start: int
    end: int
    active_qubits: List[int]
    connected_active_subgraph: bool
    support_size: int
    original_cx: int
    phase_support_cx: int
    residual_cx: int
    candidate_cx: int
    final_cx: int
    kept_original: bool
    residual_method: str = ""
    residual_reduce_order: Optional[List[int]] = None


@dataclass
class OptimizationRunReport:
    circuit_num_qubits: int
    num_blocks: int
    block_reports: List[BlockOptimizationReport]

    @property
    def total_original_cx(self) -> int:
        return sum(block.original_cx for block in self.block_reports)

    @property
    def total_candidate_cx(self) -> int:
        return sum(block.candidate_cx for block in self.block_reports)

    @property
    def total_final_cx(self) -> int:
        return sum(block.final_cx for block in self.block_reports)


@dataclass
class LinearSynthesisResult:
    circuit: QuantumCircuit
    method: str
    reduce_order: Optional[List[int]]


def _qubit_index_map(circuit: QuantumCircuit) -> Dict[Qubit, int]:
    return {qubit: i for i, qubit in enumerate(circuit.qubits)}


def _clbit_index_map(circuit: QuantumCircuit) -> Dict[Clbit, int]:
    return {clbit: i for i, clbit in enumerate(circuit.clbits)}


def _is_phase_polynomial_instruction(instruction, *, allow_barriers: bool = True) -> bool:
    name = instruction.operation.name
    if name == "barrier" and allow_barriers:
        return True
    if name == "rz":
        return len(instruction.qubits) == 1 and len(instruction.clbits) == 0
    if name == "cx":
        return len(instruction.qubits) == 2 and len(instruction.clbits) == 0
    return False


def _is_soft_passthrough_instruction(instruction) -> bool:
    """
    Return whether this instruction may be moved across a phase-polynomial block
    when it acts on qubits disjoint from the block's phase support.
    """

    return (
        instruction.operation.name in {"x", "sx"}
        and len(instruction.clbits) == 0
        and len(instruction.qubits) == 1
    )


def find_phase_polynomial_blocks(
    circuit: QuantumCircuit,
    *,
    allow_barriers: bool = True,
) -> List[PhasePolynomialBlock]:
    """
    Find maximal contiguous phase-polynomial blocks in a mixed circuit.
    """

    qmap = _qubit_index_map(circuit)
    blocks: List[PhasePolynomialBlock] = []
    data = circuit.data
    num_instructions = len(data)
    index = 0

    while index < num_instructions:
        while index < num_instructions and not _is_phase_polynomial_instruction(data[index], allow_barriers=allow_barriers):
            index += 1
        if index >= num_instructions:
            break

        segment_end = index
        while (
            segment_end + 1 < num_instructions
            and (
                _is_phase_polynomial_instruction(data[segment_end + 1], allow_barriers=allow_barriers)
                or _is_soft_passthrough_instruction(data[segment_end + 1])
            )
        ):
            segment_end += 1

        suffix_phase_qubits: Dict[int, set[int]] = {segment_end + 1: set()}
        for pos in range(segment_end, index - 1, -1):
            suffix_phase_qubits[pos] = set(suffix_phase_qubits[pos + 1])
            if _is_phase_polynomial_instruction(data[pos], allow_barriers=allow_barriers):
                suffix_phase_qubits[pos].update(qmap[qubit] for qubit in data[pos].qubits)

        cursor = index
        while cursor <= segment_end:
            while (
                cursor <= segment_end
                and not _is_phase_polynomial_instruction(data[cursor], allow_barriers=allow_barriers)
            ):
                cursor += 1
            if cursor > segment_end:
                break

            start = cursor
            active_phase_qubits: set[int] = set()
            phase_instruction_indices: List[int] = []
            passthrough_instruction_indices: List[int] = []

            while cursor <= segment_end:
                instruction = data[cursor]
                if _is_phase_polynomial_instruction(instruction, allow_barriers=allow_barriers):
                    phase_instruction_indices.append(cursor)
                    active_phase_qubits.update(qmap[qubit] for qubit in instruction.qubits)
                    cursor += 1
                    continue

                if _is_soft_passthrough_instruction(instruction):
                    instruction_qubits = {qmap[qubit] for qubit in instruction.qubits}
                    future_phase_qubits = suffix_phase_qubits[cursor + 1]
                    if instruction_qubits.isdisjoint(active_phase_qubits | future_phase_qubits):
                        passthrough_instruction_indices.append(cursor)
                        cursor += 1
                        continue
                break

            if phase_instruction_indices:
                blocks.append(
                    PhasePolynomialBlock(
                        start=start,
                        end=cursor - 1,
                        active_qubits=sorted(active_phase_qubits),
                        phase_instruction_indices=phase_instruction_indices,
                        passthrough_instruction_indices=passthrough_instruction_indices,
                    )
                )
            else:
                cursor += 1

        index = segment_end + 1

    return blocks


def _extract_compact_block(
    circuit: QuantumCircuit,
    block: PhasePolynomialBlock,
) -> QuantumCircuit:
    local_of_global = {
        global_index: local_index
        for local_index, global_index in enumerate(block.active_qubits)
    }
    qmap = _qubit_index_map(circuit)
    compact = QuantumCircuit(
        len(block.active_qubits),
        name=f"phasepoly_{block.start}_{block.end}",
    )

    for index in block.phase_instruction_indices:
        instruction = circuit.data[index]
        local_qargs = [
            compact.qubits[local_of_global[qmap[qubit]]]
            for qubit in instruction.qubits
        ]
        compact.append(instruction.operation.copy(), local_qargs, [])

    return compact


def _local_coupling_map(coupling_map: CouplingMap, active_qubits: Sequence[int]) -> CouplingMap:
    active_set = set(active_qubits)
    local_of_global = {
        global_index: local_index
        for local_index, global_index in enumerate(active_qubits)
    }
    local_edges = [
        (local_of_global[int(u)], local_of_global[int(v)])
        for u, v in coupling_map.get_edges()
        if int(u) in active_set and int(v) in active_set and int(u) != int(v)
    ]
    return CouplingMap(local_edges)


def _active_qubit_subgraph_is_connected(
    coupling_map: CouplingMap,
    active_qubits: Sequence[int],
) -> bool:
    if len(active_qubits) <= 1:
        return True
    active_set = set(active_qubits)
    graph = nx.Graph()
    graph.add_nodes_from(active_qubits)
    graph.add_edges_from(
        (int(u), int(v))
        for u, v in coupling_map.get_edges()
        if int(u) in active_set and int(v) in active_set and int(u) != int(v)
    )
    return nx.is_connected(graph)


def _new_like_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    out = circuit.copy_empty_like()
    out.global_phase = circuit.global_phase
    out.name = circuit.name
    if circuit.metadata is not None:
        out.metadata = dict(circuit.metadata)
    return out


def _append_original_instruction(
    out: QuantumCircuit,
    source: QuantumCircuit,
    instruction,
) -> None:
    qmap = _qubit_index_map(source)
    cmap = _clbit_index_map(source)
    qargs = [out.qubits[qmap[qubit]] for qubit in instruction.qubits]
    cargs = [out.clbits[cmap[clbit]] for clbit in instruction.clbits]
    out.append(instruction.operation.copy(), qargs, cargs)


def _append_compact_circuit(
    out: QuantumCircuit,
    compact: QuantumCircuit,
    active_qubits: Sequence[int],
) -> None:
    for instruction in compact.data:
        qargs = [
            out.qubits[active_qubits[compact.find_bit(qubit).index]]
            for qubit in instruction.qubits
        ]
        out.append(instruction.operation.copy(), qargs, [])


def _append_instruction_indices(
    out: QuantumCircuit,
    source: QuantumCircuit,
    instruction_indices: Sequence[int],
) -> None:
    for instruction_index in instruction_indices:
        _append_original_instruction(out, source, source.data[instruction_index])


def _append_instruction_range(
    out: QuantumCircuit,
    source: QuantumCircuit,
    start: int,
    end: int,
) -> None:
    for instruction_index in range(start, end + 1):
        _append_original_instruction(out, source, source.data[instruction_index])


def is_phase_polynomial_block(
    circuit: QuantumCircuit,
    *,
    allow_barriers: bool = True,
) -> bool:
    """Return True iff the circuit contains only CX, RZ, and optionally barriers."""

    for instruction in circuit.data:
        if not _is_phase_polynomial_instruction(instruction, allow_barriers=allow_barriers):
            return False
    return True


def extract_phase_polynomial(block: QuantumCircuit) -> PhasePolynomial:
    """
    Extract a phase-polynomial representation from a Qiskit CX/RZ block.
    """

    if not is_phase_polynomial_block(block):
        raise ValueError("Block is not a valid phase-polynomial block (allowed: cx, rz, barrier).")

    num_qubits = block.num_qubits
    qubit_to_index = _qubit_index_map(block)
    current_parities: List[BitVec] = gf2_identity(num_qubits)
    zphases: "OrderedDict[BitVec, Any]" = OrderedDict()

    for instruction in block.data:
        op = instruction.operation
        name = op.name

        if name == "barrier":
            continue

        if name == "rz":
            target_idx = qubit_to_index[instruction.qubits[0]]
            parity = current_parities[target_idx]
            angle = op.params[0]
            zphases[parity] = zphases.get(parity, 0.0) + angle
            continue

        if name == "cx":
            control_idx = qubit_to_index[instruction.qubits[0]]
            target_idx = qubit_to_index[instruction.qubits[1]]
            current_parities[target_idx] = gf2_add_rows(
                current_parities[control_idx],
                current_parities[target_idx],
            )
            continue

        raise ValueError(f"Unsupported gate in phase-polynomial extraction: {name}")

    phase_poly = PhasePolynomial(
        zphases=zphases,
        out_parities=current_parities,
        num_qubits=num_qubits,
    )
    phase_poly.remove_zero_phases()
    return phase_poly


def _apply_row_add(matrix: Matrix, control: int, target: int) -> None:
    row_control = matrix[control]
    row_target = matrix[target]
    matrix[target] = [a ^ b for a, b in zip(row_control, row_target)]


def _apply_row_swap_via_xors(matrix: Matrix, a: int, b: int, ops: List[Tuple[int, int]]) -> None:
    _apply_row_add(matrix, b, a)
    ops.append((b, a))
    _apply_row_add(matrix, a, b)
    ops.append((a, b))
    _apply_row_add(matrix, b, a)
    ops.append((b, a))


def _row_add_sequence_for_linear_transform(matrix_rows: Sequence[BitVec]) -> List[Tuple[int, int]]:
    """
    Return all-to-all row-add operations whose CNOT circuit implements `matrix_rows`.
    """

    matrix = gf2_matrix_from_rows(matrix_rows)
    n = len(matrix)
    if n == 0:
        return []
    if any(len(row) != n for row in matrix):
        raise ValueError("Linear transform synthesis requires a square matrix.")

    work = [row[:] for row in matrix]
    reduction_ops: List[Tuple[int, int]] = []

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if work[row][col] == 1:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Residual linear transform is not invertible over GF(2).")
        if pivot != col:
            _apply_row_swap_via_xors(work, col, pivot, reduction_ops)

        for row in range(n):
            if row != col and work[row][col] == 1:
                _apply_row_add(work, col, row)
                reduction_ops.append((col, row))

    return list(reversed(reduction_ops))


def residual_linear_transform(
    desired_out_parities: Sequence[BitVec],
    emitted_out_parities: Sequence[BitVec],
) -> List[BitVec]:
    desired = gf2_matrix_from_rows(desired_out_parities)
    emitted = gf2_matrix_from_rows(emitted_out_parities)
    emitted_inv = gf2_inverse(emitted)
    residual = gf2_matmul(desired, emitted_inv)
    return gf2_matrix_to_rows(residual)


def _phasepoly_cost_tuple(circuit: QuantumCircuit) -> Tuple[int, int, int]:
    """
    Cost tuple used to compare two phase-polynomial block implementations.
    """

    ops = circuit.count_ops()
    cx = int(ops.get("cx", 0))
    depth = int(circuit.depth() or 0)
    size = int(sum(int(v) for v in ops.values()))
    return (cx, depth, size)


def _choose_better_phasepoly_block(
    original: QuantumCircuit,
    candidate: QuantumCircuit,
) -> QuantumCircuit:
    if _phasepoly_cost_tuple(candidate) < _phasepoly_cost_tuple(original):
        return candidate
    return original


def _cx_gates_respect_coupling_map(circuit: QuantumCircuit, coupling_map: CouplingMap) -> bool:
    """
    Return whether every CX in `circuit` is adjacent in the undirected architecture.
    """

    graph = nx.Graph()
    graph.add_nodes_from(range(circuit.num_qubits))
    graph.add_edges_from(
        (int(u), int(v))
        for u, v in coupling_map.get_edges()
        if u < circuit.num_qubits and v < circuit.num_qubits
    )
    qmap = _qubit_index_map(circuit)

    for instruction in circuit.data:
        if instruction.operation.name != "cx":
            continue
        control = qmap[instruction.qubits[0]]
        target = qmap[instruction.qubits[1]]
        if not graph.has_edge(control, target):
            return False
    return True


def unitary_equiv_up_to_global_phase(
    circuit_a: QuantumCircuit,
    circuit_b: QuantumCircuit,
    *,
    atol: float = 1e-9,
) -> bool:
    """
    Compare two small circuits by dense unitaries, up to global phase.
    """

    u_a = Operator(circuit_a).data
    u_b = Operator(circuit_b).data

    if u_a.shape != u_b.shape:
        return False

    idx = np.unravel_index(np.argmax(np.abs(u_b)), u_b.shape)
    denom = u_b[idx]
    if abs(denom) < atol:
        return np.allclose(u_a, u_b, atol=atol, rtol=0.0)

    phase = u_a[idx] / denom
    if abs(phase) < atol:
        return False
    phase /= abs(phase)
    return np.allclose(u_a, phase * u_b, atol=atol, rtol=0.0)
