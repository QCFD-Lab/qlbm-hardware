from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Operator


BitVec = Tuple[int, ...]
Matrix = List[List[int]]


# =========================
# GF(2) utility functions
# =========================


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
                acc ^= (a[i][t] & b[t][j])
            out[i][j] = acc
    return out



def gf2_inverse(matrix: Matrix) -> Matrix:
    n = len(matrix)
    if n == 0:
        return []
    if any(len(row) != n for row in matrix):
        raise ValueError("GF(2) inverse requires a square matrix.")

    aug = [row[:] + ident_row[:] for row, ident_row in zip(matrix, gf2_matrix_from_rows(gf2_identity(n)))]

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


# =========================
# Data model
# =========================


@dataclass
class PhasePolynomial:
    """
    Qiskit-native representation of a phase polynomial block.

    Attributes
    ----------
    zphases:
        Maps parity bit-vectors to accumulated RZ angles.
        The parity describes the Boolean linear function currently carried by a qubit.
    out_parities:
        The final linear reversible transform, represented as one parity per output qubit.
    num_qubits:
        Number of logical qubits in the block.
    """

    zphases: "OrderedDict[BitVec, float]"
    out_parities: List[BitVec]
    num_qubits: int

    def support(self) -> List[BitVec]:
        return list(self.zphases.keys())

    def remove_zero_phases(self, atol: float = 1e-12) -> None:
        kept = OrderedDict()
        for parity, phase in self.zphases.items():
            if abs(float(phase)) > atol:
                kept[parity] = float(phase)
        self.zphases = kept


@dataclass
class PhaseColumn:
    column_id: int
    bits: List[int]
    angle: float


@dataclass
class ArchitectureGraph:
    """
    Minimal undirected architecture wrapper built from a Qiskit CouplingMap.

    Current limitation:
    if the coupling map contains more qubits than the circuit, this wrapper uses only
    qubits [0, ..., num_qubits-1]. That is enough to implement and test the synthesis
    algorithm itself, but a separate placement stage should be added later.
    """

    graph: nx.Graph

    @classmethod
    def from_coupling_map(cls, coupling_map: CouplingMap, num_qubits: int) -> "ArchitectureGraph":
        g = nx.Graph()
        g.add_nodes_from(range(num_qubits))
        for u, v in coupling_map.get_edges():
            if u < num_qubits and v < num_qubits and u != v:
                g.add_edge(u, v)

        if num_qubits > 1 and not nx.is_connected(g):
            raise ValueError(
                "The induced architecture on qubits [0..n-1] is disconnected. "
                "Add a placement stage or pass a coupling map whose first n qubits are connected."
            )
        return cls(graph=g)

    def neighbors(self, qubit: int, allowed: Optional[Sequence[int]] = None) -> List[int]:
        neigh = list(self.graph.neighbors(qubit))
        if allowed is None:
            return neigh
        allowed_set = set(allowed)
        return [q for q in neigh if q in allowed_set]

    def shortest_path(self, start: int, end: int, allowed: Optional[Sequence[int]] = None) -> List[int]:
        g = self.graph if allowed is None else self.graph.subgraph(list(allowed))
        return nx.shortest_path(g, source=start, target=end)

    def rooted_steiner_tree_children(
        self,
        root: int,
        terminals: Sequence[int],
        allowed: Optional[Sequence[int]] = None,
    ) -> Dict[int, List[int]]:
        """
        Approximate Steiner tree rooted at `root` by taking the union of shortest
        paths from `root` to all terminals inside the allowed subgraph, then
        extracting a rooted BFS tree.
        """
        allowed_nodes = list(self.graph.nodes) if allowed is None else list(allowed)
        sub = self.graph.subgraph(allowed_nodes)
        if root not in sub:
            raise ValueError(f"Root {root} is not present in the allowed architecture subgraph.")

        terminals = sorted(set(terminals))
        for t in terminals:
            if t not in sub:
                raise ValueError(f"Terminal {t} is not present in the allowed architecture subgraph.")

        union_graph = nx.Graph()
        union_graph.add_node(root)
        for terminal in terminals:
            path = nx.shortest_path(sub, source=root, target=terminal)
            union_graph.add_nodes_from(path)
            union_graph.add_edges_from((path[i], path[i + 1]) for i in range(len(path) - 1))

        bfs_tree = nx.bfs_tree(union_graph, source=root)
        children: Dict[int, List[int]] = {node: [] for node in bfs_tree.nodes}
        for parent, child in bfs_tree.edges:
            children[parent].append(child)
            children.setdefault(child, [])
        return children

    def non_cutting_vertices(self, qubits: Sequence[int]) -> List[int]:
        sub_nodes = list(qubits)
        if not sub_nodes:
            return []
        if len(sub_nodes) <= 2:
            return sub_nodes[:]
        sub = self.graph.subgraph(sub_nodes)
        articulation = set(nx.articulation_points(sub))
        out = [q for q in sub_nodes if q not in articulation]
        return out if out else sub_nodes[:]


# =========================
# Block validation/extraction
# =========================


ALLOWED_BLOCK_GATES = {"cx", "rz", "barrier"}



def _qubit_index_map(circuit: QuantumCircuit) -> Dict[Qubit, int]:
    return {qubit: i for i, qubit in enumerate(circuit.qubits)}



def is_phase_polynomial_block(circuit: QuantumCircuit, *, allow_barriers: bool = True) -> bool:
    """Return True iff the circuit contains only CX, RZ, and optionally barriers."""
    for instruction in circuit.data:
        op = instruction.operation
        name = op.name
        if name == "barrier" and allow_barriers:
            continue
        if name not in {"cx", "rz"}:
            return False
        if name == "rz" and len(instruction.qubits) != 1:
            return False
        if name == "cx" and len(instruction.qubits) != 2:
            return False
    return True



def extract_phase_polynomial(block: QuantumCircuit) -> PhasePolynomial:
    """
    Extract a phase-polynomial representation from a Qiskit block containing only CX and RZ.

    This follows the same high-level logic as the reference implementation's
    `PhasePoly.fromCircuit(...)`: track the parity carried by each wire segment,
    accumulate RZ phases on the parity currently present on the target qubit,
    and update the carried parity under CX.
    """
    if not is_phase_polynomial_block(block):
        raise ValueError("Block is not a valid phase-polynomial block (allowed: cx, rz, barrier).")

    num_qubits = block.num_qubits
    qubit_to_index = _qubit_index_map(block)

    # One parity per currently-carried wire label.
    current_parities: List[BitVec] = gf2_identity(num_qubits)
    zphases: "OrderedDict[BitVec, float]" = OrderedDict()

    for instruction in block.data:
        op = instruction.operation
        name = op.name

        if name == "barrier":
            continue

        if name == "rz":
            target_idx = qubit_to_index[instruction.qubits[0]]
            parity = current_parities[target_idx]
            angle = float(op.params[0])
            zphases[parity] = zphases.get(parity, 0.0) + angle
            continue

        if name == "cx":
            control_idx = qubit_to_index[instruction.qubits[0]]
            target_idx = qubit_to_index[instruction.qubits[1]]
            control_parity = current_parities[control_idx]
            target_parity = current_parities[target_idx]
            current_parities[target_idx] = gf2_add_rows(control_parity, target_parity)
            continue

        raise ValueError(f"Unsupported gate in phase-polynomial extraction: {name}")

    phase_poly = PhasePolynomial(
        zphases=zphases,
        out_parities=current_parities,
        num_qubits=num_qubits,
    )
    phase_poly.remove_zero_phases()
    return phase_poly


# =========================
# Recursive synthesizer
# =========================


@dataclass
class _RecursiveSynthState:
    num_qubits: int
    columns: Dict[int, PhaseColumn]
    circuit: QuantumCircuit
    emitted_out_parities: List[BitVec]
    debug: bool = False

    @classmethod
    def from_phase_poly(cls, phase_poly: PhasePolynomial, *, debug: bool = False) -> "_RecursiveSynthState":
        columns: Dict[int, PhaseColumn] = {}
        for cid, (parity, angle) in enumerate(phase_poly.zphases.items()):
            columns[cid] = PhaseColumn(column_id=cid, bits=list(parity), angle=float(angle))
        return cls(
            num_qubits=phase_poly.num_qubits,
            columns=columns,
            circuit=QuantumCircuit(phase_poly.num_qubits),
            emitted_out_parities=gf2_identity(phase_poly.num_qubits),
            debug=debug,
        )

    def active_column_ids(self) -> List[int]:
        return list(self.columns.keys())

    def place_cnot(self, control: int, target: int) -> None:
        """
        Emit CX(control, target).

        Important convention:
        - For the remaining phase-polynomial matrix P, commuting this CNOT through the frontier
          updates the CONTROL row as control ^= target.
        - For the emitted circuit's actual forward linear transform, Qiskit CX updates the TARGET
          row as target ^= control.
        """
        self.circuit.cx(control, target)

        # Update the remaining phase-gadget matrix (control row ^= target row).
        for col in self.columns.values():
            col.bits[control] ^= col.bits[target]

        # Update the emitted circuit's forward linear transform (target row ^= control row).
        self.emitted_out_parities[target] = gf2_add_rows(
            self.emitted_out_parities[control], self.emitted_out_parities[target]
        )

    def reduce_trivial_columns(self, column_ids: Sequence[int]) -> List[int]:
        kept: List[int] = []
        for cid in column_ids:
            col = self.columns.get(cid)
            if col is None:
                continue
            support = [i for i, bit in enumerate(col.bits) if bit == 1]
            if len(support) == 1:
                self.circuit.rz(col.angle, support[0])
                del self.columns[cid]
            else:
                kept.append(cid)
        return kept



def _choose_skew_row(state: _RecursiveSynthState, column_ids: Sequence[int], candidate_rows: Sequence[int]) -> int:
    def score(row: int) -> int:
        count1 = sum(state.columns[cid].bits[row] for cid in column_ids)
        count0 = len(column_ids) - count1
        return max(count0, count1)

    return max(candidate_rows, key=score)



def _base_recurse(
    state: _RecursiveSynthState,
    architecture: ArchitectureGraph,
    column_ids: Sequence[int],
    qubit_ids: Sequence[int],
) -> None:
    column_ids = state.reduce_trivial_columns(column_ids)
    if not column_ids or not qubit_ids:
        return

    eligible_rows = architecture.non_cutting_vertices(qubit_ids)
    chosen_row = _choose_skew_row(state, column_ids, eligible_rows)

    cols1 = [cid for cid in column_ids if state.columns[cid].bits[chosen_row] == 1]
    cols0 = [cid for cid in column_ids if state.columns[cid].bits[chosen_row] == 0]

    reduced_qubits = [q for q in qubit_ids if q != chosen_row]
    _base_recurse(state, architecture, cols0, reduced_qubits)
    _one_recurse(state, architecture, cols1, qubit_ids, chosen_row)



def _one_recurse(
    state: _RecursiveSynthState,
    architecture: ArchitectureGraph,
    column_ids: Sequence[int],
    qubit_ids: Sequence[int],
    chosen_qubit: int,
) -> None:
    column_ids = state.reduce_trivial_columns(column_ids)
    if not column_ids:
        return

    neighbors = architecture.neighbors(chosen_qubit, allowed=qubit_ids)
    if not neighbors:
        raise ValueError(f"No allowed neighbor found for qubit {chosen_qubit} in the active architecture subgraph.")

    def neighbor_score(q: int) -> int:
        return sum(state.columns[cid].bits[q] for cid in column_ids)

    chosen_neighbor = max(neighbors, key=neighbor_score)
    neighbor_ones = neighbor_score(chosen_neighbor)

    if neighbor_ones != 0:
        # Matches the paper/repo convention: emit CX(chosen_qubit, chosen_neighbor),
        # which updates the remaining matrix row for chosen_qubit as ^= chosen_neighbor.
        state.place_cnot(chosen_qubit, chosen_neighbor)
        column_ids = state.reduce_trivial_columns(column_ids)
    else:
        # Effective swap used in the reference implementation when the best neighbor row
        # is all-zeros on the active columns.
        state.place_cnot(chosen_neighbor, chosen_qubit)
        state.place_cnot(chosen_qubit, chosen_neighbor)

    cols0 = [cid for cid in column_ids if state.columns[cid].bits[chosen_qubit] == 0]
    cols1 = [cid for cid in column_ids if state.columns[cid].bits[chosen_qubit] == 1]

    reduced_qubits = [q for q in qubit_ids if q != chosen_qubit]
    _base_recurse(state, architecture, cols0, reduced_qubits)
    _one_recurse(state, architecture, cols1, qubit_ids, chosen_qubit)



def synthesize_phase_support_architecture_aware(
    phase_poly: PhasePolynomial,
    coupling_map: CouplingMap,
    *,
    debug: bool = False,
) -> Tuple[QuantumCircuit, List[BitVec]]:
    """
    Synthesize only the phase-support part using the paper's architecture-aware recursion.

    Returns
    -------
    circuit:
        Circuit that realises all required parities and places the RZ phases.
    emitted_out_parities:
        The linear transform induced by the emitted CX network so far. This is needed
        to compute the final residual transform A * P'^-1.
    """
    architecture = ArchitectureGraph.from_coupling_map(coupling_map, phase_poly.num_qubits)
    state = _RecursiveSynthState.from_phase_poly(phase_poly, debug=debug)
    initial_columns = state.reduce_trivial_columns(state.active_column_ids())
    _base_recurse(state, architecture, initial_columns, list(range(phase_poly.num_qubits)))
    return state.circuit, state.emitted_out_parities


# =========================
# Residual linear transform
# =========================


def _steiner_fill_and_eliminate_column(
    work: Matrix,
    col: int,
    root: int,
    children: Dict[int, List[int]],
    emit_row_add,
) -> None:
    """
    Reduce one GF(2) column to the root using a rooted Steiner tree.

    After completion, the column has value 1 on `root` and 0 on every other node
    in the tree. The row operations are emitted through `emit_row_add(control, target)`.
    """

    def fill(node: int) -> None:
        for child in children.get(node, []):
            fill(child)
        if work[node][col] == 1:
            return
        for child in children.get(node, []):
            if work[child][col] == 1:
                emit_row_add(child, node)
                return

    def eliminate(node: int) -> None:
        for child in children.get(node, []):
            eliminate(child)
            if work[child][col] == 1:
                emit_row_add(node, child)

    fill(root)
    if work[root][col] != 1:
        raise ValueError(
            f"Failed to create pivot 1 at row {root} for column {col}. "
            "The active architecture subgraph may be unsuitable for this reduction step."
        )
    eliminate(root)



def synthesize_linear_transform_steiner_gauss(
    matrix_rows: Sequence[BitVec],
    coupling_map: CouplingMap,
) -> QuantumCircuit:
    """
    Architecture-aware synthesis of an invertible GF(2) linear transform using a
    Steiner-tree-based Gaussian elimination.

    This is a Qiskit-native replacement for the temporary "all-to-all + transpile"
    residual fallback. It keeps every emitted CNOT architecture-compliant directly.

    Notes
    -----
    This first port focuses on correctness and mirrors the Steiner-tree column
    reduction idea from the reference implementation. It is not yet a line-by-line
    reproduction of the repo's recursive `rec_steiner_gauss(...)`, but it removes
    the routed fallback and keeps the residual stage natively architecture-aware.
    """
    work = gf2_matrix_from_rows(matrix_rows)
    n = len(work)
    if n == 0:
        return QuantumCircuit(0)
    if any(len(row) != n for row in work):
        raise ValueError("Steiner-Gauss synthesis requires a square matrix.")

    architecture = ArchitectureGraph.from_coupling_map(coupling_map, n)
    ops: List[Tuple[int, int]] = []

    def emit_row_add(control: int, target: int) -> None:
        if not architecture.graph.has_edge(control, target):
            raise ValueError(f"Attempted non-adjacent row add {control}->{target} in Steiner-Gauss synthesis.")
        _apply_row_add(work, control, target)
        ops.append((control, target))

    # Forward elimination: clear below each pivot.
    for col in range(n):
        active_rows = list(range(col, n))
        ones = [r for r in active_rows if work[r][col] == 1]
        if not ones:
            raise ValueError("Residual linear transform is not invertible over GF(2).")
        terminals = sorted(set(ones) | {col})
        children = architecture.rooted_steiner_tree_children(root=col, terminals=terminals, allowed=active_rows)
        _steiner_fill_and_eliminate_column(work, col, root=col, children=children, emit_row_add=emit_row_add)
        if work[col][col] != 1 or any(work[r][col] != 0 for r in range(col + 1, n)):
            raise ValueError(f"Forward Steiner-Gauss step failed on column {col}.")

    # Backward elimination: clear above each pivot.
    for col in reversed(range(n)):
        active_rows = list(range(0, col + 1))
        ones = [r for r in active_rows if work[r][col] == 1]
        terminals = sorted(set(ones) | {col})
        children = architecture.rooted_steiner_tree_children(root=col, terminals=terminals, allowed=active_rows)
        _steiner_fill_and_eliminate_column(work, col, root=col, children=children, emit_row_add=emit_row_add)
        if any(work[r][col] != 0 for r in range(0, col)) or work[col][col] != 1:
            raise ValueError(f"Backward Steiner-Gauss step failed on column {col}.")

    if work != gf2_matrix_from_rows(gf2_identity(n)):
        raise ValueError("Steiner-Gauss synthesis ended in a non-identity matrix.")

    qc = QuantumCircuit(n)
    for control, target in reversed(ops):
        qc.cx(control, target)
    return qc


def _apply_row_add(matrix: Matrix, control: int, target: int) -> None:
    row_control = matrix[control]
    row_target = matrix[target]
    matrix[target] = [a ^ b for a, b in zip(row_control, row_target)]



def _apply_row_swap_via_xors(matrix: Matrix, a: int, b: int, ops: List[Tuple[int, int]]) -> None:
    # swap(a, b) = CX(b->a), CX(a->b), CX(b->a) in row-add notation
    _apply_row_add(matrix, b, a)
    ops.append((b, a))
    _apply_row_add(matrix, a, b)
    ops.append((a, b))
    _apply_row_add(matrix, b, a)
    ops.append((b, a))



def synthesize_linear_transform_all_to_all(matrix_rows: Sequence[BitVec]) -> QuantumCircuit:
    """
    Synthesize an invertible GF(2) linear transform using CNOTs on full connectivity.

    The returned circuit is not architecture-aware. For now this is used only as a
    temporary fallback for the final residual transform; it can then be routed by Qiskit.
    """
    matrix = gf2_matrix_from_rows(matrix_rows)
    n = len(matrix)
    if n == 0:
        return QuantumCircuit(0)
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

    qc = QuantumCircuit(n)
    for control, target in reversed(reduction_ops):
        qc.cx(control, target)
    return qc



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

    Priority:
    1. fewer CX gates
    2. lower depth
    3. fewer total operations
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
    """
    Keep the candidate only if it is strictly better than the original under the
    phase-polynomial block cost metric.
    """
    if _phasepoly_cost_tuple(candidate) < _phasepoly_cost_tuple(original):
        return candidate
    return original


# =========================
# Emission helpers
# =========================



def emit_phase_polynomial_naive(phase_poly: PhasePolynomial) -> QuantumCircuit:
    """
    Naive emitter used only for sanity checks during development.

    For each parity in support, build the parity onto one pivot qubit with a CX ladder,
    apply RZ(angle), then uncompute.
    """
    qc = QuantumCircuit(phase_poly.num_qubits)

    for parity, angle in phase_poly.zphases.items():
        support = [i for i, bit in enumerate(parity) if bit == 1]
        if not support:
            continue
        pivot = support[-1]
        others = support[:-1]

        for q in others:
            qc.cx(q, pivot)
        qc.rz(angle, pivot)
        for q in reversed(others):
            qc.cx(q, pivot)

    return qc


# =========================
# Public optimizer
# =========================


@dataclass
class ArchitectureAwarePhasePolyOptimizer:
    """
    Stepwise Qiskit port of the Meijer-van de Griend / Duncan architecture-aware
    phase-polynomial synthesis algorithm.

    Current stage:
    - validates phase-polynomial blocks
    - extracts a faithful internal representation
    - synthesizes the phase-support part using the paper's recursion
    - computes the final residual linear transform A * P'^-1
    - synthesizes that residual natively with a Steiner-Gauss-style architecture-aware routine
    - optionally keeps the original block if the synthesized result is worse by cost

    Remaining work:
    - maximal block extraction from mixed circuits
    - making the Steiner-Gauss port closer to the reference implementation for better CX counts
    - optional placement when coupling_map has more qubits than the circuit
    """

    allow_barriers: bool = True
    debug: bool = False
    keep_original_if_worse: bool = True

    def optimize(self, circuit: QuantumCircuit, coupling_map: CouplingMap) -> QuantumCircuit:
        if not isinstance(coupling_map, CouplingMap):
            raise TypeError("coupling_map must be a qiskit.transpiler.CouplingMap")

        if not is_phase_polynomial_block(circuit, allow_barriers=self.allow_barriers):
            raise NotImplementedError(
                "This stepwise implementation currently supports only full circuits that are already phase-polynomial blocks. "
                "Block partitioning for mixed circuits will be added later."
            )

        phase_poly = extract_phase_polynomial(circuit)
        phase_circuit, emitted_out_parities = synthesize_phase_support_architecture_aware(
            phase_poly,
            coupling_map,
            debug=self.debug,
        )

        residual_rows = residual_linear_transform(phase_poly.out_parities, emitted_out_parities)
        residual_circuit = synthesize_linear_transform_steiner_gauss(residual_rows, coupling_map)

        candidate = QuantumCircuit(circuit.num_qubits)
        candidate.compose(phase_circuit, inplace=True)
        candidate.compose(residual_circuit, inplace=True)

        if self.debug:
            equivalent = unitary_equiv_up_to_global_phase(circuit, candidate)
            if not equivalent:
                raise ValueError("Synthesized circuit is not equivalent to the input block.")

        if self.keep_original_if_worse:
            return _choose_better_phasepoly_block(circuit, candidate)

        return candidate


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