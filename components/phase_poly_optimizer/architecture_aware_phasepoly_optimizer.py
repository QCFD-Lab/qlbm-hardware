from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, Qubit
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


@dataclass
class PhaseColumn:
    column_id: int
    bits: List[int]
    angle: Any


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

    def rec_steiner_tree_edges(
            self,
            root: int,
            nodes: Sequence[int],
            usable_nodes: Sequence[int],
            rec_nodes: Sequence[int],
            upper: bool = True,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Qiskit-native analogue of the repo's `architecture.rec_steiner_tree(...)`.

        Returns
        -------
        top_down_edges:
            Edges emitted in the first phase of the tree walk.
        bottom_up_edges:
            Edges emitted in the second phase of the tree walk.

        Notes
        -----
        In the full-reduction phase (`upper=False`), edges are only bidirectional inside
        `rec_nodes`. Outside that set, we keep only the orientation from larger index
        to smaller index, which mirrors the reference implementation's restriction.
        """
        usable = list(dict.fromkeys(int(x) for x in usable_nodes))
        required = list(dict.fromkeys(int(x) for x in nodes))
        rec_set = set(int(x) for x in rec_nodes)

        if root not in usable:
            raise ValueError(f"Root {root} is not present in usable_nodes.")
        for node in required:
            if node not in usable:
                raise ValueError(f"Required node {node} is not present in usable_nodes.")

        # Build the directed connectivity used for shortest paths.
        dg = nx.DiGraph()
        dg.add_nodes_from(usable)

        sub = self.graph.subgraph(usable)
        for u, v in sub.edges():
            if upper or (u in rec_set and v in rec_set):
                dg.add_edge(u, v)
                dg.add_edge(v, u)
            else:
                if u > v:
                    dg.add_edge(u, v)
                else:
                    dg.add_edge(v, u)

        # Build the spanning tree of shortest paths with `root` as start.
        vertices = [root]
        steiner_points: List[int] = []
        pending = [node for node in required if node != root]
        all_edges: List[Tuple[int, int]] = []

        while pending:
            options: List[Tuple[int, int, int, List[Tuple[int, int]]]] = []
            frontier = vertices + steiner_points

            for node in pending:
                for v in frontier:
                    if nx.has_path(dg, v, node):
                        path = nx.shortest_path(dg, source=v, target=node)
                        edge_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                        options.append((node, v, len(edge_path), edge_path))

            if not options:
                raise ValueError(
                    f"Could not connect pending Steiner node(s) {pending} from root {root} "
                    f"inside usable_nodes={usable} with rec_nodes={list(rec_set)}."
                )

            node, _, _, edge_path = min(options, key=lambda x: x[2])

            vertices.append(node)
            all_edges.extend(edge_path)

            for a, b in edge_path:
                if a not in vertices and a not in steiner_points:
                    steiner_points.append(a)
                if b not in vertices and b not in steiner_points:
                    steiner_points.append(b)

            pending.remove(node)

        all_edges = _dedup_edges_preserve_order(all_edges)

        # First phase: walk top-down from the root.
        top_down: List[Tuple[int, int]] = []
        active = {root}
        yielded = set()

        while len(yielded) < len(all_edges):
            emitted_this_round = False
            old_active = list(active)

            for edge in all_edges:
                if edge in yielded:
                    continue
                src, dst = edge
                if src in active:
                    top_down.append(edge)
                    yielded.add(edge)
                    active.add(dst)
                    emitted_this_round = True

            for v in old_active:
                active.discard(v)

            if not emitted_this_round:
                raise ValueError("Top-down Steiner tree walk got stuck.")

        # Second phase: walk bottom-up from leaves.
        remaining = list(all_edges)
        bottom_up: List[Tuple[int, int]] = []

        while remaining:
            parents = {src for src, _ in remaining}
            leaves = [v for v in vertices if v not in parents] + [v for v in steiner_points if v not in parents]

            emitted_this_round = False
            for leaf in leaves:
                leaf_edges = [edge for edge in remaining if edge[1] == leaf]
                for edge in leaf_edges:
                    bottom_up.append(edge)
                    remaining.remove(edge)
                    emitted_this_round = True

            if not emitted_this_round:
                raise ValueError("Bottom-up Steiner tree walk got stuck.")

        return top_down, bottom_up

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

    def heuristic_reduce_order(self, nodes: Optional[Sequence[int]] = None) -> List[int]:
        """
        Compute an elimination order by repeatedly removing low-degree non-cutting
        vertices. This is a generic analogue of the repo's architecture-specific
        `reduce_order` and is used for residual Steiner-Gauss synthesis.
        """
        remaining = list(self.graph.nodes if nodes is None else nodes)
        order: List[int] = []

        while remaining:
            candidates = self.non_cutting_vertices(remaining)
            sub = self.graph.subgraph(remaining)
            chosen = min(candidates, key=lambda q: (sub.degree[q], -q))
            order.append(chosen)
            remaining.remove(chosen)

        return order


# =========================
# Block validation/extraction
# =========================


ALLOWED_BLOCK_GATES = {"cx", "rz", "barrier"}



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

    Blocks are contiguous ranges containing only CX/RZ instructions, plus barriers
    when `allow_barriers` is enabled. The returned `active_qubits` list is a stable
    local-to-global qubit map for compact block synthesis.
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
    local_of_global = {global_index: local_index for local_index, global_index in enumerate(block.active_qubits)}
    qmap = _qubit_index_map(circuit)
    compact = QuantumCircuit(len(block.active_qubits), name=f"phasepoly_{block.start}_{block.end}")

    for index in block.phase_instruction_indices:
        instruction = circuit.data[index]
        local_qargs = [compact.qubits[local_of_global[qmap[qubit]]] for qubit in instruction.qubits]
        compact.append(instruction.operation.copy(), local_qargs, [])

    return compact


def _local_coupling_map(coupling_map: CouplingMap, active_qubits: Sequence[int]) -> CouplingMap:
    active_set = set(active_qubits)
    local_of_global = {global_index: local_index for local_index, global_index in enumerate(active_qubits)}
    local_edges = [
        (local_of_global[int(u)], local_of_global[int(v)])
        for u, v in coupling_map.get_edges()
        if int(u) in active_set and int(v) in active_set and int(u) != int(v)
    ]
    return CouplingMap(local_edges)


def _active_qubit_subgraph_is_connected(coupling_map: CouplingMap, active_qubits: Sequence[int]) -> bool:
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
        qargs = [out.qubits[active_qubits[compact.find_bit(qubit).index]] for qubit in instruction.qubits]
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

def is_phase_polynomial_block(circuit: QuantumCircuit, *, allow_barriers: bool = True) -> bool:
    """Return True iff the circuit contains only CX, RZ, and optionally barriers."""
    for instruction in circuit.data:
        if not _is_phase_polynomial_instruction(instruction, allow_barriers=allow_barriers):
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
            columns[cid] = PhaseColumn(column_id=cid, bits=list(parity), angle=angle)
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


def _steiner_reduce_column_recursive(
    work: Matrix,
    col: int,
    root: int,
    nodes: Sequence[int],
    usable_nodes: Sequence[int],
    rec_nodes: Sequence[int],
    upper: bool,
    architecture: ArchitectureGraph,
    emit_row_add,
) -> None:
    """
    Closer Qiskit analogue of the repo's `steiner_reduce_column(...)`.

    It uses the repo-style two-phase tree walk:
    1. top-down pass
    2. bottom-up pass
    """
    if len(nodes) <= 1:
        return

    top_down, bottom_up = architecture.rec_steiner_tree_edges(
        root=root,
        nodes=list(nodes),
        usable_nodes=list(usable_nodes),
        rec_nodes=list(rec_nodes),
        upper=upper,
    )

    if upper:
        zeros: List[Tuple[int, int]] = []
        for s0, s1 in top_down:
            if work[s0][col] == 0:
                zeros.append((s0, s1))

        for s0, s1 in reversed(zeros):
            if work[s0][col] == 0:
                emit_row_add(s1, s0)

        if work[root][col] != 1:
            raise ValueError(
                f"Upper Steiner reduction failed to create pivot 1 at row {root} for column {col}."
            )
    else:
        # Repo behaviour in the full-reduction phase.
        for s0, s1 in top_down:
            if work[s1][col] == 0:
                emit_row_add(s0, s1)

    for s0, s1 in bottom_up:
        if work[s1][col] == 1:
            emit_row_add(s0, s1)



def synthesize_linear_transform_steiner_gauss(
    matrix_rows: Sequence[BitVec],
    coupling_map: CouplingMap,
    *,
    reduce_order: Optional[Sequence[int]] = None,
) -> QuantumCircuit:
    """
    Architecture-aware synthesis of an invertible GF(2) linear transform using a
    recursive Steiner-Gauss structure closer to the reference repo.

    This version mirrors the repo at a higher level:
    - first perform an upper-triangular pass
    - collect pivot columns
    - then do the full reduction recursively on shortest-path subproblems
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
            raise ValueError(
                f"Attempted non-adjacent row add {control}->{target} in Steiner-Gauss synthesis."
            )
        _apply_row_add(work, control, target)
        ops.append((control, target))

    def rec_step(cols: List[int], rows: List[int]) -> None:
        if not cols or not rows:
            return

        size = len(rows)
        pivot = 0
        pivot_cols: List[int] = []

        rows2 = [r for r in range(n) if r in rows]
        cols2 = [c for c in range(n) if c in cols]

        # Upper-triangular pass.
        for i, c in enumerate(cols2):
            if pivot >= size:
                break

            root = rows2[pivot]
            nodes = [r for r in rows2[pivot:] if r == root or work[r][c] == 1]

            _steiner_reduce_column_recursive(
                work=work,
                col=c,
                root=root,
                nodes=nodes,
                usable_nodes=cols2[i:],
                rec_nodes=[],
                upper=True,
                architecture=architecture,
                emit_row_add=emit_row_add,
            )

            if work[root][c] == 1:
                pivot_cols.append(c)
                pivot += 1

        # Full reduction / recursive phase.
        pivot -= 1
        for i, c in enumerate(cols):
            if c not in pivot_cols:
                continue

            root = rows[pivot]
            nodes = [r for r in rows if r == root or work[r][c] == 1]
            usable_nodes = cols[i:]

            if not usable_nodes:
                pivot -= 1
                continue

            path_end = max(usable_nodes)
            rec_nodes = architecture.shortest_path(c, path_end, usable_nodes)

            if len(nodes) > 1:
                _steiner_reduce_column_recursive(
                    work=work,
                    col=c,
                    root=root,
                    nodes=nodes,
                    usable_nodes=cols,
                    rec_nodes=rec_nodes,
                    upper=False,
                    architecture=architecture,
                    emit_row_add=emit_row_add,
                )

            if len(rec_nodes) > 1:
                rec_step(list(reversed(rec_nodes)), rec_nodes)

            pivot -= 1

    if reduce_order is None:
        reduce_order = architecture.heuristic_reduce_order()
    else:
        reduce_order = list(reduce_order)
    if sorted(reduce_order) != list(range(n)):
        raise ValueError("reduce_order must be a permutation of range(n).")
    rec_step(reduce_order, list(reversed(reduce_order)))

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


def _tree_top_down_edges(children: Dict[int, List[int]], root: int) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []

    def walk(node: int) -> None:
        for child in children.get(node, []):
            edges.append((node, child))
            walk(child)

    walk(root)
    return edges


def _tree_bottom_up_edges(children: Dict[int, List[int]], root: int) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []

    def walk(node: int) -> None:
        for child in children.get(node, []):
            walk(child)
            edges.append((node, child))

    walk(root)
    return edges

def _dedup_edges_preserve_order(edges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out: List[Tuple[int, int]] = []
    for edge in edges:
        if edge not in seen:
            out.append(edge)
            seen.add(edge)
    return out


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


def _cx_gates_respect_coupling_map(circuit: QuantumCircuit, coupling_map: CouplingMap) -> bool:
    """
    Return whether every CX in `circuit` is adjacent in the undirected architecture.

    The paper treats the architecture graph as undirected. This helper follows the
    same convention, so either edge orientation in a Qiskit CouplingMap is accepted.
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
    - finds maximal phase-polynomial blocks in mixed circuits
    - extracts a faithful internal representation
    - synthesizes the phase-support part using the paper's recursion
    - computes the final residual linear transform A * P'^-1
    - synthesizes that residual natively with a Steiner-Gauss-style architecture-aware routine
    - optionally keeps the original block if the synthesized result is worse by cost

    Remaining work:
    - making the Steiner-Gauss port closer to the reference implementation for better CX counts
    """

    allow_barriers: bool = True
    debug: bool = False
    keep_original_if_worse: bool = True
    last_run_report: Optional[OptimizationRunReport] = None

    def _optimize_phase_polynomial_block(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
    ) -> Tuple[QuantumCircuit, BlockOptimizationReport]:
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

        kept_original = False
        final_circuit = candidate
        if self.keep_original_if_worse and _cx_gates_respect_coupling_map(circuit, coupling_map):
            final_circuit = _choose_better_phasepoly_block(circuit, candidate)
            kept_original = final_circuit is circuit

        report = BlockOptimizationReport(
            start=0,
            end=max(len(circuit.data) - 1, 0),
            active_qubits=list(range(circuit.num_qubits)),
            connected_active_subgraph=True,
            support_size=len(phase_poly.support()),
            original_cx=int(circuit.count_ops().get("cx", 0)),
            phase_support_cx=int(phase_circuit.count_ops().get("cx", 0)),
            residual_cx=int(residual_circuit.count_ops().get("cx", 0)),
            candidate_cx=int(candidate.count_ops().get("cx", 0)),
            final_cx=int(final_circuit.count_ops().get("cx", 0)),
            kept_original=kept_original,
        )

        return final_circuit, report

    def optimize(self, circuit: QuantumCircuit, coupling_map: CouplingMap) -> QuantumCircuit:
        if not isinstance(coupling_map, CouplingMap):
            raise TypeError("coupling_map must be a qiskit.transpiler.CouplingMap")

        if is_phase_polynomial_block(circuit, allow_barriers=self.allow_barriers):
            optimized, report = self._optimize_phase_polynomial_block(circuit, coupling_map)
            self.last_run_report = OptimizationRunReport(
                circuit_num_qubits=circuit.num_qubits,
                num_blocks=1,
                block_reports=[report],
            )
            return optimized

        blocks = find_phase_polynomial_blocks(circuit, allow_barriers=self.allow_barriers)
        if not blocks:
            self.last_run_report = OptimizationRunReport(
                circuit_num_qubits=circuit.num_qubits,
                num_blocks=0,
                block_reports=[],
            )
            return circuit.copy()

        block_by_start = {block.start: block for block in blocks}
        out = _new_like_circuit(circuit)
        index = 0
        block_reports: List[BlockOptimizationReport] = []

        while index < len(circuit.data):
            block = block_by_start.get(index)
            if block is None:
                _append_original_instruction(out, circuit, circuit.data[index])
                index += 1
                continue

            compact = _extract_compact_block(circuit, block)
            local_coupling_map = _local_coupling_map(coupling_map, block.active_qubits)

            if _active_qubit_subgraph_is_connected(coupling_map, block.active_qubits):
                optimized_block, report = self._optimize_phase_polynomial_block(compact, local_coupling_map)
            else:
                optimized_block = compact
                support_size = len(extract_phase_polynomial(compact).support())
                report = BlockOptimizationReport(
                    start=block.start,
                    end=block.end,
                    active_qubits=list(block.active_qubits),
                    connected_active_subgraph=False,
                    support_size=support_size,
                    original_cx=int(compact.count_ops().get("cx", 0)),
                    phase_support_cx=0,
                    residual_cx=0,
                    candidate_cx=int(compact.count_ops().get("cx", 0)),
                    final_cx=int(compact.count_ops().get("cx", 0)),
                    kept_original=True,
                )

            report.start = block.start
            report.end = block.end
            report.active_qubits = list(block.active_qubits)
            report.connected_active_subgraph = _active_qubit_subgraph_is_connected(coupling_map, block.active_qubits)
            block_reports.append(report)

            if block.passthrough_instruction_indices and report.candidate_cx >= report.original_cx:
                report.final_cx = report.original_cx
                report.kept_original = True
                _append_instruction_range(out, circuit, block.start, block.end)
                index = block.end + 1
                continue

            _append_instruction_indices(out, circuit, block.passthrough_instruction_indices)
            _append_compact_circuit(out, optimized_block, block.active_qubits)
            index = block.end + 1

        self.last_run_report = OptimizationRunReport(
            circuit_num_qubits=circuit.num_qubits,
            num_blocks=len(blocks),
            block_reports=block_reports,
        )
        return out


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
