from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from .tools_for_optimization import (
    BitVec,
    Matrix,
    BlockOptimizationReport,
    LinearSynthesisResult,
    OptimizationRunReport,
    PhasePolynomial,
    PhasePolynomialBlock,
    _active_qubit_subgraph_is_connected,
    _apply_row_add,
    _append_compact_circuit,
    _append_instruction_indices,
    _append_instruction_range,
    _append_original_instruction,
    _choose_better_phasepoly_block,
    _cx_gates_respect_coupling_map,
    _extract_compact_block,
    _local_coupling_map,
    _new_like_circuit,
    _phasepoly_cost_tuple,
    _qubit_index_map,
    _row_add_sequence_for_linear_transform,
    extract_phase_polynomial,
    find_phase_polynomial_blocks,
    gf2_add_rows,
    gf2_identity,
    gf2_inverse,
    gf2_matmul,
    gf2_matrix_from_rows,
    gf2_matrix_to_rows,
    is_phase_polynomial_block,
    parity_to_str,
    residual_linear_transform,
    unitary_equiv_up_to_global_phase,
)


@dataclass
class PhaseColumn:
    column_id: int
    bits: List[int]
    angle: Any


@dataclass
class ArchitectureGraph:
    """
    Minimal undirected architecture wrapper built from a Qiskit CouplingMap.
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
# Recursive synthesizer
# =========================


@dataclass
class _RecursiveSynthState:
    num_qubits: int
    columns: Dict[int, PhaseColumn]
    circuit: QuantumCircuit
    emitted_out_parities: List[BitVec]

    @classmethod
    def from_phase_poly(cls, phase_poly: PhasePolynomial) -> "_RecursiveSynthState":
        columns: Dict[int, PhaseColumn] = {}
        for cid, (parity, angle) in enumerate(phase_poly.zphases.items()):
            columns[cid] = PhaseColumn(column_id=cid, bits=list(parity), angle=angle)
        return cls(
            num_qubits=phase_poly.num_qubits,
            columns=columns,
            circuit=QuantumCircuit(phase_poly.num_qubits),
            emitted_out_parities=gf2_identity(phase_poly.num_qubits),
        )

    def active_column_ids(self) -> List[int]:
        return list(self.columns.keys())

    def place_cnot(self, control: int, target: int) -> None:
        """
        Emit CX(control, target).

        Important convention:
        - For the remaining phase-polynomial matrix P, commuting this CNOT
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
    state = _RecursiveSynthState.from_phase_poly(phase_poly)
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
    Qiskit analogue of the repo's `steiner_reduce_column(...)`.
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
        for s0, s1 in top_down:
            if work[s1][col] == 0:
                emit_row_add(s0, s1)

    for s0, s1 in bottom_up:
        if work[s1][col] == 1:
            emit_row_add(s0, s1)



def synthesize_linear_transform_steiner_gauss(
    matrix_rows: Sequence[BitVec],
    coupling_map: CouplingMap,
    reduce_order: Optional[Sequence[int]] = None,
) -> QuantumCircuit:
    """
    Architecture-aware synthesis of an invertible GF(2) linear transform using a
    recursive Steiner-Gauss structure.
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

    reduce_order = list(range(n - 1, -1, -1)) if reduce_order is None else list(reduce_order)
    if sorted(reduce_order) != list(range(n)):
        raise ValueError("reduce_order must be a permutation of range(n).")
    rec_step(reduce_order, list(reversed(reduce_order)))

    if work != gf2_matrix_from_rows(gf2_identity(n)):
        raise ValueError("Steiner-Gauss synthesis ended in a non-identity matrix.")

    qc = QuantumCircuit(n)
    for control, target in reversed(ops):
        qc.cx(control, target)
    return qc


def _append_swap_via_cx(circuit: QuantumCircuit, a: int, b: int) -> None:
    circuit.cx(a, b)
    circuit.cx(b, a)
    circuit.cx(a, b)


def _append_remote_cx_via_restored_swaps(
    circuit: QuantumCircuit,
    control: int,
    target: int,
    architecture: ArchitectureGraph,
) -> None:
    """
    Append a logical CX(control, target) using only architecture-local CX gates.

    The temporary SWAP chain restores the original logical placement before return,
    so a sequence of these routed CNOTs implements the same all-to-all row-add
    sequence on the original logical qubit indices.
    """
    if control == target:
        return
    path = architecture.shortest_path(control, target)
    if len(path) == 2:
        circuit.cx(control, target)
        return

    for i in range(len(path) - 2):
        _append_swap_via_cx(circuit, path[i], path[i + 1])

    circuit.cx(path[-2], path[-1])

    for i in range(len(path) - 3, -1, -1):
        _append_swap_via_cx(circuit, path[i], path[i + 1])


def synthesize_linear_transform_graph_exact(
    matrix_rows: Sequence[BitVec],
    coupling_map: CouplingMap,
) -> QuantumCircuit:
    """
    Correctness-first graph-constrained synthesis of an invertible GF(2) transform.

    This is used as a fallback when the recursive Steiner-Gauss heuristic cannot
    find a directed reduction subproblem for the chosen order. It is usually larger
    than a successful Steiner-Gauss circuit, but it preserves the requested linear
    transform exactly and uses only edges of the undirected architecture graph.
    """
    n = len(matrix_rows)
    architecture = ArchitectureGraph.from_coupling_map(coupling_map, n)
    qc = QuantumCircuit(n)
    for control, target in _row_add_sequence_for_linear_transform(matrix_rows):
        _append_remote_cx_via_restored_swaps(qc, control, target, architecture)
    return qc


def _non_cutting_peel_order(
    architecture: ArchitectureGraph,
    prefer_high_index: bool,
    dfs_priority: Optional[Dict[int, int]] = None,
) -> List[int]:
    remaining = list(architecture.graph.nodes)
    order: List[int] = []

    while remaining:
        candidates = architecture.non_cutting_vertices(remaining)
        sub = architecture.graph.subgraph(remaining)

        if dfs_priority is not None:
            chosen = max(
                candidates,
                key=lambda q: (dfs_priority.get(q, -1), -sub.degree[q], q),
            )
        elif prefer_high_index:
            chosen = max(candidates, key=lambda q: (q, -sub.degree[q]))
        else:
            chosen = min(candidates, key=lambda q: (sub.degree[q], -q))

        order.append(chosen)
        remaining.remove(chosen)

    return order


def _candidate_reduce_orders(architecture: ArchitectureGraph, n: int) -> List[Tuple[str, List[int]]]:
    orders: List[Tuple[str, List[int]]] = [
        ("reverse", list(range(n - 1, -1, -1))),
        ("natural", list(range(n))),
        (
            "degree_ascending",
            sorted(range(n), key=lambda q: (architecture.graph.degree[q], -q)),
        ),
        (
            "non_cutting_low_degree",
            _non_cutting_peel_order(architecture, prefer_high_index=False),
        ),
        (
            "non_cutting_high_index",
            _non_cutting_peel_order(architecture, prefer_high_index=True),
        ),
    ]

    try:
        dfs_nodes = list(nx.dfs_postorder_nodes(architecture.graph, source=0))
    except nx.NetworkXError:
        dfs_nodes = list(range(n))
    dfs_priority = {node: index for index, node in enumerate(dfs_nodes)}
    orders.append(
        (
            "dfs_postorder_non_cutting",
            _non_cutting_peel_order(
                architecture,
                prefer_high_index=True,
                dfs_priority=dfs_priority,
            ),
        )
    )

    seen = set()
    unique_orders: List[Tuple[str, List[int]]] = []
    for name, order in orders:
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        unique_orders.append((name, order))
    return unique_orders


def synthesize_linear_transform_architecture_aware_result(
    matrix_rows: Sequence[BitVec],
    coupling_map: CouplingMap,
    reduce_order: Optional[Sequence[int]] = None,
) -> LinearSynthesisResult:
    n = len(matrix_rows)
    architecture = ArchitectureGraph.from_coupling_map(coupling_map, n)
    if reduce_order is None:
        orders = _candidate_reduce_orders(architecture, n)
    else:
        orders = [("specified", list(reduce_order))]

    best: Optional[LinearSynthesisResult] = None
    best_cost: Optional[Tuple[int, int, int]] = None

    for name, order in orders:
        try:
            circuit = synthesize_linear_transform_steiner_gauss(
                matrix_rows,
                coupling_map,
                reduce_order=order,
            )
        except (ValueError, nx.NetworkXException):
            continue

        cost = _phasepoly_cost_tuple(circuit)
        if best is None or best_cost is None or cost < best_cost:
            best = LinearSynthesisResult(
                circuit=circuit,
                method=f"steiner_gauss:{name}",
                reduce_order=list(order),
            )
            best_cost = cost

    if best is not None:
        return best

    return LinearSynthesisResult(
        circuit=synthesize_linear_transform_graph_exact(matrix_rows, coupling_map),
        method="graph_exact_fallback",
        reduce_order=None,
    )


def synthesize_linear_transform_architecture_aware(
    matrix_rows: Sequence[BitVec],
    coupling_map: CouplingMap,
    reduce_order: Optional[Sequence[int]] = None,
) -> QuantumCircuit:
    """
    Synthesize the residual basis transform using the paper's Steiner-Gauss step.

    The reference implementation treats Steiner-Gauss as the post-processing for
    ``A * P'^-1``. This port first tries that reducer. If its order-dependent
    directed Steiner walk fails, it falls back to an exact local-CX construction
    rather than rejecting an otherwise valid phase-polynomial block.
    """
    return synthesize_linear_transform_architecture_aware_result(
        matrix_rows,
        coupling_map,
        reduce_order=reduce_order,
    ).circuit


def _dedup_edges_preserve_order(edges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out: List[Tuple[int, int]] = []
    for edge in edges:
        if edge not in seen:
            out.append(edge)
            seen.add(edge)
    return out


# =========================
# Public optimizer
# =========================


@dataclass
class ArchitectureAwarePhasePolyOptimizer:
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
        )

        residual_rows = residual_linear_transform(phase_poly.out_parities, emitted_out_parities)
        residual_result = synthesize_linear_transform_architecture_aware_result(
            residual_rows,
            coupling_map,
        )
        residual_circuit = residual_result.circuit

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
            residual_method=residual_result.method,
            residual_reduce_order=residual_result.reduce_order,
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
                    residual_method="skipped_disconnected",
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
