from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from .architecture_aware_phasepoly_optimizer import (
    BitVec,
    BlockOptimizationReport,
    OptimizationRunReport,
    PhasePolynomial,
    _append_compact_circuit,
    _append_instruction_indices,
    _append_instruction_range,
    _append_original_instruction,
    _choose_better_phasepoly_block,
    _extract_compact_block,
    _new_like_circuit,
    _phasepoly_cost_tuple,
    _row_add_sequence_for_linear_transform,
    extract_phase_polynomial,
    find_phase_polynomial_blocks,
    gf2_add_rows,
    gf2_identity,
    is_phase_polynomial_block,
    residual_linear_transform,
    unitary_equiv_up_to_global_phase,
)


@dataclass
class _AllToAllSynthState:
    """
    Mutable state for Vandaele et al.'s all-to-all parity-network algorithm.

    `columns` is the remaining parity table. Emitting a Qiskit CX(control, target)
    updates this table as row `control ^= target`, matching the convention used in
    section 2 of the paper for parity-table basis changes.
    """

    num_qubits: int
    columns: Dict[int, Tuple[List[int], Any]]
    circuit: QuantumCircuit
    emitted_out_parities: List[BitVec]

    @classmethod
    def from_phase_poly(cls, phase_poly: PhasePolynomial) -> "_AllToAllSynthState":
        columns = {
            column_id: ([int(bit) for bit in parity], angle)
            for column_id, (parity, angle) in enumerate(phase_poly.zphases.items())
        }
        return cls(
            num_qubits=phase_poly.num_qubits,
            columns=columns,
            circuit=QuantumCircuit(phase_poly.num_qubits),
            emitted_out_parities=gf2_identity(phase_poly.num_qubits),
        )

    def place_cnot(self, control: int, target: int) -> None:
        self.circuit.cx(control, target)

        for bits, _ in self.columns.values():
            bits[control] ^= bits[target]

        self.emitted_out_parities[target] = gf2_add_rows(
            self.emitted_out_parities[control],
            self.emitted_out_parities[target],
        )


def _parity_integer_key(bits: Sequence[int]) -> Tuple[int, ...]:
    """
    Tie-break equal-weight parities as in section 3.1: choose the parity
    representing the smallest integer, with row 0 treated as the most
    significant bit.
    """

    return tuple(int(bit) for bit in bits)


def _choose_minimum_weight_column(state: _AllToAllSynthState) -> int:
    if not state.columns:
        raise ValueError("Cannot choose a parity from an empty table.")

    return min(
        state.columns,
        key=lambda column_id: (
            sum(state.columns[column_id][0]),
            _parity_integer_key(state.columns[column_id][0]),
        ),
    )


def _build_parity_graph(
    selected_bits: Sequence[int],
    remaining_columns: Dict[int, Tuple[List[int], Any]],
) -> nx.DiGraph:
    """
    Build Vandaele et al.'s complete directed parity graph for all-to-all coupling.

    An arc parent -> child is implemented as CX(child, parent). In the paper's
    row-addition notation this is `P_child = P_child xor P_parent`, so the arc
    weight is `h(P_child xor P_parent) - h(P_child)` over the remaining table.
    """

    support = [index for index, bit in enumerate(selected_bits) if bit]
    graph = nx.DiGraph()
    graph.add_nodes_from(support)

    for parent in support:
        for child in support:
            if parent == child:
                continue
            weight = 0
            for bits, _ in remaining_columns.values():
                weight += (bits[child] ^ bits[parent]) - bits[child]
            graph.add_edge(parent, child, weight=weight)

    return graph


def _minimum_weight_arborescence(graph: nx.DiGraph) -> nx.DiGraph:
    if graph.number_of_nodes() <= 1:
        arborescence = nx.DiGraph()
        arborescence.add_nodes_from(graph.nodes)
        return arborescence
    return nx.algorithms.tree.branchings.minimum_spanning_arborescence(graph)


def _arborescence_root(arborescence: nx.DiGraph) -> int:
    roots = [node for node in arborescence.nodes if arborescence.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError("Expected exactly one root in the minimum arborescence.")
    return int(roots[0])


def synthesize_phase_support_all_to_all(
    phase_poly: PhasePolynomial,
) -> Tuple[QuantumCircuit, List[BitVec]]:
    """
    Synthesize the phase-support part using Algorithm 1 of Vandaele et al. (2022).

    The paper focuses on parity-network synthesis and intentionally leaves the
    final linear reversible function as a separate problem. The returned
    `emitted_out_parities` records the linear map introduced by the generated CNOT
    network so the caller can append that residual transform.
    """

    state = _AllToAllSynthState.from_phase_poly(phase_poly)

    while state.columns:
        column_id = _choose_minimum_weight_column(state)
        selected_bits, angle = state.columns.pop(column_id)
        support = [index for index, bit in enumerate(selected_bits) if bit]

        if len(support) == 0:
            continue
        if len(support) == 1:
            state.circuit.rz(angle, support[0])
            continue

        graph = _build_parity_graph(selected_bits, state.columns)
        arborescence = _minimum_weight_arborescence(graph)
        root = _arborescence_root(arborescence)

        for node in nx.dfs_postorder_nodes(arborescence, source=root):
            if node == root:
                continue
            parent = next(iter(arborescence.predecessors(node)))
            state.place_cnot(int(node), int(parent))

        state.circuit.rz(angle, root)

    return state.circuit, state.emitted_out_parities


def synthesize_linear_transform_all_to_all(matrix_rows: Sequence[BitVec]) -> QuantumCircuit:
    """
    Synthesize an invertible GF(2) linear transform with unrestricted CNOTs.
    """

    circuit = QuantumCircuit(len(matrix_rows))
    for control, target in _row_add_sequence_for_linear_transform(matrix_rows):
        circuit.cx(control, target)
    return circuit


@dataclass
class A2APhasePoly:
    """
    All-to-all phase-polynomial optimizer with the same public shape as
    `ArchitectureAwarePhasePolyOptimizer`.

    The class accepts the old `A2APhasePoly(circuit).optimize()` style as a
    compatibility convenience, but the preferred interface is
    `A2APhasePoly().optimize(circuit, coupling_map=None)`.
    """

    circuit: Optional[QuantumCircuit] = None
    allow_barriers: bool = True
    debug: bool = False
    keep_original_if_worse: bool = True
    last_run_report: Optional[OptimizationRunReport] = None

    def _optimize_phase_polynomial_block(
        self,
        circuit: QuantumCircuit,
    ) -> Tuple[QuantumCircuit, BlockOptimizationReport]:
        phase_poly = extract_phase_polynomial(circuit)
        phase_circuit, emitted_out_parities = synthesize_phase_support_all_to_all(
            phase_poly
        )

        residual_rows = residual_linear_transform(
            phase_poly.out_parities,
            emitted_out_parities,
        )
        residual_circuit = synthesize_linear_transform_all_to_all(residual_rows)

        candidate = QuantumCircuit(circuit.num_qubits)
        candidate.compose(phase_circuit, inplace=True)
        candidate.compose(residual_circuit, inplace=True)

        if self.debug and not unitary_equiv_up_to_global_phase(circuit, candidate):
            raise ValueError("Synthesized all-to-all block is not equivalent to input.")

        final_circuit = candidate
        kept_original = False
        if self.keep_original_if_worse:
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
            residual_method="all_to_all_gauss",
            residual_reduce_order=None,
        )
        return final_circuit, report

    def optimize(
        self,
        circuit: Optional[QuantumCircuit] = None,
        coupling_map: Optional[CouplingMap] = None,
        *,
        debug: Optional[bool] = None,
    ) -> QuantumCircuit:
        """
        Optimize all maximal {cx, rz} phase-polynomial blocks.

        `coupling_map` is accepted for interface parity with the architecture-aware
        optimizer and intentionally ignored because this optimizer assumes all-to-all
        coupling.
        """

        del coupling_map
        if debug is not None:
            self.debug = debug

        source = circuit if circuit is not None else self.circuit
        if source is None:
            raise TypeError("A circuit must be passed to optimize().")

        if is_phase_polynomial_block(source, allow_barriers=self.allow_barriers):
            optimized, report = self._optimize_phase_polynomial_block(source)
            self.last_run_report = OptimizationRunReport(
                circuit_num_qubits=source.num_qubits,
                num_blocks=1,
                block_reports=[report],
            )
            return optimized

        blocks = find_phase_polynomial_blocks(
            source,
            allow_barriers=self.allow_barriers,
        )
        if not blocks:
            self.last_run_report = OptimizationRunReport(
                circuit_num_qubits=source.num_qubits,
                num_blocks=0,
                block_reports=[],
            )
            return source.copy()

        block_by_start = {block.start: block for block in blocks}
        out = _new_like_circuit(source)
        block_reports: List[BlockOptimizationReport] = []
        index = 0

        while index < len(source.data):
            block = block_by_start.get(index)
            if block is None:
                _append_original_instruction(out, source, source.data[index])
                index += 1
                continue

            compact = _extract_compact_block(source, block)
            optimized_block, report = self._optimize_phase_polynomial_block(compact)
            report.start = block.start
            report.end = block.end
            report.active_qubits = list(block.active_qubits)
            report.connected_active_subgraph = True
            block_reports.append(report)

            if (
                block.passthrough_instruction_indices
                and _phasepoly_cost_tuple(optimized_block) >= _phasepoly_cost_tuple(compact)
            ):
                report.final_cx = report.original_cx
                report.kept_original = True
                _append_instruction_range(out, source, block.start, block.end)
                index = block.end + 1
                continue

            _append_instruction_indices(out, source, block.passthrough_instruction_indices)
            _append_compact_circuit(out, optimized_block, block.active_qubits)
            index = block.end + 1

        self.last_run_report = OptimizationRunReport(
            circuit_num_qubits=source.num_qubits,
            num_blocks=len(blocks),
            block_reports=block_reports,
        )
        return out


AllToAllPhasePolyOptimizer = A2APhasePoly
