from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap


class TopologyAwarePhasePolyOptimizer:
    """
    Uses the algorithm of Meijer-van de Griend / Duncan for the phase part, and then appends a
    routed CNOT-only correction circuit so the optimized block is exactly
    equivalent to the original block.

    Public workflow:
        optimizer = TopologyAwarePhasePolyOptimizer(circuit, coupling_map)
        optimized_circuit = optimizer.optimize()

    Notes
    -----
    - Only contiguous {cx, rz} blocks are optimized.
    - The optimizer only uses the active qubits of each block. It does not
      route through inactive qubits, because those qubits may carry unknown
      state in the surrounding circuit.
    - If the active induced subgraph is disconnected, the original block is
      kept unchanged.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        coupling_map: Union[CouplingMap, Sequence[Tuple[int, int]]],
        routing_method: str = "basic",
        optimization_level: int = 0,
    ) -> None:
        self.circuit = circuit
        self.blocks: List[Tuple[int, int, List[int], List[Any]]] = []
        self.routing_method = routing_method
        self.optimization_level = optimization_level

        if isinstance(coupling_map, CouplingMap):
            edges = list(coupling_map.get_edges())
        else:
            edges = [(int(u), int(v)) for u, v in coupling_map]

        self.coupling_map = CouplingMap(edges)
        self.global_graph = nx.Graph()
        self.global_graph.add_edges_from((int(u), int(v)) for u, v in edges)

    @staticmethod
    def _inst_fields(inst: Any) -> Tuple[Any, Any, Any]:
        return inst.operation, inst.qubits, inst.clbits

    def find_blocks(self) -> List[Tuple[int, int, List[int], List[Any]]]:
        """
        Find maximal contiguous blocks containing only CX and RZ gates.
        """
        blocks: List[Tuple[int, int, List[int], List[Any]]] = []
        current_block: Optional[dict] = None

        for i, inst in enumerate(self.circuit.data):
            gate, qargs, _ = self._inst_fields(inst)

            if gate.name in {"cx", "rz"}:
                indices = [self.circuit.find_bit(q).index for q in qargs]

                if current_block is None:
                    current_block = {
                        "start": i,
                        "end": i,
                        "qubits": set(indices),
                        "gates": [(gate, qargs)],
                    }
                else:
                    current_block["end"] = i
                    current_block["qubits"].update(indices)
                    current_block["gates"].append((gate, qargs))
            else:
                if current_block is not None:
                    blocks.append(
                        (
                            current_block["start"],
                            current_block["end"],
                            list(current_block["qubits"]),
                            current_block["gates"],
                        )
                    )
                    current_block = None

        if current_block is not None:
            blocks.append(
                (
                    current_block["start"],
                    current_block["end"],
                    list(current_block["qubits"]),
                    current_block["gates"],
                )
            )

        self.blocks = blocks
        return blocks

    def extract_block_compact(self, start: int, end: int) -> Tuple[QuantumCircuit, List[int]]:
        """
        Extract [start, end] to a compact circuit over only the active qubits.

        Returns
        -------
        block: QuantumCircuit
            Compact local circuit.
        active_qubits: list[int]
            Local-to-global qubit map.
        """
        active_set = set()

        for i in range(start, end + 1):
            gate, qargs, _ = self._inst_fields(self.circuit.data[i])
            if gate.name not in {"cx", "rz"}:
                continue
            for q in qargs:
                active_set.add(self.circuit.find_bit(q).index)

        active_qubits = sorted(active_set)
        local_of_global = {g: i for i, g in enumerate(active_qubits)}
        block = QuantumCircuit(len(active_qubits), name=f"block_{start}_{end}")

        for i in range(start, end + 1):
            gate, qargs, _ = self._inst_fields(self.circuit.data[i])
            if gate.name not in {"cx", "rz"}:
                continue

            local_qargs = [
                block.qubits[local_of_global[self.circuit.find_bit(q).index]]
                for q in qargs
            ]
            block.append(gate, local_qargs, [])

        return block, active_qubits

    @staticmethod
    def _gf2_eye(n: int) -> np.ndarray:
        return np.eye(n, dtype=bool)

    @staticmethod
    def _gf2_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A_u8 = A.astype(np.uint8)
        B_u8 = B.astype(np.uint8)
        return ((A_u8 @ B_u8) % 2).astype(bool)

    def _gf2_invert(self, M: np.ndarray) -> np.ndarray:
        M = np.array(M, dtype=bool, copy=True)
        n = M.shape[0]
        A = np.hstack([M, self._gf2_eye(n)])

        for col in range(n):
            pivot = None
            for row in range(col, n):
                if A[row, col]:
                    pivot = row
                    break

            if pivot is None:
                raise ValueError("Matrix is not invertible over GF(2).")

            if pivot != col:
                A[[col, pivot]] = A[[pivot, col]]

            for row in range(n):
                if row != col and A[row, col]:
                    A[row] ^= A[col]

        if not np.array_equal(A[:, :n], self._gf2_eye(n)):
            raise ValueError("GF(2) inversion failed.")

        return A[:, n:]

    def _extract_phase_polynomial_and_linear_map(
        self,
        block: QuantumCircuit,
    ) -> Tuple[np.ndarray, List[Any], np.ndarray]:
        """
        Extract the phase polynomial support and the block's linear map.

        Conventions
        -----------
        If the current computational-basis values are q = A x, then row k of A
        is the Boolean linear form carried by qubit k in terms of the original
        inputs x.
        """
        n = block.num_qubits
        A = self._gf2_eye(n)
        parity_dict = {}

        for inst in block.data:
            gate, qargs, _ = self._inst_fields(inst)

            if gate.name == "cx":
                control = block.find_bit(qargs[0]).index
                target = block.find_bit(qargs[1]).index
                A[target] ^= A[control]
            elif gate.name == "rz":
                qubit = block.find_bit(qargs[0]).index
                parity = tuple(bool(v) for v in A[qubit])
                angle = gate.params[0]
                parity_dict[parity] = parity_dict.get(parity, 0) + angle
            else:
                raise ValueError(f"Unexpected gate {gate.name} in phase-polynomial block.")

        parities = list(parity_dict.keys())
        m = len(parities)
        P_abs = np.zeros((n, m), dtype=bool)
        angles: List[Any] = []

        for j, parity in enumerate(parities):
            P_abs[:, j] = np.array(parity, dtype=bool)
            angles.append(parity_dict[parity])

        return P_abs, angles, A.copy()

    def _make_local_graph(self, active_global_qubits: List[int]) -> nx.Graph:
        active = list(active_global_qubits)
        active_set = set(active)
        relabel = {gq: i for i, gq in enumerate(active)}

        g = nx.Graph()
        g.add_nodes_from(active)
        for u, v in self.global_graph.edges():
            if u in active_set and v in active_set:
                g.add_edge(u, v)

        return nx.relabel_nodes(g, relabel, copy=True)

    @staticmethod
    def _non_cutting_vertices(graph: nx.Graph, subset: List[int]) -> List[int]:
        if len(subset) <= 2:
            return list(subset)

        sub = graph.subgraph(subset)
        cuts = set(nx.articulation_points(sub))
        non_cuts = [q for q in subset if q not in cuts]
        return non_cuts if non_cuts else list(subset)

    def _reduce_finished_columns(
        self,
        P: np.ndarray,
        angles: List[Any],
        cols: List[int],
        circ: QuantumCircuit,
    ) -> List[int]:
        """
        Emit RZ gates for trivial columns (support size 1) and remove them.
        """
        remaining = list(cols)
        for col in list(remaining):
            support = np.flatnonzero(P[:, col])
            if len(support) == 1:
                circ.rz(angles[col], int(support[0]))
                remaining.remove(col)
        return remaining

    @staticmethod
    def _choose_base_row(P: np.ndarray, cols: List[int], candidate_rows: List[int]) -> int:
        """
        Pick the non-cutting vertex with the most skewed split.
        """

        def score(row: int) -> int:
            ones = int(np.count_nonzero(P[row, cols]))
            zeros = len(cols) - ones
            return max(ones, zeros)

        return int(max(candidate_rows, key=score))

    @staticmethod
    def _choose_neighbor(P: np.ndarray, cols: List[int], neighbors: List[int]) -> int:
        def score(row: int) -> int:
            return int(np.count_nonzero(P[row, cols]))

        return int(max(neighbors, key=score))

    def _place_phase_cx(
        self,
        circ: QuantumCircuit,
        P: np.ndarray,
        A_phase: np.ndarray,
        control: int,
        target: int,
    ) -> None:
        """
        Apply CX(control, target).

        Important: the phase-gadget support matrix P and the computational
        basis linear map A_phase update differently.

        - For the phase-gadget support matrix (commuting the CNOT through the
          remaining phase gadgets), the control row changes:
              P[control, :] ^= P[target, :]
        - For the actual computational-basis linear map of the circuit, the
          target row changes:
              A_phase[target, :] ^= A_phase[control, :]
        """
        circ.cx(control, target)
        P[control, :] ^= P[target, :]
        A_phase[target, :] ^= A_phase[control, :]

    def _synthesize_phase_polynomial_topology_aware(
        self,
        block: QuantumCircuit,
        P_abs: np.ndarray,
        angles: List[Any],
        graph: nx.Graph,
    ) -> Tuple[QuantumCircuit, np.ndarray]:
        """
        Architecture-aware synthesis of the phase part only.
        """
        n = block.num_qubits
        circ = QuantumCircuit(n, name=f"{block.name}_phase_topo")
        P = np.array(P_abs, dtype=bool, copy=True)
        A_phase = self._gf2_eye(n)

        active_cols = self._reduce_finished_columns(P, angles, list(range(P.shape[1])), circ)

        def base_recurse(cols_to_use: List[int], qubits_to_use: List[int]) -> None:
            cols_to_use[:] = self._reduce_finished_columns(P, angles, cols_to_use, circ)
            if not cols_to_use or not qubits_to_use:
                return

            candidate_rows = self._non_cutting_vertices(graph, qubits_to_use)
            chosen_row = self._choose_base_row(P, cols_to_use, candidate_rows)

            cols0 = [c for c in cols_to_use if not P[chosen_row, c]]
            cols1 = [c for c in cols_to_use if P[chosen_row, c]]

            base_recurse(cols0, [q for q in qubits_to_use if q != chosen_row])
            one_recurse(cols1, qubits_to_use, chosen_row)

        def one_recurse(cols_to_use: List[int], qubits_to_use: List[int], row: int) -> None:
            cols_to_use[:] = self._reduce_finished_columns(P, angles, cols_to_use, circ)
            if not cols_to_use:
                return

            neighbors = [q for q in graph.neighbors(row) if q in qubits_to_use]
            if not neighbors:
                raise RuntimeError(
                    f"No available neighbor for local qubit {row} in subset {qubits_to_use}."
                )

            chosen_neighbor = self._choose_neighbor(P, cols_to_use, neighbors)
            neighbor_ones = int(np.count_nonzero(P[chosen_neighbor, cols_to_use]))

            if neighbor_ones > 0:
                # Paper / reference implementation case:
                #   PlaceCNOT(row, chosen_neighbor)
                # which updates the PHASE matrix as P[row] ^= P[chosen_neighbor].
                self._place_phase_cx(
                    circ=circ,
                    P=P,
                    A_phase=A_phase,
                    control=row,
                    target=chosen_neighbor,
                )
                cols_to_use[:] = self._reduce_finished_columns(P, angles, cols_to_use, circ)
            else:
                # If the chosen neighbor is all-zero on the active columns,
                # the paper uses two CNOTs that act like a restricted swap.
                self._place_phase_cx(
                    circ=circ,
                    P=P,
                    A_phase=A_phase,
                    control=chosen_neighbor,
                    target=row,
                )
                self._place_phase_cx(
                    circ=circ,
                    P=P,
                    A_phase=A_phase,
                    control=row,
                    target=chosen_neighbor,
                )

            cols0 = [c for c in cols_to_use if not P[row, c]]
            cols1 = [c for c in cols_to_use if P[row, c]]
            base_recurse(cols0, [q for q in qubits_to_use if q != row])
            one_recurse(cols1, qubits_to_use, row)

        base_recurse(active_cols, list(range(n)))
        return circ, A_phase

    def _route_cnot_only_circuit(self, circuit: QuantumCircuit, local_graph: nx.Graph) -> QuantumCircuit:
        if circuit.num_qubits <= 1 or len(circuit.data) == 0:
            return circuit.copy()

        bidir_edges = []
        for u, v in local_graph.edges():
            bidir_edges.append((u, v))
            bidir_edges.append((v, u))

        routed = transpile(
            circuit,
            basis_gates=["cx"],
            coupling_map=CouplingMap(bidir_edges),
            initial_layout=list(range(circuit.num_qubits)),
            layout_method="trivial",
            routing_method=self.routing_method,
            optimization_level=self.optimization_level,
        )
        routed.global_phase = 0
        return routed

    @staticmethod
    def _record_row_add(
        B: np.ndarray,
        ops: List[Tuple[int, int]],
        control: int,
        target: int,
    ) -> None:
        if control == target:
            return
        ops.append((control, target))
        B[target] ^= B[control]

    @staticmethod
    def _approx_steiner_tree(graph: nx.Graph, root: int, terminals: List[int]) -> nx.Graph:
        if root not in graph:
            raise RuntimeError(f"Root {root} not present in working graph.")
        for t in terminals:
            if t not in graph:
                raise RuntimeError(f"Terminal {t} not present in working graph.")

        tree = nx.Graph()
        tree.add_node(root)

        remaining = set(terminals)
        remaining.discard(root)

        while remaining:
            lengths, paths = nx.multi_source_dijkstra(graph, sources=list(tree.nodes()))
            reachable = [t for t in remaining if t in lengths]
            if not reachable:
                raise RuntimeError(
                    f"Cannot connect remaining terminals {sorted(remaining)} to root {root}."
                )

            best_t = min(reachable, key=lambda t: lengths[t])
            nx.add_path(tree, paths[best_t])
            remaining.remove(best_t)

        if tree.number_of_edges() == 0:
            return tree

        weighted = nx.Graph()
        for u, v in tree.edges():
            weighted.add_edge(u, v, weight=1)
        return nx.minimum_spanning_tree(weighted, weight="weight")

    def _steiner_reduce_column(
        self,
        B: np.ndarray,
        ops: List[Tuple[int, int]],
        tree: nx.Graph,
        root: int,
        col: int,
    ) -> None:
        if tree.number_of_nodes() == 0:
            raise RuntimeError("Steiner tree is empty.")

        oriented = nx.bfs_tree(tree, root)

        # Fill phase: child -> parent when parent is 0 and child is 1.
        for node in nx.dfs_postorder_nodes(oriented, source=root):
            if node == root:
                continue
            parent = next(oriented.predecessors(node))
            if (not B[parent, col]) and B[node, col]:
                self._record_row_add(B, ops, control=node, target=parent)

        for node in tree.nodes():
            if not B[node, col]:
                raise RuntimeError(
                    f"Fill phase failed for column {col}: node {node} does not carry 1."
                )

        # Elimination phase: parent -> child in postorder.
        for node in nx.dfs_postorder_nodes(oriented, source=root):
            if node == root:
                continue
            parent = next(oriented.predecessors(node))
            if B[node, col]:
                self._record_row_add(B, ops, control=parent, target=node)

        for node in tree.nodes():
            expected = node == root
            if bool(B[node, col]) != expected:
                raise RuntimeError(
                    f"Column reduction failed for column {col}: node {node} has "
                    f"{int(B[node, col])}, expected {int(expected)}."
                )

    def _steiner_gauss_synthesize(
        self,
        block: QuantumCircuit,
        A_target: np.ndarray,
        graph: nx.Graph,
    ) -> QuantumCircuit:
        """
        Synthesize a routed CNOT-only circuit for A_target.
        """
        n = A_target.shape[0]
        B = np.array(A_target, dtype=bool, copy=True)
        ops: List[Tuple[int, int]] = []

        # Forward elimination.
        for col in range(n):
            active_rows = list(range(col, n))
            working_graph = graph.subgraph(active_rows).copy()

            if col not in working_graph:
                raise RuntimeError(f"Pivot row {col} missing in forward subgraph.")

            terminals = [r for r in active_rows if B[r, col]]
            if not terminals:
                raise RuntimeError(f"A_target is singular or unreachable in column {col}.")
            if col not in terminals:
                terminals.append(col)

            tree = self._approx_steiner_tree(working_graph, root=col, terminals=terminals)
            self._steiner_reduce_column(B, ops, tree, root=col, col=col)

        # Backward elimination.
        for col in range(n - 1, -1, -1):
            if not B[col, col]:
                raise RuntimeError(f"Lost diagonal 1 at column {col} during backward phase.")

            active_rows = list(range(0, col + 1))
            working_graph = graph.subgraph(active_rows).copy()

            if col not in working_graph:
                raise RuntimeError(f"Pivot row {col} missing in backward subgraph.")

            terminals = [r for r in active_rows if B[r, col]]
            if terminals == [col]:
                continue
            if col not in terminals:
                terminals.append(col)

            tree = self._approx_steiner_tree(working_graph, root=col, terminals=terminals)
            self._steiner_reduce_column(B, ops, tree, root=col, col=col)

        if not np.array_equal(B, self._gf2_eye(n)):
            raise RuntimeError("Steiner-Gauss failed to reduce A_target to identity.")

        circ = QuantumCircuit(block.num_qubits, name=f"{block.name}_steiner_gauss")
        for control, target in reversed(ops):
            circ.cx(control, target)
        return circ

    def _synthesize_linear_map_all_to_all(
        self,
        block: QuantumCircuit,
        A_target: np.ndarray,
    ) -> QuantumCircuit:
        """
        All-to-all fallback for the correction circuit before final routing.
        """
        n = A_target.shape[0]
        B = np.array(A_target, dtype=bool, copy=True)
        ops: List[Tuple[int, int]] = []

        for col in range(n):
            if not B[col, col]:
                pivot = None
                for row in range(col + 1, n):
                    if B[row, col]:
                        pivot = row
                        break
                if pivot is None:
                    raise ValueError("A_target is not invertible over GF(2).")

                B[col] ^= B[pivot]
                ops.append((pivot, col))
                B[pivot] ^= B[col]
                ops.append((col, pivot))
                B[col] ^= B[pivot]
                ops.append((pivot, col))

            for row in range(col + 1, n):
                if B[row, col]:
                    B[row] ^= B[col]
                    ops.append((col, row))

        for col in range(n - 1, -1, -1):
            for row in range(col):
                if B[row, col]:
                    B[row] ^= B[col]
                    ops.append((col, row))

        if not np.array_equal(B, self._gf2_eye(n)):
            raise RuntimeError("Failed to reduce linear map to identity.")

        circ = QuantumCircuit(block.num_qubits, name=f"{block.name}_all2all_fix")
        for control, target in reversed(ops):
            circ.cx(control, target)
        return circ

    def optimize_block(
        self,
        block: QuantumCircuit,
        active_qubits: List[int],
    ) -> QuantumCircuit:
        """
        Optimize one compact block while preserving exact block equivalence.
        """
        if block.num_qubits <= 1:
            return block.copy()

        local_graph = self._make_local_graph(active_qubits)
        if not local_graph.nodes:
            return block.copy()
        if not nx.is_connected(local_graph):
            return block.copy()

        P_abs, angles, A_orig = self._extract_phase_polynomial_and_linear_map(block)

        phase_circ, A_phase = self._synthesize_phase_polynomial_topology_aware(
            block=block,
            P_abs=P_abs,
            angles=angles,
            graph=local_graph,
        )

        A_fix = self._gf2_matmul(A_orig, self._gf2_invert(A_phase))

        try:
            fix_circ = self._steiner_gauss_synthesize(
                block=block,
                A_target=A_fix,
                graph=local_graph,
            )
        except RuntimeError:
            fix_all_to_all = self._synthesize_linear_map_all_to_all(block, A_fix)
            fix_circ = self._route_cnot_only_circuit(fix_all_to_all, local_graph)

        out = QuantumCircuit(block.num_qubits, name=f"{block.name}_topology_aware")
        out.compose(phase_circ, inplace=True)
        out.compose(fix_circ, inplace=True)
        return out

    @staticmethod
    def _is_better_block(orig_block: QuantumCircuit, cand_block: QuantumCircuit) -> Tuple[bool, dict]:
        orig_ops = orig_block.count_ops()
        cand_ops = cand_block.count_ops()

        orig_cx = int(orig_ops.get("cx", 0))
        cand_cx = int(cand_ops.get("cx", 0))
        orig_depth = int(orig_block.depth())
        cand_depth = int(cand_block.depth())

        if cand_cx < orig_cx:
            decision = "optimized (fewer cx)"
            use_candidate = True
        elif cand_cx == orig_cx and cand_depth < orig_depth:
            decision = "optimized (same cx, lower depth)"
            use_candidate = True
        else:
            decision = "kept original"
            use_candidate = False

        return use_candidate, {
            "orig_cx": orig_cx,
            "cand_cx": cand_cx,
            "orig_depth": orig_depth,
            "cand_depth": cand_depth,
            "decision": decision,
        }

    @staticmethod
    def _log_block_decision(
        block_id: int,
        start: int,
        end: int,
        qubits: List[int],
        stats: dict,
        debug: bool = False,
    ) -> None:
        if not debug:
            return
        print(
            f"[Block {block_id}] gates {start}-{end}, qubits={qubits} | "
            f"orig: cx={stats['orig_cx']}, depth={stats['orig_depth']} | "
            f"cand: cx={stats['cand_cx']}, depth={stats['cand_depth']} -> "
            f"{stats['decision']}"
        )

    def replace_blocks(
        self,
        blocks: List[Tuple[int, int, List[int], List[Any]]],
        debug: bool = False,
    ) -> QuantumCircuit:
        block_map = {}

        for block_id, (start, end, _, _) in enumerate(blocks, start=1):
            orig_block, active_qubits = self.extract_block_compact(start, end)
            cand_block = self.optimize_block(orig_block, active_qubits=active_qubits)

            use_candidate, stats = self._is_better_block(orig_block, cand_block)
            self._log_block_decision(
                block_id=block_id,
                start=start,
                end=end,
                qubits=active_qubits,
                stats=stats,
                debug=debug,
            )

            chosen_block = cand_block if use_candidate else orig_block
            block_map[start] = (end, chosen_block, active_qubits)

        new_circuit = self.circuit.copy_empty_like()
        i = 0

        while i < len(self.circuit.data):
            if i in block_map:
                end, chosen_block, active_qubits = block_map[i]

                for inst in chosen_block.data:
                    instr, qargs, _ = self._inst_fields(inst)
                    mapped_qargs = [
                        new_circuit.qubits[active_qubits[chosen_block.find_bit(q).index]]
                        for q in qargs
                    ]
                    new_circuit.append(instr, mapped_qargs, [])

                i = end + 1
            else:
                instr, qargs, cargs = self._inst_fields(self.circuit.data[i])
                mapped_qargs = [new_circuit.qubits[self.circuit.find_bit(q).index] for q in qargs]
                mapped_cargs = [new_circuit.clbits[self.circuit.find_bit(c).index] for c in cargs]
                new_circuit.append(instr, mapped_qargs, mapped_cargs)
                i += 1

        return new_circuit

    def optimize(self, debug: bool = False) -> QuantumCircuit:
        blocks = self.find_blocks()
        if not blocks:
            return self.circuit.copy()
        return self.replace_blocks(blocks, debug=debug)
