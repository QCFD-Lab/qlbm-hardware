import numpy as np
import networkx as nx
from typing import List, Tuple, Any, Optional
from qiskit import QuantumCircuit

class A2APhasePoly:
    """
    Find and optimize maximal non-overlapping {cx, rz} blocks in a Qiskit circuit.

    Important:
    - This implementation preserves full block equivalence.
    - It does so by:
        1) extracting the phase polynomial p(x),
        2) synthesizing a new phase-polynomial circuit,
        3) extracting/tracking the linear reversible map g(x),
        4) appending a CNOT-only correction circuit so the total block matches
           the original unitary exactly (up to global phase).
    """

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.blocks = []

    @staticmethod
    def _inst_fields(inst: Any):
        """Return (operation, qubits, clbits) for a CircuitInstruction."""
        return inst.operation, inst.qubits, inst.clbits

    def find_blocks(self) -> List[Tuple[int, int, List[int], List]]:
        """
        Finds all maximal non-overlapping contiguous blocks consisting only of
        CNOT and Rz gates.

        Returns:
            List of tuples:
              (start_index, end_index, qubit_indices, gate_list)
        """
        blocks = []
        current_block = None

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
                    blocks.append((
                        current_block["start"],
                        current_block["end"],
                        list(current_block["qubits"]),
                        current_block["gates"],
                    ))
                    current_block = None

        if current_block is not None:
            blocks.append((
                current_block["start"],
                current_block["end"],
                list(current_block["qubits"]),
                current_block["gates"],
            ))

        self.blocks = blocks
        return blocks

    def extract_block_full(self, start: int, end: int) -> QuantumCircuit:
        """
        Extract a subcircuit for the block [start, end], keeping the same quantum
        registers as the original circuit.
        """
        subcircuit = QuantumCircuit(*self.circuit.qregs, *self.circuit.cregs)
        for i in range(start, end + 1):
            gate, qargs, cargs = self._inst_fields(self.circuit.data[i])
            subcircuit.append(gate, qargs, cargs)
        return subcircuit

    def extract_block_compact(self, start: int, end: int) -> Tuple[QuantumCircuit, List[int]]:
        """
        Extract block [start, end] onto a compact local circuit.

        Returns:
            block_circuit: circuit on only the active qubits of the block
            active_qubits: local->global qubit index map
        """
        # active_qubits: List[int] = []
        # seen = set()
        #
        # # Preserve first-seen order for stable local indexing
        # for i in range(start, end + 1):
        #     gate, qargs, _ = self._inst_fields(self.circuit.data[i])
        #     if gate.name not in {"cx", "rz"}:
        #         continue
        #     for q in qargs:
        #         qidx = self.circuit.find_bit(q).index
        #         if qidx not in seen:
        #             seen.add(qidx)
        #             active_qubits.append(qidx)

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
        """
        Matrix multiplication over GF(2), returning bool.
        """
        A_u8 = A.astype(np.uint8)
        B_u8 = B.astype(np.uint8)
        return ((A_u8 @ B_u8) % 2).astype(bool)

    def _gf2_invert(self, M: np.ndarray) -> np.ndarray:
        """
        Invert an invertible matrix over GF(2).
        Returns a bool matrix.
        """
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

        left = A[:, :n]
        if not np.array_equal(left, self._gf2_eye(n)):
            raise ValueError("GF(2) inversion failed.")

        return A[:, n:]

    def _extract_phase_polynomial_and_linear_map(
        self, block: QuantumCircuit
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        Extract:
          - P_abs: absolute parity table, shape (n, m), dtype=bool
          - angles: corresponding list of Rz angles
          - A_orig: final computational-basis linear map of the original block

        Convention:
          If current qubit values are q = A x, then row k of A is the Boolean
          linear form carried by qubit k in terms of the original input bits x.
        """
        n = block.num_qubits
        A = self._gf2_eye(n)
        parity_dict = {}

        for inst in block.data:
            gate, qargs, _ = self._inst_fields(inst)

            if gate.name == "cx":
                control = block.find_bit(qargs[0]).index
                target = block.find_bit(qargs[1]).index

                # Actual computational-basis update:
                # x_target <- x_target xor x_control
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
        angles = []

        for j, parity in enumerate(parities):
            P_abs[:, j] = np.array(parity, dtype=bool)
            angles.append(parity_dict[parity])

        A_orig = A.copy()
        return P_abs, angles, A_orig

    def _choose_parity(self, P: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Choose a parity of minimum Hamming weight.
        Tie-break by smallest bitstring, treating row 0 as MSB.
        """
        if P.shape[1] == 0:
            raise ValueError("Cannot choose a parity from an empty table.")

        weights = np.count_nonzero(P, axis=0)
        min_weight = int(np.min(weights))
        candidates = np.where(weights == min_weight)[0]

        best_idx = min(
            (int(j) for j in candidates),
            key=lambda j: tuple(int(b) for b in P[:, j])
        )
        return best_idx, P[:, best_idx].copy()

    def _build_parity_graph(self, y: np.ndarray, P: np.ndarray) -> nx.DiGraph:
        """
        Build the all-to-all parity graph for the chosen parity y.

        Convention used here:
          edge parent -> child corresponds to applying CX(child, parent)
          during the postorder traversal.

        Under CX(child, parent), the parity-table update is:
          P[child, :] ^= P[parent, :]

        So the weight is:
          h(P[child] xor P[parent]) - h(P[child])

        Negative weights are expected and valid.
        """
        vertices = [i for i, bit in enumerate(y) if bit]
        G = nx.DiGraph()
        G.add_nodes_from(vertices)

        for parent in vertices:
            for child in vertices:
                if parent == child:
                    continue

                weight = (
                    int(np.count_nonzero(P[child] ^ P[parent]))
                    - int(np.count_nonzero(P[child]))
                )
                G.add_edge(parent, child, weight=weight)

        return G

    def _find_min_weight_arborescence(self, G: nx.DiGraph) -> nx.DiGraph:
        if len(G.nodes) <= 1:
            H = nx.DiGraph()
            H.add_nodes_from(G.nodes)
            return H

        return nx.algorithms.tree.branchings.minimum_spanning_arborescence(G)

    def _synthesize_phase_polynomial_all_to_all(
        self,
        block: QuantumCircuit,
        P_abs: np.ndarray,
        angles: List,
    ) -> Tuple[QuantumCircuit, np.ndarray]:
        """
        Synthesize only the phase-polynomial part.

        Returns:
          phase_circuit, A_phase

        where A_phase is the computational-basis linear map produced by the
        synthesized phase-polynomial circuit.
        """
        n = block.num_qubits
        circ = QuantumCircuit(*block.qregs, *block.cregs)

        # Current parity table in the current basis
        P = P_abs.copy()

        # Current computational-basis linear map of synthesized circuit
        A_phase = self._gf2_eye(n)

        angles = list(angles)

        while P.shape[1] > 0:
            col_idx, y = self._choose_parity(P)
            angle = angles.pop(col_idx)
            P = np.delete(P, col_idx, axis=1)

            support = np.flatnonzero(y)
            if len(support) == 0:
                continue

            if len(support) == 1:
                root = int(support[0])
                circ.rz(angle, root)
                continue

            G = self._build_parity_graph(y, P)
            arb = self._find_min_weight_arborescence(G)

            roots = [node for node in arb.nodes if arb.in_degree(node) == 0]
            if len(roots) != 1:
                raise RuntimeError("Expected exactly one arborescence root.")
            root = int(roots[0])

            # Successors-first traversal via DFS postorder
            for node in nx.dfs_postorder_nodes(arb, source=root):
                if node == root:
                    continue

                parent = next(iter(arb.predecessors(node)))

                # Gate direction chosen to match the parity-table update below
                circ.cx(node, parent)

                # Parity-table update in the current basis
                P[node, :] ^= P[parent, :]

                # Computational-basis linear-map update
                # CX(node, parent): x_parent <- x_parent xor x_node
                A_phase[parent, :] ^= A_phase[node, :]

            circ.rz(angle, root)

        return circ, A_phase

    def _synthesize_linear_map_all_to_all(
        self,
        block: QuantumCircuit,
        A_target: np.ndarray,
    ) -> QuantumCircuit:
        """
        Synthesize a CNOT-only circuit whose computational-basis linear map is A_target.
        Works for all-to-all connectivity.
        """
        n = A_target.shape[0]
        B = np.array(A_target, dtype=bool, copy=True)
        ops = []

        # Forward elimination
        for col in range(n):
            if not B[col, col]:
                pivot = None
                for row in range(col + 1, n):
                    if B[row, col]:
                        pivot = row
                        break

                if pivot is None:
                    raise ValueError("A_target is not invertible over GF(2).")

                # Swap rows col <-> pivot using 3 row additions
                B[col] ^= B[pivot]
                ops.append((pivot, col))   # CX(pivot, col)

                B[pivot] ^= B[col]
                ops.append((col, pivot))   # CX(col, pivot)

                B[col] ^= B[pivot]
                ops.append((pivot, col))   # CX(pivot, col)

            for row in range(col + 1, n):
                if B[row, col]:
                    B[row] ^= B[col]
                    ops.append((col, row))  # CX(col, row)

        # Backward elimination
        for col in range(n - 1, -1, -1):
            for row in range(col):
                if B[row, col]:
                    B[row] ^= B[col]
                    ops.append((col, row))  # CX(col, row)

        if not np.array_equal(B, self._gf2_eye(n)):
            raise RuntimeError("Failed to reduce linear map to identity.")

        circ = QuantumCircuit(block.num_qubits, name=f"{block.name}_all2all")

        # If G_k ... G_1 A = I, then A = G_1 ... G_k
        for control, target in reversed(ops):
            circ.cx(control, target)

        return circ

    def _is_better_block(self, orig_block: QuantumCircuit, cand_block: QuantumCircuit) -> Tuple[bool, dict]:
        """
        Decide whether cand_block should replace orig_block.

        Rule:
          1) fewer CX gates wins
          2) if CX counts tie, lower depth wins
          3) otherwise keep the original

        Returns:
          (use_candidate, stats_dict)
        """
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

        stats = {
            "orig_cx": orig_cx,
            "cand_cx": cand_cx,
            "orig_depth": orig_depth,
            "cand_depth": cand_depth,
            "decision": decision,
        }
        return use_candidate, stats

    def _log_block_decision(
            self,
            block_id: int,
            start: int,
            end: int,
            qubits: List[int],
            stats: dict,
            debug: bool = False,
    ) -> None:
        """
        Print per-block selection diagnostics when debug=True.
        """
        if not debug:
            return

        print(
            f"[Block {block_id}] gates {start}-{end}, qubits={qubits} | "
            f"orig: cx={stats['orig_cx']}, depth={stats['orig_depth']} | "
            f"cand: cx={stats['cand_cx']}, depth={stats['cand_depth']} -> "
            f"{stats['decision']}"
        )

    def optimize_block(self, block: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize a single compact {cx, rz} block while preserving block equivalence.
        """
        P_abs, angles, A_orig = self._extract_phase_polynomial_and_linear_map(block)

        phase_circ, A_phase = self._synthesize_phase_polynomial_all_to_all(
            block, P_abs, angles
        )

        A_phase_inv = self._gf2_invert(A_phase)
        A_fix = self._gf2_matmul(A_orig, A_phase_inv)
        fix_circ = self._synthesize_linear_map_all_to_all(block, A_fix)

        opt_circ = QuantumCircuit(*block.qregs, *block.cregs)
        opt_circ.compose(phase_circ, inplace=True)
        opt_circ.compose(fix_circ, inplace=True)
        return opt_circ

    def replace_blocks(
            self,
            blocks: List[Tuple[int, int, List[int], List]],
            debug: bool = False,
    ) -> QuantumCircuit:
        block_map = {}

        for block_id, (start, end, qubits, _) in enumerate(blocks, start=1):
            orig_block, active_qubits = self.extract_block_compact(start, end)
            cand_block = self.optimize_block(orig_block)

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
                    instr, qargs, cargs = self._inst_fields(inst)
                    mapped_qargs = [
                        new_circuit.qubits[active_qubits[chosen_block.find_bit(q).index]]
                        for q in qargs
                    ]
                    new_circuit.append(instr, mapped_qargs, []) #optimized phase polynomials contain only quantum ops
                i = end + 1
            else:
                instr, qargs, cargs = self._inst_fields(self.circuit.data[i])
                mapped_qargs = [new_circuit.qubits[self.circuit.find_bit(q).index] for q in qargs]
                mapped_cargs = [new_circuit.clbits[self.circuit.find_bit(c).index] for c in cargs]
                new_circuit.append(instr, mapped_qargs, mapped_cargs)
                i += 1

        return new_circuit

    def optimize(self, debug: bool = False) -> QuantumCircuit:
        """
        Find and replace all phase-polynomial blocks.
        """
        blocks = self.find_blocks()
        if not blocks:
            return self.circuit.copy()
        return self.replace_blocks(blocks, debug=debug)