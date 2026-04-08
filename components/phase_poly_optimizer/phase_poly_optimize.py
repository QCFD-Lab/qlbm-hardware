import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union
from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearFunction

class PhasePolyOptimizer:
    """
    Finds and optimises maximal non‑overlapping phase‑polynomial blocks in a Qiskit circuit.
    A phase‑polynomial block consists only of CNOT and single‑qubit Rz gates.
    """

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.blocks = []  # list of (start_idx, end_idx, qubits, gates)

    def find_blocks(self) -> List[Tuple[int, int, List[int], List]]:
        """
        Finds all maximal non‑overlapping phase‑polynomial blocks.

        Returns:
            List of tuples (start_index, end_index, qubit_indices, gate_list)
            where gate_list contains (gate, qargs) for each gate in the block.
        """
        blocks = []
        current_block = None
        for i, (gate, qargs, _) in enumerate(self.circuit.data):
            if gate.name == 'cx' or gate.name == 'rz':
                # Convert Qubit objects to integer indices
                indices = [self.circuit.find_bit(q)[0] for q in qargs]
                if current_block is None:
                    current_block = {
                        'start': i,
                        'end': i,
                        'qubits': set(indices),
                        'gates': [(gate, qargs)]
                    }
                else:
                    current_block['end'] = i
                    current_block['qubits'].update(indices)
                    current_block['gates'].append((gate, qargs))
            else:
                if current_block is not None:
                    blocks.append((
                        current_block['start'],
                        current_block['end'],
                        list(current_block['qubits']),
                        current_block['gates']
                    ))
                    current_block = None
        if current_block is not None:
            blocks.append((
                current_block['start'],
                current_block['end'],
                list(current_block['qubits']),
                current_block['gates']
            ))
        self.blocks = blocks
        return blocks

    def extract_block(self, start: int, end: int) -> QuantumCircuit:
        """
        Extracts a subcircuit for the phase polynomial block.

        Args:
            start: start instruction index
            end:   end instruction index (inclusive)

        Returns:
            A QuantumCircuit containing only the gates of the block.
        """
        subcircuit = QuantumCircuit(*self.circuit.qregs)
        for i in range(start, end + 1):
            subcircuit.append(self.circuit.data[i][0], self.circuit.data[i][1])
        return subcircuit

    def _extract_parity_table(self, block: QuantumCircuit):
        """
        Extract the parity table and the corresponding angles from a block
        that consists only of CNOT and Rz gates.
        """
        n = block.num_qubits
        M = np.eye(n, dtype=int)  # current linear mapping
        parity_dict = {}   # parity -> total angle

        for gate, qargs, _ in block.data:
            if gate.name == 'cx':
                control = block.find_bit(qargs[0])[0]
                target = block.find_bit(qargs[1])[0]
                M[target] = (M[target] ^ M[control]) % 2
            elif gate.name == 'rz':
                qubit = block.find_bit(qargs[0])[0]
                parity = tuple(M[qubit]) # tuple of ints (0/1)
                angle = gate.params[0]
                if parity in parity_dict:
                    parity_dict[parity] += angle
                else:
                    parity_dict[parity] = angle
            else:
                raise ValueError(f"Unexpected gate {gate.name} in phase polynomial block")
        # Build parity table and angle list
        parities = list(parity_dict.keys())
        m = len(parities)
        P = np.zeros((n, m), dtype=int)
        angles = [0.0] * m   # placeholder, will fill with actual angles
        for j, parity in enumerate(parities):
            P[:, j] = list(parity)
            angles[j] = parity_dict[parity]

        return P, angles, M

    def _choose_parity(self, P: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Choose the parity with minimal Hamming weight. Tie‑break by smallest
        integer representation (treating the column as binary with qubit 0 as LSB).
        """
        weights = np.sum(P, axis=0)          # shape (m,)
        min_weight = np.min(weights)
        candidates = np.where(weights == min_weight)[0]
        # Compute integer value for each candidate column
        powers = 1 << np.arange(P.shape[0])
        int_vals = (P.T @ powers)           # shape (m,)f
        best_idx = candidates[np.argmin(int_vals[candidates])]
        return best_idx, P[:, best_idx]

    def _build_parity_graph(self, y: np.ndarray, P: np.ndarray) -> nx.DiGraph:
        """
        Build the directed parity graph G_y for the given parity y and current parity table.
        Vertices are indices where y has a 1. Edge weight from i to j is
            h(P[i] XOR P[j]) - h(P[j]).
        """
        vertices = [i for i, val in enumerate(y) if val]
        G = nx.DiGraph()
        G.add_nodes_from(vertices)
        for i in vertices:
            for j in vertices:
                if i != j:
                    weight = np.sum(P[i] ^ P[j]) - np.sum(P[j])
                    G.add_edge(i, j, weight=weight)
        return G

    def _find_min_weight_arborescence(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Find a minimum weight spanning arborescence (optimum branching) in the directed graph.
        """
        # NetworkX's minimum_spanning_arborescence returns a branching (a forest).
        # For a complete directed graph it will be a single arborescence.
        return nx.algorithms.tree.branchings.minimum_spanning_arborescence(G)

    def _gf2_invert(self, M: np.ndarray) -> np.ndarray:
        n = len(M)
        A = np.hstack((M.copy(), np.eye(n, dtype=int)))
        for i in range(n):
            if A[i, i] == 0:
                for j in range(i + 1, n):
                    if A[j, i] == 1:
                        A[[i, j]] = A[[j, i]]
                        break
            for j in range(n):
                if i != j and A[j, i] == 1:
                    A[j] ^= A[i]
        return A[:, n:] % 2

    def synthesize_all_to_all(self, P: np.ndarray, angles: List[float], M_target: np.ndarray) -> QuantumCircuit:
        n = P.shape[0]
        circ = QuantumCircuit(n)

        # Track the linear transformation of the new optimized circuit
        M_opt = np.eye(n, dtype=int)

        P = P.copy()
        angles = list(angles)

        while P.shape[1] > 0:
            col_idx, y = self._choose_parity(P)
            angle = angles.pop(col_idx)
            P = np.delete(P, col_idx, axis=1)

            # Skip synthesis if parity is all zeros (can happen if angles map to parity 0)
            if np.sum(y) == 0:
                continue

            G = self._build_parity_graph(y, P)
            arborescence = self._find_min_weight_arborescence(G)

            roots = [node for node in arborescence.nodes if arborescence.in_degree(node) == 0]
            root = roots[0] if roots else next(iter(arborescence.nodes))

            for node in nx.dfs_postorder_nodes(arborescence, source=root):
                if node == root:
                    continue
                parent = next(iter(arborescence.predecessors(node)))

                circ.cx(parent, node)
                P[node, :] ^= P[parent, :]
                # Track the basis change exactly as the parity table
                M_opt[node, :] ^= M_opt[parent, :]

            if angle != 0.0:
                circ.rz(angle, root)

        # --- BASIS CORRECTION STEP ---
        # need a transformation T such that T @ M_opt = M_target (modulo 2).
        # so T = M_target @ inv(M_opt) (modulo 2).
        M_opt_inv = self._gf2_invert(M_opt)
        T = (M_target @ M_opt_inv) % 2

        # Qiskit's LinearFunction automatically synthesizes a CNOT network for this matrix
        correction_circ = LinearFunction(T).definition
        if correction_circ and len(correction_circ.data) > 0:
            circ = circ.compose(correction_circ)

        return circ


    def optimize_block(self, block: QuantumCircuit) -> QuantumCircuit:
        """
        Optimise a single phase‑polynomial block using the all‑to‑all synthesis algorithm.

        Args:
            block: a QuantumCircuit that consists only of CNOT and Rz gates.

        Returns:
            An optimised QuantumCircuit implementing the same phase polynomial.
        """
        P, angles, M_target = self._extract_parity_table(block)
        opt_circ = self.synthesize_all_to_all(P, angles, M_target)
        return opt_circ

    def replace_blocks(self, blocks: List[Tuple[int, int, List[int], List]]) -> QuantumCircuit:
        # Build a dictionary from start index to (end index, optimized circuit)
        block_map = {}
        for start, end, _, _ in blocks:
            block_circ = self.extract_block(start, end)
            opt_circ = self.optimize_block(block_circ)
            block_map[start] = (end, opt_circ)

        # Create a new circuit with the same registers as the original
        new_circuit = self.circuit.copy_empty_like()

        def map_qargs(source_circuit: QuantumCircuit, target_circuit: QuantumCircuit, qargs):
            return [target_circuit.qubits[source_circuit.find_bit(q).index] for q in qargs]

        def map_cargs(source_circuit: QuantumCircuit, target_circuit: QuantumCircuit, cargs):
            return [target_circuit.clbits[source_circuit.find_bit(c).index] for c in cargs]

        i = 0
        while i < len(self.circuit.data):
            if i in block_map:
                end, opt_circ = block_map[i]
                # append instructions from the optimized circuit remapping bits
                for instr, qargs, cargs in opt_circ.data:
                    mapped_qargs = map_qargs(opt_circ, new_circuit, qargs)
                    mapped_cargs = map_cargs(opt_circ, new_circuit, cargs)
                    new_circuit.append(instr, mapped_qargs, mapped_cargs)
                i = end + 1  # jump past the original block
            else:
                instr, qargs, cargs = self.circuit.data[i]
                mapped_qargs = map_qargs(self.circuit, new_circuit, qargs)
                mapped_cargs = map_cargs(self.circuit, new_circuit, cargs)
                new_circuit.append(instr, mapped_qargs, mapped_cargs)
                i += 1

        return new_circuit


    def optimize(self) -> QuantumCircuit:
        """Find and replace all phase‑polynomial blocks, returning the optimized circuit."""
        blocks = self.find_blocks()
        if not blocks:
            print("No blocks to optimize!")
            return self.circuit.copy()  # nothing to optimize
        optimized_circuit = self.replace_blocks(blocks)
        return optimized_circuit