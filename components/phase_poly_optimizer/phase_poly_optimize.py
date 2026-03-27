import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression

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

    def _extract_parity_table(self, block: QuantumCircuit) -> Tuple[np.ndarray, List[float]]:
        """
        Extract the parity table and the corresponding angles from a block
        that consists only of CNOT and Rz gates.

        Returns:
            P: binary matrix of shape (n, m) where each column is a parity
            angles: list of floats, same length as number of columns
        """
        n = block.num_qubits
        # M: current linear mapping, initially identity
        M = np.eye(n, dtype=int)
        # Store parities as tuples and accumulated angles
        parity_dict = {}
        for gate, qargs, _ in block.data:
            if gate.name == 'cx':
                control = block.find_bit(qargs[0])[0]
                target = block.find_bit(qargs[1])[0]
                M[target] = (M[target] ^ M[control]) % 2
            elif gate.name == 'rz':
                qubit = block.find_bit(qargs[0])[0]
                # Get the parity vector for this qubit
                parity = tuple(M[qubit])  # convert to tuple for hashing
                angle = gate.params[0]  # keep symbolic if present
                if parity in parity_dict:
                    parity_dict[parity] += angle
                else:
                    parity_dict[parity] = angle
            else:
                raise ValueError(f"Unexpected gate {gate.name} in phase polynomial block")
        # Build parity table as a matrix (n x m)
        parities = list(parity_dict.keys())
        m = len(parities)
        P = np.zeros((n, m), dtype=int)
        angles = np.zeros(m, dtype=float)
        for j, parity in enumerate(parities):
            P[:, j] = list(parity)
            angles[j] = parity_dict[parity]
        return P, angles

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

    def synthesize_all_to_all(self, P: np.ndarray, angles: List[float]) -> QuantumCircuit:
        """
        Synthesise a phase polynomial using the all‑to‑all algorithm (Algorithm 1 of the paper).

        Args:
            P: parity table (n x m)
            angles: list of angles, same length as columns of P

        Returns:
            A QuantumCircuit implementing the phase polynomial.
        """
        n = P.shape[0]
        circ = QuantumCircuit(n)

        P = P.copy()
        angles = list(angles)

        while P.shape[1] > 0:
            # Step 1: choose a parity y with minimal Hamming weight
            col_idx, y = self._choose_parity(P)
            angle = angles.pop(col_idx)
            P = np.delete(P, col_idx, axis=1)   # remove column

            # Step 2: build parity graph and find minimum arborescence
            G = self._build_parity_graph(y, P)
            arborescence = self._find_min_weight_arborescence(G)

            # Find the root (node with indegree 0)
            roots = [n for n in arborescence.nodes if arborescence.in_degree(n) == 0]
            if not roots:
                # fallback to first node, although should not happen
                print("No roots found")
                root = next(iter(arborescence.nodes))
            else:
                root = roots[0]
            for node in arborescence.nodes:
                if len(list(arborescence.predecessors(node))) == 0:
                    root = node
                    break

            # Process the arborescence in a successors‑first (postorder) traversal
            for i in nx.dfs_postorder_nodes(arborescence, source=root):
                if i == root:
                    continue
                # Get the unique predecessor
                pred = next(iter(arborescence.predecessors(i)))
                circ.cx(pred, i)          # control = pred, target = i
                # Update the parity table rows
                P[i, :] ^= P[pred, :]

            # After the arborescence, the parity y is stored on the root qubit.
            # Apply the corresponding Rz gate.
            circ.rz(angle, root)

        return circ


    def optimize_block(self, block: QuantumCircuit) -> QuantumCircuit:
        """
        Optimise a single phase‑polynomial block using the all‑to‑all synthesis algorithm.

        Args:
            block: a QuantumCircuit that consists only of CNOT and Rz gates.

        Returns:
            An optimised QuantumCircuit implementing the same phase polynomial.
        """
        P, angles = self._extract_parity_table(block)
        opt_circ = self.synthesize_all_to_all(P, angles)
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

        i = 0
        while i < len(self.circuit.data):
            if i in block_map:
                end, opt_circ = block_map[i]
                # Append all instructions from the optimized circuit
                for instr, qargs, cargs in opt_circ.data:
                    new_circuit.append(instr, qargs, cargs)
                i = end + 1  # jump past the original block
            else:
                instr, qargs, cargs = self.circuit.data[i]
                new_circuit.append(instr, qargs, cargs)
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