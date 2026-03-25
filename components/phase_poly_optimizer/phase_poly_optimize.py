import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from qiskit import QuantumCircuit


class PhasePolyOptimizer:
    """
    A class to find and optimize maximal non-overlapping phase polynomial blocks in a Qiskit circuit.
    A phase polynomial block consists only of CNOT and Rz gates.
    """

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.blocks = []  # List of (start_index, end_index, qubits, gates)

    def find_blocks(self) -> List[Tuple[int, int, List[int], List]]:
        """
        Finds all maximal non-overlapping phase polynomial blocks in the circuit.

        Returns:
            List of tuples (start_index, end_index, qubits, gates) for each block.
        """
        blocks = []
        current_block = None
        for i, (gate, qargs, _) in enumerate(self.circuit.data):
            if gate.name == 'cx' or gate.name == 'rz':
                # Convert qargs to integer indices
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
            start: Start index of the block in the original circuit.
            end: End index of the block in the original circuit.

        Returns:
            A new QuantumCircuit containing only the gates in the block.
        """
        subcircuit = QuantumCircuit(*self.circuit.qregs)
        for i in range(start, end + 1):
            subcircuit.append(self.circuit.data[i][0], self.circuit.data[i][1])
        return subcircuit

    def _extract_parity_table(self, block: QuantumCircuit) -> Tuple[np.ndarray, List[float]]:
        """
        Extracts the parity table from a block of CNOT and Rz gates.
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
                angle = float(gate.params[0])  # assume parameter is a float (or sympy)
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
        Chooses the parity with minimal Hamming weight.

        Args:
            P: Parity table.

        Returns:
            Index of the chosen parity and the parity itself.
        """
        pass

    def _build_parity_graph(self, y: np.ndarray, P: np.ndarray) -> nx.DiGraph:
        """
        Builds the parity graph G_y for a given parity y.

        Args:
            y: Parity to synthesize.
            P: Parity table.

        Returns:
            Directed graph G_y with weights.
        """
        vertices = [i for i, val in enumerate(y) if val]
        G = nx.DiGraph()
        G.add_nodes_from(vertices)

        for i in vertices:
            for j in vertices:
                if i != j:
                    # Weight: h(P_i XOR P_j) - h(P_j)
                    weight = np.sum(P[i] ^ P[j]) - np.sum(P[j])
                    G.add_edge(i, j, weight=weight)

        return G

    def _find_min_weight_arborescence(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Finds the minimum weight spanning arborescence in G.

        Args:
            G: Directed graph.

        Returns:
            Minimum weight spanning arborescence.
        """
        # Tarjan's algorithm for minimum spanning arborescence
        return nx.algorithms.tree.branchings.minimum_spanning_arborescence(G)

    def _synthesize_block_all_to_all(self, P: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Synthesizes a phase polynomial block with all-to-all connectivity.

        Args:
            P: Parity table.

        Returns:
            Updated parity table and list of CNOT gates (control, target).
        """
        cnot_sequence = []
        while P.shape[1] > 0:
            col_idx, y = self._choose_parity(P)
            P = np.delete(P, col_idx, axis=1)
            G = self._build_parity_graph(y, P)
            arborescence = self._find_min_weight_arborescence(G)

            # Perform depth-first postorder traversal
            for i in nx.dfs_postorder_nodes(arborescence, source=next(iter(arborescence.nodes))):
                predecessors = list(arborescence.predecessors(i))
                if predecessors:
                    j = predecessors[0]
                    cnot_sequence.append((j, i))
                    # Update the parity table
                    P[i] ^= P[j]

        return P, cnot_sequence

    def optimize_block(self, block, connectivity="all-to-all"):
        """
        Optimizes a phase polynomial block using the specified algorithm.
        """
        parity_table = self._extract_parity_table(block)
        _, cnot_sequence = self._synthesize_block_all_to_all(parity_table)

        # Build the optimized circuit
        from qiskit import QuantumCircuit
        optimized_block = QuantumCircuit(*block.qregs)
        for control_idx, target_idx in cnot_sequence:
            optimized_block.cx(control_idx, target_idx)

        # Re-add Rz gates from the original block
        for gate, qargs, cargs in block.data:
            if gate.name.startswith('rz'):
                # Use find_bit to get the index of the qubit
                qubit = qargs[0]
                qubit_idx = block.find_bit(qubit)[0]
                optimized_block.rz(gate.params[0], qubit_idx)

        return optimized_block

    def replace_block(self, circuit: QuantumCircuit, start: int, end: int,
                      optimized_block: QuantumCircuit ) -> QuantumCircuit:
        """
        Replaces a phase polynomial block in the original circuit with an optimized version.

        Args:
            circuit: The original QuantumCircuit.
            start: Start index of the block to replace.
            end: End index of the block to replace.
            optimized_block: The optimized QuantumCircuit to insert.

        Returns:
            A new QuantumCircuit with the block replaced.
        """
        new_circuit = circuit.copy()
        new_circuit.data = (
            circuit.data[:start] +
            optimized_block.data +
            circuit.data[end + 1:]
        )
        return new_circuit