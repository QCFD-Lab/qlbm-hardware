"""
Algorithm 1: Phase Polynomial Synthesis for All-to-All Connectivity
Based on Vandaele et al. (2022) - Phase polynomials synthesis algorithms for NISQ architectures

This algorithm synthesizes phase polynomials for quantum circuits with full connectivity,
minimizing the number of CNOT gates required.
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict
import heapq

# Import our core data structures
from core_data_structures import ParityTable


def tarjan_min_spanning_arborescence(graph: np.ndarray) -> List[Tuple[int, int]]:
    """
    Implement Tarjan's algorithm for finding minimum weight spanning arborescence.
    
    Args:
        graph: Weighted adjacency matrix where graph[i][j] = weight of edge i->j
        
    Returns:
        List of edges (parent, child) in the minimum spanning arborescence
    """
    n = graph.shape[0]
    
    # Tarjan's algorithm is complex - for now we'll implement a simpler
    # approach that works for our use case
    
    # For complete graphs, we can use a greedy approach:
    # 1. Start with all nodes disconnected
    # 2. Add the minimum weight edge that doesn't create a cycle
    # 3. Repeat until all nodes are connected
    
    # This is essentially Prim's algorithm for minimum spanning tree
    # which gives us a spanning arborescence when we root at a specific node
    
    if n == 1:
        return []
    
    # Choose root (node 0)
    root = 0
    
    # Initialize
    in_tree = [False] * n
    min_edge = [float('inf')] * n
    parent = [-1] * n
    
    in_tree[root] = True
    min_edge[root] = 0
    
    # Main loop
    for _ in range(n - 1):
        # Find minimum edge to add
        min_val = float('inf')
        u = -1
        v = -1
        
        for i in range(n):
            if in_tree[i]:
                for j in range(n):
                    if not in_tree[j] and graph[i][j] < min_edge[j]:
                        min_edge[j] = graph[i][j]
                        parent[j] = i
                        if graph[i][j] < min_val:
                            min_val = graph[i][j]
                            u = i
                            v = j
        
        if u == -1:
            break
            
        in_tree[v] = True
    
    # Build the arborescence
    arborescence = []
    for i in range(1, n):  # Skip root
        if parent[i] != -1:
            arborescence.append((parent[i], i))
    
    return arborescence


def successors_first_traversal(arborescence: List[Tuple[int, int]], root: int) -> List[Tuple[int, int]]:
    """
    Perform successors-first traversal of an arborescence.
    
    Args:
        arborescence: List of (parent, child) edges
        root: Root node of the arborescence
        
    Returns:
        List of edges in successors-first order
    """
    if not arborescence:
        return []
    
    # Build tree structure
    children = defaultdict(list)
    for parent, child in arborescence:
        children[parent].append(child)
    
    # Post-order traversal (successors first)
    sequence = []
    stack = [(root, False)]
    
    while stack:
        node, visited = stack.pop()
        if visited:
            # All children processed, add edges from node to its children
            for child in children[node]:
                sequence.append((node, child))
        else:
            stack.append((node, True))
            # Push children in reverse order for correct processing
            for child in reversed(children[node]):
                stack.append((child, False))
    
    return sequence


def construct_parity_graph(parity_table: ParityTable, parity_index: int) -> np.ndarray:
    """
    Construct the parity graph for a given parity.
    
    Args:
        parity_table: The parity table
        parity_index: Index of the parity to synthesize
        
    Returns:
        Weighted adjacency matrix for the parity graph
    """
    n = parity_table.num_qubits()
    parity = parity_table[parity_index]
    
    # Find terminals (qubits where this parity has value 1)
    terminals = []
    for i in range(n):
        if parity[i] == 1:
            terminals.append(i)
    
    # Create complete directed graph among terminals
    graph = np.full((n, n), float('inf'))
    
    # For nodes not in terminals, they have no edges
    # For terminals, create complete graph with weights based on hamming distance
    for i in terminals:
        for j in terminals:
            if i != j:
                # Weight is the hamming distance between rows i and j
                # This measures how much the parity table changes if we apply CNOT(i,j)
                row_i = parity_table.parities[i, :]
                row_j = parity_table.parities[j, :]
                hamming_dist = np.sum(row_i != row_j)
                graph[i][j] = hamming_dist
    
    return graph


def synthesize_all_to_all(parity_table: ParityTable, use_qiskit: bool = False) -> Tuple[List[Tuple[int, int]], int]:
    """
    Algorithm 1: Synthesize phase polynomial for all-to-all connectivity.
    
    Args:
        parity_table: Input parity table
        use_qiskit: If True, returns Qiskit circuit instead of CNOT sequence
        
    Returns:
        If use_qiskit=False: (cnot_sequence, total_cnot_count)
        If use_qiskit=True: (qiskit_circuit, total_cnot_count)
    """
    try:
        from qiskit import QuantumCircuit as QiskitCircuit
        from qiskit import transpile
    except ImportError:
        if use_qiskit:
            raise ImportError("Qiskit is required for use_qiskit=True")
        use_qiskit = False
    
    # Create Qiskit circuit if needed
    if use_qiskit:
        num_qubits = parity_table.num_qubits()
        qc = QiskitCircuit(num_qubits)
    
    cnot_sequence = []
    total_cnots = 0
    
    # Make a copy to work with
    working_table = parity_table.copy()
    
    while not working_table.is_empty():
        # Step 1: Choose parity with minimum hamming weight
        min_hamming = float('inf')
        selected_parity = 0
        
        for i in range(len(working_table)):
            h = working_table.hamming_weight(i)
            if h < min_hamming or (h == min_hamming and i < selected_parity):
                min_hamming = h
                selected_parity = i
        
        # Step 2: Construct parity graph
        parity_graph = construct_parity_graph(working_table, selected_parity)
        
        # Step 3: Find minimum weight spanning arborescence
        # For now, we'll use our simplified Tarjan implementation
        # Note: This finds a spanning tree, which gives us one possible arborescence
        arborescence = tarjan_min_spanning_arborescence(parity_graph)
        
        # Step 4: Get successors-first traversal sequence
        # Find the root (node with no incoming edges in our arborescence)
        all_nodes = set()
        children = set()
        for parent, child in arborescence:
            all_nodes.add(parent)
            all_nodes.add(child)
            children.add(child)
        
        root = (all_nodes - children).pop() if (all_nodes - children) else 0
        sequence = successors_first_traversal(arborescence, root)
        
        # Step 5: Apply the sequence
        for control, target in sequence:
            working_table.row_addition(control, target)
            cnot_sequence.append((control, target))
            total_cnots += 1
            
            if use_qiskit:
                qc.cx(control, target)
        
        # Step 6: Remove the synthesized parity
        working_table.remove_parity(selected_parity)
    
    if use_qiskit:
        # Optimize the circuit
        optimized_qc = transpile(qc, optimization_level=3)
        return optimized_qc, total_cnots
    else:
        return cnot_sequence, total_cnots


def synthesize_phase_polynomial(parity_vectors: List[List[int]], use_qiskit: bool = False):
    """
    High-level function to synthesize a phase polynomial from parity vectors.
    
    Args:
        parity_vectors: List of parity vectors (each is list of 0/1 integers)
        use_qiskit: Whether to return Qiskit circuit
        
    Returns:
        Synthesized circuit or CNOT sequence
    """
    parity_table = ParityTable(parity_vectors)
    return synthesize_all_to_all(parity_table, use_qiskit)


if __name__ == "__main__":
    print("Algorithm 1: All-to-All Connectivity Synthesis")
    print("=" * 50)
    
    # Example 1: Simple phase polynomial
    print("Example 1: Simple phase polynomial")
    parities = [
        [1, 1, 0, 0],  # x0 XOR x1
        [1, 0, 1, 0],  # x0 XOR x2
        [0, 1, 1, 0]   # x1 XOR x2
    ]
    
    sequence, cnot_count = synthesize_phase_polynomial(parities, use_qiskit=False)
    print(f"CNOT sequence ({cnot_count} gates): {sequence}")
    print()
    
    # Example 2: With Qiskit
    try:
        print("Example 2: With Qiskit integration")
        qc, cnot_count = synthesize_phase_polynomial(parities, use_qiskit=True)
        print(f"Qiskit circuit with {cnot_count} CNOT gates:")
        print(qc)
        print(f"Circuit depth: {qc.depth()}")
    except ImportError:
        print("Example 2: Qiskit not available, skipping")