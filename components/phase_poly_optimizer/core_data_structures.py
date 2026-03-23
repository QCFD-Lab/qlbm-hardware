"""
Core data structures for phase polynomial optimization framework.
Based on Vandaele et al. (2022) - Phase polynomials synthesis algorithms for NISQ architectures
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import heapq

class ParityTable:
    """
    Represents a parity table for phase polynomials.
    Each column represents a parity (Boolean vector).
    """
    
    def __init__(self, parities: Optional[List[List[int]]] = None):
        """
        Initialize parity table.
        
        Args:
            parities: List of parity vectors (each vector is list of 0/1 integers)
                     Each inner list represents a parity (column in the table)
        """
        self.parities = []
        if parities is not None:
            # Convert to numpy array for efficient operations
            # Each parity is a column, so we transpose the input
            self.parities = np.array(parities, dtype=int).T
            if len(self.parities.shape) == 1:
                self.parities = self.parities.reshape(-1, 1)
    
    def __len__(self) -> int:
        """Return number of parities."""
        return self.parities.shape[1] if len(self.parities.shape) > 1 else 0
    
    def __getitem__(self, index: int) -> np.ndarray:
        """Get parity at given index."""
        return self.parities[:, index]
    
    def num_qubits(self) -> int:
        """Return number of qubits."""
        return self.parities.shape[0]
    
    def hamming_weight(self, parity_index: int) -> int:
        """Calculate hamming weight of a parity."""
        return int(np.sum(self.parities[:, parity_index]))
    
    def row_addition(self, control: int, target: int):
        """
        Perform row addition: P[control] = P[control] XOR P[target]
        This corresponds to applying a CNOT gate CNOT(control, target).
        
        According to the paper: CNOT_{i,j} performs P_i = P_i ⊕ P_j
        where P_i and P_j are rows representing qubit states.
        
        Args:
            control: Control qubit index (row to be updated)
            target: Target qubit index (row to XOR with control)
        """
        # XOR the target row into the control row (row-wise operation)
        self.parities[control, :] ^= self.parities[target, :]
    
    def remove_parity(self, index: int):
        """Remove parity at given index."""
        self.parities = np.delete(self.parities, index, axis=1)
    
    def is_empty(self) -> bool:
        """Check if parity table is empty."""
        return len(self) == 0
    
    def copy(self) -> 'ParityTable':
        """Return a copy of this parity table."""
        # Create a new ParityTable with the same parities
        # Need to convert back to list format for the constructor
        parities_list = self.parities.T.tolist()
        return ParityTable(parities_list)
    
    def to_list(self) -> List[List[int]]:
        """Convert to list of lists."""
        return self.parities.T.tolist()
    
    def __str__(self) -> str:
        return str(self.parities.T)


class ArchitectureGraph:
    """
    Represents quantum hardware architecture connectivity.
    """
    
    def __init__(self, num_qubits: int, edges: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize architecture graph.
        
        Args:
            num_qubits: Number of qubits
            edges: List of (source, target) edges representing connectivity
        """
        self.num_qubits = num_qubits
        self.adjacency = defaultdict(set)
        
        if edges is None:
            # Default: all-to-all connectivity
            for i in range(num_qubits):
                for j in range(num_qubits):
                    if i != j:
                        self.adjacency[i].add(j)
        else:
            for src, tgt in edges:
                self.adjacency[src].add(tgt)
                self.adjacency[tgt].add(src)  # Undirected graph
    
    def is_connected(self, u: int, v: int) -> bool:
        """Check if two qubits are directly connected."""
        return v in self.adjacency[u]
    
    def get_neighbors(self, qubit: int) -> Set[int]:
        """Get neighbors of a qubit."""
        return self.adjacency[qubit]
    
    def shortest_path(self, start: int, end: int) -> List[int]:
        """
        Find shortest path between two qubits using BFS.
        Returns list of qubits in path, or empty list if no path exists.
        """
        if start == end:
            return [start]
        
        if not self.is_connected(start, end):
            # Find path via BFS
            queue = deque([(start, [start])])
            visited = set([start])
            
            while queue:
                current, path = queue.popleft()
                for neighbor in self.get_neighbors(current):
                    if neighbor == end:
                        return path + [neighbor]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        return [start, end]  # Direct connection
    
    def all_pairs_shortest_paths(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Compute all pairs shortest paths.
        Returns dictionary mapping (src, tgt) to path.
        """
        paths = {}
        for src in range(self.num_qubits):
            for tgt in range(self.num_qubits):
                if src != tgt:
                    paths[(src, tgt)] = self.shortest_path(src, tgt)
        return paths
    
    def is_all_to_all(self) -> bool:
        """Check if this is an all-to-all connected architecture."""
        for i in range(self.num_qubits):
            if len(self.adjacency[i]) != self.num_qubits - 1:
                return False
        return True

class SteinerTree:
    """
    Represents a Steiner tree for partial connectivity synthesis.
    """
    
    def __init__(self, terminals: Set[int], steiner_nodes: Set[int], edges: List[Tuple[int, int]]):
        """
        Initialize Steiner tree.
        
        Args:
            terminals: Set of terminal vertices
            steiner_nodes: Set of Steiner nodes
            edges: List of edges in the tree
        """
        self.terminals = set(terminals)
        self.steiner_nodes = set(steiner_nodes)
        self.edges = list(edges)
        self.vertices = self.terminals.union(self.steiner_nodes)
    
    def size(self) -> int:
        """Return number of vertices in tree."""
        return len(self.vertices)
    
    def cost(self) -> int:
        """
        Calculate cost of Steiner tree: C(y) = 2|V_T| - |S| - 1
        where |V_T| is number of vertices, |S| is number of terminals.
        """
        return 2 * self.size() - len(self.terminals) - 1
    
    def get_spanning_arborescences(self) -> List['SpanningArborescence']:
        """
        Get all possible spanning arborescences of this tree.
        Each arborescence is rooted at a different vertex.
        """
        arborescences = []
        for root in self.vertices:
            # Create arborescence rooted at 'root'
            edges = self._create_arborescence_edges(root)
            arborescences.append(SpanningArborescence(root, edges, self))
        return arborescences
    
    def _create_arborescence_edges(self, root: int) -> List[Tuple[int, int]]:
        """
        Create edges for arborescence rooted at given vertex.
        Uses BFS to establish parent-child relationships.
        """
        if len(self.vertices) == 1:
            return []
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in self.edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from root to establish arborescence
        edges = []
        visited = {root}
        queue = deque([root])
        parent = {}
        
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        # Create directed edges (parent -> child)
        for child, p in parent.items():
            edges.append((p, child))
        
        return edges


class SpanningArborescence:
    """
    Represents a spanning arborescence (directed tree) of a Steiner tree.
    """
    
    def __init__(self, root: int, edges: List[Tuple[int, int]], steiner_tree: SteinerTree):
        self.root = root
        self.edges = edges
        self.steiner_tree = steiner_tree
    
    def get_traversal_sequence(self) -> List[Tuple[int, int]]:
        """
        Get successors-first traversal sequence.
        Returns list of (control, target) pairs for CNOT operations.
        """
        # Build tree structure
        children = defaultdict(list)
        for parent, child in self.edges:
            children[parent].append(child)
        
        # Post-order traversal (successors first)
        sequence = []
        stack = [(self.root, False)]
        
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
    
    def apply_to_parity_table(self, parity_table: ParityTable, parity_index: int) -> ParityTable:
        """
        Apply this arborescence to synthesize the given parity.
        Returns modified parity table.
        """
        # Get the parity to synthesize
        original_parity = parity_table[parity_index]
        
        # Create a copy to work with
        modified_table = parity_table.copy()
        
        # Apply the traversal sequence
        # Note: The sequence contains (control, target) pairs for CNOT gates
        # which correspond to row additions P[control] = P[control] XOR P[target]
        sequence = self.get_traversal_sequence()
        for control, target in sequence:
            modified_table.row_addition(control, target)
        
        return modified_table
