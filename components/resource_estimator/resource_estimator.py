from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap
from collections import Counter
from typing import Dict, Any, Optional, List

class ResourceEstimator:
    """
    A modular resource estimator for quantum circuits.
    
    Takes a hardware config in JSON and a Qiskit quantum circuit,
    and returns an object with various metrics.
    """
    
    def __init__(self, hardware_config: Dict[str, Any], force_no_coupling: bool = False, remove_idle_qubits: bool = False):
        """
        Initialize the resource estimator with a hardware config.
        
        Args:
            hardware_config: A dictionary containing hardware configuration.
                Expected keys:
                - basis_gates: List of basis gates for the platform.
                - coupling_map: Coupling map for the platform (optional).
                - coupling_type: Type of coupling map (e.g., "linear_chain", "2d_grid").
                - coupling_params: Parameters for generating the coupling map.
                - gate_times: Dictionary of gate times (optional).
        """
        self.hardware_config = hardware_config
        self.basis_gates = hardware_config.get("basis_gates", [])
        self.coupling_map = hardware_config.get("coupling_map", None)
        self.coupling_type = hardware_config.get("coupling_type", None)
        self.coupling_params = hardware_config.get("coupling_params", {})
        self.gate_times = hardware_config.get("gate_times", None)
        self.remove_idle_qubits = remove_idle_qubits

        if force_no_coupling:
            self.coupling_map = None
            self.coupling_type = None
        else:
            # Generate coupling map if coupling_type is provided
            if self.coupling_type is not None:
                self.coupling_map = self._generate_coupling_map()
    
    def transpile_for_platform(self, circuit: QuantumCircuit, optimization_level: int = 0,
        seed_transpiler: Optional[int] = None) -> QuantumCircuit:
        """
        Transpile the circuit for the given hardware platform.
        
        Args:
            circuit: The quantum circuit to transpile.
            optimization_level: Optimization level for transpilation.
            seed_transpiler: Seed for transpiler (optional).
            
        Returns:
            Transpiled quantum circuit.
        """
        coupling = CouplingMap(self.coupling_map) if self.coupling_map is not None else None
        
        tc = transpile(
            circuit,
            basis_gates=self.basis_gates,
            coupling_map=coupling,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )
        return tc
    
    def extract_metrics(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Extract metrics from a transpiled circuit.
        
        Args:
            circuit: The transpiled quantum circuit.
            
        Returns:
            Dictionary of metrics.
        """
        ops = circuit.count_ops()
        depth = circuit.depth()
        size = circuit.size()
        num_qubits = circuit.num_qubits
        active_qubits = self._count_active_qubits(circuit)
        
        by_arity = self._count_gates_by_arity(ops)
        
        naive_total_time = self._calculate_naive_total_time(ops)
        
        metrics = {
            "num_qubits": num_qubits,
            "active_qubits": active_qubits,
            "depth": depth,
            "size": size,
            "ops_per_gate": dict(ops),
            "ops_1q": by_arity["1q"],
            "ops_2q": by_arity["2q"],
            "ops_3q": by_arity["3q"],
            "ops_other": by_arity["other"],
            "naive_total_time_s": naive_total_time,
        }
        return metrics
    
    def _count_gates_by_arity(self, ops: Dict[str, int]) -> Dict[str, int]:
        """
        Count gates by their arity (1q, 2q, 3q, other).
        
        Args:
            ops: Dictionary of gate counts.
            
        Returns:
            Dictionary of gate counts by arity.
        """
        ONE_Q_GATES = {
            "x", "y", "z", "h", "s", "sdg", "t", "tdg",
            "rx", "ry", "rz", "sx", "sxdg", "u", "u1", "u2", "u3",
        }
        TWO_Q_GATES = {
            "cx", "cz", "cy", "swap", "rxx", "ryy", "rzz", "rzx", "iswap", "ecr",
        }
        THREE_Q_GATES = {"ccx", "cswap", "c3x"}
        
        by_arity = Counter()
        for name, count in ops.items():
            lname = name.lower()
            if lname in ONE_Q_GATES:
                by_arity["1q"] += count
            elif lname in TWO_Q_GATES:
                by_arity["2q"] += count
            elif lname in THREE_Q_GATES:
                by_arity["3q"] += count
            else:
                by_arity["other"] += count
        
        return by_arity
    
    def _generate_coupling_map(self) -> Optional[List[List[int]]]:
        """
        Generate a coupling map based on the coupling_type and coupling_params.
        
        Returns:
            Generated coupling map, or None if coupling_type is not supported.
        """
        if self.coupling_type == "linear_chain":
            num_qubits = self.coupling_params.get("num_qubits", 0)
            return self._make_linear_chain_edges(num_qubits)
        elif self.coupling_type == "2d_grid":
            rows = self.coupling_params.get("rows", 0)
            cols = self.coupling_params.get("cols", 0)
            return self._make_2d_grid_edges(rows, cols)
        else:
            return None
    
    def _make_linear_chain_edges(self, num_qubits: int) -> List[List[int]]:
        """
        Generate edges for a linear chain coupling map.
        
        Args:
            num_qubits: Number of qubits in the chain.
            
        Returns:
            List of edges for the linear chain.
        """
        edges = []
        for i in range(num_qubits - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        return edges
    
    def _make_2d_grid_edges(self, rows: int, cols: int) -> List[List[int]]:
        """
        Generate edges for a 2D grid coupling map.
        
        Args:
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.
            
        Returns:
            List of edges for the 2D grid.
        """
        edges = []
        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                # Right neighbor
                if c + 1 < cols:
                    q_right = r * cols + (c + 1)
                    edges.append([q, q_right])
                    edges.append([q_right, q])
                # Down neighbor
                if r + 1 < rows:
                    q_down = (r + 1) * cols + c
                    edges.append([q, q_down])
                    edges.append([q_down, q])
        return edges
    
    def _calculate_naive_total_time(self, ops: Dict[str, int]) -> Optional[float]:
        """
        Calculate naive total time (sum of gate_time * count).
        
        Args:
            ops: Dictionary of gate counts.
            
        Returns:
            Total time in seconds, or None if gate_times is not provided.
        """
        if self.gate_times is None:
            return None
        
        total_time = 0.0
        for name, count in ops.items():
            if name in self.gate_times:
                total_time += count * self.gate_times[name]
        
        return total_time

    @staticmethod
    def _count_active_qubits(circuit: QuantumCircuit) -> int:
        dag = circuit_to_dag(circuit)
        return circuit.num_qubits - len(list(dag.idle_wires()))

    @staticmethod
    def _remove_idle_qubits(qc: QuantumCircuit) -> QuantumCircuit:
        dag = circuit_to_dag(qc)
        idle_qubits = list(dag.idle_wires())
        if idle_qubits:
            dag.remove_qubits(*idle_qubits)
        return dag_to_circuit(dag)
    
    def estimate(self, circuit: QuantumCircuit, optimization_level: int = 3,  seed_transpiler: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Estimate resources for the given circuit.
        
        Args:
            circuit: The quantum circuit to estimate.
            optimization_level: Optimization level for transpilation.
            seed_transpiler: Seed for transpiler (optional).
            
        Returns:
            Dictionary containing the metrics and the transpiled circuit.
        """
        platform_num_qubits = self.hardware_config.get("num_qubits", float("inf"))

        transpiled_circuit_raw = self.transpile_for_platform(
            circuit,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )

        if self.remove_idle_qubits:
            transpiled_circuit_without_idle = self._remove_idle_qubits(transpiled_circuit_raw)
            metrics = self.extract_metrics(transpiled_circuit_without_idle)
            transpiled_circuit = transpiled_circuit_without_idle
        else:
            metrics = self.extract_metrics(transpiled_circuit_raw)
            transpiled_circuit = transpiled_circuit_raw

        if transpiled_circuit.num_qubits > platform_num_qubits:
           print(f"WARNING: Circuit requires {transpiled_circuit.num_qubits} qubits, but the platform only supports "
                f"{platform_num_qubits} qubits. Transpilation may fail or produce incorrect results.\n",
            )

        result = {
            "metrics": metrics,
            "transpiled_circuit": transpiled_circuit,
        }

        return result