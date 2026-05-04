"""Resource estimation for QLBM circuits with NISQ hardware platforms in mind."""

from __future__ import annotations

from collections import Counter
from math import prod
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap


NON_UNITARY_OPS = {"measure", "reset", "barrier", "delay"}


class QLBMResourceEstimator:
    """Resource estimator for Qiskit QLBM circuits on NISQ hardware models."""

    def __init__(
        self,
        hardware_config: Dict[str, Any],
        remove_idle_qubits: bool = False,
        force_no_coupling: bool = False,
    ) -> None:
        self.hardware_config = hardware_config
        self.remove_idle_qubits = remove_idle_qubits

        self.basis_gates = [g.lower() for g in hardware_config.get("basis_gates", [])]
        self.coupling_type = None if force_no_coupling else hardware_config.get("coupling_type")
        self.coupling_params = hardware_config.get("coupling_params", {})
        self.coupling_map = None if force_no_coupling else hardware_config.get("coupling_map")
        if self.coupling_map is None and self.coupling_type is not None:
            self.coupling_map = self._generate_coupling_map()

        self.directed_coupling = bool(hardware_config.get("directed_coupling", False))
        self.gate_times_s = self._normalize_gate_dict(
            hardware_config.get("gate_times_s", {})
        )
        self.gate_fidelities = self._normalize_gate_dict(
            hardware_config.get("gate_fidelities", {})
        )
        self.measurement_time_s = hardware_config.get("measurement_time_s")
        self.measurement_fidelity = hardware_config.get("measurement_fidelity")
        self.coherence = hardware_config.get("coherence", {})

    def estimate(
        self,
        circuit: QuantumCircuit,
        label: Optional[str] = None,
        qlbm_metadata: Optional[Dict[str, Any]] = None,
        optimization_level: int = 1,
        seed_transpiler: Optional[int] = 42,
        transpile_circuit: bool = True,
    ) -> Dict[str, Any]:
        """Return logical, transpiled, compatibility, timing, and noise-proxy data."""
        logical_metrics = self.extract_metrics(circuit)
        report: Dict[str, Any] = {
            "label": label,
            "hardware": self.hardware_summary(),
            "qlbm": qlbm_metadata or {},
            "logical": logical_metrics,
            "logical_compatibility": self.check_compatibility(
                circuit,
                include_coupling_violations=False,
            ),
            "logical_time": self.estimate_time(circuit),
            "logical_fidelity": self.estimate_fidelity(circuit),
            "logical_coherence": self.estimate_coherence(circuit),
            "transpiled": None,
            "transpiled_circuit": None,
            "transpiled_compatibility": None,
            "transpiled_time": None,
            "transpiled_fidelity": None,
            "transpiled_coherence": None,
            "overheads": None,
        }

        if not transpile_circuit:
            return report

        try:
            transpiled_qc = self.transpile_circuit(
                circuit,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
            )
            transpiled_metrics = self.extract_metrics(transpiled_qc)
            report.update(
                {
                    "transpiled": transpiled_metrics,
                    "transpiled_circuit": transpiled_qc,
                    "transpiled_compatibility": self.check_compatibility(
                        transpiled_qc,
                        include_coupling_violations=True,
                    ),
                    "transpiled_time": self.estimate_time(transpiled_qc),
                    "transpiled_fidelity": self.estimate_fidelity(transpiled_qc),
                    "transpiled_coherence": self.estimate_coherence(transpiled_qc),
                    "overheads": self._calculate_overheads(
                        logical_metrics, transpiled_metrics
                    ),
                }
            )
        except Exception as exc:
            report["transpile_error"] = repr(exc)

        return report

    def transpile_circuit(
        self,
        circuit: QuantumCircuit,
        optimization_level: int = 1,
        seed_transpiler: Optional[int] = 42,
    ) -> QuantumCircuit:
        """Transpile a circuit for this estimator's hardware config."""
        coupling = CouplingMap(self.coupling_map) if self.coupling_map else None
        basis = self.basis_gates or None
        transpiled_qc = transpile(
            circuit,
            basis_gates=basis,
            coupling_map=coupling,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )
        if self.remove_idle_qubits:
            return self._remove_idle_qubits(transpiled_qc)
        return transpiled_qc

    def extract_metrics(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Extract circuit resource metrics without changing the circuit."""
        op_counts = {name.lower(): count for name, count in circuit.count_ops().items()}
        arity_counts = Counter()

        for inst in circuit.data:
            name = inst.operation.name.lower()
            arity = len(inst.qubits)
            if name == "measure":
                arity_counts["measurement"] += 1
            elif name == "reset":
                arity_counts["reset"] += 1
            elif name == "barrier":
                arity_counts["barrier"] += 1
            elif arity == 1:
                arity_counts["1q"] += 1
            elif arity == 2:
                arity_counts["2q"] += 1
            elif arity >= 3:
                arity_counts["3q_plus"] += 1
            else:
                arity_counts["other"] += 1

        non_measure_ops = (
            arity_counts["1q"] + arity_counts["2q"] + arity_counts["3q_plus"]
        )
        two_qubit_fraction = (
            arity_counts["2q"] / non_measure_ops if non_measure_ops else 0.0
        )

        return {
            "num_qubits": circuit.num_qubits,
            "active_qubits": self._count_active_qubits(circuit),
            "num_clbits": circuit.num_clbits,
            "depth": circuit.depth(),
            "size": circuit.size(),
            "width": circuit.width(),
            "op_counts": op_counts,
            "num_1q_ops": arity_counts["1q"],
            "num_2q_ops": arity_counts["2q"],
            "num_3q_plus_ops": arity_counts["3q_plus"],
            "num_measure": arity_counts["measurement"],
            "num_reset": arity_counts["reset"],
            "num_barrier": arity_counts["barrier"],
            "num_other_ops": arity_counts["other"],
            "two_qubit_fraction": two_qubit_fraction,
        }

    def obeys_basis_gates(self, circuit: QuantumCircuit) -> bool:
        """Return True if all unitary operations are in the configured basis gates."""
        return not self.unsupported_gates(circuit)

    def unsupported_gates(self, circuit: QuantumCircuit) -> List[str]:
        """Return sorted unitary operation names not present in basis_gates."""
        if not self.basis_gates:
            return []
        unsupported = set()
        for name in circuit.count_ops():
            lname = name.lower()
            if lname not in self.basis_gates and lname not in NON_UNITARY_OPS:
                unsupported.add(lname)
        return sorted(unsupported)

    def obeys_coupling_map(self, circuit: QuantumCircuit) -> bool:
        """Return True if all two-qubit operations obey the configured coupling map."""
        return not self.coupling_violations(circuit)

    def coupling_violations(self, circuit: QuantumCircuit) -> List[Dict[str, Any]]:
        """List two-qubit operations whose physical qubit pair is not coupled."""
        if not self.coupling_map:
            return []

        allowed_edges = self._edge_set(self.coupling_map)
        violations = []
        for index, inst in enumerate(circuit.data):
            if len(inst.qubits) != 2:
                continue
            q0 = circuit.find_bit(inst.qubits[0]).index
            q1 = circuit.find_bit(inst.qubits[1]).index
            if self._edge_allowed(q0, q1, allowed_edges):
                continue
            violations.append(
                {
                    "index": index,
                    "gate": inst.operation.name.lower(),
                    "qubits": [q0, q1],
                }
            )
        return violations

    def check_compatibility(
        self,
        circuit: QuantumCircuit,
        include_coupling_violations: bool = False,
    ) -> Dict[str, Any]:
        """Check qubit count, basis gates, and coupling-map compatibility."""
        hardware_qubits = self.hardware_config.get("num_qubits")
        qubit_fit = hardware_qubits is None or circuit.num_qubits <= hardware_qubits
        unsupported = self.unsupported_gates(circuit)
        coupling_violations = self.coupling_violations(circuit)

        compatibility = {
            "compatible": qubit_fit and not unsupported and not coupling_violations,
            "qubit_fit": qubit_fit,
            "required_qubits": circuit.num_qubits,
            "available_qubits": hardware_qubits,
            "basis_gates_ok": not unsupported,
            "unsupported_gates": unsupported,
            "coupling_map_checked": self.coupling_map is not None,
            "coupling_map_ok": not coupling_violations,
        }
        if include_coupling_violations:
            compatibility["num_coupling_violations"] = len(coupling_violations)
            compatibility["coupling_violations"] = coupling_violations
        return compatibility

    def estimate_time(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Estimate serial and layer-critical-path circuit duration."""
        warnings = []
        op_counts = {name.lower(): count for name, count in circuit.count_ops().items()}
        unknown_gates = sorted(
            name
            for name in op_counts
            if name not in NON_UNITARY_OPS and name not in self.gate_times_s
        )
        if unknown_gates:
            warnings.append(f"Missing gate times for: {', '.join(unknown_gates)}")

        serial_time_s = 0.0
        for name, count in op_counts.items():
            gate_name = name.lower()
            if gate_name == "measure":
                serial_time_s += count * (self.measurement_time_s or 0.0)
            else:
                serial_time_s += count * self.gate_times_s.get(gate_name, 0.0)

        if op_counts.get("measure", 0) and self.measurement_time_s is None:
            warnings.append("Missing measurement_time_s for measured circuit")

        critical_path_time_s, critical_unknown = self._critical_path_time(circuit)
        for name in critical_unknown:
            if name not in unknown_gates and name not in NON_UNITARY_OPS:
                warnings.append(f"Missing gate time on critical path for: {name}")

        return {
            "serial_time_s": serial_time_s,
            "critical_path_time_s": critical_path_time_s,
            "unknown_gate_times": unknown_gates,
            "measurement_time_included": bool(op_counts.get("measure", 0))
            and self.measurement_time_s is not None,
            "warnings": warnings,
        }

    def estimate_fidelity(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Estimate first-order gate and readout success probability."""
        warnings = []
        op_counts = {name.lower(): count for name, count in circuit.count_ops().items()}
        missing = sorted(
            name
            for name in op_counts
            if name not in NON_UNITARY_OPS and name not in self.gate_fidelities
        )
        if missing:
            warnings.append(f"Missing gate fidelities for: {', '.join(missing)}")

        gate_factors = [
            self.gate_fidelities[name] ** count
            for name, count in op_counts.items()
            if name in self.gate_fidelities
        ]
        gate_success_probability = prod(gate_factors) if gate_factors else None

        num_measure = op_counts.get("measure", 0)
        if num_measure == 0:
            readout_success_probability = None
            readout_applicable = False
        elif self.measurement_fidelity is None:
            readout_success_probability = None
            readout_applicable = True
            warnings.append("Missing measurement_fidelity for measured circuit")
        else:
            readout_success_probability = self.measurement_fidelity**num_measure
            readout_applicable = True

        if gate_success_probability is None:
            total_success_probability = readout_success_probability
        elif readout_success_probability is None:
            total_success_probability = gate_success_probability
        else:
            total_success_probability = (
                gate_success_probability * readout_success_probability
            )

        return {
            "model": "independent_product_estimate",
            "gate_success_probability": gate_success_probability,
            "readout_success_probability": readout_success_probability,
            "readout_applicable": readout_applicable,
            "total_success_probability": total_success_probability,
            "missing_gate_fidelities": missing,
            "warnings": warnings,
        }

    def estimate_coherence(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Compare estimated circuit duration to configured T1/T2 coherence times."""
        time_report = self.estimate_time(circuit)
        duration_s = time_report["critical_path_time_s"]
        t1_s = self.coherence.get("t1_s")
        t2_s = self.coherence.get("t2_s")
        warnings = []

        t1_ratio = duration_s / t1_s if t1_s else None
        t2_ratio = duration_s / t2_s if t2_s else None
        if t1_s is None:
            warnings.append("Missing coherence.t1_s")
        if t2_s is None:
            warnings.append("Missing coherence.t2_s")

        return {
            "duration_model": "critical_path_time_s",
            "duration_s": duration_s,
            "t1_s": t1_s,
            "t2_s": t2_s,
            "duration_over_t1": t1_ratio,
            "duration_over_t2": t2_ratio,
            "passes_t1": None if t1_ratio is None else t1_ratio < 1.0,
            "passes_t2": None if t2_ratio is None else t2_ratio < 1.0,
            "warnings": warnings + time_report["warnings"],
        }

    def compose_for_measurement(
        self, main_circuit: QuantumCircuit, measurement_circuit: QuantumCircuit
    ) -> QuantumCircuit:
        """Compose a QLBM evolution circuit with a compatible measurement circuit."""
        circuit = QuantumCircuit(*(measurement_circuit.qregs + measurement_circuit.cregs))
        circuit.compose(
            main_circuit,
            qubits=list(range(main_circuit.num_qubits)),
            inplace=True,
        )
        circuit.compose(measurement_circuit.copy(), inplace=True)
        return circuit

    def hardware_summary(self) -> Dict[str, Any]:
        """Return a compact hardware summary used in estimate reports."""
        return {
            "id": self.hardware_config.get("id"),
            "architecture": self.hardware_config.get("architecture"),
            "device_name": self.hardware_config.get("device_name"),
            "year_reported": self.hardware_config.get("year_reported"),
            "num_qubits": self.hardware_config.get("num_qubits"),
            "basis_gates": self.basis_gates,
            "coupling_type": self.coupling_type,
            "num_couplings": len(self.coupling_map or []),
            "directed_coupling": self.directed_coupling,
        }

    def _critical_path_time(self, circuit: QuantumCircuit) -> Tuple[float, List[str]]:
        dag = circuit_to_dag(circuit)
        total_time_s = 0.0
        unknown = set()
        for layer in dag.layers():
            layer_time_s = 0.0
            for node in layer["graph"].op_nodes():
                name = node.op.name.lower()
                if name == "measure":
                    duration = self.measurement_time_s
                elif name in NON_UNITARY_OPS:
                    duration = 0.0
                else:
                    duration = self.gate_times_s.get(name)
                if duration is None:
                    unknown.add(name)
                    duration = 0.0
                layer_time_s = max(layer_time_s, duration)
            total_time_s += layer_time_s
        return total_time_s, sorted(unknown)

    def _calculate_overheads(self, logical: Dict[str, Any], transpiled: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "depth_ratio": self._safe_ratio(transpiled["depth"], logical["depth"]),
            "size_ratio": self._safe_ratio(transpiled["size"], logical["size"]),
            "two_qubit_gate_ratio": self._safe_ratio(
                transpiled["num_2q_ops"], logical["num_2q_ops"]
            ),
            "added_depth": transpiled["depth"] - logical["depth"],
            "added_size": transpiled["size"] - logical["size"],
            "added_two_qubit_gates": transpiled["num_2q_ops"]
            - logical["num_2q_ops"],
        }

    def _generate_coupling_map(self) -> Optional[List[List[int]]]:
        if self.coupling_type == "linear_chain":
            return self._make_linear_chain_edges(
                int(self.coupling_params.get("num_qubits", 0))
            )
        if self.coupling_type == "2d_grid":
            return self._make_2d_grid_edges(
                int(self.coupling_params.get("rows", 0)),
                int(self.coupling_params.get("cols", 0)),
            )
        return None

    @staticmethod
    def _make_linear_chain_edges(num_qubits: int) -> List[List[int]]:
        edges = []
        for i in range(num_qubits - 1):
            edges.extend([[i, i + 1], [i + 1, i]])
        return edges

    @staticmethod
    def _make_2d_grid_edges(rows: int, cols: int) -> List[List[int]]:
        edges = []
        for row in range(rows):
            for col in range(cols):
                qubit = row * cols + col
                if col + 1 < cols:
                    right = row * cols + col + 1
                    edges.extend([[qubit, right], [right, qubit]])
                if row + 1 < rows:
                    down = (row + 1) * cols + col
                    edges.extend([[qubit, down], [down, qubit]])
        return edges

    @staticmethod
    def _normalize_gate_dict(values: Dict[str, Any]) -> Dict[str, Any]:
        return {str(name).lower(): value for name, value in values.items()}

    @staticmethod
    def _edge_set(coupling_map: Iterable[Iterable[int]]) -> set[Tuple[int, int]]:
        return {(int(edge[0]), int(edge[1])) for edge in coupling_map}

    def _edge_allowed(
        self, q0: int, q1: int, allowed_edges: set[Tuple[int, int]]
    ) -> bool:
        if (q0, q1) in allowed_edges:
            return True
        return not self.directed_coupling and (q1, q0) in allowed_edges

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
        if denominator == 0:
            return None
        return numerator / denominator

    @staticmethod
    def _count_active_qubits(circuit: QuantumCircuit) -> int:
        dag = circuit_to_dag(circuit)
        return circuit.num_qubits - len(list(dag.idle_wires()))

    @staticmethod
    def _remove_idle_qubits(circuit: QuantumCircuit) -> QuantumCircuit:
        dag = circuit_to_dag(circuit)
        idle_qubits = list(dag.idle_wires())
        if idle_qubits:
            dag.remove_qubits(*idle_qubits)
        return dag_to_circuit(dag)
