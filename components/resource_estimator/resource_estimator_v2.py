"""Resource estimation for QLBM circuits with NISQ hardware platforms in mind."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, InstructionDurations, PassManager
from qiskit.transpiler.passes import ASAPScheduleAnalysis


NON_UNITARY_OPS = {"measure", "reset", "barrier", "delay"}
SECTION_BOUNDARY_PREFIX = "section_boundary::"


class QLBMResourceEstimator:
    """Resource estimator for Qiskit QLBM circuits on NISQ hardware models."""

    def __init__(self, hardware_config: Dict[str, Any], force_no_coupling: bool = False,
    ) -> None:
        self.hardware_config = hardware_config
        self.basis_gates = [g.lower() for g in hardware_config.get("basis_gates", [])]
        self.coupling_type = None if force_no_coupling else hardware_config.get("coupling_type")
        self.coupling_params = hardware_config.get("coupling_params", {})
        self.coupling_map = None if force_no_coupling else hardware_config.get("coupling_map")
        if self.coupling_map is None and self.coupling_type is not None:
            self.coupling_map = self._generate_coupling_map()

        self.directed_coupling = bool(hardware_config.get("directed_coupling", False))
        self.gate_times_s = self._normalize_gate_dict(hardware_config.get("gate_times_s", {}))
        self.measurement_time_s = hardware_config.get("measurement_time_s")

    def estimate(
        self,
        circuit: QuantumCircuit,
        label: Optional[str] = None,
        qlbm_metadata: Optional[Dict[str, Any]] = None,
        sectioned_circuit: Optional[QuantumCircuit] = None,
        section_names: Optional[List[str]] = None,
        optimization_level: int = 1,
        seed_transpiler: Optional[int] = 42,
        transpile_circuit: bool = True,
    ) -> Dict[str, Any]:
        """Return logical, transpiled, compatibility, and timing resource data."""
        logical_metrics = self.extract_metrics(circuit,
            include_active_qubits=False,
            include_active_qubit_indices=False,
        )
        report: Dict[str, Any] = {
            "label": label,
            "hardware": self.hardware_summary(),
            "qlbm": qlbm_metadata or {},
            "logical": logical_metrics,
            "logical_capacity": self.check_logical_capacity(circuit),
            "transpiled": None,
            "transpiled_circuit": None,
            "transpiled_compatibility": None,
            "transpiled_time": None,
            "transpiled_compact": None,
            "transpiled_compact_circuit": None,
            "section_analysis": None,
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
            transpiled_compact_qc = self._remove_idle_qubits(transpiled_qc)
            report.update(
                {
                    "transpiled": transpiled_metrics,
                    "transpiled_circuit": transpiled_qc,
                    "transpiled_compatibility": self.check_compatibility(
                        transpiled_qc,
                        include_coupling_violations=True,
                    ),
                    "transpiled_time": self.estimate_time(transpiled_qc),
                    "transpiled_compact": self.extract_metrics(
                        transpiled_compact_qc,
                        include_active_qubit_indices=False,
                    ),
                    "transpiled_compact_circuit": transpiled_compact_qc,
                    "overheads": self._calculate_overheads(
                        logical_metrics, transpiled_metrics
                    ),
                }
            )
            if sectioned_circuit is not None and section_names:
                try:
                    report["section_analysis"] = self.analyze_sections(
                        sectioned_circuit,
                        section_names,
                        optimization_level=optimization_level,
                        seed_transpiler=seed_transpiler,
                    )
                except Exception as section_exc:
                    report["section_analysis_error"] = repr(section_exc)
        except Exception as exc:
            report["transpile_error"] = repr(exc)

        return report

    def estimate_pretranspiled(
        self,
        circuit: QuantumCircuit,
        label: Optional[str] = None,
        qlbm_metadata: Optional[Dict[str, Any]] = None,
        logical_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return resource data for a circuit that has already been transpiled.

        This method intentionally does not invoke Qiskit's transpiler. The supplied
        circuit is treated as the physical circuit to validate, calculate time, and compact by removing idle qubits.
        """
        transpiled_metrics = self.extract_metrics(circuit)
        transpiled_compact_qc = self.compact_circuit(circuit)
        report: Dict[str, Any] = {
            "label": label,
            "hardware": self.hardware_summary(),
            "qlbm": qlbm_metadata or {},
            "logical": logical_metrics,
            "logical_capacity": None,
            "transpiled": transpiled_metrics,
            "transpiled_circuit": circuit,
            "transpiled_compatibility": self.check_compatibility(
                circuit,
                include_coupling_violations=True,
            ),
            "transpiled_time": self.estimate_time(circuit),
            "transpiled_compact": self.extract_metrics(
                transpiled_compact_qc,
                include_active_qubit_indices=False,
            ),
            "transpiled_compact_circuit": transpiled_compact_qc,
            "section_analysis": None,
            "overheads": self._calculate_overheads(logical_metrics, transpiled_metrics)
            if logical_metrics is not None
            else None,
        }
        return report

    def compact_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Return a copy of the circuit with idle qubits removed."""
        return self._remove_idle_qubits(circuit)

    def transpile_circuit(
        self,
        circuit: QuantumCircuit,
        optimization_level: int = 1,
        seed_transpiler: Optional[int] = 42,
    ) -> QuantumCircuit:
        """Transpile a circuit for this estimator's hardware config."""
        coupling = CouplingMap(self.coupling_map) if self.coupling_map else None
        basis = self.basis_gates or None
        return transpile(
            circuit,
            basis_gates=basis,
            coupling_map=coupling,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )

    def analyze_sections(
        self,
        sectioned_circuit: QuantumCircuit,
        section_names: List[str],
        optimization_level: int = 1,
        seed_transpiler: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Attribute transpiled resource metrics to labeled top-level sections."""
        transpiled = self.transpile_circuit(
            sectioned_circuit,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )
        section_circuits = self._split_by_section_boundaries(transpiled, section_names)
        sections = [self._section_metrics(name, circuit) for name, circuit in section_circuits]

        return {
            "sections": sections,
            "max_critical_path_time_section": self._max_section(
                sections,
                "critical_path_time_s",
            ),
            "max_two_qubit_gate_section": self._max_section(
                sections,
                "num_2q_ops",
            ),
        }

    def extract_metrics(
        self,
        circuit: QuantumCircuit,
        include_active_qubits: bool = True,
        include_active_qubit_indices: bool = True,
    ) -> Dict[str, Any]:
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

        metrics = {
            "num_qubits": circuit.num_qubits,
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
        if include_active_qubits:
            metrics["active_qubits"] = self._count_active_qubits(circuit)
        if include_active_qubit_indices:
            metrics["active_qubit_indices"] = self._active_qubit_indices(circuit)
        return metrics

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
        """Check used-qubit capacity, basis gates, and coupling-map compatibility."""
        hardware_qubits = self.hardware_config.get("num_qubits")
        used_qubits = self._count_active_qubits(circuit)
        qubit_capacity_ok = hardware_qubits is None or used_qubits <= hardware_qubits
        unsupported = self.unsupported_gates(circuit)
        coupling_violations = self.coupling_violations(circuit)
        compatibility = qubit_capacity_ok and not unsupported and not coupling_violations

        compatibility = {
            "compatible": compatibility,
            "qubit_capacity_ok": qubit_capacity_ok,
            "used_qubits": used_qubits,
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

    def check_logical_capacity(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Check whether the logical circuit qubit count fits the hardware size."""
        hardware_qubits = self.hardware_config.get("num_qubits")
        logical_qubits = circuit.num_qubits
        return {
            "logical_qubits": logical_qubits,
            "available_qubits": hardware_qubits,
            "qubit_capacity_ok": (
                hardware_qubits is None or logical_qubits <= hardware_qubits
            ),
        }

    def estimate_time(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Estimate scheduled wall-clock duration and serial gate-work time."""
        warnings = []
        op_counts = {name.lower(): count for name, count in circuit.count_ops().items()}
        unknown_gates = sorted(
            name
            for name in op_counts
            if name not in NON_UNITARY_OPS and name not in self.gate_times_s
        )
        if unknown_gates:
            warnings.append(f"Missing gate times for: {', '.join(unknown_gates)}")

        serial_time_s = self._serial_time(circuit)

        if op_counts.get("measure", 0) and self.measurement_time_s is None:
            warnings.append("Missing measurement_time_s for measured circuit")

        scheduled_duration_s = None
        max_idle_time_s = None
        if unknown_gates:
            warnings.append(
                "Scheduled duration unavailable because gate durations are missing"
            )
        elif op_counts.get("measure", 0) and self.measurement_time_s is None:
            warnings.append(
                "Scheduled duration unavailable because measurement_time_s is missing"
            )
        else:
            scheduled_duration_s,  max_idle_time_s, schedule_warnings = self._scheduled_timing(circuit)
            warnings.extend(schedule_warnings)

        critical_path_time_s, critical_unknown = self._critical_path_time(circuit)
        for name in critical_unknown:
            if name not in unknown_gates and name not in NON_UNITARY_OPS:
                warnings.append(f"Missing gate time on critical path for: {name}")

        return {
            "timing_model": "qiskit_asap_schedule",
            "scheduled_duration_s": scheduled_duration_s,
            "max_idle_time_s": max_idle_time_s,
            "serial_time_s": serial_time_s,
            "critical_path_time_s": critical_path_time_s,
            "unknown_gate_times": unknown_gates,
            "measurement_time_included": bool(op_counts.get("measure", 0))
            and self.measurement_time_s is not None,
            "warnings": warnings,
        }

    def _section_metrics(self, section_name: str, circuit: QuantumCircuit) -> Dict[str, Any]:
        metrics = self.extract_metrics(
            circuit,
            include_active_qubits=False,
            include_active_qubit_indices=False,
        )
        critical_path_time_s, _ = self._critical_path_time(circuit)
        return {
            "section": section_name,
            "num_1q_ops": metrics["num_1q_ops"],
            "num_2q_ops": metrics["num_2q_ops"],
            "depth": metrics["depth"],
            "size": metrics["size"],
            "critical_path_time_s": critical_path_time_s,
            "serial_time_s": self._serial_time(circuit),
            "op_counts": metrics["op_counts"],
        }

    def _split_by_section_boundaries(
        self,
        circuit: QuantumCircuit,
        section_names: List[str],
    ) -> List[Tuple[str, QuantumCircuit]]:
        section_circuits = [
            QuantumCircuit(*(circuit.qregs + circuit.cregs)) for _ in section_names
        ]
        section_index = 0
        for inst in circuit.data:
            if self._is_section_boundary(inst.operation):
                section_index += 1
                continue
            if section_index >= len(section_circuits):
                continue
            section_circuits[section_index].append(
                inst.operation.copy(),
                inst.qubits,
                inst.clbits,
            )
        return list(zip(section_names, section_circuits))

    @staticmethod
    def _is_section_boundary(operation: Any) -> bool:
        return (
            operation.name.lower() == "barrier"
            and isinstance(operation.label, str)
            and operation.label.startswith(SECTION_BOUNDARY_PREFIX)
        )

    @staticmethod
    def _max_section(sections: List[Dict[str, Any]], metric_name: str) -> Optional[Dict[str, Any]]:
        if not sections:
            return None
        return max(sections, key=lambda section: section[metric_name])

    def _serial_time(self, circuit: QuantumCircuit) -> float:
        serial_time_s = 0.0
        for name, count in circuit.count_ops().items():
            gate_name = name.lower()
            if gate_name == "measure":
                serial_time_s += count * (self.measurement_time_s or 0.0)
            else:
                serial_time_s += count * self.gate_times_s.get(gate_name, 0.0)
        return serial_time_s

    def _scheduled_timing(self, circuit: QuantumCircuit) -> Tuple[Optional[float], Optional[float], List[str]]:
        durations = self._instruction_durations()
        try:
            pass_manager = PassManager([ASAPScheduleAnalysis(durations)])
            pass_manager.run(circuit)
        except Exception as exc:
            return None, None, [f"Scheduled duration unavailable: {exc!r}"]

        node_start_times = pass_manager.property_set.get("node_start_time", {})
        scheduled_duration_s = 0.0
        intervals_by_qubit: Dict[int, List[Tuple[float, float]]] = {}
        missing = set()
        for node, start_time_s in node_start_times.items():
            if not hasattr(node, "op"):
                continue
            duration_s = self._node_duration_s(node)
            if duration_s is None:
                missing.add(node.op.name.lower())
                continue
            end_time_s = start_time_s + duration_s
            scheduled_duration_s = max(scheduled_duration_s, end_time_s)
            for qubit in node.qargs:
                qubit_index = circuit.find_bit(qubit).index
                intervals_by_qubit.setdefault(qubit_index, []).append(
                    (start_time_s, end_time_s)
                )

        if missing:
            return None, None, [
                "Scheduled duration unavailable because durations are missing for: "
                + ", ".join(sorted(missing))
            ]
        return scheduled_duration_s, self._max_idle_time(
            intervals_by_qubit,
            scheduled_duration_s,
        ), []

    @staticmethod
    def _max_idle_time(
        intervals_by_qubit: Dict[int, List[Tuple[float, float]]],
        scheduled_duration_s: float,
    ) -> float:
        max_idle_time_s = 0.0
        for intervals in intervals_by_qubit.values():
            previous_end_s = 0.0
            for start_time_s, end_time_s in sorted(intervals):
                max_idle_time_s = max(max_idle_time_s, start_time_s - previous_end_s)
                previous_end_s = max(previous_end_s, end_time_s)
            max_idle_time_s = max(
                max_idle_time_s,
                scheduled_duration_s - previous_end_s,
            )
        return max_idle_time_s

    def _instruction_durations(self) -> InstructionDurations:
        duration_entries = [
            (name, None, duration_s, "s")
            for name, duration_s in self.gate_times_s.items()
        ]
        if self.measurement_time_s is not None:
            duration_entries.append(("measure", None, self.measurement_time_s, "s"))
        duration_entries.append(("reset", None, 0.0, "s"))
        return InstructionDurations(duration_entries)

    def _node_duration_s(self, node: Any) -> Optional[float]:
        name = node.op.name.lower()
        if name == "measure":
            return self.measurement_time_s
        if name in {"barrier", "reset"}:
            return 0.0
        if name == "delay":
            return self._delay_duration_s(node.op)
        return self.gate_times_s.get(name)

    @staticmethod
    def _delay_duration_s(operation: Any) -> Optional[float]:
        duration = getattr(operation, "duration", None)
        unit = getattr(operation, "unit", None)
        if duration is None:
            return None
        if unit == "s":
            return float(duration)
        if unit == "ms":
            return float(duration) * 1e-3
        if unit == "us":
            return float(duration) * 1e-6
        if unit == "ns":
            return float(duration) * 1e-9
        return None

    def hardware_summary(self) -> Dict[str, Any]:
        """Return a compact hardware summary used in estimate reports."""
        validation = self.validate_hardware_config()
        return {
            "id": self.hardware_config.get("id"),
            "architecture": self.hardware_config.get("architecture"),
            "device_name": self.hardware_config.get("device_name"),
            "year_reported": self.hardware_config.get("year_reported"),
            "num_qubits": self.hardware_config.get("num_qubits"),
            "topology_num_qubits": validation["topology_num_qubits"],
            "basis_gates": self.basis_gates,
            "coupling_type": self.coupling_type,
            "num_couplings": len(self.coupling_map or []),
            "directed_coupling": self.directed_coupling,
            "validation_warnings": validation["warnings"],
        }

    def validate_hardware_config(self) -> Dict[str, Any]:
        """Validate consistency between declared qubits and topology qubits."""
        hardware_qubits = self.hardware_config.get("num_qubits")
        topology_num_qubits = self._topology_num_qubits()
        warnings = []
        if (
            hardware_qubits is not None
            and topology_num_qubits is not None
            and topology_num_qubits != hardware_qubits
        ):
            warnings.append(
                "Topology qubit count "
                f"({topology_num_qubits}) differs from hardware num_qubits "
                f"({hardware_qubits})"
            )
        return {
            "topology_num_qubits": topology_num_qubits,
            "topology_matches_num_qubits": (
                None
                if hardware_qubits is None or topology_num_qubits is None
                else topology_num_qubits == hardware_qubits
            ),
            "warnings": warnings,
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
                self.coupling_params.get("disabled_qubits", []),
            )
        return None

    @staticmethod
    def _make_linear_chain_edges(num_qubits: int) -> List[List[int]]:
        edges = []
        for i in range(num_qubits - 1):
            edges.extend([[i, i + 1], [i + 1, i]])
        return edges

    @staticmethod
    def _make_2d_grid_edges(rows: int, cols: int, disabled_qubits: Optional[List[int]] = None) -> List[List[int]]:
        edges = []
        disabled = set(disabled_qubits or [])
        for row in range(rows):
            for col in range(cols):
                qubit = row * cols + col
                if qubit in disabled:
                    continue
                if col + 1 < cols:
                    right = row * cols + col + 1
                    if right not in disabled:
                        edges.extend([[qubit, right], [right, qubit]])
                if row + 1 < rows:
                    down = (row + 1) * cols + col
                    if down not in disabled:
                        edges.extend([[qubit, down], [down, qubit]])
        return edges

    def _topology_num_qubits(self) -> Optional[int]:
        if not self.coupling_map:
            return None
        qubits = set()
        for q0, q1 in self.coupling_map:
            qubits.add(int(q0))
            qubits.add(int(q1))
        return len(qubits)

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
    def _active_qubit_indices(circuit: QuantumCircuit) -> List[int]:
        active = set()
        for inst in circuit.data:
            if inst.operation.name.lower() == "barrier":
                continue
            for qubit in inst.qubits:
                active.add(circuit.find_bit(qubit).index)
        return sorted(active)

    @classmethod
    def _count_active_qubits(cls, circuit: QuantumCircuit) -> int:
        return len(cls._active_qubit_indices(circuit))

    @staticmethod
    def _remove_idle_qubits(circuit: QuantumCircuit) -> QuantumCircuit:
        dag = circuit_to_dag(circuit)
        idle_qubits = [wire for wire in dag.idle_wires() if wire in circuit.qubits]
        if idle_qubits:
            dag.remove_qubits(*idle_qubits)
        return dag_to_circuit(dag)
