#!/usr/bin/env python3
"""End-to-end phase-polynomial optimizer simulation harness for QLBM circuits."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_SOURCE_ROOT = PROJECT_ROOT / "qlbm"
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"
COMPONENT_DIR = Path(__file__).resolve().parent

for path in (QLBM_SOURCE_ROOT, QLBM_HARDWARE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.phase_poly_optimizer import (
    A2APhasePoly,
    ArchitectureAwarePhasePolyOptimizer,
)
from components.resource_estimator import QLBMResourceEstimator
from components.resource_estimator.benchmark_qlbm_resources_v2 import (
    build_full_logical_circuit,
    build_sectioned_logical_circuit,
)
from qlbm.components import ABGridMeasurement, ABInitialConditions, ABQLBM, EmptyPrimitive
from qlbm.lattice import ABLattice


NUM_SHOTS = 2**12
OPTIMIZATION_LEVEL = 0
SEED_TRANSPILER = 42
SEED_SIMULATOR = 42
COUNTS_MAX_PROBABILITY_DELTA_TOLERANCE = 0.01
DEFAULT_MAX_COUNT_SIMULATION_QUBITS = 24
DEFAULT_PHASE_POLY_INTERMEDIATE_BASIS = ["rz", "sx", "x", "cx"]
DEFAULT_OUTPUT_ROOT = COMPONENT_DIR / "output"


def load_resource_estimator_config(config_path: Path) -> Dict[str, Any]:
    """Load resource-estimator hardware configurations from JSON."""
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_case_label(
    algorithm_name: str,
    lattice_data: Dict[str, Any],
    num_steps: int,
) -> str:
    """Build a stable output/report label from algorithm, lattice, and timestep."""
    lattice_config = lattice_data["lattice"]
    dimensions = lattice_config["dim"]
    dimension_label = "x".join(str(dimensions[axis]) for axis in sorted(dimensions))
    velocity_label = str(lattice_config["velocities"]).lower()
    geometry_count = len(lattice_data.get("geometry", []))
    return (
        f"{algorithm_name.lower()}-{velocity_label}-{dimension_label}"
        f"-g{geometry_count}-t{num_steps}"
    )


def build_ab_8x4_case(num_steps: int) -> Dict[str, Any]:
    """Build the AB QLBM lattice and full measured logical circuits."""
    algorithm_name = "ABQLBM"
    lattice_data = {
        "lattice": {"dim": {"x": 8, "y": 4}, "velocities": "d2q9"},
        "geometry": [
            {
                "shape": "cuboid",
                "x": [5, 7],
                "y": [2, 3],
                "boundary": "bounceback",
            },
        ],
    }
    lattice = ABLattice(lattice_data)
    initial_conditions = ABInitialConditions(lattice)
    algorithm = ABQLBM(lattice, use_agnostic_bcs=False)
    postprocessing = EmptyPrimitive(lattice)
    measurement = ABGridMeasurement(lattice)
    sectioned_circuit, section_names = build_sectioned_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        num_steps,
    )

    return {
        "label": build_case_label(algorithm_name, lattice_data, num_steps),
        "lattice": lattice,
        "lattice_data": lattice_data,
        "metadata": {
            "algorithm": algorithm_name,
            "lattice": lattice_data,
            "num_timesteps": num_steps,
        },
        "logical_circuit": build_full_logical_circuit(
            initial_conditions,
            algorithm,
            postprocessing,
            measurement,
            num_steps,
        ),
        "sectioned_circuit": sectioned_circuit,
        "section_names": section_names,
    }


def run_logical_resource_estimation(
    estimator: QLBMResourceEstimator,
    case: Dict[str, Any],
) -> Dict[str, Any]:
    """Transpile the full logical circuit and return resource data."""
    return estimator.estimate(
        case["logical_circuit"],
        label=case["label"],
        qlbm_metadata=case["metadata"],
        sectioned_circuit=case["sectioned_circuit"],
        section_names=case["section_names"],
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
        transpile_circuit=True,
        phase_polynomial_analysis=True,
    )


def representative_two_qubit_gate_time_s(hardware_config: Dict[str, Any]) -> Optional[float]:
    """Return a representative native two-qubit duration for intermediate CX reports."""
    gate_times = {
        str(name).lower(): value
        for name, value in hardware_config.get("gate_times_s", {}).items()
    }
    for gate_name in ("cx", "ecr", "cz", "iswap", "rxx", "ryy", "rzz", "rzx"):
        if gate_name in gate_times:
            return gate_times[gate_name]
    return None


def build_phase_poly_intermediate_hardware_config(
    hardware_config: Dict[str, Any],
    intermediate_basis_gates: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Build a hardware-shaped config for the CX/RZ phase-polynomial IR.

    The connectivity and qubit capacity are copied from the target hardware, while
    the basis is changed to a CX-capable intermediate basis. This lets the optimizer
    work on an architecture-aware CX/RZ representation before the result is compiled
    back to the true native basis.
    """
    basis = [
        gate.lower()
        for gate in (intermediate_basis_gates or DEFAULT_PHASE_POLY_INTERMEDIATE_BASIS)
    ]
    config = dict(hardware_config)
    config["id"] = f"{hardware_config.get('id', 'hardware')}-phasepoly-cx-ir"
    config["basis_gates"] = basis

    gate_times = {
        str(name).lower(): value
        for name, value in hardware_config.get("gate_times_s", {}).items()
    }
    if "cx" in basis and "cx" not in gate_times:
        representative_time = representative_two_qubit_gate_time_s(hardware_config)
        if representative_time is not None:
            gate_times["cx"] = representative_time
    config["gate_times_s"] = gate_times
    return config


def optimize_physical_circuit(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
) -> tuple[QuantumCircuit, Any]:
    """Run the default architecture-aware phase-polynomial optimizer."""
    return optimize_circuit_with_phase_poly(
        circuit,
        coupling_map,
        optimizer_name="architecture_aware",
    )


def optimize_circuit_with_phase_poly(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    optimizer_name: str = "architecture_aware",
) -> tuple[QuantumCircuit, Any]:
    """Run a selected phase-polynomial optimizer."""
    if optimizer_name == "architecture_aware":
        optimizer = ArchitectureAwarePhasePolyOptimizer()
    elif optimizer_name in {"all_to_all"}:
        optimizer = A2APhasePoly()
    else:
        raise ValueError(
            "optimizer_name must be one of: architecture_aware, all_to_all"
        )

    optimized_circuit = optimizer.optimize(
        circuit=circuit.copy(),
        coupling_map=coupling_map,
    )
    return optimized_circuit, optimizer


def run_intermediate_native_phase_poly_pipeline(
    circuit: QuantumCircuit,
    hardware_config: Dict[str, Any],
    optimizer_name: str = "architecture_aware",
    intermediate_basis_gates: Optional[list[str]] = None,
    optimization_level: int = OPTIMIZATION_LEVEL,
    seed_transpiler: int = SEED_TRANSPILER,
    qlbm_metadata: Optional[Dict[str, Any]] = None,
    phase_polynomial_analysis: bool = True,
) -> Dict[str, Any]:
    """
    Optimize in a CX/RZ phase-polynomial IR, then compile to native hardware basis.

    This is the recommended pipeline for hardware configurations whose native
    entangling gate is not CX. It evaluates whether reducing the intermediate CX
    structure also reduces final native two-qubit resources.
    """
    native_estimator = QLBMResourceEstimator(hardware_config)
    intermediate_config = build_phase_poly_intermediate_hardware_config(
        hardware_config,
        intermediate_basis_gates=intermediate_basis_gates,
    )
    intermediate_estimator = QLBMResourceEstimator(intermediate_config)

    baseline_native_report = native_estimator.estimate(
        circuit,
        label="baseline-native",
        qlbm_metadata=qlbm_metadata,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        transpile_circuit=True,
        phase_polynomial_analysis=phase_polynomial_analysis,
    )
    if baseline_native_report.get("transpile_error"):
        raise RuntimeError(baseline_native_report["transpile_error"])

    intermediate_report = intermediate_estimator.estimate(
        circuit,
        label="baseline-phasepoly-cx-ir",
        qlbm_metadata=qlbm_metadata,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        transpile_circuit=True,
        phase_polynomial_analysis=phase_polynomial_analysis,
    )
    if intermediate_report.get("transpile_error"):
        raise RuntimeError(intermediate_report["transpile_error"])

    intermediate_circuit = intermediate_report["transpiled_circuit"]
    optimized_intermediate_circuit, optimizer = optimize_circuit_with_phase_poly(
        intermediate_circuit,
        CouplingMap(intermediate_estimator.coupling_map),
        optimizer_name=optimizer_name,
    )
    optimized_intermediate_report = intermediate_estimator.estimate_pretranspiled(
        optimized_intermediate_circuit,
        label="optimized-phasepoly-cx-ir",
        qlbm_metadata=qlbm_metadata,
        logical_metrics=baseline_native_report["logical"],
        phase_polynomial_analysis=phase_polynomial_analysis,
    )

    optimized_native_circuit = native_estimator.transpile_circuit(
        optimized_intermediate_circuit,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    optimized_native_report = native_estimator.estimate_pretranspiled(
        optimized_native_circuit,
        label="optimized-native-after-phasepoly-cx-ir",
        qlbm_metadata=qlbm_metadata,
        logical_metrics=baseline_native_report["logical"],
        phase_polynomial_analysis=phase_polynomial_analysis,
    )

    return {
        "pipeline": "intermediate_native",
        "optimizer": optimizer,
        "intermediate_basis_gates": intermediate_config["basis_gates"],
        "native_basis_gates": native_estimator.basis_gates,
        "baseline_native_report": baseline_native_report,
        "intermediate_report": intermediate_report,
        "optimized_intermediate_report": optimized_intermediate_report,
        "optimized_native_report": optimized_native_report,
    }


def run_all_to_all_intermediate_native_phase_poly_pipeline(
    circuit: QuantumCircuit,
    hardware_config: Dict[str, Any],
    intermediate_basis_gates: Optional[list[str]] = None,
    optimization_level: int = OPTIMIZATION_LEVEL,
    seed_transpiler: int = SEED_TRANSPILER,
    qlbm_metadata: Optional[Dict[str, Any]] = None,
    phase_polynomial_analysis: bool = True,
) -> Dict[str, Any]:
    """
    Optimize an all-to-all CX/RZ phase-polynomial IR, then compile to native hardware.

    This is the baseline for comparing architecture-aware phase-polynomial
    optimization against all-to-all phase-polynomial optimization followed by
    hardware routing and native-basis translation.
    """
    native_estimator = QLBMResourceEstimator(hardware_config)
    intermediate_config = build_phase_poly_intermediate_hardware_config(
        hardware_config,
        intermediate_basis_gates=intermediate_basis_gates,
    )
    intermediate_estimator = QLBMResourceEstimator(
        intermediate_config,
        force_no_coupling=True,
    )

    baseline_native_report = native_estimator.estimate(
        circuit,
        label="baseline-native",
        qlbm_metadata=qlbm_metadata,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        transpile_circuit=True,
        phase_polynomial_analysis=phase_polynomial_analysis,
    )
    if baseline_native_report.get("transpile_error"):
        raise RuntimeError(baseline_native_report["transpile_error"])

    intermediate_report = intermediate_estimator.estimate(
        circuit,
        label="baseline-all-to-all-phasepoly-cx-ir",
        qlbm_metadata=qlbm_metadata,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        transpile_circuit=True,
        phase_polynomial_analysis=phase_polynomial_analysis,
    )
    if intermediate_report.get("transpile_error"):
        raise RuntimeError(intermediate_report["transpile_error"])

    intermediate_circuit = intermediate_report["transpiled_circuit"]
    optimized_intermediate_circuit, optimizer = optimize_circuit_with_phase_poly(
        intermediate_circuit,
        CouplingMap.from_full(intermediate_circuit.num_qubits),
        optimizer_name="all_to_all",
    )
    optimized_intermediate_report = intermediate_estimator.estimate_pretranspiled(
        optimized_intermediate_circuit,
        label="optimized-all-to-all-phasepoly-cx-ir",
        qlbm_metadata=qlbm_metadata,
        logical_metrics=baseline_native_report["logical"],
        phase_polynomial_analysis=phase_polynomial_analysis,
    )

    optimized_native_circuit = native_estimator.transpile_circuit(
        optimized_intermediate_circuit,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    optimized_native_report = native_estimator.estimate_pretranspiled(
        optimized_native_circuit,
        label="optimized-native-after-all-to-all-phasepoly-cx-ir",
        qlbm_metadata=qlbm_metadata,
        logical_metrics=baseline_native_report["logical"],
        phase_polynomial_analysis=phase_polynomial_analysis,
    )

    return {
        "pipeline": "all_to_all_intermediate_native",
        "optimizer": optimizer,
        "intermediate_basis_gates": intermediate_config["basis_gates"],
        "native_basis_gates": native_estimator.basis_gates,
        "baseline_native_report": baseline_native_report,
        "intermediate_report": intermediate_report,
        "optimized_intermediate_report": optimized_intermediate_report,
        "optimized_native_report": optimized_native_report,
    }


def require_compatible(report: Dict[str, Any], label: str) -> None:
    """Raise if a reported physical circuit no longer matches hardware constraints."""
    compatibility = report["transpiled_compatibility"]
    if compatibility["compatible"]:
        return

    failures = {
        "basis_gates_ok": compatibility["basis_gates_ok"],
        "unsupported_gates": compatibility["unsupported_gates"],
        "coupling_map_ok": compatibility["coupling_map_ok"],
        "num_coupling_violations": compatibility.get("num_coupling_violations"),
        "qubit_capacity_ok": compatibility["qubit_capacity_ok"],
        "used_qubits": compatibility["used_qubits"],
        "available_qubits": compatibility["available_qubits"],
    }
    raise RuntimeError(f"{label} circuit is not hardware compatible: {failures}")


def run_final_counts_simulation(
    circuit: QuantumCircuit,
    lattice: Any,
    output_dir: Path,
    final_timestep: int,
    num_shots: int,
    seed_simulator: int = SEED_SIMULATOR,
) -> Dict[str, int]:
    """Run one measured circuit and save only the final QLBM result timestep."""
    backend = AerSimulator(method="statevector", seed_simulator=seed_simulator)
    simulation_circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=0,
        seed_transpiler=seed_simulator,
    )
    result = backend.run(simulation_circuit, shots=num_shots).result()
    counts = result.get_counts()

    qlbm_result = lattice.create_result(str(output_dir), "step")
    qlbm_result.visualize_geometry()
    qlbm_result.save_timestep_counts(counts, final_timestep)
    return dict(counts)


def normalize_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def compare_normalized_counts(
    reference_counts: Dict[str, int],
    candidate_counts: Dict[str, int],
    max_probability_delta_tolerance: float,
) -> Dict[str, Any]:
    """Compare two sampled distributions using normalized count probabilities."""
    reference = normalize_counts(reference_counts)
    candidate = normalize_counts(candidate_counts)
    keys = sorted(set(reference) | set(candidate))
    deltas = {
        key: abs(reference.get(key, 0.0) - candidate.get(key, 0.0))
        for key in keys
    }
    max_delta = max(deltas.values(), default=0.0)
    total_variation_distance = 0.5 * sum(deltas.values())
    return {
        "match": max_delta <= max_probability_delta_tolerance,
        "max_probability_delta": max_delta,
        "total_variation_distance": total_variation_distance,
        "tolerance": max_probability_delta_tolerance,
        "num_reference_keys": len(reference),
        "num_candidate_keys": len(candidate),
        "deltas": deltas,
    }


def counts_simulation_skip_reason(
    reference_circuit: QuantumCircuit,
    candidate_circuit: QuantumCircuit,
    max_count_simulation_qubits: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Return a skip reason when final count simulation is too large."""
    if max_count_simulation_qubits is None:
        return None

    largest_num_qubits = max(reference_circuit.num_qubits, candidate_circuit.num_qubits)
    if largest_num_qubits <= max_count_simulation_qubits:
        return None

    return {
        "skipped": True,
        "reason": (
            "Final count simulation skipped because the compact routed circuit "
            "is too large for statevector simulation in this harness."
        ),
        "max_count_simulation_qubits": max_count_simulation_qubits,
        "unoptimized_compact_qubits": reference_circuit.num_qubits,
        "optimized_compact_qubits": candidate_circuit.num_qubits,
    }


def json_safe(value: Any) -> Any:
    """Convert report objects into JSON-serializable summaries."""
    if isinstance(value, QuantumCircuit):
        return {
            "num_qubits": value.num_qubits,
            "num_clbits": value.num_clbits,
            "depth": value.depth(),
            "size": value.size(),
            "op_counts": dict(value.count_ops()),
        }
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def summarize_optimizer_report(report: Any) -> Optional[Dict[str, Any]]:
    """Summarize optimizer diagnostics without serializing every block report."""
    if report is None:
        return None

    block_reports = list(getattr(report, "block_reports", []) or [])
    original_cx = sum(block.original_cx for block in block_reports)
    candidate_cx = sum(block.candidate_cx for block in block_reports)
    final_cx = sum(block.final_cx for block in block_reports)
    kept_original = sum(1 for block in block_reports if block.kept_original)
    disconnected = sum(
        1 for block in block_reports if not block.connected_active_subgraph
    )
    residual_methods: Dict[str, int] = {}
    residual_cx_by_method: Dict[str, int] = {}
    for block in block_reports:
        method = getattr(block, "residual_method", "") or "unknown"
        residual_methods[method] = residual_methods.get(method, 0) + 1
        residual_cx_by_method[method] = (
            residual_cx_by_method.get(method, 0) + block.residual_cx
        )

    return {
        "circuit_num_qubits": report.circuit_num_qubits,
        "num_blocks": report.num_blocks,
        "total_original_cx": original_cx,
        "total_candidate_cx": candidate_cx,
        "total_final_cx": final_cx,
        "cx_delta_vs_original": final_cx - original_cx,
        "num_blocks_kept_original": kept_original,
        "num_disconnected_blocks": disconnected,
        "residual_methods": residual_methods,
        "residual_cx_by_method": residual_cx_by_method,
    }


def compare_metric_summary(
    before: Optional[Dict[str, Any]],
    after: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return compact before/after resource deltas for thesis plots."""
    if before is None or after is None:
        return None

    def metric_delta(name: str) -> Optional[int]:
        before_value = before.get(name)
        after_value = after.get(name)
        if before_value is None or after_value is None:
            return None
        return int(after_value) - int(before_value)

    before_ops = before.get("op_counts", {}) or {}
    after_ops = after.get("op_counts", {}) or {}
    op_names = sorted(set(before_ops) | set(after_ops))
    return {
        "depth_delta": metric_delta("depth"),
        "size_delta": metric_delta("size"),
        "num_1q_ops_delta": metric_delta("num_1q_ops"),
        "num_2q_ops_delta": metric_delta("num_2q_ops"),
        "op_count_deltas": {
            name: int(after_ops.get(name, 0)) - int(before_ops.get(name, 0))
            for name in op_names
        },
    }


def write_json_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, indent=2)


def resolve_output_paths(
    case_label: str,
    output_root: Path,
    output_file_path_base: Optional[Path],
    output_file_path_optimized: Optional[Path],
) -> tuple[Path, Path]:
    """Resolve baseline and optimized output directories for a case."""
    base_path = output_file_path_base or output_root / case_label
    optimized_path = output_file_path_optimized or output_root / f"{case_label}-optimized"
    return base_path, optimized_path


def run_phase_poly_harness(
    hardware_name: str,
    optimizer_name: str = "architecture_aware",
    pipeline: str = "intermediate_native",
    intermediate_basis_gates: Optional[list[str]] = None,
    num_steps: int = 1,
    num_shots: int = NUM_SHOTS,
    run_counts_simulation: bool = True,
    max_count_simulation_qubits: Optional[int] = DEFAULT_MAX_COUNT_SIMULATION_QUBITS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output_file_path_base: Optional[Path] = None,
    output_file_path_optimized: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    config_path = config_path or (
        QLBM_HARDWARE_ROOT / "components" / "resource_estimator" / "config.json"
    )
    hardware_configs = load_resource_estimator_config(config_path)
    hardware_config = hardware_configs[hardware_name]
    estimator = QLBMResourceEstimator(hardware_config)
    case = build_ab_8x4_case(num_steps)
    effective_optimizer_name = (
        "all_to_all"
        if pipeline == "all_to_all_intermediate_native"
        else optimizer_name
    )
    output_case_label = (
        case["label"]
        if effective_optimizer_name == "architecture_aware"
        else f"{case['label']}-{effective_optimizer_name}"
    )
    output_file_path_base, output_file_path_optimized = resolve_output_paths(
        output_case_label,
        output_root,
        output_file_path_base,
        output_file_path_optimized,
    )

    if pipeline == "post_transpile":
        unoptimized_report = run_logical_resource_estimation(estimator, case)
        if unoptimized_report.get("transpile_error"):
            raise RuntimeError(unoptimized_report["transpile_error"])
        require_compatible(unoptimized_report, "Unoptimized")

        physical_circuit = unoptimized_report["transpiled_circuit"]
        optimized_circuit, optimizer = optimize_circuit_with_phase_poly(
            physical_circuit,
            CouplingMap(estimator.coupling_map),
            optimizer_name=optimizer_name,
        )
        optimized_report = estimator.estimate_pretranspiled(
            optimized_circuit,
            label=f"{output_case_label}-phasepoly-optimized",
            qlbm_metadata=case["metadata"],
            logical_metrics=unoptimized_report["logical"],
            phase_polynomial_analysis=True,
        )
        if optimizer_name == "architecture_aware":
            require_compatible(optimized_report, "Optimized")

        intermediate_report = None
        optimized_intermediate_report = None
        intermediate_basis = None
        native_basis = estimator.basis_gates
    elif pipeline == "intermediate_native":
        pipeline_reports = run_intermediate_native_phase_poly_pipeline(
            case["logical_circuit"],
            hardware_config,
            optimizer_name=optimizer_name,
            intermediate_basis_gates=intermediate_basis_gates,
            optimization_level=OPTIMIZATION_LEVEL,
            seed_transpiler=SEED_TRANSPILER,
            qlbm_metadata=case["metadata"],
            phase_polynomial_analysis=True,
        )
        unoptimized_report = pipeline_reports["baseline_native_report"]
        optimized_report = pipeline_reports["optimized_native_report"]
        intermediate_report = pipeline_reports["intermediate_report"]
        optimized_intermediate_report = pipeline_reports[
            "optimized_intermediate_report"
        ]
        optimizer = pipeline_reports["optimizer"]
        intermediate_basis = pipeline_reports["intermediate_basis_gates"]
        native_basis = pipeline_reports["native_basis_gates"]

        require_compatible(unoptimized_report, "Unoptimized native")
        require_compatible(optimized_report, "Optimized native")
    elif pipeline == "all_to_all_intermediate_native":
        pipeline_reports = run_all_to_all_intermediate_native_phase_poly_pipeline(
            case["logical_circuit"],
            hardware_config,
            intermediate_basis_gates=intermediate_basis_gates,
            optimization_level=OPTIMIZATION_LEVEL,
            seed_transpiler=SEED_TRANSPILER,
            qlbm_metadata=case["metadata"],
            phase_polynomial_analysis=True,
        )
        unoptimized_report = pipeline_reports["baseline_native_report"]
        optimized_report = pipeline_reports["optimized_native_report"]
        intermediate_report = pipeline_reports["intermediate_report"]
        optimized_intermediate_report = pipeline_reports[
            "optimized_intermediate_report"
        ]
        optimizer = pipeline_reports["optimizer"]
        intermediate_basis = pipeline_reports["intermediate_basis_gates"]
        native_basis = pipeline_reports["native_basis_gates"]

        require_compatible(unoptimized_report, "Unoptimized native")
        require_compatible(optimized_report, "Optimized native")
    else:
        raise ValueError(
            "pipeline must be one of: post_transpile, intermediate_native, "
            "all_to_all_intermediate_native"
        )

    if run_counts_simulation:
        count_skip = counts_simulation_skip_reason(
            unoptimized_report["transpiled_compact_circuit"],
            optimized_report["transpiled_compact_circuit"],
            max_count_simulation_qubits,
        )
    else:
        count_skip = {
            "skipped": True,
            "reason": "Final count simulation disabled by run_counts_simulation=False.",
        }
    if count_skip is None:
        unoptimized_counts = run_final_counts_simulation(
            unoptimized_report["transpiled_compact_circuit"],
            case["lattice"],
            output_file_path_base,
            num_steps,
            num_shots,
        )
        optimized_counts = run_final_counts_simulation(
            optimized_report["transpiled_compact_circuit"],
            case["lattice"],
            output_file_path_optimized,
            num_steps,
            num_shots,
        )
        counts_comparison = compare_normalized_counts(
            unoptimized_counts,
            optimized_counts,
            COUNTS_MAX_PROBABILITY_DELTA_TOLERANCE,
        )
        counts_comparison["skipped"] = False
    else:
        counts_comparison = {
            "match": None,
            "max_probability_delta": None,
            "total_variation_distance": None,
            "tolerance": COUNTS_MAX_PROBABILITY_DELTA_TOLERANCE,
            **count_skip,
        }

    summary = {
        "case": case["label"],
        "hardware": hardware_name,
        "optimizer_name": effective_optimizer_name,
        "pipeline": pipeline,
        "native_basis_gates": native_basis,
        "intermediate_basis_gates": intermediate_basis,
        "num_steps": num_steps,
        "num_shots": num_shots,
        "run_counts_simulation": run_counts_simulation,
        "max_count_simulation_qubits": max_count_simulation_qubits,
        "output_file_path_base": str(output_file_path_base),
        "output_file_path_optimized": str(output_file_path_optimized),
        "unoptimized_metrics": unoptimized_report["transpiled"],
        "optimized_metrics": optimized_report["transpiled"],
        "unoptimized_compact_metrics": unoptimized_report["transpiled_compact"],
        "optimized_compact_metrics": optimized_report["transpiled_compact"],
        "native_metric_comparison": compare_metric_summary(
            unoptimized_report["transpiled"],
            optimized_report["transpiled"],
        ),
        "native_compact_metric_comparison": compare_metric_summary(
            unoptimized_report["transpiled_compact"],
            optimized_report["transpiled_compact"],
        ),
        "intermediate_metrics": (
            intermediate_report["transpiled"] if intermediate_report else None
        ),
        "optimized_intermediate_metrics": (
            optimized_intermediate_report["transpiled"]
            if optimized_intermediate_report
            else None
        ),
        "intermediate_compact_metrics": (
            intermediate_report["transpiled_compact"] if intermediate_report else None
        ),
        "optimized_intermediate_compact_metrics": (
            optimized_intermediate_report["transpiled_compact"]
            if optimized_intermediate_report
            else None
        ),
        "intermediate_metric_comparison": compare_metric_summary(
            intermediate_report["transpiled"] if intermediate_report else None,
            optimized_intermediate_report["transpiled"]
            if optimized_intermediate_report
            else None,
        ),
        "intermediate_compact_metric_comparison": compare_metric_summary(
            intermediate_report["transpiled_compact"] if intermediate_report else None,
            optimized_intermediate_report["transpiled_compact"]
            if optimized_intermediate_report
            else None,
        ),
        "unoptimized_compatibility": unoptimized_report["transpiled_compatibility"],
        "optimized_compatibility": optimized_report["transpiled_compatibility"],
        "section_analysis": unoptimized_report["section_analysis"],
        "unoptimized_phase_polynomial_analysis": unoptimized_report[
            "phase_polynomial_analysis"
        ],
        "optimized_phase_polynomial_analysis": optimized_report[
            "phase_polynomial_analysis"
        ],
        "intermediate_phase_polynomial_analysis": (
            intermediate_report["phase_polynomial_analysis"]
            if intermediate_report
            else None
        ),
        "optimized_intermediate_phase_polynomial_analysis": (
            optimized_intermediate_report["phase_polynomial_analysis"]
            if optimized_intermediate_report
            else None
        ),
        "optimizer_report": summarize_optimizer_report(
            getattr(optimizer, "last_run_report", None)
        ),
        "counts_comparison": counts_comparison,
    }
    write_json_report(output_file_path_base / "resource_report.json", unoptimized_report)
    write_json_report(output_file_path_optimized / "resource_report.json", optimized_report)
    write_json_report(output_file_path_optimized / "comparison_report.json", summary)
    return summary


def main() -> None:
    hardware_name = "superconducting_google_willow_2024"
    optimizer_name = "architecture_aware" # CHOOSE BETWEEN architecture_aware AND all_to_all
    pipeline = "all_to_all_intermediate_native" # CHOOSE BETWEEN post_transpile, intermediate_native, all_to_all_intermediate_native
    num_steps = 1
    num_shots = NUM_SHOTS
    run_counts_simulation = True
    output_root = DEFAULT_OUTPUT_ROOT
    config_path = None

    summary = run_phase_poly_harness(
        hardware_name=hardware_name,
        optimizer_name=optimizer_name,
        pipeline=pipeline,
        num_steps=num_steps,
        num_shots=num_shots,
        run_counts_simulation=run_counts_simulation,
        output_root=output_root,
        config_path=config_path,
    )
    print(json.dumps(json_safe(summary), indent=2))
    if summary["counts_comparison"].get("skipped"):
        print(summary["counts_comparison"]["reason"])
    elif not summary["counts_comparison"]["match"]:
        raise SystemExit("Optimized final counts differ from baseline beyond tolerance")


if __name__ == "__main__":
    main()
