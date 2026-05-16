#!/usr/bin/env python3
"""End-to-end phase-polynomial optimizer simulation harness for QLBM circuits."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_SOURCE_ROOT = PROJECT_ROOT / "qlbm"
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"
COMPONENT_DIR = Path(__file__).resolve().parent

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qlbm-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/qlbm-cache")

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
    elif optimizer_name in {"all_to_all", "a2a"}:
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
    result = backend.run(circuit, shots=num_shots).result()
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
    num_steps: int = 1,
    num_shots: int = NUM_SHOTS,
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
    output_case_label = (
        case["label"]
        if optimizer_name == "architecture_aware"
        else f"{case['label']}-{optimizer_name}"
    )
    output_file_path_base, output_file_path_optimized = resolve_output_paths(
        output_case_label,
        output_root,
        output_file_path_base,
        output_file_path_optimized,
    )

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

    summary = {
        "case": case["label"],
        "hardware": hardware_name,
        "optimizer_name": optimizer_name,
        "num_steps": num_steps,
        "num_shots": num_shots,
        "output_file_path_base": str(output_file_path_base),
        "output_file_path_optimized": str(output_file_path_optimized),
        "unoptimized_metrics": unoptimized_report["transpiled"],
        "optimized_metrics": optimized_report["transpiled"],
        "unoptimized_compact_metrics": unoptimized_report["transpiled_compact"],
        "optimized_compact_metrics": optimized_report["transpiled_compact"],
        "unoptimized_compatibility": unoptimized_report["transpiled_compatibility"],
        "optimized_compatibility": optimized_report["transpiled_compatibility"],
        "section_analysis": unoptimized_report["section_analysis"],
        "unoptimized_phase_polynomial_analysis": unoptimized_report[
            "phase_polynomial_analysis"
        ],
        "optimized_phase_polynomial_analysis": optimized_report[
            "phase_polynomial_analysis"
        ],
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
    # Edit these values for benchmark/simulation runs.
    hardware_name = "superconducting_google_willow_2024"
    optimizer_name = "architecture_aware"
    num_steps = 1
    num_shots = NUM_SHOTS
    output_root = DEFAULT_OUTPUT_ROOT
    config_path = None

    summary = run_phase_poly_harness(
        hardware_name=hardware_name,
        optimizer_name=optimizer_name,
        num_steps=num_steps,
        num_shots=num_shots,
        output_root=output_root,
        config_path=config_path,
    )
    print(json.dumps(json_safe(summary), indent=2))
    if not summary["counts_comparison"]["match"]:
        raise SystemExit("Optimized final counts differ from baseline beyond tolerance")


if __name__ == "__main__":
    main()
