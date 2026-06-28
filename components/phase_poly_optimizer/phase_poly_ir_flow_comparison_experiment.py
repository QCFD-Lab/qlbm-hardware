"""Compare phase-polynomial IR optimization flows on selected hardware targets."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_SOURCE_ROOT = PROJECT_ROOT / "qlbm"
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"

for path in (QLBM_SOURCE_ROOT, QLBM_HARDWARE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.phase_poly_optimizer.optimizer_benchmark_simulation import (
    DEFAULT_PHASE_POLY_INTERMEDIATE_BASIS,
    OPTIMIZATION_LEVEL,
    SEED_TRANSPILER,
    load_resource_estimator_config,
    run_all_to_all_intermediate_native_phase_poly_pipeline,
    run_intermediate_native_phase_poly_pipeline,
)
from components.resource_estimator.experiment_resource_sweeps import (
    DEFAULT_ALGORITHMS,
    build_experiment_case,
    make_case_spec,
)


DEFAULT_CONFIG_PATH = (
    QLBM_HARDWARE_ROOT / "components" / "resource_estimator" / "config.json"
)
DEFAULT_OUTPUT_DIR = (
    QLBM_HARDWARE_ROOT
    / "components"
    / "phase_poly_optimizer"
    / "output"
    / "phase-poly-ir-flow-comparison"
)
DEFAULT_GRID_SIZE = 8
DEFAULT_NUM_OBSTACLES = 1
DEFAULT_NUM_TIMESTEPS = 1
DEFAULT_RUNS = [
    {
        "hardware_key": "neutral_atom_aws_2023",
        "optimizer": "all_to_all",
        "pipeline": "all_to_all_intermediate_native",
    },
    {
        "hardware_key": "superconducting_google_willow_2024",
        "optimizer": "architecture_aware",
        "pipeline": "intermediate_native",
    },
    {
        "hardware_key": "superconducting_ibm_nighthawk_r1_2025",
        "optimizer": "architecture_aware",
        "pipeline": "intermediate_native",
    },
]


FIELDNAMES = [
    "hardware_key",
    "device_name",
    "optimizer",
    "pipeline",
    "algorithm",
    "qlbm_grid",
    "num_obstacles",
    "num_timesteps",
    "native_basis_gates",
    "intermediate_basis_gates",
    "coupling_type",
    "hardware_qubits",
    "direct_hardware_depth",
    "ir_optimized_hardware_depth",
    "direct_hardware_two_qubit",
    "ir_optimized_hardware_two_qubit",
    "hardware_two_qubit_reduction_percent",
    "direct_hardware_rz",
    "ir_optimized_hardware_rz",
    "direct_hardware_total_gates",
    "ir_optimized_hardware_total_gates",
    "diagnostic_ir_cx_baseline",
    "diagnostic_ir_cx_optimized",
    "diagnostic_ir_cx_reduction_percent",
    "diagnostic_ir_depth_baseline",
    "diagnostic_ir_depth_optimized",
    "diagnostic_ir_total_gates_baseline",
    "diagnostic_ir_total_gates_optimized",
    "optimization_error",
]


def _percent_reduction(baseline: int, optimized: int) -> Optional[float]:
    if baseline == 0:
        return None
    return 100.0 * (baseline - optimized) / baseline


def _op_count(metrics: Dict[str, Any], gate: str) -> int:
    return int((metrics.get("op_counts") or {}).get(gate, 0))


def _hardware_label(hardware_key: str) -> str:
    label = hardware_key
    for prefix in ("superconducting_", "neutral_atom_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    return label.replace("_", "-")


def _run_label(run: Dict[str, str]) -> str:
    return f"{_hardware_label(run['hardware_key'])}-{run['optimizer'].replace('_', '-')}"


def _require_compatible(report: Dict[str, Any], label: str) -> None:
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


def _make_row(
    run: Dict[str, str],
    hardware_config: Dict[str, Any],
    algorithm: str,
    pipeline_report: Optional[Dict[str, Any]],
    optimization_error: Optional[str],
    grid_size: int,
    num_obstacles: int,
    num_timesteps: int,
) -> Dict[str, Any]:
    baseline_native = (
        (pipeline_report or {}).get("baseline_native_report", {}).get("transpiled")
        or {}
    )
    optimized_native = (
        (pipeline_report or {}).get("optimized_native_report", {}).get("transpiled")
        or {}
    )
    intermediate = (
        (pipeline_report or {}).get("intermediate_report", {}).get("transpiled")
        or {}
    )
    optimized_intermediate = (
        (pipeline_report or {})
        .get("optimized_intermediate_report", {})
        .get("transpiled")
        or {}
    )

    direct_hardware_2q = int(baseline_native.get("num_2q_ops", 0) or 0)
    ir_optimized_hardware_2q = int(optimized_native.get("num_2q_ops", 0) or 0)
    intermediate_cx_baseline = _op_count(intermediate, "cx")
    intermediate_cx_optimized = _op_count(optimized_intermediate, "cx")

    return {
        "hardware_key": run["hardware_key"],
        "device_name": hardware_config.get("device_name"),
        "optimizer": run["optimizer"],
        "pipeline": run["pipeline"],
        "algorithm": algorithm,
        "qlbm_grid": f"{grid_size}x{grid_size}",
        "num_obstacles": num_obstacles,
        "num_timesteps": num_timesteps,
        "native_basis_gates": " ".join(hardware_config.get("basis_gates", [])),
        "intermediate_basis_gates": " ".join(DEFAULT_PHASE_POLY_INTERMEDIATE_BASIS),
        "coupling_type": hardware_config.get("coupling_type"),
        "hardware_qubits": hardware_config.get("num_qubits"),
        "direct_hardware_depth": baseline_native.get("depth"),
        "ir_optimized_hardware_depth": optimized_native.get("depth"),
        "direct_hardware_two_qubit": direct_hardware_2q,
        "ir_optimized_hardware_two_qubit": ir_optimized_hardware_2q,
        "hardware_two_qubit_reduction_percent": _percent_reduction(
            direct_hardware_2q,
            ir_optimized_hardware_2q,
        ),
        "direct_hardware_rz": _op_count(baseline_native, "rz"),
        "ir_optimized_hardware_rz": _op_count(optimized_native, "rz"),
        "direct_hardware_total_gates": baseline_native.get("size"),
        "ir_optimized_hardware_total_gates": optimized_native.get("size"),
        "diagnostic_ir_cx_baseline": intermediate_cx_baseline,
        "diagnostic_ir_cx_optimized": intermediate_cx_optimized,
        "diagnostic_ir_cx_reduction_percent": _percent_reduction(
            intermediate_cx_baseline,
            intermediate_cx_optimized,
        ),
        "diagnostic_ir_depth_baseline": intermediate.get("depth"),
        "diagnostic_ir_depth_optimized": optimized_intermediate.get("depth"),
        "diagnostic_ir_total_gates_baseline": intermediate.get("size"),
        "diagnostic_ir_total_gates_optimized": optimized_intermediate.get("size"),
        "optimization_error": optimization_error,
    }


def run_ir_flow_comparison(
    runs: Sequence[Dict[str, str]] = DEFAULT_RUNS,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_obstacles: int = DEFAULT_NUM_OBSTACLES,
    num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    config_path: Path = DEFAULT_CONFIG_PATH,
    optimization_level: int = OPTIMIZATION_LEVEL,
    seed_transpiler: int = SEED_TRANSPILER,
) -> List[Dict[str, Any]]:
    """Run all requested hardware/optimizer IR-flow comparisons."""
    hardware_configs = load_resource_estimator_config(config_path)
    rows: List[Dict[str, Any]] = []

    for run in runs:
        hardware_config = hardware_configs[run["hardware_key"]]
        for algorithm in algorithms:
            spec = make_case_spec(
                "phase_poly_ir_flow_comparison",
                algorithm,
                grid_size,
                num_timesteps=num_timesteps,
                num_obstacles=num_obstacles,
                obstacle_boundary="bounceback",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                built_case = build_experiment_case(spec)

            pipeline_report = None
            optimization_error = None
            try:
                if run["pipeline"] == "all_to_all_intermediate_native":
                    pipeline_report = run_all_to_all_intermediate_native_phase_poly_pipeline(
                        built_case.circuit,
                        hardware_config,
                        intermediate_basis_gates=DEFAULT_PHASE_POLY_INTERMEDIATE_BASIS,
                        optimization_level=optimization_level,
                        seed_transpiler=seed_transpiler,
                        qlbm_metadata=spec.metadata(),
                        phase_polynomial_analysis=False,
                    )
                elif run["pipeline"] == "intermediate_native":
                    pipeline_report = run_intermediate_native_phase_poly_pipeline(
                        built_case.circuit,
                        hardware_config,
                        optimizer_name=run["optimizer"],
                        intermediate_basis_gates=DEFAULT_PHASE_POLY_INTERMEDIATE_BASIS,
                        optimization_level=optimization_level,
                        seed_transpiler=seed_transpiler,
                        qlbm_metadata=spec.metadata(),
                        phase_polynomial_analysis=False,
                    )
                else:
                    raise ValueError(f"Unknown pipeline: {run['pipeline']}")

                _require_compatible(
                    pipeline_report["baseline_native_report"],
                    f"{run['hardware_key']} {algorithm} baseline native",
                )
                _require_compatible(
                    pipeline_report["optimized_native_report"],
                    f"{run['hardware_key']} {algorithm} optimized native",
                )
            except Exception as exc:
                optimization_error = repr(exc)

            rows.append(
                _make_row(
                    run,
                    hardware_config,
                    algorithm,
                    pipeline_report,
                    optimization_error,
                    grid_size,
                    num_obstacles,
                    num_timesteps,
                )
            )

    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write all IR-flow comparison rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Write CSV output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    stem = (
        "phase_poly_ir_flow_comparison_"
        f"{first['qlbm_grid']}_obs{first['num_obstacles']}_t{first['num_timesteps']}"
    )
    paths = [output_dir / f"{stem}.csv"]
    write_csv(rows, paths[0])
    return paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare phase-polynomial IR optimization flows.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--num-obstacles", type=int, default=DEFAULT_NUM_OBSTACLES)
    parser.add_argument("--num-timesteps", type=int, default=DEFAULT_NUM_TIMESTEPS)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        choices=DEFAULT_ALGORITHMS,
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=OPTIMIZATION_LEVEL,
    )
    parser.add_argument("--seed-transpiler", type=int, default=SEED_TRANSPILER)
    return parser.parse_args()


def main() -> None:
    """Run the experiment and write result files."""
    args = parse_args()
    rows = run_ir_flow_comparison(
        algorithms=args.algorithms,
        grid_size=args.grid_size,
        num_obstacles=args.num_obstacles,
        num_timesteps=args.num_timesteps,
        config_path=args.config_path,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
    )
    paths = write_outputs(rows, args.output_dir)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
