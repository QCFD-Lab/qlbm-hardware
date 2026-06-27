"""Compare unoptimized and phase-polynomial-optimized QLBM circuits."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qiskit.transpiler import CouplingMap

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_SOURCE_ROOT = PROJECT_ROOT / "qlbm"
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qlbm-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/qlbm-cache")

for path in (QLBM_SOURCE_ROOT, QLBM_HARDWARE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.phase_poly_optimizer import (  # noqa: E402
    A2APhasePoly,
    ArchitectureAwarePhasePolyOptimizer,
)
from components.resource_estimator import QLBMResourceEstimator  # noqa: E402
from components.resource_estimator.experiment_resource_sweeps import (  # noqa: E402
    DEFAULT_ALGORITHMS,
    build_experiment_case,
    make_case_spec,
)


DEFAULT_OUTPUT_DIR = (
    QLBM_HARDWARE_ROOT
    / "components"
    / "phase_poly_optimizer"
    / "output"
    / "phase-poly-optimization-comparison"
)
DEFAULT_BASIS_GATES = ["rz", "sx", "x", "cx"]
DEFAULT_GRID_SIZE = 8
DEFAULT_NUM_OBSTACLES = 1
DEFAULT_NUM_TIMESTEPS = 1
DEFAULT_NUM_QUBITS = 49
DEFAULT_HARDWARE_GRID_ROWS = 7
DEFAULT_HARDWARE_GRID_COLS = 7
DEFAULT_OPTIMIZATION_LEVEL = 0
DEFAULT_SEED_TRANSPILER = 42


FIELDNAMES = [
    "topology",
    "optimizer",
    "algorithm",
    "qlbm_grid",
    "num_obstacles",
    "num_timesteps",
    "basis_gates",
    "hardware_qubits",
    "coupling_type",
    "coupling_params",
    "baseline_depth",
    "optimized_depth",
    "baseline_cx",
    "optimized_cx",
    "cx_reduction_percent",
    "baseline_rz",
    "optimized_rz",
    "baseline_total_gates",
    "optimized_total_gates",
    "optimization_error",
]


def hypothetical_cx_config(
    topology: str,
    num_qubits: int,
    hardware_grid_rows: int,
    hardware_grid_cols: int,
    basis_gates: Sequence[str] = DEFAULT_BASIS_GATES,
) -> Dict[str, Any]:
    """Return a hypothetical CX/RZ hardware config for one experiment topology."""
    if topology == "all_to_all":
        coupling_type = "all_to_all"
        coupling_params = {"num_qubits": num_qubits}
        description = "all-to-all"
    elif topology == "2d_grid":
        grid_qubits = hardware_grid_rows * hardware_grid_cols
        if grid_qubits != num_qubits:
            raise ValueError(
                "2D hardware grid size must match num_qubits: "
                f"{hardware_grid_rows}x{hardware_grid_cols}={grid_qubits}, "
                f"num_qubits={num_qubits}"
            )
        coupling_type = "2d_grid"
        coupling_params = {
            "rows": hardware_grid_rows,
            "cols": hardware_grid_cols,
        }
        description = f"{hardware_grid_rows}x{hardware_grid_cols} 2D nearest-neighbour"
    else:
        raise ValueError("topology must be one of: all_to_all, 2d_grid")

    return {
        "id": f"hypothetical_{num_qubits}q_{coupling_type}_phasepoly_cx",
        "architecture": f"hypothetical {description} CX/RZ device",
        "device_name": f"Hypothetical {num_qubits}-qubit {description} CX/RZ",
        "num_qubits": num_qubits,
        "basis_gates": list(basis_gates),
        "coupling_type": coupling_type,
        "coupling_params": coupling_params,
        "gate_times_s": {
            "rz": 0.0,
            "sx": 1.0,
            "x": 1.0,
            "cx": 1.0,
        },
        "measurement_time_s": 1.0,
    }


def _op_count(metrics: Dict[str, Any], gate: str) -> int:
    return int((metrics.get("op_counts") or {}).get(gate, 0))


def _cx_reduction_percent(baseline_cx: int, optimized_cx: int) -> Optional[float]:
    if baseline_cx == 0:
        return None
    return 100.0 * (baseline_cx - optimized_cx) / baseline_cx


def _optimizer_for_topology(topology: str) -> tuple[str, Any]:
    if topology == "all_to_all":
        return "all_to_all", A2APhasePoly()
    if topology == "2d_grid":
        return "architecture_aware", ArchitectureAwarePhasePolyOptimizer()
    raise ValueError("topology must be one of: all_to_all, 2d_grid")


def _coupling_map_for_topology(
    topology: str,
    estimator: QLBMResourceEstimator,
    num_qubits: int,
) -> CouplingMap:
    if topology == "all_to_all":
        return CouplingMap.from_full(num_qubits)
    return CouplingMap(estimator.coupling_map)


def require_compatible(report: Dict[str, Any], label: str) -> None:
    """Raise if a reported circuit does not match the configured hardware."""
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


def make_comparison_row(
    topology: str,
    optimizer_name: str,
    algorithm: str,
    baseline_report: Dict[str, Any],
    optimized_report: Optional[Dict[str, Any]],
    hardware_config: Dict[str, Any],
    optimization_error: Optional[str],
    qlbm_grid_size: int,
    num_obstacles: int,
    num_timesteps: int,
) -> Dict[str, Any]:
    """Flatten one before/after optimization comparison into a table row."""
    baseline = baseline_report.get("transpiled") or {}
    optimized = (optimized_report or {}).get("transpiled") or {}
    baseline_cx = _op_count(baseline, "cx")
    optimized_cx = _op_count(optimized, "cx")

    return {
        "topology": topology,
        "optimizer": optimizer_name,
        "algorithm": algorithm,
        "qlbm_grid": f"{qlbm_grid_size}x{qlbm_grid_size}",
        "num_obstacles": num_obstacles,
        "num_timesteps": num_timesteps,
        "basis_gates": " ".join(hardware_config["basis_gates"]),
        "hardware_qubits": hardware_config["num_qubits"],
        "coupling_type": hardware_config["coupling_type"],
        "coupling_params": json.dumps(hardware_config["coupling_params"]),
        "baseline_depth": baseline.get("depth"),
        "optimized_depth": optimized.get("depth"),
        "baseline_cx": baseline_cx,
        "optimized_cx": optimized_cx if optimized_report else None,
        "cx_reduction_percent": (
            _cx_reduction_percent(baseline_cx, optimized_cx)
            if optimized_report
            else None
        ),
        "baseline_rz": _op_count(baseline, "rz"),
        "optimized_rz": _op_count(optimized, "rz") if optimized_report else None,
        "baseline_total_gates": baseline.get("size"),
        "optimized_total_gates": optimized.get("size"),
        "optimization_error": optimization_error,
    }


def run_topology_comparison(
    topology: str,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    qlbm_grid_size: int = DEFAULT_GRID_SIZE,
    num_obstacles: int = DEFAULT_NUM_OBSTACLES,
    num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    num_qubits: int = DEFAULT_NUM_QUBITS,
    hardware_grid_rows: int = DEFAULT_HARDWARE_GRID_ROWS,
    hardware_grid_cols: int = DEFAULT_HARDWARE_GRID_COLS,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
) -> List[Dict[str, Any]]:
    """Run one topology's unoptimized-vs-optimized comparison."""
    hardware_config = hypothetical_cx_config(
        topology=topology,
        num_qubits=num_qubits,
        hardware_grid_rows=hardware_grid_rows,
        hardware_grid_cols=hardware_grid_cols,
    )
    estimator = QLBMResourceEstimator(hardware_config)
    optimizer_name, optimizer = _optimizer_for_topology(topology)
    rows: List[Dict[str, Any]] = []

    for algorithm in algorithms:
        spec = make_case_spec(
            "phase_poly_optimization_comparison",
            algorithm,
            qlbm_grid_size,
            num_timesteps=num_timesteps,
            num_obstacles=num_obstacles,
            obstacle_boundary="bounceback",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            built_case = build_experiment_case(spec)

        baseline_report = estimator.estimate(
            built_case.circuit,
            label=f"{spec.label}-{topology}-baseline",
            qlbm_metadata=spec.metadata(),
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            transpile_circuit=True,
            phase_polynomial_analysis=False,
        )
        if baseline_report.get("transpile_error"):
            raise RuntimeError(str(baseline_report["transpile_error"]))
        require_compatible(baseline_report, f"{algorithm} {topology} baseline")

        optimized_report = None
        optimization_error = None
        try:
            baseline_circuit = baseline_report["transpiled_circuit"]
            coupling_map = _coupling_map_for_topology(
                topology,
                estimator,
                baseline_circuit.num_qubits,
            )
            optimized_circuit = optimizer.optimize(
                circuit=baseline_circuit.copy(),
                coupling_map=coupling_map,
            )
            optimized_report = estimator.estimate_pretranspiled(
                optimized_circuit,
                label=f"{spec.label}-{topology}-{optimizer_name}-optimized",
                qlbm_metadata=spec.metadata(),
                logical_metrics=baseline_report["logical"],
                phase_polynomial_analysis=False,
            )
            require_compatible(
                optimized_report,
                f"{algorithm} {topology} optimized",
            )
        except Exception as exc:
            optimization_error = repr(exc)

        rows.append(
            make_comparison_row(
                topology=topology,
                optimizer_name=optimizer_name,
                algorithm=algorithm,
                baseline_report=baseline_report,
                optimized_report=optimized_report,
                hardware_config=hardware_config,
                optimization_error=optimization_error,
                qlbm_grid_size=qlbm_grid_size,
                num_obstacles=num_obstacles,
                num_timesteps=num_timesteps,
            )
        )

    return rows


def run_comparison(
    topologies: Sequence[str],
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    qlbm_grid_size: int = DEFAULT_GRID_SIZE,
    num_obstacles: int = DEFAULT_NUM_OBSTACLES,
    num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    num_qubits: int = DEFAULT_NUM_QUBITS,
    hardware_grid_rows: int = DEFAULT_HARDWARE_GRID_ROWS,
    hardware_grid_cols: int = DEFAULT_HARDWARE_GRID_COLS,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
) -> List[Dict[str, Any]]:
    """Run the comparison for all requested topologies."""
    rows: List[Dict[str, Any]] = []
    for topology in topologies:
        rows.extend(
            run_topology_comparison(
                topology=topology,
                algorithms=algorithms,
                qlbm_grid_size=qlbm_grid_size,
                num_obstacles=num_obstacles,
                num_timesteps=num_timesteps,
                num_qubits=num_qubits,
                hardware_grid_rows=hardware_grid_rows,
                hardware_grid_cols=hardware_grid_cols,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
            )
        )
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write all comparison rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Write CSV output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    grid = str(first["qlbm_grid"]).replace("x", "x")
    stem = (
        "phase_poly_optimization_comparison_"
        f"{grid}_obs{first['num_obstacles']}_t{first['num_timesteps']}"
    )
    paths = [output_dir / f"{stem}.csv"]
    write_csv(rows, paths[0])
    return paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare phase-polynomial optimization resource metrics.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qlbm-grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--num-obstacles", type=int, default=DEFAULT_NUM_OBSTACLES)
    parser.add_argument("--num-timesteps", type=int, default=DEFAULT_NUM_TIMESTEPS)
    parser.add_argument("--num-qubits", type=int, default=DEFAULT_NUM_QUBITS)
    parser.add_argument(
        "--hardware-grid-rows",
        type=int,
        default=DEFAULT_HARDWARE_GRID_ROWS,
    )
    parser.add_argument(
        "--hardware-grid-cols",
        type=int,
        default=DEFAULT_HARDWARE_GRID_COLS,
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["all_to_all", "2d_grid"],
        choices=["all_to_all", "2d_grid"],
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        choices=DEFAULT_ALGORITHMS,
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=DEFAULT_OPTIMIZATION_LEVEL,
    )
    parser.add_argument("--seed-transpiler", type=int, default=DEFAULT_SEED_TRANSPILER)
    return parser.parse_args()


def main() -> None:
    """Run the experiment and write result files."""
    args = parse_args()
    rows = run_comparison(
        topologies=args.topologies,
        algorithms=args.algorithms,
        qlbm_grid_size=args.qlbm_grid_size,
        num_obstacles=args.num_obstacles,
        num_timesteps=args.num_timesteps,
        num_qubits=args.num_qubits,
        hardware_grid_rows=args.hardware_grid_rows,
        hardware_grid_cols=args.hardware_grid_cols,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
    )
    paths = write_outputs(rows, args.output_dir)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
