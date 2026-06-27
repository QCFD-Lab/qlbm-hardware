"""Experiment QLBM resource-estimation sweeps.

This module builds parameterized QLBM cases, estimates resources for each
hardware configuration, and writes a CSV plus raw JSON reports suitable
for plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_SOURCE_ROOT = PROJECT_ROOT / "qlbm"
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qlbm-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/qlbm-cache")

for path in (QLBM_SOURCE_ROOT, QLBM_HARDWARE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.resource_estimator import QLBMResourceEstimator  # noqa: E402
from components.resource_estimator.benchmark_qlbm_resources_v2 import (  # noqa: E402
    build_full_logical_circuit,
    build_sectioned_logical_circuit,
    json_safe,
    load_hardware_configs,
)
from qlbm import ABLattice, MSLattice, SpaceTimeLattice  # noqa: E402
from qlbm.components import (  # noqa: E402
    ABGridMeasurement,
    ABInitialConditions,
    ABQLBM,
    EmptyPrimitive,
    GridMeasurement,
    MSInitialConditions,
    MSQLBM,
)
from qlbm.components.spacetime import (  # noqa: E402
    SpaceTimeGridVelocityMeasurement,
    SpaceTimeQLBM,
)
from qlbm.components.spacetime.initial.pointwise import (  # noqa: E402
    PointWiseSpaceTimeInitialConditions,
)


DEFAULT_CONFIG_PATH = (
    QLBM_HARDWARE_ROOT / "components" / "resource_estimator" / "config.json"
)
DEFAULT_OUTPUT_DIR = (
    QLBM_HARDWARE_ROOT / "qlbm-hardware-output" / "resource-estimates" / "v2"
)
DEFAULT_ALGORITHMS = ["ABQLBM", "MSQLBM", "SpaceTimeQLBM"]
DEFAULT_EXPERIMENTS = [
    "grid_size",
    "obstacle_count",
    "timesteps",
    "routing_overhead",
    "scheduled_duration",
]
DEFAULT_GRID_SIZES = [4, 8, 16]
DEFAULT_TIMESTEPS = [1, 2, 4, 8]
DEFAULT_OBSTACLE_COUNTS = [0, 1, 2, 3, 4]
DEFAULT_OBSTACLE_GRID = 8
DEFAULT_TIMESTEP_GRID = 8
DEFAULT_ROUTING_GRID = 8


@dataclass(frozen=True)
class ExperimentCaseSpec:
    """A parameterized QLBM case before circuits are built."""

    experiment: str
    algorithm: str
    grid_x: int
    grid_y: int
    num_timesteps: int
    obstacles: Tuple[Dict[str, Any], ...]

    @property
    def num_obstacles(self) -> int:
        """Return the number of geometry objects in this case."""
        return len(self.obstacles)

    @property
    def grid_points(self) -> int:
        """Return the number of spatial grid points."""
        return self.grid_x * self.grid_y

    @property
    def grid_label(self) -> str:
        """Return a compact grid label."""
        return f"{self.grid_x}x{self.grid_y}"

    @property
    def label(self) -> str:
        """Return a stable case label for CSV rows and report files."""
        return (
            f"{self.experiment}__{self.algorithm.lower()}__"
            f"{self.grid_label}__obs{self.num_obstacles}__t{self.num_timesteps}"
        )

    def metadata(self) -> Dict[str, Any]:
        """Return QLBM metadata saved into each estimator report."""
        return {
            "experiment": self.experiment,
            "algorithm": self.algorithm,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "grid_points": self.grid_points,
            "num_obstacles": self.num_obstacles,
            "num_timesteps": self.num_timesteps,
            "lattice": self.lattice_data(),
        }

    def lattice_data(self) -> Dict[str, Any]:
        """Return framework-specific lattice data for this case."""
        if self.algorithm == "ABQLBM":
            velocities: Any = "d2q9"
        elif self.algorithm == "MSQLBM":
            velocities = {"x": 4, "y": 4}
        elif self.algorithm == "SpaceTimeQLBM":
            velocities = "D2Q4"
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        return {
            "lattice": {
                "dim": {"x": self.grid_x, "y": self.grid_y},
                "velocities": velocities,
            },
            "geometry": [dict(obstacle) for obstacle in self.obstacles],
        }


@dataclass
class BuiltExperimentCase:
    """A QLBM case with built full and sectioned logical circuits."""

    spec: ExperimentCaseSpec
    circuit: QuantumCircuit
    sectioned_circuit: QuantumCircuit
    section_names: List[str]


def deterministic_obstacles(
    grid_x: int,
    grid_y: int,
    count: int,
    boundary: str = "bounceback",
) -> Tuple[Dict[str, Any], ...]:
    """Create deterministic non-overlapping one-cell cuboid obstacles."""
    if count < 0:
        raise ValueError("count must be non-negative")

    candidates = [
        (x, y)
        for y in range(1, max(1, grid_y - 1))
        for x in range(1, max(1, grid_x - 1))
    ]
    if count > len(candidates):
        raise ValueError(
            f"Cannot place {count} non-overlapping obstacles in {grid_x}x{grid_y}"
        )

    obstacles = []
    for x, y in candidates[:count]:
        obstacles.append(
            {
                "shape": "cuboid",
                "x": [x, x],
                "y": [y, y],
                "boundary": boundary,
            }
        )
    return tuple(obstacles)


def select_names(raw_names: Sequence[str], available: Iterable[str]) -> List[str]:
    """Resolve explicit names or ``all`` against available names."""
    available_names = list(available)
    if list(raw_names) == ["all"]:
        return available_names
    unknown = sorted(set(raw_names) - set(available_names))
    if unknown:
        raise ValueError(f"Unknown names: {', '.join(unknown)}")
    return list(raw_names)


def make_case_spec(
    experiment: str,
    algorithm: str,
    grid_size: int,
    num_timesteps: int = 1,
    num_obstacles: int = 0,
    obstacle_boundary: str = "bounceback",
) -> ExperimentCaseSpec:
    """Create a square-grid case specification."""
    return ExperimentCaseSpec(
        experiment=experiment,
        algorithm=algorithm,
        grid_x=grid_size,
        grid_y=grid_size,
        num_timesteps=num_timesteps,
        obstacles=deterministic_obstacles(
            grid_size,
            grid_size,
            num_obstacles,
            boundary=obstacle_boundary,
        ),
    )


def experiment_specs(
    experiments: Sequence[str],
    algorithms: Sequence[str],
    grid_sizes: Sequence[int] = DEFAULT_GRID_SIZES,
    obstacle_counts: Sequence[int] = DEFAULT_OBSTACLE_COUNTS,
    timesteps: Sequence[int] = DEFAULT_TIMESTEPS,
    obstacle_grid: int = DEFAULT_OBSTACLE_GRID,
    timestep_grid: int = DEFAULT_TIMESTEP_GRID,
    routing_grid: int = DEFAULT_ROUTING_GRID,
) -> List[ExperimentCaseSpec]:
    """Generate case specifications for the selected resource experiments."""
    specs: List[ExperimentCaseSpec] = []
    for experiment in experiments:
        for algorithm in algorithms:
            if experiment == "grid_size":
                for grid_size in grid_sizes:
                    specs.append(make_case_spec(experiment, algorithm, grid_size))
            elif experiment == "obstacle_count":
                for count in obstacle_counts:
                    specs.append(
                        make_case_spec(
                            experiment,
                            algorithm,
                            obstacle_grid,
                            num_obstacles=count,
                        )
                    )
            elif experiment == "timesteps":
                for num_timesteps in timesteps:
                    specs.append(
                        make_case_spec(
                            experiment,
                            algorithm,
                            timestep_grid,
                            num_timesteps=num_timesteps,
                        )
                    )
            elif experiment in {"routing_overhead", "scheduled_duration"}:
                specs.append(make_case_spec(experiment, algorithm, routing_grid))
            else:
                raise ValueError(f"Unsupported experiment: {experiment}")
    return specs


def build_experiment_case(spec: ExperimentCaseSpec) -> BuiltExperimentCase:
    """Build logical and sectioned circuits for a case specification."""
    lattice_data = spec.lattice_data()
    section_timesteps = spec.num_timesteps
    logical_timesteps = spec.num_timesteps

    if spec.algorithm == "ABQLBM":
        lattice = ABLattice(lattice_data)
        initial_conditions = ABInitialConditions(lattice)
        algorithm = ABQLBM(lattice)
        postprocessing = EmptyPrimitive(lattice)
        measurement = ABGridMeasurement(lattice)
    elif spec.algorithm == "MSQLBM":
        lattice = MSLattice(lattice_data)
        initial_conditions = MSInitialConditions(lattice)
        algorithm = MSQLBM(lattice)
        postprocessing = EmptyPrimitive(lattice)
        measurement = GridMeasurement(lattice)
    elif spec.algorithm == "SpaceTimeQLBM":
        lattice = SpaceTimeLattice(
            num_timesteps=spec.num_timesteps,
            lattice_data=lattice_data,
        )
        initial_conditions = PointWiseSpaceTimeInitialConditions(lattice)
        algorithm = SpaceTimeQLBM(lattice)
        postprocessing = EmptyPrimitive(lattice)
        measurement = SpaceTimeGridVelocityMeasurement(lattice)
        section_timesteps = 1
        logical_timesteps = 1
    else:
        raise ValueError(f"Unsupported algorithm: {spec.algorithm}")

    sectioned_circuit, section_names = build_sectioned_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        section_timesteps,
    )
    circuit = build_full_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        logical_timesteps,
    )
    return BuiltExperimentCase(
        spec=spec,
        circuit=circuit,
        sectioned_circuit=sectioned_circuit,
        section_names=section_names,
    )


def hardware_row_metadata(hardware_key: str, hardware_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return stable hardware metadata for CSV rows."""
    return {
        "hardware_key": hardware_key,
        "hardware": hardware_config.get("id", hardware_key),
        "hardware_label": hardware_config.get("device_name")
        or hardware_config.get("id", hardware_key),
        "architecture": hardware_config.get("architecture"),
        "coupling_type": hardware_config.get("coupling_type"),
        "num_hardware_qubits": hardware_config.get("num_qubits"),
        "measurement_time_s": hardware_config.get("measurement_time_s"),
    }


def report_timing_warnings(report: Dict[str, Any]) -> str:
    """Return semicolon-delimited timing warnings from an estimator report."""
    timing = report.get("transpiled_time") or {}
    return "; ".join(timing.get("warnings") or [])


def make_experiment_csv_row(
    report: Dict[str, Any],
    spec: ExperimentCaseSpec,
    hardware_key: str,
    hardware_config: Dict[str, Any],
    build_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Flatten an estimator report into one tidy experiment CSV row."""
    logical = report.get("logical") or {}
    transpiled = report.get("transpiled") or {}
    transpiled_compact = report.get("transpiled_compact") or {}
    overheads = report.get("overheads") or {}
    compatibility = report.get("transpiled_compatibility") or {}
    timing = report.get("transpiled_time") or {}
    section_analysis = report.get("section_analysis") or {}
    max_time_section = section_analysis.get("max_critical_path_time_section") or {}
    max_2q_section = section_analysis.get("max_two_qubit_gate_section") or {}

    row = {
        "case": spec.label,
        "experiment": spec.experiment,
        "algorithm": spec.algorithm,
        "grid_x": spec.grid_x,
        "grid_y": spec.grid_y,
        "grid_label": spec.grid_label,
        "grid_points": spec.grid_points,
        "num_obstacles": spec.num_obstacles,
        "num_timesteps": spec.num_timesteps,
        "build_error": build_error,
        "logical_qubits": logical.get("num_qubits"),
        "logical_depth": logical.get("depth"),
        "logical_size": logical.get("size"),
        "logical_2q_ops": logical.get("num_2q_ops"),
        "transpiled_qubits": transpiled.get("num_qubits"),
        "transpiled_active_qubits": transpiled.get("active_qubits"),
        "transpiled_compact_qubits": transpiled_compact.get("num_qubits"),
        "transpiled_depth": transpiled.get("depth"),
        "transpiled_size": transpiled.get("size"),
        "transpiled_2q_ops": transpiled.get("num_2q_ops"),
        "depth_ratio": overheads.get("depth_ratio"),
        "size_ratio": overheads.get("size_ratio"),
        "two_qubit_gate_ratio": overheads.get("two_qubit_gate_ratio"),
        "transpiled_compatible": compatibility.get("compatible"),
        "transpiled_qubit_capacity_ok": compatibility.get("qubit_capacity_ok"),
        "transpiled_basis_gates_ok": compatibility.get("basis_gates_ok"),
        "transpiled_coupling_map_ok": compatibility.get("coupling_map_ok"),
        "scheduled_duration_s": timing.get("scheduled_duration_s"),
        "max_idle_time_s": timing.get("max_idle_time_s"),
        "critical_path_time_s": timing.get("critical_path_time_s"),
        "serial_time_s": timing.get("serial_time_s"),
        "timing_warnings": report_timing_warnings(report),
        "max_time_section": max_time_section.get("section"),
        "max_time_section_critical_path_s": max_time_section.get(
            "critical_path_time_s"
        ),
        "max_2q_section": max_2q_section.get("section"),
        "max_2q_section_count": max_2q_section.get("num_2q_ops"),
        "transpile_error": report.get("transpile_error"),
        "routing_reference_2q_ops": None,
        "routing_2q_overhead": None,
        "routing_added_2q": None,
        "routing_reference_transpile_error": None,
    }
    row.update(hardware_row_metadata(hardware_key, hardware_config))
    return row


def build_error_row(
    spec: ExperimentCaseSpec,
    hardware_key: str,
    hardware_config: Dict[str, Any],
    build_error: str,
) -> Dict[str, Any]:
    """Create a CSV row for a case that failed before estimation."""
    return make_experiment_csv_row(
        report={},
        spec=spec,
        hardware_key=hardware_key,
        hardware_config=hardware_config,
        build_error=build_error,
    )


def estimate_case(
    built_case: BuiltExperimentCase,
    hardware_config: Dict[str, Any],
    optimization_level: int,
    seed_transpiler: int,
    force_no_coupling: bool = False,
) -> Dict[str, Any]:
    """Run the resource estimator for one built case and hardware config."""
    estimator = QLBMResourceEstimator(
        hardware_config,
        force_no_coupling=force_no_coupling,
    )
    return estimator.estimate(
        built_case.circuit,
        label=built_case.spec.label,
        qlbm_metadata=built_case.spec.metadata(),
        sectioned_circuit=built_case.sectioned_circuit,
        section_names=built_case.section_names,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        transpile_circuit=True,
    )


def routing_row_from_pair(
    coupled_row: Dict[str, Any],
    uncoupled_row: Dict[str, Any],
) -> Dict[str, Any]:
    """Add routing-overhead columns to a coupled-row/uncoupled-row pair."""
    row = dict(coupled_row)
    reference_2q = uncoupled_row.get("transpiled_2q_ops")
    coupled_2q = coupled_row.get("transpiled_2q_ops")
    row["routing_reference_2q_ops"] = reference_2q
    row["routing_reference_transpile_error"] = uncoupled_row.get("transpile_error")

    if reference_2q in (None, "", 0) or coupled_2q in (None, ""):
        row["routing_2q_overhead"] = None
        row["routing_added_2q"] = None
        return row

    row["routing_2q_overhead"] = float(coupled_2q) / float(reference_2q)
    row["routing_added_2q"] = int(coupled_2q) - int(reference_2q)
    return row


def write_json_report(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON report, replacing unserializable circuit objects with null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, indent=2)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write CSV rows with a stable union of all columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment_sweeps(
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    experiments: Sequence[str] = DEFAULT_EXPERIMENTS,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    hardware_names: Sequence[str] = ("all",),
    grid_sizes: Sequence[int] = DEFAULT_GRID_SIZES,
    obstacle_counts: Sequence[int] = DEFAULT_OBSTACLE_COUNTS,
    timesteps: Sequence[int] = DEFAULT_TIMESTEPS,
    optimization_level: int = 1,
    seed_transpiler: int = 42,
) -> List[Dict[str, Any]]:
    """Run selected resource sweeps and return tidy CSV rows."""
    configs = load_hardware_configs(config_path)
    selected_hardware = select_names(hardware_names, configs.keys())
    selected_algorithms = select_names(algorithms, DEFAULT_ALGORITHMS)
    selected_experiments = select_names(experiments, DEFAULT_EXPERIMENTS)
    specs = experiment_specs(
        selected_experiments,
        selected_algorithms,
        grid_sizes=grid_sizes,
        obstacle_counts=obstacle_counts,
        timesteps=timesteps,
    )

    rows: List[Dict[str, Any]] = []
    reports_dir = output_dir / "reports"

    for spec in specs:
        try:
            built_case = build_experiment_case(spec)
        except Exception as exc:
            for hardware_key in selected_hardware:
                write_json_report(
                    reports_dir / f"{spec.label}__{hardware_key}__build_error.json",
                    {
                        "label": spec.label,
                        "qlbm": spec.metadata(),
                        "hardware": hardware_row_metadata(
                            hardware_key,
                            configs[hardware_key],
                        ),
                        "build_error": repr(exc),
                    },
                )
                rows.append(
                    build_error_row(
                        spec,
                        hardware_key,
                        configs[hardware_key],
                        build_error=repr(exc),
                    )
                )
            continue

        for hardware_key in selected_hardware:
            hardware_config = configs[hardware_key]

            coupled_report = estimate_case(
                built_case,
                hardware_config,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
                force_no_coupling=False,
            )
            report_path = reports_dir / f"{spec.label}__{hardware_key}.json"
            write_json_report(report_path, coupled_report)
            coupled_row = make_experiment_csv_row(
                coupled_report,
                spec,
                hardware_key,
                hardware_config,
            )

            if spec.experiment != "routing_overhead":
                rows.append(coupled_row)
                continue

            uncoupled_report = estimate_case(
                built_case,
                hardware_config,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
                force_no_coupling=True,
            )
            uncoupled_path = (
                reports_dir / f"{spec.label}__{hardware_key}__no_coupling.json"
            )
            write_json_report(uncoupled_path, uncoupled_report)
            uncoupled_row = make_experiment_csv_row(
                uncoupled_report,
                spec,
                hardware_key,
                hardware_config,
            )
            rows.append(routing_row_from_pair(coupled_row, uncoupled_row))

    csv_path = output_dir / "experiment_resource_estimates.csv"
    write_csv(rows, csv_path)
    return rows


def parse_ints(values: Sequence[str]) -> List[int]:
    """Parse CLI integer lists."""
    return [int(value) for value in values]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run QLBM resource-estimation experiment sweeps.",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        help=f"Experiment names or all. Available: {', '.join(DEFAULT_EXPERIMENTS)}",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["all"],
        help=f"Algorithm names or all. Available: {', '.join(DEFAULT_ALGORITHMS)}",
    )
    parser.add_argument(
        "--hardware",
        nargs="+",
        default=["all"],
        help="Hardware config keys or all.",
    )
    parser.add_argument("--grid-sizes", nargs="+", default=DEFAULT_GRID_SIZES)
    parser.add_argument(
        "--obstacle-counts",
        nargs="+",
        default=DEFAULT_OBSTACLE_COUNTS,
    )
    parser.add_argument("--timesteps", nargs="+", default=DEFAULT_TIMESTEPS)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--seed-transpiler", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small 4x4 grid-size sweep on the first selected hardware.",
    )
    return parser.parse_args()


def main() -> None:
    """Run experiment sweeps from the command line."""
    args = parse_args()
    configs = load_hardware_configs(args.config_path)
    hardware_names = args.hardware

    if args.smoke:
        hardware_names = [select_names(hardware_names, configs.keys())[0]]
        experiments = ["grid_size"]
        grid_sizes = [4]
        obstacle_counts = [0]
        timesteps = [1]
    else:
        experiments = args.experiments
        grid_sizes = parse_ints(args.grid_sizes)
        obstacle_counts = parse_ints(args.obstacle_counts)
        timesteps = parse_ints(args.timesteps)

    rows = run_experiment_sweeps(
        config_path=args.config_path,
        output_dir=args.output_dir,
        experiments=experiments,
        algorithms=args.algorithms,
        hardware_names=hardware_names,
        grid_sizes=grid_sizes,
        obstacle_counts=obstacle_counts,
        timesteps=timesteps,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
    )
    print(
        "Wrote "
        f"{len(rows)} rows to {args.output_dir / 'experiment_resource_estimates.csv'}"
    )


if __name__ == "__main__":
    main()
