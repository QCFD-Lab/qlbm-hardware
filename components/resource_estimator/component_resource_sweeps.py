"""Component-level QLBM resource-estimation sweeps.

This module estimates resources for the algorithm-body primitives/operators that
make up each QLBM method, for example streaming, reflection, collision, and
ancilla preparation."""

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
    SECTION_BOUNDARY_PREFIX,
    json_safe,
    load_hardware_configs,
)
from components.resource_estimator.experiment_resource_sweeps import (  # noqa: E402
    DEFAULT_ALGORITHMS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_GRID_SIZES,
    DEFAULT_OBSTACLE_COUNTS,
    DEFAULT_TIMESTEPS,
    ExperimentCaseSpec,
    experiment_specs,
    hardware_row_metadata,
    report_timing_warnings,
    select_names,
)
from qlbm import ABLattice, MSLattice, SpaceTimeLattice  # noqa: E402
from qlbm.components.ab import (  # noqa: E402
    ABInitialConditions,
    ABReflectionOperator,
    ABStreamingOperator,
)
from qlbm.components.ms import (  # noqa: E402
    BounceBackReflectionOperator,
    MSInitialConditions,
    MSStreamingOperator,
    SpecularReflectionOperator,
    StreamingAncillaPreparation,
)
from qlbm.components.spacetime import (  # noqa: E402
    PointWiseSpaceTimeInitialConditions,
    PointWiseSpaceTimeReflectionOperator,
    SpaceTimeD2Q4CollisionOperator,
    SpaceTimeStreamingOperator,
)
from qlbm.lattice.geometry.shapes.block import Block  # noqa: E402
from qlbm.tools.exceptions import LatticeException  # noqa: E402
from qlbm.tools.utils import get_time_series  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    QLBM_HARDWARE_ROOT
    / "qlbm-hardware-output"
    / "resource-estimates"
    / "v2"
    / "components"
)
DEFAULT_COMPONENT_EXPERIMENTS = ["grid_size", "obstacle_count", "timesteps"]
GENERIC_ALL_TO_ALL_HARDWARE_KEY = "generic_all_to_all_rz_sx_x_cx"
GENERIC_ALL_TO_ALL_HARDWARE_CONFIG = {
    "id": "generic all-to-all rz sx x cx",
    "architecture": "generic all-to-all native-gate model",
    "device_name": "Generic all-to-all rz/sx/x/cx",
    "year_reported": 2026,
    "num_qubits": 64,
    "basis_gates": ["rz", "sx", "x", "cx"],
    "coupling_type": "all_to_all",
    "coupling_params": {"num_qubits": 64},
    "gate_times_s": {
        "rz": 0.0,
        "sx": 25e-9,
        "x": 25e-9,
        "cx": 200e-9,
    },
    "measurement_time_s": 1e-6,
}


@dataclass(frozen=True)
class ComponentSection:
    """A single logical component instance in an algorithm body."""

    section: str
    component_group: str
    component_kind: str
    component_class: str
    circuit: QuantumCircuit
    sequence_index: int
    logical_timestep: Optional[int] = None
    algorithm_substep: Optional[int] = None
    encoded_timestep: Optional[int] = None
    dimension: Optional[int] = None
    velocities: Tuple[int, ...] = ()
    boundary_condition: Optional[str] = None


@dataclass(frozen=True)
class BuiltComponentCase:
    """A QLBM component case with full and sectioned algorithm-body circuits."""

    spec: ExperimentCaseSpec
    circuit: QuantumCircuit
    sectioned_circuit: QuantumCircuit
    sections: Tuple[ComponentSection, ...]

    @property
    def section_names(self) -> List[str]:
        """Return section labels in sequence order."""
        return [section.section for section in self.sections]


def _component_circuit(component: Any) -> QuantumCircuit:
    return component.circuit if hasattr(component, "circuit") else component


def _section_name(
    sequence_index: int,
    component_group: str,
    component_kind: str,
    logical_timestep: Optional[int] = None,
    algorithm_substep: Optional[int] = None,
    encoded_timestep: Optional[int] = None,
    dimension: Optional[int] = None,
    boundary_condition: Optional[str] = None,
) -> str:
    tokens = [f"{sequence_index:03d}", component_group, component_kind]
    if logical_timestep is not None:
        tokens.append(f"step{logical_timestep}")
    if algorithm_substep is not None:
        tokens.append(f"sub{algorithm_substep}")
    if encoded_timestep is not None:
        tokens.append(f"encoded{encoded_timestep}")
    if dimension is not None:
        tokens.append(f"dim{dimension}")
    if boundary_condition is not None:
        tokens.append(boundary_condition)
    return "__".join(tokens)


def _make_section(
    component: Any,
    sequence_index: int,
    component_group: str,
    component_kind: str,
    logical_timestep: Optional[int] = None,
    algorithm_substep: Optional[int] = None,
    encoded_timestep: Optional[int] = None,
    dimension: Optional[int] = None,
    velocities: Sequence[int] = (),
    boundary_condition: Optional[str] = None,
) -> ComponentSection:
    return ComponentSection(
        section=_section_name(
            sequence_index,
            component_group,
            component_kind,
            logical_timestep=logical_timestep,
            algorithm_substep=algorithm_substep,
            encoded_timestep=encoded_timestep,
            dimension=dimension,
            boundary_condition=boundary_condition,
        ),
        component_group=component_group,
        component_kind=component_kind,
        component_class=component.__class__.__name__,
        circuit=_component_circuit(component),
        sequence_index=sequence_index,
        logical_timestep=logical_timestep,
        algorithm_substep=algorithm_substep,
        encoded_timestep=encoded_timestep,
        dimension=dimension,
        velocities=tuple(velocities),
        boundary_condition=boundary_condition,
    )


def abqlbm_component_sections(spec: ExperimentCaseSpec) -> Tuple[ComponentSection, ...]:
    """Build ABQLBM initialization plus streaming/reflection component sections."""
    lattice = ABLattice(spec.lattice_data())
    sections: List[ComponentSection] = []

    sections.append(
        _make_section(
            ABInitialConditions(lattice),
            len(sections),
            "initialization",
            "ab_initial_conditions",
        )
    )

    for logical_timestep in range(1, spec.num_timesteps + 1):
        sections.append(
            _make_section(
                ABStreamingOperator(lattice),
                len(sections),
                "streaming",
                "ab_streaming",
                logical_timestep=logical_timestep,
            )
        )
        sections.append(
            _make_section(
                ABReflectionOperator(lattice, None),
                len(sections),
                "reflection",
                "ab_reflection",
                logical_timestep=logical_timestep,
            )
        )

    return tuple(sections)


def msqlbm_component_sections(spec: ExperimentCaseSpec) -> Tuple[ComponentSection, ...]:
    """Build MSQLBM initialization and CFL sections."""
    lattice = MSLattice(spec.lattice_data())
    time_series = get_time_series(
        2 ** lattice.num_velocities[0].bit_length(),
        group_velocities=False,
    )
    sections: List[ComponentSection] = []

    sections.append(
        _make_section(
            MSInitialConditions(lattice),
            len(sections),
            "initialization",
            "ms_initial_conditions",
        )
    )

    for logical_timestep in range(1, spec.num_timesteps + 1):
        for algorithm_substep, velocities_to_increment in enumerate(time_series, start=1):
            sections.append(
                _make_section(
                    MSStreamingOperator(lattice, velocities_to_increment),
                    len(sections),
                    "streaming",
                    "ms_streaming",
                    logical_timestep=logical_timestep,
                    algorithm_substep=algorithm_substep,
                    velocities=velocities_to_increment,
                )
            )

            if lattice.shapes["specular"]:
                if not all(
                    isinstance(shape, Block) for shape in lattice.shapes["specular"]
                ):
                    raise LatticeException(
                        "All specular MSQLBM shapes must be cuboids."
                    )
                sections.append(
                    _make_section(
                        SpecularReflectionOperator(
                            lattice,
                            lattice.shapes["specular"],  # type: ignore[arg-type]
                        ),
                        len(sections),
                        "reflection",
                        "ms_specular_reflection",
                        logical_timestep=logical_timestep,
                        algorithm_substep=algorithm_substep,
                        velocities=velocities_to_increment,
                        boundary_condition="specular",
                    )
                )

            for boundary_condition in ("bounceback", "specular"):
                if lattice.shapes[boundary_condition]:
                    if not all(
                        isinstance(shape, Block)
                        for shape in lattice.shapes[boundary_condition]
                    ):
                        raise LatticeException(
                            f"All {boundary_condition} MSQLBM shapes must be cuboids."
                        )
                sections.append(
                    _make_section(
                        BounceBackReflectionOperator(
                            lattice,
                            lattice.shapes["bounceback"],  # type: ignore[arg-type]
                        ),
                        len(sections),
                        "reflection",
                        "ms_bounceback_reflection",
                        logical_timestep=logical_timestep,
                        algorithm_substep=algorithm_substep,
                        velocities=velocities_to_increment,
                        boundary_condition=boundary_condition,
                    )
                )

            for dim in range(lattice.num_dims):
                sections.append(
                    _make_section(
                        StreamingAncillaPreparation(
                            lattice,
                            velocities_to_increment,
                            dim,
                        ),
                        len(sections),
                        "ancilla_preparation",
                        "ms_streaming_ancilla_preparation",
                        logical_timestep=logical_timestep,
                        algorithm_substep=algorithm_substep,
                        dimension=dim,
                        velocities=velocities_to_increment,
                    )
                )

    return tuple(sections)


def spacetimeqlbm_component_sections(spec: ExperimentCaseSpec) -> Tuple[ComponentSection, ...]:
    """Build SpaceTimeQLBM initialization plus streaming/reflection/collision sections."""
    lattice = SpaceTimeLattice(
        num_timesteps=spec.num_timesteps,
        lattice_data=spec.lattice_data(),
    )
    if lattice.shapes["specular"]:
        raise LatticeException(
            "Currently, the Space-Time QLBM algorithm only supports bounceback "
            "boundary conditions."
        )

    sections: List[ComponentSection] = []
    sections.append(
        _make_section(
            PointWiseSpaceTimeInitialConditions(lattice),
            len(sections),
            "initialization",
            "spacetime_pointwise_initial_conditions",
        )
    )
    for encoded_timestep in range(lattice.num_timesteps, 0, -1):
        sections.append(
            _make_section(
                SpaceTimeStreamingOperator(lattice, encoded_timestep),
                len(sections),
                "streaming",
                "spacetime_streaming",
                encoded_timestep=encoded_timestep,
            )
        )
        sections.append(
            _make_section(
                PointWiseSpaceTimeReflectionOperator(
                    lattice,
                    encoded_timestep,
                    lattice.shapes["bounceback"],
                ),
                len(sections),
                "reflection",
                "spacetime_pointwise_reflection",
                encoded_timestep=encoded_timestep,
                boundary_condition="bounceback",
            )
        )
        if lattice.num_dims > 1:
            sections.append(
                _make_section(
                    SpaceTimeD2Q4CollisionOperator(lattice, encoded_timestep),
                    len(sections),
                    "collision",
                    "spacetime_d2q4_collision",
                    encoded_timestep=encoded_timestep,
                )
            )

    return tuple(sections)


def component_sections_for_spec(spec: ExperimentCaseSpec) -> Tuple[ComponentSection, ...]:
    """Build component sections for the selected algorithm."""
    if spec.algorithm == "ABQLBM":
        return abqlbm_component_sections(spec)
    if spec.algorithm == "MSQLBM":
        return msqlbm_component_sections(spec)
    if spec.algorithm == "SpaceTimeQLBM":
        return spacetimeqlbm_component_sections(spec)
    raise ValueError(f"Unsupported algorithm: {spec.algorithm}")


def build_component_circuits(
    sections: Sequence[ComponentSection],
) -> Tuple[QuantumCircuit, QuantumCircuit]:
    """Build full and diagnostic sectioned circuits from component sections."""
    if not sections:
        raise ValueError("At least one component section is required.")

    first_circuit = sections[0].circuit
    circuit = QuantumCircuit(*(first_circuit.qregs + first_circuit.cregs))
    sectioned_circuit = QuantumCircuit(*(first_circuit.qregs + first_circuit.cregs))

    for index, section in enumerate(sections):
        circuit.compose(section.circuit, inplace=True, qubits=range(circuit.num_qubits))
        sectioned_circuit.compose(
            section.circuit,
            inplace=True,
            qubits=range(sectioned_circuit.num_qubits),
        )
        if index < len(sections) - 1:
            sectioned_circuit.barrier(
                label=f"{SECTION_BOUNDARY_PREFIX}{section.section}"
            )

    return circuit, sectioned_circuit


def build_component_case(spec: ExperimentCaseSpec) -> BuiltComponentCase:
    """Build component-level full and sectioned circuits for one case."""
    sections = component_sections_for_spec(spec)
    circuit, sectioned_circuit = build_component_circuits(sections)
    return BuiltComponentCase(
        spec=spec,
        circuit=circuit,
        sectioned_circuit=sectioned_circuit,
        sections=sections,
    )


def _metrics_prefix(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        f"{prefix}_qubits": metrics.get("num_qubits"),
        f"{prefix}_depth": metrics.get("depth"),
        f"{prefix}_size": metrics.get("size"),
        f"{prefix}_1q_ops": metrics.get("num_1q_ops"),
        f"{prefix}_2q_ops": metrics.get("num_2q_ops"),
        f"{prefix}_3q_plus_ops": metrics.get("num_3q_plus_ops"),
    }


def make_component_csv_rows(
    report: Dict[str, Any],
    built_case: BuiltComponentCase,
    hardware_key: str,
    hardware_config: Dict[str, Any],
    build_error: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Flatten component section analysis into CSV rows."""
    spec = built_case.spec
    estimator = QLBMResourceEstimator(hardware_config)
    logical_sections = {
        section.section: estimator.extract_metrics(
            section.circuit,
            include_active_qubits=False,
            include_active_qubit_indices=False,
        )
        for section in built_case.sections
    }
    transpiled_sections = {
        section["section"]: section
        for section in (report.get("section_analysis") or {}).get("sections", [])
    }

    logical_algorithm = report.get("logical") or {}
    transpiled_algorithm = report.get("transpiled") or {}
    compatibility = report.get("transpiled_compatibility") or {}
    timing = report.get("transpiled_time") or {}

    rows: List[Dict[str, Any]] = []
    for section in built_case.sections:
        logical = logical_sections.get(section.section) or {}
        transpiled = transpiled_sections.get(section.section) or {}
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
            "component_section": section.section,
            "component_group": section.component_group,
            "component_kind": section.component_kind,
            "component_class": section.component_class,
            "sequence_index": section.sequence_index,
            "logical_timestep": section.logical_timestep,
            "algorithm_substep": section.algorithm_substep,
            "encoded_timestep": section.encoded_timestep,
            "dimension": section.dimension,
            "velocities": " ".join(str(velocity) for velocity in section.velocities),
            "boundary_condition": section.boundary_condition,
            "component_critical_path_time_s": transpiled.get("critical_path_time_s"),
            "component_serial_time_s": transpiled.get("serial_time_s"),
            "component_op_counts_json": json.dumps(
                transpiled.get("op_counts") or {},
                sort_keys=True,
            ),
            "algorithm_logical_depth": logical_algorithm.get("depth"),
            "algorithm_logical_2q_ops": logical_algorithm.get("num_2q_ops"),
            "algorithm_transpiled_depth": transpiled_algorithm.get("depth"),
            "algorithm_transpiled_2q_ops": transpiled_algorithm.get("num_2q_ops"),
            "algorithm_scheduled_duration_s": timing.get("scheduled_duration_s"),
            "algorithm_critical_path_time_s": timing.get("critical_path_time_s"),
            "algorithm_serial_time_s": timing.get("serial_time_s"),
            "timing_warnings": report_timing_warnings(report),
            "transpiled_compatible": compatibility.get("compatible"),
            "transpiled_qubit_capacity_ok": compatibility.get("qubit_capacity_ok"),
            "transpiled_basis_gates_ok": compatibility.get("basis_gates_ok"),
            "transpiled_coupling_map_ok": compatibility.get("coupling_map_ok"),
            "transpile_error": report.get("transpile_error"),
            "section_analysis_error": report.get("section_analysis_error"),
        }
        row.update(_metrics_prefix(logical, "logical_component"))
        row.update(_metrics_prefix(transpiled, "transpiled_component"))
        row.update(hardware_row_metadata(hardware_key, hardware_config))
        rows.append(row)

    return rows


def _build_error_rows(
    spec: ExperimentCaseSpec,
    hardware_key: str,
    hardware_config: Dict[str, Any],
    build_error: str,
) -> List[Dict[str, Any]]:
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
    }
    row.update(hardware_row_metadata(hardware_key, hardware_config))
    return [row]


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


def load_component_hardware_configs(config_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load configured hardware plus the generic component backend."""
    configs = load_hardware_configs(config_path)
    configs[GENERIC_ALL_TO_ALL_HARDWARE_KEY] = dict(
        GENERIC_ALL_TO_ALL_HARDWARE_CONFIG
    )
    return configs


def run_component_sweeps(
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    experiments: Sequence[str] = DEFAULT_COMPONENT_EXPERIMENTS,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    hardware_names: Sequence[str] = ("all",),
    grid_sizes: Sequence[int] = DEFAULT_GRID_SIZES,
    obstacle_counts: Sequence[int] = DEFAULT_OBSTACLE_COUNTS,
    timesteps: Sequence[int] = DEFAULT_TIMESTEPS,
    optimization_level: int = 1,
    seed_transpiler: int = 42,
) -> List[Dict[str, Any]]:
    """Run component-level sweeps and return tidy CSV rows."""
    configs = load_component_hardware_configs(config_path)
    selected_hardware = select_names(hardware_names, configs.keys())
    selected_algorithms = select_names(algorithms, DEFAULT_ALGORITHMS)
    selected_experiments = select_names(experiments, DEFAULT_COMPONENT_EXPERIMENTS)
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
            built_case = build_component_case(spec)
        except Exception as exc:
            for hardware_key in selected_hardware:
                hardware_config = configs[hardware_key]
                write_json_report(
                    reports_dir / f"{spec.label}__{hardware_key}__build_error.json",
                    {
                        "label": spec.label,
                        "qlbm": spec.metadata(),
                        "hardware": hardware_row_metadata(hardware_key, hardware_config),
                        "build_error": repr(exc),
                    },
                )
                rows.extend(
                    _build_error_rows(
                        spec,
                        hardware_key,
                        hardware_config,
                        build_error=repr(exc),
                    )
                )
            continue

        for hardware_key in selected_hardware:
            hardware_config = configs[hardware_key]
            estimator = QLBMResourceEstimator(hardware_config)
            report = estimator.estimate(
                built_case.circuit,
                label=spec.label,
                qlbm_metadata=spec.metadata(),
                sectioned_circuit=built_case.sectioned_circuit,
                section_names=built_case.section_names,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
                transpile_circuit=True,
            )
            write_json_report(
                reports_dir / f"{spec.label}__{hardware_key}.json",
                report,
            )
            rows.extend(
                make_component_csv_rows(
                    report,
                    built_case,
                    hardware_key,
                    hardware_config,
                )
            )

    csv_path = output_dir / "component_resource_estimates.csv"
    write_csv(rows, csv_path)
    return rows


def parse_ints(values: Sequence[str]) -> List[int]:
    """Parse CLI integer lists."""
    return [int(value) for value in values]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run component-level QLBM resource-estimation sweeps.",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        help=(
            "Experiment names or all. Available: "
            f"{', '.join(DEFAULT_COMPONENT_EXPERIMENTS)}"
        ),
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
        default=[GENERIC_ALL_TO_ALL_HARDWARE_KEY],
        help=(
            "Hardware config keys or all. Defaults to the generic all-to-all "
            "rz/sx/x/cx backend."
        ),
    )
    parser.add_argument("--grid-sizes", nargs="+", default=DEFAULT_GRID_SIZES)
    parser.add_argument("--obstacle-counts", nargs="+", default=DEFAULT_OBSTACLE_COUNTS)
    parser.add_argument("--timesteps", nargs="+", default=DEFAULT_TIMESTEPS)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--seed-transpiler", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small ABQLBM 4x4 component sweep on the first selected hardware.",
    )
    return parser.parse_args()


def main() -> None:
    """Run component sweeps from the command line."""
    args = parse_args()
    configs = load_component_hardware_configs(args.config_path)
    hardware_names = args.hardware

    if args.smoke:
        hardware_names = [select_names(hardware_names, configs.keys())[0]]
        experiments = ["grid_size"]
        algorithms = ["ABQLBM"]
        grid_sizes = [4]
        obstacle_counts = [0]
        timesteps = [1]
    else:
        experiments = args.experiments
        algorithms = args.algorithms
        grid_sizes = parse_ints(args.grid_sizes)
        obstacle_counts = parse_ints(args.obstacle_counts)
        timesteps = parse_ints(args.timesteps)

    rows = run_component_sweeps(
        config_path=args.config_path,
        output_dir=args.output_dir,
        experiments=experiments,
        algorithms=algorithms,
        hardware_names=hardware_names,
        grid_sizes=grid_sizes,
        obstacle_counts=obstacle_counts,
        timesteps=timesteps,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
    )
    print(
        "Wrote "
        f"{len(rows)} rows to {args.output_dir / 'component_resource_estimates.csv'}"
    )


if __name__ == "__main__":
    main()
