"""QLBM resource-estimation sweeps using QLBMResourceEstimator."""

from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from qiskit import QuantumCircuit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_SOURCE_ROOT = PROJECT_ROOT / "qlbm"
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qlbm-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/qlbm-cache")

for path in (QLBM_SOURCE_ROOT, QLBM_HARDWARE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.resource_estimator import QLBMResourceEstimator
from qlbm import ABLattice, MSLattice, SpaceTimeLattice
from qlbm.components import (
    ABGridMeasurement,
    ABInitialConditions,
    ABQLBM,
    EmptyPrimitive,
    GridMeasurement,
    MSInitialConditions,
    MSQLBM,
)
from qlbm.components.spacetime import SpaceTimeGridVelocityMeasurement, SpaceTimeQLBM
from qlbm.components.spacetime.initial.pointwise import (
    PointWiseSpaceTimeInitialConditions,
)


def _component_circuit(component: Any) -> QuantumCircuit:
    return component.circuit if hasattr(component, "circuit") else component


def build_full_logical_circuit(
    initial_conditions: Any,
    algorithm: Any,
    postprocessing: Any,
    measurement: Any,
    num_timesteps: int,
) -> QuantumCircuit:
    """Build the measured logical circuit, the entire logical QLBM circuit.."""
    initial_circuit = _component_circuit(initial_conditions)
    algorithm_circuit = _component_circuit(algorithm)
    postprocessing_circuit = _component_circuit(postprocessing)
    measurement_circuit = _component_circuit(measurement)

    circuit = QuantumCircuit(*(measurement_circuit.qregs + measurement_circuit.cregs))
    circuit.compose(
        initial_circuit.copy(),
        inplace=True,
        qubits=range(circuit.num_qubits),
    )
    for _ in range(num_timesteps):
        circuit.compose(algorithm_circuit.copy(), inplace=True)
    circuit.compose(postprocessing_circuit.copy(), inplace=True)
    circuit.compose(measurement_circuit.copy(), inplace=True)
    return circuit


def build_abqlbm_4x4_d2q9(num_timesteps: int):
    lattice_data = {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
        "geometry": [],
    }
    lattice = ABLattice(lattice_data)
    initial_conditions = ABInitialConditions(lattice)
    algorithm = ABQLBM(lattice)
    postprocessing = EmptyPrimitive(lattice)
    measurement = ABGridMeasurement(lattice)
    return {
        "label": f"abqlbm_4x4_d2q9_t{num_timesteps}_no_obstacles",
        "algorithm": "ABQLBM",
        "lattice": lattice_data,
        "num_timesteps": num_timesteps,
        "circuit": build_full_logical_circuit(
            initial_conditions,
            algorithm,
            postprocessing,
            measurement,
            num_timesteps,
        ),
    }


def build_msqlbm_4x4_v4x4(num_timesteps: int):
    lattice_data = {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
        "geometry": [],
    }
    lattice = MSLattice(lattice_data)
    initial_conditions = MSInitialConditions(lattice)
    algorithm = MSQLBM(lattice)
    postprocessing = EmptyPrimitive(lattice)
    measurement = GridMeasurement(lattice)
    return {
        "label": f"msqlbm_4x4_v4x4_t{num_timesteps}_no_obstacles",
        "algorithm": "MSQLBM",
        "lattice": lattice_data,
        "num_timesteps": num_timesteps,
        "circuit": build_full_logical_circuit(
            initial_conditions,
            algorithm,
            postprocessing,
            measurement,
            num_timesteps,
        ),
    }


def build_spacetime_4x4_d2q4(num_timesteps: int):
    lattice_data = {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q4"},
        "geometry": [],
    }
    lattice = SpaceTimeLattice(num_timesteps=num_timesteps, lattice_data=lattice_data)
    initial_conditions = PointWiseSpaceTimeInitialConditions(lattice)
    algorithm = SpaceTimeQLBM(lattice)
    postprocessing = EmptyPrimitive(lattice)
    measurement = SpaceTimeGridVelocityMeasurement(lattice)
    return {
        "label": f"spacetime_4x4_d2q4_t{num_timesteps}_no_obstacles",
        "algorithm": "SpaceTimeQLBM",
        "lattice": lattice_data,
        "num_timesteps": num_timesteps,
        "circuit": build_full_logical_circuit(
            initial_conditions,
            algorithm,
            postprocessing,
            measurement,
            1,
        ),
    }


CASE_BUILDERS = {
    "abqlbm_4x4": build_abqlbm_4x4_d2q9,
    "msqlbm_4x4": build_msqlbm_4x4_v4x4,
    "spacetime_4x4": build_spacetime_4x4_d2q4,
}


def load_hardware_configs(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def select_names(raw_names: List[str], available: Iterable[str]) -> List[str]:
    available_names = list(available)
    if raw_names == ["all"]:
        return available_names
    unknown = sorted(set(raw_names) - set(available_names))
    if unknown:
        raise ValueError(f"Unknown names: {', '.join(unknown)}")
    return raw_names


def json_safe(value: Any) -> Any:
    if hasattr(value, "qasm") or value.__class__.__name__ == "QuantumCircuit":
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def make_csv_row(report: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    logical = report["logical"]
    transpiled = report.get("transpiled") or {}
    simulation = report.get("simulation") or {}
    overheads = report.get("overheads") or {}
    compatibility = report.get("transpiled_compatibility") or {}
    timing = report.get("transpiled_time") or {}
    fidelity = report.get("transpiled_fidelity") or {}

    return {
        "case": case["label"],
        "algorithm": case["algorithm"],
        "hardware": report["hardware"]["id"],
        "logical_qubits": logical["num_qubits"],
        "logical_depth": logical["depth"],
        "logical_size": logical["size"],
        "logical_2q_ops": logical["num_2q_ops"],
        "transpiled_qubits": transpiled.get("num_qubits"),
        "transpiled_active_qubits": transpiled.get("active_qubits"),
        "simulation_qubits": simulation.get("num_qubits"),
        "transpiled_depth": transpiled.get("depth"),
        "transpiled_size": transpiled.get("size"),
        "transpiled_2q_ops": transpiled.get("num_2q_ops"),
        "depth_ratio": overheads.get("depth_ratio"),
        "size_ratio": overheads.get("size_ratio"),
        "two_qubit_gate_ratio": overheads.get("two_qubit_gate_ratio"),
        "compatible": compatibility.get("compatible"),
        "qubit_capacity_ok": compatibility.get("qubit_capacity_ok"),
        "basis_gates_ok": compatibility.get("basis_gates_ok"),
        "coupling_map_ok": compatibility.get("coupling_map_ok"),
        "critical_path_time_s": timing.get("critical_path_time_s"),
        "serial_time_s": timing.get("serial_time_s"),
        "total_success_probability": fidelity.get("total_success_probability"),
        "transpile_error": report.get("transpile_error"),
    }


def run_sweep(
    config_path: Path,
    output_dir: Path,
    case_names: List[str],
    hardware_names: List[str],
    optimization_level: int,
    seed_transpiler: int,
    transpile_circuit: bool,
    num_timesteps: int,
) -> List[Dict[str, Any]]:
    configs = load_hardware_configs(config_path)
    selected_hardware = select_names(hardware_names, configs.keys())
    selected_cases = select_names(case_names, CASE_BUILDERS.keys())

    rows = []
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for case_name in selected_cases:
        case = CASE_BUILDERS[case_name](num_timesteps)
        circuit = case["circuit"]
        metadata = {
            "algorithm": case["algorithm"],
            "lattice": case["lattice"],
            "num_timesteps": case.get("num_timesteps"),
        }

        for hardware_name in selected_hardware:
            estimator = QLBMResourceEstimator(configs[hardware_name])
            report = estimator.estimate(
                circuit,
                label=case["label"],
                qlbm_metadata=metadata,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
                transpile_circuit=transpile_circuit,
            )
            row = make_csv_row(report, case)
            rows.append(row)

            report_path = reports_dir / f"{case['label']}__{hardware_name}.json"
            with report_path.open("w", encoding="utf-8") as file:
                json.dump(json_safe(report), file, indent=2)

    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config_path = QLBM_HARDWARE_ROOT / "components" / "resource_estimator" / "config.json"
    output_dir = QLBM_HARDWARE_ROOT / "qlbm-hardware-output" / "resource-estimates" / "v2"

    # Use ["all"] or choose from: abqlbm_4x4, msqlbm_4x4, spacetime_4x4.
    case_names = ["all"]

    # Use ["all"] or choose keys from config.json, e.g. ["superconducting_google_willow_2024"].
    hardware_names = ["all"]

    optimization_level = 1
    seed_transpiler = 42
    transpile_circuit = True
    num_timesteps = 1

    rows = run_sweep(
        config_path=config_path,
        output_dir=output_dir,
        case_names=case_names,
        hardware_names=hardware_names,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        transpile_circuit=transpile_circuit,
        num_timesteps=num_timesteps,
    )
    csv_path = output_dir / "qlbm_resource_estimates_v2.csv"
    write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
