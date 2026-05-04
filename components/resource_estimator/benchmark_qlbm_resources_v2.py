"""Small QLBM resource-estimation sweeps using QLBMResourceEstimator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
from qlbm.components import ABQLBM, MSQLBM
from qlbm.components.spacetime import SpaceTimeQLBM


DEFAULT_CONFIG_PATH = (
    QLBM_HARDWARE_ROOT / "components" / "resource_estimator" / "config.json"
)
DEFAULT_OUTPUT_DIR = (
    QLBM_HARDWARE_ROOT / "qlbm-hardware-output" / "resource-estimates" / "v2"
)


def build_abqlbm_4x4_d2q9():
    lattice_data = {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
        "geometry": [],
    }
    lattice = ABLattice(lattice_data)
    return {
        "label": "abqlbm_4x4_d2q9_no_obstacles",
        "algorithm": "ABQLBM",
        "lattice": lattice_data,
        "circuit": ABQLBM(lattice).circuit,
    }


def build_msqlbm_4x4_v4x4():
    lattice_data = {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
        "geometry": [],
    }
    lattice = MSLattice(lattice_data)
    return {
        "label": "msqlbm_4x4_v4x4_no_obstacles",
        "algorithm": "MSQLBM",
        "lattice": lattice_data,
        "circuit": MSQLBM(lattice).circuit,
    }


def build_spacetime_4x4_d2q4_t1():
    lattice_data = {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q4"},
        "geometry": [],
    }
    lattice = SpaceTimeLattice(num_timesteps=1, lattice_data=lattice_data)
    return {
        "label": "spacetime_4x4_d2q4_t1_no_obstacles",
        "algorithm": "SpaceTimeQLBM",
        "lattice": lattice_data,
        "num_timesteps": 1,
        "circuit": SpaceTimeQLBM(lattice).circuit,
    }


CASE_BUILDERS = {
    "abqlbm_4x4": build_abqlbm_4x4_d2q9,
    "msqlbm_4x4": build_msqlbm_4x4_v4x4,
    "spacetime_4x4": build_spacetime_4x4_d2q4_t1,
}


def load_hardware_configs(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_names(raw_names: List[str], available: Iterable[str]) -> List[str]:
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
    overheads = report.get("overheads") or {}
    compatibility = report.get("transpiled_compatibility") or {}
    timing = report.get("transpiled_time") or {}
    fidelity = report.get("transpiled_fidelity") or {}
    coherence = report.get("transpiled_coherence") or {}

    return {
        "case": case["label"],
        "algorithm": case["algorithm"],
        "hardware": report["hardware"]["id"],
        "logical_qubits": logical["num_qubits"],
        "logical_depth": logical["depth"],
        "logical_size": logical["size"],
        "logical_2q_ops": logical["num_2q_ops"],
        "transpiled_qubits": transpiled.get("num_qubits"),
        "transpiled_depth": transpiled.get("depth"),
        "transpiled_size": transpiled.get("size"),
        "transpiled_2q_ops": transpiled.get("num_2q_ops"),
        "depth_ratio": overheads.get("depth_ratio"),
        "size_ratio": overheads.get("size_ratio"),
        "two_qubit_gate_ratio": overheads.get("two_qubit_gate_ratio"),
        "compatible": compatibility.get("compatible"),
        "qubit_fit": compatibility.get("qubit_fit"),
        "basis_gates_ok": compatibility.get("basis_gates_ok"),
        "coupling_map_ok": compatibility.get("coupling_map_ok"),
        "critical_path_time_s": timing.get("critical_path_time_s"),
        "serial_time_s": timing.get("serial_time_s"),
        "total_success_probability": fidelity.get("total_success_probability"),
        "duration_over_t1": coherence.get("duration_over_t1"),
        "duration_over_t2": coherence.get("duration_over_t2"),
        "transpile_error": report.get("transpile_error"),
    }


def run_sweep(args: argparse.Namespace) -> List[Dict[str, Any]]:
    configs = load_hardware_configs(args.config)
    hardware_names = parse_names(args.hardware, configs.keys())
    case_names = parse_names(args.cases, CASE_BUILDERS.keys())

    rows = []
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for case_name in case_names:
        case = CASE_BUILDERS[case_name]()
        circuit = case["circuit"]
        metadata = {
            "algorithm": case["algorithm"],
            "lattice": case["lattice"],
            "num_timesteps": case.get("num_timesteps"),
        }

        for hardware_name in hardware_names:
            estimator = QLBMResourceEstimator(configs[hardware_name])
            report = estimator.estimate(
                circuit,
                label=case["label"],
                qlbm_metadata=metadata,
                optimization_level=args.optimization_level,
                seed_transpiler=args.seed_transpiler,
                transpile_circuit=not args.no_transpile,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["all"],
        choices=["all", *CASE_BUILDERS.keys()],
    )
    parser.add_argument("--hardware", nargs="+", default=["all"])
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--seed-transpiler", type=int, default=42)
    parser.add_argument("--no-transpile", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_sweep(args)
    csv_path = args.output_dir / "qlbm_resource_estimates_v2.csv"
    write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
