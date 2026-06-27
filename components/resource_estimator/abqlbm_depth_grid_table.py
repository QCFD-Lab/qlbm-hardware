"""Generate an ABQLBM transpiled-depth table versus grid size."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from qlbm import ABLattice  # noqa: E402
from qlbm.components import (  # noqa: E402
    ABGridMeasurement,
    ABInitialConditions,
    ABQLBM,
    EmptyPrimitive,
)


DEFAULT_CONFIG_PATH = (
    QLBM_HARDWARE_ROOT / "components" / "resource_estimator" / "config.json"
)
DEFAULT_OUTPUT_DIR = (
    QLBM_HARDWARE_ROOT
    / "qlbm-hardware-output"
    / "resource-estimates"
    / "abqlbm-depth-grid"
)
DEFAULT_GRIDS = [(8, 8), (16, 16), (32, 32)]


def build_abqlbm_case(grid_x: int, grid_y: int, num_timesteps: int = 1):
    """Build full and sectioned ABQLBM circuits for a grid with no obstacles."""
    lattice_data = {
        "lattice": {"dim": {"x": grid_x, "y": grid_y}, "velocities": "d2q9"},
        "geometry": [],
    }
    lattice = ABLattice(lattice_data)
    initial_conditions = ABInitialConditions(lattice)
    algorithm = ABQLBM(lattice)
    postprocessing = EmptyPrimitive(lattice)
    measurement = ABGridMeasurement(lattice)
    sectioned_circuit, section_names = build_sectioned_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        num_timesteps,
    )
    full_circuit = build_full_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        num_timesteps,
    )
    return full_circuit, sectioned_circuit, section_names


def section_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact section-level metrics from an estimator report."""
    section_analysis = report.get("section_analysis") or {}
    max_2q = section_analysis.get("max_two_qubit_gate_section") or {}
    max_time = section_analysis.get("max_critical_path_time_section") or {}
    return {
        "max_2q_section": max_2q.get("section"),
        "max_2q_section_count": max_2q.get("num_2q_ops"),
        "max_time_section": max_time.get("section"),
        "max_time_section_critical_path_s": max_time.get("critical_path_time_s"),
    }


def select_hardware(
    hardware_names: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Resolve selected hardware names."""
    if list(hardware_names) == ["all"]:
        return list(configs)
    unknown = sorted(set(hardware_names) - set(configs))
    if unknown:
        raise ValueError(f"Unknown hardware configs: {', '.join(unknown)}")
    return list(hardware_names)


def estimate_depth_rows(
    config_path: Path = DEFAULT_CONFIG_PATH,
    grids: Sequence[Tuple[int, int]] = DEFAULT_GRIDS,
    hardware_names: Sequence[str] = ("all",),
    optimization_level: int = 1,
    seed_transpiler: int = 42,
) -> List[Dict[str, Any]]:
    """Estimate ABQLBM depth for each grid and hardware platform."""
    configs = load_hardware_configs(config_path)
    selected_hardware = select_hardware(hardware_names, configs)
    rows: List[Dict[str, Any]] = []

    for grid_x, grid_y in grids:
        grid_label = f"{grid_x}x{grid_y}"
        circuit, sectioned_circuit, section_names = build_abqlbm_case(
            grid_x,
            grid_y,
            num_timesteps=1,
        )
        logical = QLBMResourceEstimator({}).extract_metrics(
            circuit,
            include_active_qubits=False,
            include_active_qubit_indices=False,
        )

        for hardware_key in selected_hardware:
            hardware_config = configs[hardware_key]
            estimator = QLBMResourceEstimator(hardware_config)
            report = estimator.estimate(
                circuit,
                label=f"abqlbm_{grid_label}_d2q9_t1_no_obstacles",
                qlbm_metadata={
                    "algorithm": "ABQLBM",
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "grid_points": grid_x * grid_y,
                    "num_obstacles": 0,
                    "num_timesteps": 1,
                },
                sectioned_circuit=sectioned_circuit,
                section_names=section_names,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
                transpile_circuit=True,
            )
            transpiled = report.get("transpiled") or {}
            transpiled_compact = report.get("transpiled_compact") or {}
            compatibility = report.get("transpiled_compatibility") or {}
            timing = report.get("transpiled_time") or {}
            sections = section_summary(report)

            rows.append(
                {
                    "algorithm": "ABQLBM",
                    "grid": grid_label,
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "grid_points": grid_x * grid_y,
                    "num_obstacles": 0,
                    "num_timesteps": 1,
                    "hardware_key": hardware_key,
                    "hardware": hardware_config.get("id", hardware_key),
                    "hardware_label": hardware_config.get("device_name")
                    or hardware_config.get("id", hardware_key),
                    "architecture": hardware_config.get("architecture"),
                    "coupling_type": hardware_config.get("coupling_type"),
                    "hardware_qubits": hardware_config.get("num_qubits"),
                    "logical_qubits": logical["num_qubits"],
                    "transpiled_compact_qubits": transpiled_compact.get("num_qubits"),
                    "logical_gate_count": logical["size"],
                    "transpiled_gate_count": transpiled.get("size"),
                    "logical_two_qubit_count": logical["num_2q_ops"],
                    "transpiled_two_qubit_count": transpiled.get("num_2q_ops"),
                    "logical_circuit_depth": logical["depth"],
                    "transpiled_circuit_depth": transpiled.get("depth"),
                    "transpiled_active_qubits": transpiled.get("active_qubits"),
                    "transpiled_compatible": compatibility.get("compatible"),
                    "transpiled_qubit_capacity_ok": compatibility.get(
                        "qubit_capacity_ok"
                    ),
                    **sections,
                    "scheduled_duration_s": timing.get("scheduled_duration_s"),
                    "transpile_error": report.get("transpile_error"),
                    "report": json_safe(report),
                }
            )

    return rows


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Write CSV and raw JSON outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "abqlbm_depth_vs_grid.csv"

    fieldnames = [
        "algorithm",
        "grid",
        "num_obstacles",
        "num_timesteps",
        "hardware_label",
        "logical_qubits",
        "transpiled_compact_qubits",
        "logical_gate_count",
        "transpiled_gate_count",
        "logical_two_qubit_count",
        "transpiled_two_qubit_count",
        "logical_circuit_depth",
        "transpiled_circuit_depth",
        "max_2q_section",
        "max_2q_section_count",
        "max_time_section",
        "max_time_section_critical_path_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    for row in rows:
        report_path = reports_dir / f"{row['algorithm'].lower()}_{row['grid']}__{row['hardware_key']}.json"
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(row["report"], file, indent=2)

    return csv_path


def parse_grid(value: str) -> Tuple[int, int]:
    """Parse a grid value such as 8x4."""
    raw_x, raw_y = value.lower().split("x", maxsplit=1)
    return int(raw_x), int(raw_y)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create an ABQLBM depth-vs-grid table for transpiled circuits.",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--grids",
        nargs="+",
        default=[f"{x}x{y}" for x, y in DEFAULT_GRIDS],
        help="Grid sizes, e.g. 8x8 16x16 32x32.",
    )
    parser.add_argument(
        "--hardware",
        nargs="+",
        default=["all"],
        help="Hardware config keys, or all.",
    )
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--seed-transpiler", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Generate the ABQLBM depth table."""
    args = parse_args()
    rows = estimate_depth_rows(
        config_path=args.config_path,
        grids=[parse_grid(grid) for grid in args.grids],
        hardware_names=args.hardware,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
    )
    csv_path = write_outputs(rows, args.output_dir)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
