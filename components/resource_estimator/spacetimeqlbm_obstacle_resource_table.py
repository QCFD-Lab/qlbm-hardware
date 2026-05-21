"""Generate SpaceTimeQLBM resource tables versus bounceback obstacle count."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
from components.resource_estimator.thesis_resource_sweeps import (  # noqa: E402
    deterministic_obstacles,
)
from qlbm import SpaceTimeLattice  # noqa: E402
from qlbm.components import EmptyPrimitive  # noqa: E402
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
    QLBM_HARDWARE_ROOT
    / "qlbm-hardware-output"
    / "resource-estimates"
    / "spacetimeqlbm-obstacle-resources"
)
DEFAULT_GRID_SIZE = 16
DEFAULT_OBSTACLE_COUNTS = [0, 1, 2, 3, 4]


def build_spacetimeqlbm_obstacle_case(
    grid_size: int,
    num_obstacles: int,
    num_timesteps: int = 1,
):
    """Build full and sectioned SpaceTimeQLBM circuits for bounceback obstacles."""
    lattice_data = {
        "lattice": {
            "dim": {"x": grid_size, "y": grid_size},
            "velocities": "D2Q4",
        },
        "geometry": list(
            deterministic_obstacles(
                grid_size,
                grid_size,
                num_obstacles,
                boundary="bounceback",
            )
        ),
    }
    lattice = SpaceTimeLattice(
        num_timesteps=num_timesteps,
        lattice_data=lattice_data,
    )
    initial_conditions = PointWiseSpaceTimeInitialConditions(lattice)
    algorithm = SpaceTimeQLBM(lattice)
    postprocessing = EmptyPrimitive(lattice)
    measurement = SpaceTimeGridVelocityMeasurement(lattice)

    sectioned_circuit, section_names = build_sectioned_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        1,
    )
    full_circuit = build_full_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        1,
    )
    return full_circuit, sectioned_circuit, section_names, lattice_data


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


def estimate_obstacle_rows(
    config_path: Path = DEFAULT_CONFIG_PATH,
    grid_size: int = DEFAULT_GRID_SIZE,
    obstacle_counts: Sequence[int] = DEFAULT_OBSTACLE_COUNTS,
    hardware_names: Sequence[str] = ("all",),
    optimization_level: int = 1,
    seed_transpiler: int = 42,
) -> List[Dict[str, Any]]:
    """Estimate SpaceTimeQLBM resources for obstacle-count variation."""
    configs = load_hardware_configs(config_path)
    selected_hardware = select_hardware(hardware_names, configs)
    rows: List[Dict[str, Any]] = []

    for num_obstacles in obstacle_counts:
        circuit, sectioned_circuit, section_names, lattice_data = (
            build_spacetimeqlbm_obstacle_case(
                grid_size,
                num_obstacles,
                num_timesteps=1,
            )
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
                label=(
                    f"spacetimeqlbm_{grid_size}x{grid_size}_"
                    f"obs{num_obstacles}_t1"
                ),
                qlbm_metadata={
                    "algorithm": "SpaceTimeQLBM",
                    "grid_x": grid_size,
                    "grid_y": grid_size,
                    "grid_points": grid_size * grid_size,
                    "num_obstacles": num_obstacles,
                    "num_timesteps": 1,
                    "lattice": lattice_data,
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
            rows.append(
                {
                    "algorithm": "SpaceTimeQLBM",
                    "grid": f"{grid_size}x{grid_size}",
                    "num_obstacles": num_obstacles,
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
                    **section_summary(report),
                    "transpile_error": report.get("transpile_error"),
                    "report": json_safe(report),
                }
            )
    return rows


def table_fieldnames() -> List[str]:
    """Return the thesis-facing CSV fields."""
    return [
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


def latex_cell(value: Any) -> str:
    """Return a simple LaTeX table cell."""
    if value is None:
        return ""
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("_", "\\_")
    )


def latex_number(value: Any) -> str:
    """Return a compact LaTeX numeric cell."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    """Write thesis-facing CSV rows."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = table_fieldnames()
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_latex(rows: List[Dict[str, Any]], tex_path: Path) -> None:
    """Write a portrait pivoted LaTeX table."""
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    obstacle_counts = []
    platforms = []
    lookup = {}
    for row in rows:
        label = str(row["num_obstacles"])
        if label not in obstacle_counts:
            obstacle_counts.append(label)
        if row["hardware_label"] not in platforms:
            platforms.append(row["hardware_label"])
        lookup[(row["hardware_label"], label)] = row

    metrics = [
        ("Logical qubits", "logical_qubits"),
        ("Compact qubits", "transpiled_compact_qubits"),
        ("Logical gates", "logical_gate_count"),
        ("Transpiled gates", "transpiled_gate_count"),
        ("Logical 2q", "logical_two_qubit_count"),
        ("Transpiled 2q", "transpiled_two_qubit_count"),
        ("Logical depth", "logical_circuit_depth"),
        ("Transpiled depth", "transpiled_circuit_depth"),
        ("Max 2q section", "max_2q_section"),
        ("Max 2q section count", "max_2q_section_count"),
        ("Max time section", "max_time_section"),
        ("Max section time (s)", "max_time_section_critical_path_s"),
    ]

    with tex_path.open("w", encoding="utf-8") as file:
        file.write("\\begin{table}\n")
        file.write("\\centering\n")
        file.write("\\scriptsize\n")
        file.write("\\begin{tabular}{ll" + ("r" * len(obstacle_counts)) + "}\n")
        file.write("\\hline\n")
        file.write(
            "Platform & Metric & "
            + " & ".join(latex_cell(f"{count} obs") for count in obstacle_counts)
            + " \\\\\n"
        )
        file.write("\\hline\n")
        for platform in platforms:
            for metric_label, metric_key in metrics:
                values = [
                    lookup.get((platform, count), {}).get(metric_key)
                    for count in obstacle_counts
                ]
                formatted_values = [
                    latex_cell(value)
                    if isinstance(value, str)
                    else latex_number(value)
                    for value in values
                ]
                file.write(
                    f"{latex_cell(platform)} & {latex_cell(metric_label)} & "
                    + " & ".join(formatted_values)
                    + " \\\\\n"
                )
            file.write("\\hline\n")
        file.write("\\end{tabular}\n")
        file.write(
            "\\caption{SpaceTimeQLBM resource estimates for a fixed grid with "
            "varying bounceback obstacle count.}\n"
        )
        file.write("\\label{tab:spacetimeqlbm-obstacle-resources}\n")
        file.write("\\end{table}\n")


def write_reports(rows: List[Dict[str, Any]], reports_dir: Path) -> None:
    """Write raw JSON estimator reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        report_path = (
            reports_dir
            / (
                f"spacetimeqlbm_{row['grid']}_obs{row['num_obstacles']}__"
                f"{row['hardware_key']}.json"
            )
        )
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(row["report"], file, indent=2)


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path):
    """Write CSV, LaTeX, and raw JSON outputs."""
    csv_path = output_dir / "spacetimeqlbm_obstacle_resources.csv"
    tex_path = output_dir / "spacetimeqlbm_obstacle_resources.tex"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    write_reports(rows, output_dir / "reports")
    return csv_path, tex_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a SpaceTimeQLBM obstacle-count resource table.",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument(
        "--obstacle-counts",
        nargs="+",
        type=int,
        default=DEFAULT_OBSTACLE_COUNTS,
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
    """Generate the SpaceTimeQLBM obstacle-count resource table."""
    args = parse_args()
    rows = estimate_obstacle_rows(
        config_path=args.config_path,
        grid_size=args.grid_size,
        obstacle_counts=args.obstacle_counts,
        hardware_names=args.hardware,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
    )
    csv_path, tex_path = write_outputs(rows, args.output_dir)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote LaTeX table to {tex_path}")


if __name__ == "__main__":
    main()
