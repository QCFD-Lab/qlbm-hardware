"""Plot QLBM transpiled resources versus timestep count."""

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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    QLBM_HARDWARE_ROOT
    / "qlbm-hardware-output"
    / "resource-estimates"
    / "qlbm-timestep-resources"
)
DEFAULT_ALGORITHMS = ["ABQLBM", "MSQLBM", "SpaceTimeQLBM"]
DEFAULT_GRID = (16, 16)
DEFAULT_NUM_OBSTACLES = 2
DEFAULT_TIMESTEPS = [1, 2, 4]


def build_case(
    algorithm_name: str,
    grid_x: int,
    grid_y: int,
    num_obstacles: int,
    num_timesteps: int,
):
    """Build full and sectioned QLBM circuits for the timestep sweep."""
    geometry = list(
        deterministic_obstacles(
            grid_x,
            grid_y,
            num_obstacles,
            boundary="bounceback",
        )
    )

    if algorithm_name == "ABQLBM":
        lattice_data = {
            "lattice": {
                "dim": {"x": grid_x, "y": grid_y},
                "velocities": "d2q9",
            },
            "geometry": geometry,
        }
        lattice = ABLattice(lattice_data)
        initial_conditions = ABInitialConditions(lattice)
        algorithm = ABQLBM(lattice)
        postprocessing = EmptyPrimitive(lattice)
        measurement = ABGridMeasurement(lattice)
        composition_timesteps = num_timesteps
    elif algorithm_name == "MSQLBM":
        lattice_data = {
            "lattice": {
                "dim": {"x": grid_x, "y": grid_y},
                "velocities": {"x": 4, "y": 4},
            },
            "geometry": geometry,
        }
        lattice = MSLattice(lattice_data)
        initial_conditions = MSInitialConditions(lattice)
        algorithm = MSQLBM(lattice)
        postprocessing = EmptyPrimitive(lattice)
        measurement = GridMeasurement(lattice)
        composition_timesteps = num_timesteps
    elif algorithm_name == "SpaceTimeQLBM":
        lattice_data = {
            "lattice": {
                "dim": {"x": grid_x, "y": grid_y},
                "velocities": "D2Q4",
            },
            "geometry": geometry,
        }
        lattice = SpaceTimeLattice(
            num_timesteps=num_timesteps,
            lattice_data=lattice_data,
        )
        initial_conditions = PointWiseSpaceTimeInitialConditions(lattice)
        algorithm = SpaceTimeQLBM(lattice)
        postprocessing = EmptyPrimitive(lattice)
        measurement = SpaceTimeGridVelocityMeasurement(lattice)
        composition_timesteps = 1
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    sectioned_circuit, section_names = build_sectioned_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        composition_timesteps,
    )
    full_circuit = build_full_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        composition_timesteps,
    )
    return full_circuit, sectioned_circuit, section_names, lattice_data


def select_names(raw_names: Sequence[str], available: Sequence[str]) -> List[str]:
    """Resolve explicit names or ``all`` against available names."""
    if list(raw_names) == ["all"]:
        return list(available)
    unknown = sorted(set(raw_names) - set(available))
    if unknown:
        raise ValueError(f"Unknown names: {', '.join(unknown)}")
    return list(raw_names)


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


def estimate_rows(
    config_path: Path = DEFAULT_CONFIG_PATH,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    hardware_names: Sequence[str] = ("all",),
    timesteps: Sequence[int] = DEFAULT_TIMESTEPS,
    grid_x: int = DEFAULT_GRID[0],
    grid_y: int = DEFAULT_GRID[1],
    num_obstacles: int = DEFAULT_NUM_OBSTACLES,
    optimization_level: int = 1,
    seed_transpiler: int = 42,
) -> List[Dict[str, Any]]:
    """Estimate timestep-sweep resource rows."""
    configs = load_hardware_configs(config_path)
    selected_algorithms = select_names(algorithms, DEFAULT_ALGORITHMS)
    selected_hardware = select_names(hardware_names, list(configs))
    rows: List[Dict[str, Any]] = []

    for algorithm_name in selected_algorithms:
        for num_timesteps in timesteps:
            circuit, sectioned_circuit, section_names, lattice_data = build_case(
                algorithm_name,
                grid_x,
                grid_y,
                num_obstacles,
                num_timesteps,
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
                        f"{algorithm_name.lower()}_{grid_x}x{grid_y}_"
                        f"obs{num_obstacles}_t{num_timesteps}"
                    ),
                    qlbm_metadata={
                        "algorithm": algorithm_name,
                        "grid_x": grid_x,
                        "grid_y": grid_y,
                        "grid_points": grid_x * grid_y,
                        "num_obstacles": num_obstacles,
                        "num_timesteps": num_timesteps,
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
                timing = report.get("transpiled_time") or {}
                rows.append(
                    {
                        "algorithm": algorithm_name,
                        "grid": f"{grid_x}x{grid_y}",
                        "num_obstacles": num_obstacles,
                        "num_timesteps": num_timesteps,
                        "hardware_key": hardware_key,
                        "hardware": hardware_config.get("id", hardware_key),
                        "hardware_label": hardware_config.get("device_name")
                        or hardware_config.get("id", hardware_key),
                        "architecture": hardware_config.get("architecture"),
                        "coupling_type": hardware_config.get("coupling_type"),
                        "hardware_qubits": hardware_config.get("num_qubits"),
                        "logical_qubits": logical["num_qubits"],
                        "transpiled_compact_qubits": transpiled_compact.get(
                            "num_qubits"
                        ),
                        "logical_gate_count": logical["size"],
                        "transpiled_gate_count": transpiled.get("size"),
                        "logical_two_qubit_count": logical["num_2q_ops"],
                        "transpiled_two_qubit_count": transpiled.get("num_2q_ops"),
                        "logical_circuit_depth": logical["depth"],
                        "transpiled_circuit_depth": transpiled.get("depth"),
                        "scheduled_duration_s": timing.get("scheduled_duration_s"),
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


def csv_fieldnames() -> List[str]:
    """Return CSV fields for the timestep sweep."""
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
        "scheduled_duration_s",
        "max_2q_section",
        "max_2q_section_count",
        "max_time_section",
        "max_time_section_critical_path_s",
        "transpile_error",
    ]


def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    """Write timestep-sweep CSV rows."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames()
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_reports(rows: List[Dict[str, Any]], reports_dir: Path) -> None:
    """Write raw JSON estimator reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        report_path = (
            reports_dir
            / (
                f"{row['algorithm'].lower()}_{row['grid']}_"
                f"obs{row['num_obstacles']}_t{row['num_timesteps']}__"
                f"{row['hardware_key']}.json"
            )
        )
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(row["report"], file, indent=2)


def successful_plot_rows(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    """Return rows that can be plotted for a metric."""
    return [
        row
        for row in rows
        if row.get(metric) is not None and not row.get("transpile_error")
    ]


def save_metric_plot(
    rows: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> List[Path]:
    """Save a one-panel-per-algorithm line plot."""
    data = successful_plot_rows(rows, metric)
    if not data:
        return []

    algorithms = [algorithm for algorithm in DEFAULT_ALGORITHMS if any(
        row["algorithm"] == algorithm for row in data
    )]
    hardware_labels = []
    for row in data:
        if row["hardware_label"] not in hardware_labels:
            hardware_labels.append(row["hardware_label"])

    fig, axes = plt.subplots(
        1,
        len(algorithms),
        figsize=(5.2 * len(algorithms), 4.0),
        sharey=False,
    )
    if len(algorithms) == 1:
        axes = [axes]

    for ax, algorithm_name in zip(axes, algorithms):
        for hardware_label in hardware_labels:
            series = sorted(
                [
                    row
                    for row in data
                    if row["algorithm"] == algorithm_name
                    and row["hardware_label"] == hardware_label
                ],
                key=lambda row: row["num_timesteps"],
            )
            if not series:
                continue
            ax.plot(
                [row["num_timesteps"] for row in series],
                [row[metric] for row in series],
                marker="o",
                linewidth=1.5,
                label=hardware_label,
            )
        ax.set_title(algorithm_name)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(ylabel)
        values = [row[metric] for row in data if row["algorithm"] == algorithm_name]
        if values and max(values) / max(min(values), 1) > 50:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{filename}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path):
    """Write CSV, raw reports, and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "qlbm_timestep_resources.csv"
    write_csv(rows, csv_path)
    write_reports(rows, output_dir / "reports")
    plot_paths = []
    plot_paths.extend(
        save_metric_plot(
            rows,
            "transpiled_two_qubit_count",
            "Transpiled two-qubit gates",
            "transpiled_two_qubit_count_vs_timesteps",
            output_dir,
        )
    )
    plot_paths.extend(
        save_metric_plot(
            rows,
            "transpiled_circuit_depth",
            "Transpiled circuit depth",
            "transpiled_depth_vs_timesteps",
            output_dir,
        )
    )
    return csv_path, plot_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot QLBM transpiled resources versus timestep count.",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        choices=DEFAULT_ALGORITHMS,
    )
    parser.add_argument(
        "--hardware",
        nargs="+",
        default=["all"],
        help="Hardware config keys, or all.",
    )
    parser.add_argument(
        "--grid",
        default=f"{DEFAULT_GRID[0]}x{DEFAULT_GRID[1]}",
        help="Grid dimensions, e.g. 8x4 or 16x16.",
    )
    parser.add_argument("--num-obstacles", type=int, default=DEFAULT_NUM_OBSTACLES)
    parser.add_argument(
        "--timesteps",
        nargs="+",
        type=int,
        default=DEFAULT_TIMESTEPS,
    )
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--seed-transpiler", type=int, default=42)
    return parser.parse_args()


def parse_grid(value: str) -> tuple[int, int]:
    """Parse a grid value such as 8x4."""
    raw_x, raw_y = value.lower().split("x", maxsplit=1)
    return int(raw_x), int(raw_y)


def main() -> None:
    """Generate the timestep-sweep plots."""
    args = parse_args()
    grid_x, grid_y = parse_grid(args.grid)
    rows = estimate_rows(
        config_path=args.config_path,
        algorithms=args.algorithms,
        hardware_names=args.hardware,
        timesteps=args.timesteps,
        grid_x=grid_x,
        grid_y=grid_y,
        num_obstacles=args.num_obstacles,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
    )
    csv_path, plot_paths = write_outputs(rows, args.output_dir)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    for plot_path in plot_paths:
        print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
