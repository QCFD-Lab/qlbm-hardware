"""Generate phase-polynomial block statistics for thesis result tables."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
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
from components.resource_estimator.thesis_resource_sweeps import (  # noqa: E402
    DEFAULT_ALGORITHMS,
    build_thesis_case,
    make_case_spec,
)


DEFAULT_OUTPUT_DIR = (
    QLBM_HARDWARE_ROOT
    / "components"
    / "phase_poly_optimizer"
    / "output"
    / "phase-poly-block-statistics"
)
DEFAULT_BASIS_GATES = ["rz", "sx", "x", "cx"]
DEFAULT_COUPLING = "all_to_all"
COUPLING_CHOICES = ["all_to_all", "nearest_neighbor", "2d_grid", "linear_chain"]
DEFAULT_GRID_SIZE = 8
DEFAULT_NUM_OBSTACLES = 1
DEFAULT_NUM_TIMESTEPS = 1
DEFAULT_NUM_QUBITS = 50
DEFAULT_HARDWARE_GRID_ROWS = 5
DEFAULT_HARDWARE_GRID_COLS = 10
DEFAULT_OPTIMIZATION_LEVEL = 0
DEFAULT_SEED_TRANSPILER = 42


FIELDNAMES = [
    "algorithm",
    "grid",
    "num_obstacles",
    "num_timesteps",
    "basis_gates",
    "coupling_type",
    "coupling_params",
    "hardware_qubits",
    "logical_qubits",
    "transpiled_qubits",
    "transpiled_active_qubits",
    "transpiled_depth",
    "transpiled_size",
    "num_detected_blocks",
    "average_block_size",
    "average_active_qubits",
    "circuit_coverage_percent",
    "transpiled_total_rz",
    "transpiled_total_cx",
    "block_total_rz",
    "block_total_cx",
    "largest_block_size",
    "largest_block_active_qubits",
    "largest_block_rz",
    "largest_block_cx",
    "transpile_error",
]


def _normalize_coupling(coupling: str) -> str:
    """Map user-facing coupling names to resource-estimator topology names."""
    if coupling == "nearest_neighbor":
        return "2d_grid"
    return coupling


def hypothetical_cx_config(
    num_qubits: int = DEFAULT_NUM_QUBITS,
    basis_gates: Sequence[str] = DEFAULT_BASIS_GATES,
    coupling: str = DEFAULT_COUPLING,
    hardware_grid_rows: int = DEFAULT_HARDWARE_GRID_ROWS,
    hardware_grid_cols: int = DEFAULT_HARDWARE_GRID_COLS,
) -> Dict[str, Any]:
    """Return the hypothetical CX/RZ hardware config."""
    if coupling not in COUPLING_CHOICES:
        raise ValueError(f"coupling must be one of: {', '.join(COUPLING_CHOICES)}")

    coupling_type = _normalize_coupling(coupling)
    if coupling_type == "2d_grid":
        grid_qubits = hardware_grid_rows * hardware_grid_cols
        if grid_qubits != num_qubits:
            raise ValueError(
                "2D nearest-neighbour hardware grid size must match num_qubits: "
                f"{hardware_grid_rows}x{hardware_grid_cols}={grid_qubits}, "
                f"num_qubits={num_qubits}"
            )
        coupling_params = {
            "rows": hardware_grid_rows,
            "cols": hardware_grid_cols,
        }
        coupling_description = (
            f"{hardware_grid_rows}x{hardware_grid_cols} 2D nearest-neighbour"
        )
    else:
        coupling_params = {"num_qubits": num_qubits}
        coupling_description = coupling_type.replace("_", "-")

    return {
        "id": f"hypothetical_{num_qubits}q_{coupling_type}_cx_ir",
        "architecture": f"hypothetical {coupling_description} CX/RZ intermediate device",
        "device_name": f"Hypothetical {num_qubits}-qubit {coupling_description} CX/RZ",
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


def _safe_percent(numerator: Any, denominator: Any) -> float:
    if not denominator:
        return 0.0
    return 100.0 * float(numerator or 0) / float(denominator)


def make_phase_block_row(
    algorithm: str,
    report: Dict[str, Any],
    grid_size: int,
    num_obstacles: int,
    num_timesteps: int,
    hardware_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten phase-polynomial block analysis into a CSV/LaTeX row."""
    transpiled = report.get("transpiled") or {}
    logical = report.get("logical") or {}
    op_counts = transpiled.get("op_counts") or {}
    analysis = report.get("phase_polynomial_analysis") or {}
    largest_block = analysis.get("largest_block") or {}
    transpiled_size = transpiled.get("size")

    return {
        "algorithm": algorithm,
        "grid": f"{grid_size}x{grid_size}",
        "num_obstacles": num_obstacles,
        "num_timesteps": num_timesteps,
        "basis_gates": " ".join(hardware_config["basis_gates"]),
        "coupling_type": hardware_config["coupling_type"],
        "coupling_params": hardware_config.get("coupling_params", {}),
        "hardware_qubits": hardware_config["num_qubits"],
        "logical_qubits": logical.get("num_qubits"),
        "transpiled_qubits": transpiled.get("num_qubits"),
        "transpiled_active_qubits": transpiled.get("active_qubits"),
        "transpiled_depth": transpiled.get("depth"),
        "transpiled_size": transpiled_size,
        "num_detected_blocks": analysis.get("num_blocks", 0),
        "average_block_size": analysis.get("average_block_instructions", 0.0),
        "average_active_qubits": analysis.get("average_active_qubits", 0.0),
        "circuit_coverage_percent": _safe_percent(
            analysis.get("total_block_instructions", 0),
            transpiled_size,
        ),
        "transpiled_total_rz": op_counts.get("rz", 0),
        "transpiled_total_cx": op_counts.get("cx", 0),
        "block_total_rz": analysis.get("total_rz", 0),
        "block_total_cx": analysis.get("total_cx", 0),
        "largest_block_size": largest_block.get("instruction_count", 0),
        "largest_block_active_qubits": largest_block.get("active_qubit_count", 0),
        "largest_block_rz": largest_block.get("rz_count", 0),
        "largest_block_cx": largest_block.get("cx_count", 0),
        "transpile_error": report.get("transpile_error"),
    }


def run_phase_block_statistics(
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_obstacles: int = DEFAULT_NUM_OBSTACLES,
    num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    num_qubits: int = DEFAULT_NUM_QUBITS,
    coupling: str = DEFAULT_COUPLING,
    hardware_grid_rows: int = DEFAULT_HARDWARE_GRID_ROWS,
    hardware_grid_cols: int = DEFAULT_HARDWARE_GRID_COLS,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
) -> List[Dict[str, Any]]:
    """Run the requested phase-polynomial block statistics experiment."""
    hardware_config = hypothetical_cx_config(
        num_qubits=num_qubits,
        coupling=coupling,
        hardware_grid_rows=hardware_grid_rows,
        hardware_grid_cols=hardware_grid_cols,
    )
    estimator = QLBMResourceEstimator(hardware_config)
    rows: List[Dict[str, Any]] = []

    for algorithm in algorithms:
        spec = make_case_spec(
            "phase_poly_block_statistics",
            algorithm,
            grid_size,
            num_timesteps=num_timesteps,
            num_obstacles=num_obstacles,
            obstacle_boundary="bounceback",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            built_case = build_thesis_case(spec)
        report = estimator.estimate(
            built_case.circuit,
            label=spec.label,
            qlbm_metadata=spec.metadata(),
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            transpile_circuit=True,
            phase_polynomial_analysis=True,
        )
        rows.append(
            make_phase_block_row(
                algorithm,
                report,
                grid_size,
                num_obstacles,
                num_timesteps,
                hardware_config,
            )
        )

    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write block statistics rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    """Escape a small value for a LaTeX table cell."""
    text = "" if value is None else str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def latex_number(value: Any) -> str:
    """Format numeric values for a compact LaTeX table."""
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_latex(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write a thesis-ready transposed LaTeX tabular summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    coupling_label = str(rows[0]["coupling_type"]).replace("_", "-")
    coupling_text = str(rows[0]["coupling_type"]).replace("_", " ")
    coupling_params = rows[0].get("coupling_params") or {}
    if rows[0]["coupling_type"] == "2d_grid" and coupling_params:
        coupling_text = (
            f"{coupling_params.get('rows')}x{coupling_params.get('cols')} "
            "2D nearest-neighbour"
        )
    metric_rows = [
        ("num_detected_blocks", "Blocks"),
        ("average_block_size", "Avg. block size"),
        ("average_active_qubits", "Avg. active qubits"),
        ("circuit_coverage_percent", "Coverage (\\%)"),
        ("transpiled_total_rz", "Total RZ"),
        ("transpiled_total_cx", "Total CX"),
        ("largest_block_rz", "Largest RZ"),
        ("largest_block_cx", "Largest CX"),
    ]
    algorithms = [str(row["algorithm"]) for row in rows]
    alignment = "l" + ("r" * len(rows))

    with path.open("w", encoding="utf-8") as file:
        file.write("\\begin{table}[ht]\n")
        file.write("\\centering\n")
        file.write(f"\\begin{{tabular}}{{{alignment}}}\n")
        file.write("\\hline\n")
        file.write(
            "Metric & "
            + " & ".join(latex_escape(algorithm) for algorithm in algorithms)
            + " \\\\\n"
        )
        file.write("\\hline\n")
        for key, label in metric_rows:
            cells = [label]
            for row in rows:
                cells.append(latex_number(row.get(key)))
            file.write(" & ".join(cells) + " \\\\\n")
        file.write("\\hline\n")
        file.write("\\end{tabular}\n")
        file.write(
            "\\caption{Phase-polynomial block statistics after transpilation "
            "to a hypothetical 50-qubit "
            f"{latex_escape(coupling_text)} "
            "CX/RZ basis for 8x8 QLBM circuits with one bounceback obstacle "
            "and one timestep.}\n"
        )
        file.write(f"\\label{{tab:phase-poly-block-statistics-{coupling_label}}}\n")
        file.write("\\end{table}\n")


def write_outputs(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    coupling: str = DEFAULT_COUPLING,
) -> tuple[Path, Path]:
    """Write CSV and LaTeX outputs."""
    suffix = "" if coupling == "all_to_all" else f"_{coupling}"
    csv_path = output_dir / f"phase_poly_block_statistics_8x8_obs1_t1{suffix}.csv"
    tex_path = output_dir / f"phase_poly_block_statistics_8x8_obs1_t1{suffix}.tex"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    return csv_path, tex_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate phase-polynomial block statistics for QLBM circuits.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--num-obstacles", type=int, default=DEFAULT_NUM_OBSTACLES)
    parser.add_argument("--num-timesteps", type=int, default=DEFAULT_NUM_TIMESTEPS)
    parser.add_argument("--num-qubits", type=int, default=DEFAULT_NUM_QUBITS)
    parser.add_argument(
        "--hardware-grid-rows",
        type=int,
        default=DEFAULT_HARDWARE_GRID_ROWS,
        help="Rows for 2D nearest-neighbour hardware coupling.",
    )
    parser.add_argument(
        "--hardware-grid-cols",
        type=int,
        default=DEFAULT_HARDWARE_GRID_COLS,
        help="Columns for 2D nearest-neighbour hardware coupling.",
    )
    parser.add_argument(
        "--coupling",
        choices=COUPLING_CHOICES,
        default=DEFAULT_COUPLING,
        help="Hypothetical hardware coupling used before block analysis.",
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
    rows = run_phase_block_statistics(
        algorithms=args.algorithms,
        grid_size=args.grid_size,
        num_obstacles=args.num_obstacles,
        num_timesteps=args.num_timesteps,
        num_qubits=args.num_qubits,
        coupling=args.coupling,
        hardware_grid_rows=args.hardware_grid_rows,
        hardware_grid_cols=args.hardware_grid_cols,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
    )
    csv_path, tex_path = write_outputs(rows, args.output_dir, coupling=args.coupling)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote LaTeX table to {tex_path}")


if __name__ == "__main__":
    main()
