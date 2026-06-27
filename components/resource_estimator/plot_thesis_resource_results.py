"""Generate figures and CSV tables from resource-estimation CSV results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qlbm-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/qlbm-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"
DEFAULT_OUTPUT_DIR = (
    QLBM_HARDWARE_ROOT / "qlbm-hardware-output" / "resource-estimates" / "v2"
)
DEFAULT_CSV_PATH = DEFAULT_OUTPUT_DIR / "experiment_resource_estimates.csv"


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes and add columns used by plots."""
    df = df.copy()
    numeric_columns = [
        "grid_x",
        "grid_y",
        "grid_points",
        "num_obstacles",
        "num_timesteps",
        "logical_depth",
        "logical_2q_ops",
        "transpiled_depth",
        "transpiled_2q_ops",
        "scheduled_duration_s",
        "routing_2q_overhead",
        "routing_added_2q",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["build_error", "transpile_error", "timing_warnings"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("")

    if "hardware_label" not in df.columns:
        df["hardware_label"] = df["hardware"]

    return df


def successful_rows(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return rows with a plotted metric and no build/transpile failure."""
    return df[
        df[metric].notna()
        & df["build_error"].eq("")
        & df["transpile_error"].eq("")
    ].copy()


def save_line_plot(
    df: pd.DataFrame,
    experiment: str,
    x: str,
    y: str,
    filename: str,
    xlabel: str,
    ylabel: str,
    figures_dir: Path,
) -> List[Path]:
    """Save a line plot by algorithm."""
    data = successful_rows(df[df["experiment"].eq(experiment)], y)
    if data.empty:
        return []

    sns.set_theme(style="whitegrid", context="paper")
    plot = sns.relplot(
        data=data,
        x=x,
        y=y,
        hue="hardware_label",
        col="algorithm",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=3.2,
        aspect=1.2,
    )
    plot.set_axis_labels(xlabel, ylabel)
    plot.set_titles("{col_name}")
    for ax in plot.axes.flat:
        values = data[y].dropna()
        if not values.empty and values.max() / max(values.min(), 1) > 100:
            ax.set_yscale("log")
    plot.figure.tight_layout()

    return save_figure(plot.figure, figures_dir, filename)


def save_routing_plot(df: pd.DataFrame, figures_dir: Path) -> List[Path]:
    """Save routing overhead by topology."""
    data = successful_rows(
        df[df["experiment"].eq("routing_overhead")],
        "routing_2q_overhead",
    )
    if data.empty:
        return []

    summary = (
        data.groupby(["algorithm", "coupling_type"], dropna=False)
        .agg(routing_2q_overhead=("routing_2q_overhead", "median"))
        .reset_index()
    )
    sns.set_theme(style="whitegrid", context="paper")
    figure, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=summary,
        x="coupling_type",
        y="routing_2q_overhead",
        hue="algorithm",
        ax=ax,
    )
    ax.set_xlabel("Coupling topology")
    ax.set_ylabel("Two-qubit routing overhead")
    ax.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    return save_figure(figure, figures_dir, "routing_overhead_by_topology")


def save_duration_plot(df: pd.DataFrame, figures_dir: Path) -> List[Path]:
    """Save scheduled duration by hardware platform."""
    data = successful_rows(
        df[df["experiment"].eq("scheduled_duration")],
        "scheduled_duration_s",
    )
    if data.empty:
        return []

    sns.set_theme(style="whitegrid", context="paper")
    plot = sns.catplot(
        data=data,
        x="hardware_label",
        y="scheduled_duration_s",
        hue="algorithm",
        kind="bar",
        height=4.0,
        aspect=1.8,
    )
    plot.set_axis_labels("Hardware platform", "Scheduled duration (s)")
    for ax in plot.axes.flat:
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=25)
    plot.figure.tight_layout()
    return save_figure(plot.figure, figures_dir, "scheduled_duration_by_platform")


def save_figure(figure: plt.Figure, figures_dir: Path, filename: str) -> List[Path]:
    """Save a matplotlib figure as PNG and PDF."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = figures_dir / f"{filename}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def write_table(table: pd.DataFrame, tables_dir: Path, name: str) -> List[Path]:
    """Write a table as CSV."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{name}.csv"
    table.to_csv(csv_path, index=False)
    return [csv_path]


def save_tables(df: pd.DataFrame, tables_dir: Path) -> List[Path]:
    """Save compact resource summary tables."""
    outputs: List[Path] = []

    compatible = df[
        df["experiment"].eq("grid_size")
        & df["build_error"].eq("")
        & df["transpile_error"].eq("")
        & df["transpiled_compatible"].astype(str).eq("True")
    ].copy()
    if not compatible.empty:
        idx = compatible.groupby(["algorithm", "hardware_label"])["grid_points"].idxmax()
        max_grid = compatible.loc[
            idx,
            [
                "algorithm",
                "hardware_label",
                "grid_label",
                "grid_points",
                "transpiled_depth",
                "transpiled_2q_ops",
            ],
        ].sort_values(["algorithm", "grid_points", "hardware_label"])
        outputs.extend(write_table(max_grid, tables_dir, "maximum_compatible_grid_size"))

    failed = df[(df["build_error"].ne("")) | (df["transpile_error"].ne(""))].copy()
    if not failed.empty:
        failed_table = failed[
            [
                "experiment",
                "algorithm",
                "hardware_label",
                "grid_label",
                "num_obstacles",
                "num_timesteps",
                "build_error",
                "transpile_error",
            ]
        ]
        outputs.extend(write_table(failed_table, tables_dir, "failed_cases"))

    durations = df[
        df["experiment"].eq("scheduled_duration")
        & df["scheduled_duration_s"].notna()
        & df["build_error"].eq("")
        & df["transpile_error"].eq("")
    ].copy()
    if not durations.empty:
        duration_table = durations[
            [
                "algorithm",
                "hardware_label",
                "scheduled_duration_s",
                "critical_path_time_s",
                "serial_time_s",
                "timing_warnings",
            ]
        ].sort_values(["algorithm", "scheduled_duration_s"])
        outputs.extend(write_table(duration_table, tables_dir, "scheduled_duration_rankings"))

    routing = df[
        df["experiment"].eq("routing_overhead")
        & df["routing_2q_overhead"].notna()
        & df["build_error"].eq("")
        & df["transpile_error"].eq("")
    ].copy()
    if not routing.empty:
        routing_table = (
            routing.groupby(["algorithm", "coupling_type"], dropna=False)
            .agg(
                median_routing_2q_overhead=("routing_2q_overhead", "median"),
                min_routing_2q_overhead=("routing_2q_overhead", "min"),
                max_routing_2q_overhead=("routing_2q_overhead", "max"),
                median_routing_added_2q=("routing_added_2q", "median"),
                platforms=("hardware_label", "nunique"),
            )
            .reset_index()
            .sort_values(["algorithm", "coupling_type"])
        )
        outputs.extend(write_table(routing_table, tables_dir, "routing_overhead_by_topology"))

    return outputs


def generate_outputs(
    csv_path: Path = DEFAULT_CSV_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> List[Path]:
    """Generate all figures and CSV tables from a resource CSV."""
    df = clean_results(pd.read_csv(csv_path))
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    outputs: List[Path] = []

    outputs.extend(
        save_line_plot(
            df,
            "grid_size",
            "grid_points",
            "transpiled_depth",
            "depth_vs_grid_size",
            "Grid points",
            "Transpiled depth",
            figures_dir,
        )
    )
    outputs.extend(
        save_line_plot(
            df,
            "grid_size",
            "grid_points",
            "transpiled_2q_ops",
            "twoq_vs_grid_size",
            "Grid points",
            "Transpiled two-qubit gates",
            figures_dir,
        )
    )
    outputs.extend(
        save_line_plot(
            df,
            "obstacle_count",
            "num_obstacles",
            "transpiled_2q_ops",
            "twoq_vs_obstacles",
            "Number of obstacles",
            "Transpiled two-qubit gates",
            figures_dir,
        )
    )
    outputs.extend(
        save_line_plot(
            df,
            "timesteps",
            "num_timesteps",
            "transpiled_2q_ops",
            "twoq_vs_timesteps",
            "Number of timesteps",
            "Transpiled two-qubit gates",
            figures_dir,
        )
    )
    outputs.extend(save_routing_plot(df, figures_dir))
    outputs.extend(save_duration_plot(df, figures_dir))
    outputs.extend(save_tables(df, tables_dir))
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot QLBM resource-estimation results.",
    )
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Generate plots and tables from the command line."""
    args = parse_args()
    outputs = generate_outputs(csv_path=args.csv_path, output_dir=args.output_dir)
    print(f"Wrote {len(outputs)} plot/table files under {args.output_dir}")


if __name__ == "__main__":
    main()
