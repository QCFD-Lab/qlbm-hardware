"""Generate figures and tables from component-level resource estimates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"
DEFAULT_OUTPUT_DIR = (
    QLBM_HARDWARE_ROOT
    / "qlbm-hardware-output"
    / "resource-estimates"
    / "v2"
    / "components"
)
DEFAULT_CSV_PATH = DEFAULT_OUTPUT_DIR / "component_resource_estimates.csv"


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes and add columns used by component plots."""
    df = df.copy()
    numeric_columns = [
        "grid_x",
        "grid_y",
        "grid_points",
        "num_obstacles",
        "num_timesteps",
        "sequence_index",
        "logical_timestep",
        "algorithm_substep",
        "encoded_timestep",
        "dimension",
        "logical_component_depth",
        "logical_component_2q_ops",
        "transpiled_component_depth",
        "transpiled_component_2q_ops",
        "component_critical_path_time_s",
        "component_serial_time_s",
        "algorithm_transpiled_depth",
        "algorithm_transpiled_2q_ops",
        "algorithm_scheduled_duration_s",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["build_error", "transpile_error", "section_analysis_error"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("")

    if "hardware_label" not in df.columns and "hardware" in df.columns:
        df["hardware_label"] = df["hardware"]

    return df


def successful_rows(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return rows with a plotted metric and no build/transpile failure."""
    return df[
        df[metric].notna()
        & df["build_error"].eq("")
        & df["transpile_error"].eq("")
        & df["section_analysis_error"].eq("")
    ].copy()


def aggregate_component_totals(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Sum component instances per case/hardware/group, then median over hardware."""
    per_platform = (
        successful_rows(df, metric)
        .groupby(
            [
                "experiment",
                "algorithm",
                "grid_points",
                "num_obstacles",
                "num_timesteps",
                "component_group",
                "hardware_label",
            ],
            dropna=False,
        )[metric]
        .sum()
        .reset_index()
    )
    return (
        per_platform.groupby(
            [
                "experiment",
                "algorithm",
                "grid_points",
                "num_obstacles",
                "num_timesteps",
                "component_group",
            ],
            dropna=False,
        )[metric]
        .median()
        .reset_index()
    )


def save_component_scaling_plot(
    df: pd.DataFrame,
    experiment: str,
    x: str,
    metric: str,
    filename: str,
    xlabel: str,
    ylabel: str,
    figures_dir: Path,
) -> List[Path]:
    """Save a component-group scaling line plot faceted by algorithm."""
    data = aggregate_component_totals(df[df["experiment"].eq(experiment)], metric)
    if data.empty:
        return []

    sns.set_theme(style="whitegrid", context="paper")
    plot = sns.relplot(
        data=data,
        x=x,
        y=metric,
        hue="component_group",
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
        values = data[metric].dropna()
        if not values.empty and values.max() / max(values.min(), 1) > 100:
            ax.set_yscale("log")
    plot.figure.tight_layout()
    return save_figure(plot.figure, figures_dir, filename)


def save_component_share_plot(df: pd.DataFrame, figures_dir: Path) -> List[Path]:
    """Save component two-qubit share for the smallest successful grid-size case."""
    metric = "transpiled_component_2q_ops"
    data = successful_rows(df[df["experiment"].eq("grid_size")], metric)
    if data.empty:
        return []

    min_grid = data["grid_points"].min()
    data = data[data["grid_points"].eq(min_grid)]
    totals = (
        data.groupby(["algorithm", "component_group"], dropna=False)[metric]
        .sum()
        .reset_index()
    )
    algorithm_totals = totals.groupby("algorithm")[metric].transform("sum")
    totals["component_share"] = totals[metric] / algorithm_totals

    sns.set_theme(style="whitegrid", context="paper")
    figure, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=totals,
        x="algorithm",
        y="component_share",
        hue="component_group",
        ax=ax,
    )
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Share of transpiled two-qubit gates")
    ax.set_ylim(0, 1)
    figure.tight_layout()
    return save_figure(figure, figures_dir, "component_twoq_share_by_algorithm")


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
    """Save compact component summary tables."""
    outputs: List[Path] = []
    metric = "transpiled_component_2q_ops"
    data = successful_rows(df, metric)
    if data.empty:
        return outputs

    bottlenecks = (
        data.groupby(
            ["algorithm", "component_group", "component_kind"],
            dropna=False,
        )
        .agg(
            median_component_2q_ops=(metric, "median"),
            max_component_2q_ops=(metric, "max"),
            median_component_depth=("transpiled_component_depth", "median"),
            max_component_depth=("transpiled_component_depth", "max"),
            cases=("case", "nunique"),
        )
        .reset_index()
        .sort_values(["algorithm", "max_component_2q_ops"], ascending=[True, False])
    )
    outputs.extend(write_table(bottlenecks, tables_dir, "component_bottlenecks"))

    by_group = (
        data.groupby(["experiment", "algorithm", "component_group"], dropna=False)
        .agg(
            total_component_2q_ops=(metric, "sum"),
            total_component_depth=("transpiled_component_depth", "sum"),
            rows=("component_section", "count"),
        )
        .reset_index()
        .sort_values(["experiment", "algorithm", "total_component_2q_ops"])
    )
    outputs.extend(write_table(by_group, tables_dir, "component_totals_by_group"))
    return outputs


def generate_outputs(
    csv_path: Path = DEFAULT_CSV_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> List[Path]:
    """Generate component-level figures and tables from a resource CSV."""
    df = clean_results(pd.read_csv(csv_path))
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    outputs: List[Path] = []
    metric = "transpiled_component_2q_ops"

    outputs.extend(
        save_component_scaling_plot(
            df,
            "grid_size",
            "grid_points",
            metric,
            "component_twoq_vs_grid_size",
            "Grid points",
            "Median transpiled two-qubit gates by component",
            figures_dir,
        )
    )
    outputs.extend(
        save_component_scaling_plot(
            df,
            "obstacle_count",
            "num_obstacles",
            metric,
            "component_twoq_vs_obstacles",
            "Number of obstacles",
            "Median transpiled two-qubit gates by component",
            figures_dir,
        )
    )
    outputs.extend(
        save_component_scaling_plot(
            df,
            "timesteps",
            "num_timesteps",
            metric,
            "component_twoq_vs_timesteps",
            "Number of timesteps",
            "Median transpiled two-qubit gates by component",
            figures_dir,
        )
    )
    outputs.extend(save_component_share_plot(df, figures_dir))
    outputs.extend(save_tables(df, tables_dir))
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot component-level QLBM resource-estimation results.",
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
