"""Generate fixed-case component comparison figures and CSV tables."""

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
    QLBM_HARDWARE_ROOT
    / "qlbm-hardware-output"
    / "resource-estimates"
    / "v2"
    / "components-8x8-obs2-generic"
)
DEFAULT_CSV_PATH = DEFAULT_OUTPUT_DIR / "component_resource_estimates.csv"

ALGORITHM_ORDER = ["ABQLBM", "MSQLBM", "SpaceTimeQLBM"]
ALGORITHM_LABELS = {
    "ABQLBM": "ABQLBM",
    "MSQLBM": "MSQLBM",
    "SpaceTimeQLBM": "STQBM",
}
COMPONENT_ORDER = [
    "initialization",
    "streaming",
    "reflection",
    "collision",
    "ancilla_preparation",
]
COMPONENT_LABELS = {
    "initialization": "Initialization",
    "streaming": "Streaming",
    "reflection": "Reflection / boundary",
    "collision": "Collision",
    "ancilla_preparation": "Ancilla prep.",
}
PALETTE = {
    "initialization": "#6F4E7C",
    "streaming": "#4C78A8",
    "reflection": "#F58518",
    "collision": "#54A24B",
    "ancilla_preparation": "#B279A2",
}


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize fixed-case component results."""
    df = df.copy()
    numeric_columns = [
        "transpiled_component_2q_ops",
        "transpiled_component_depth",
        "component_critical_path_time_s",
        "component_serial_time_s",
        "algorithm_scheduled_duration_s",
        "algorithm_transpiled_2q_ops",
        "algorithm_transpiled_depth",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["build_error", "transpile_error", "section_analysis_error"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("")

    df["component_group_label"] = df["component_group"].map(
        COMPONENT_LABELS
    ).fillna(df["component_group"])
    return df


def successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows without build/transpile/section errors."""
    return df[
        df["build_error"].eq("")
        & df["transpile_error"].eq("")
        & df["section_analysis_error"].eq("")
    ].copy()


def component_platform_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated component instances per algorithm/hardware/group."""
    return (
        successful_rows(df)
        .groupby(
            ["algorithm", "hardware_label", "component_group"],
            dropna=False,
        )
        .agg(
            component_2q_ops=("transpiled_component_2q_ops", "sum"),
            component_depth=("transpiled_component_depth", "sum"),
            component_critical_path_time_s=("component_critical_path_time_s", "sum"),
            component_serial_time_s=("component_serial_time_s", "sum"),
        )
        .reset_index()
    )


def median_component_table(component_totals: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return component-group median resources pivoted by algorithm."""
    table = (
        component_totals.groupby(["component_group", "algorithm"], dropna=False)[metric]
        .median()
        .unstack("algorithm")
        .reindex(COMPONENT_ORDER)
        .dropna(how="all")
        .reindex(columns=ALGORITHM_ORDER)
        .fillna(0)
    )
    table.index = [COMPONENT_LABELS.get(index, index) for index in table.index]
    table = table.rename(columns=ALGORITHM_LABELS)
    return table.reset_index(names="Component")


def share_table(
    component_totals: pd.DataFrame,
    metric: str,
    share_column: str = "component_share",
) -> pd.DataFrame:
    """Return median component shares for the selected metric by algorithm."""
    data = component_totals.copy()
    totals = data.groupby(["algorithm", "hardware_label"])[metric].transform("sum")
    data[share_column] = data[metric] / totals
    table = (
        data.groupby(["component_group", "algorithm"], dropna=False)[share_column]
        .median()
        .unstack("algorithm")
        .reindex(COMPONENT_ORDER)
        .dropna(how="all")
        .reindex(columns=ALGORITHM_ORDER)
        .fillna(0)
    )
    table.index = [COMPONENT_LABELS.get(index, index) for index in table.index]
    table = table.rename(columns=ALGORITHM_LABELS)
    return table.reset_index(names="Component")


def write_table(table: pd.DataFrame, tables_dir: Path, name: str) -> List[Path]:
    """Write a table as CSV."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{name}.csv"
    table.to_csv(csv_path, index=False)
    return [csv_path]


def save_median_stacked_bar(
    component_totals: pd.DataFrame,
    figures_dir: Path,
    metric: str,
    ylabel: str,
    filename: str,
) -> List[Path]:
    """Save stacked bar chart of median component resources."""
    median = (
        component_totals.groupby(["algorithm", "component_group"], dropna=False)[metric]
        .median()
        .unstack("component_group")
        .reindex(ALGORITHM_ORDER)
        .fillna(0)
    )
    columns = [column for column in COMPONENT_ORDER if column in median.columns]
    median = median[columns]

    sns.set_theme(style="whitegrid", context="paper")
    figure, ax = plt.subplots(figsize=(7.0, 4.2))
    bottom = pd.Series(0.0, index=median.index)
    for component in columns:
        ax.bar(
            median.index,
            median[component],
            bottom=bottom,
            label=COMPONENT_LABELS.get(component, component),
            color=PALETTE.get(component),
            width=0.65,
        )
        bottom += median[component]

    totals = bottom
    ax.set_yscale("log")
    ax.set_ylim(1, totals.max() * 4.0)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for algorithm, total in totals.items():
        ax.text(
            algorithm,
            total * 1.18,
            f"{total:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(median.index)))
    ax.set_xticklabels([ALGORITHM_LABELS.get(index, index) for index in median.index])
    ax.legend(
        title="Component",
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
    )
    ax.set_title("8x8 lattice, 2 bounce-back obstacles", pad=12)
    figure.tight_layout()
    return save_figure(figure, figures_dir, filename)


def save_share_heatmap(
    component_totals: pd.DataFrame,
    figures_dir: Path,
    metric: str,
    title: str,
    filename: str,
) -> List[Path]:
    """Save heatmap of median component resource shares."""
    table = share_table(component_totals, metric).set_index("Component")
    table = table.rename(columns=ALGORITHM_LABELS)
    sns.set_theme(style="white", context="paper")
    figure, ax = plt.subplots(figsize=(6.5, 3.2))
    sns.heatmap(
        table,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Component")
    ax.set_title(title)
    figure.tight_layout()
    return save_figure(figure, figures_dir, filename)


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


def generate_outputs(
    csv_path: Path = DEFAULT_CSV_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> List[Path]:
    """Generate fixed-case component outputs from a component resource CSV."""
    df = clean_results(pd.read_csv(csv_path))
    component_totals = component_platform_totals(df)
    fixed_dir = output_dir / "fixed-case"
    figures_dir = fixed_dir / "figures"
    tables_dir = fixed_dir / "tables"
    outputs: List[Path] = []

    outputs.extend(
        write_table(
            median_component_table(component_totals, "component_2q_ops"),
            tables_dir,
            "median_component_twoq_by_algorithm",
        )
    )
    outputs.extend(
        write_table(
            median_component_table(component_totals, "component_depth"),
            tables_dir,
            "median_component_depth_by_algorithm",
        )
    )
    outputs.extend(
        write_table(
            share_table(component_totals, "component_2q_ops"),
            tables_dir,
            "median_component_twoq_share_by_algorithm",
        )
    )
    outputs.extend(
        write_table(
            share_table(component_totals, "component_depth"),
            tables_dir,
            "median_component_depth_share_by_algorithm",
        )
    )
    outputs.extend(
        save_median_stacked_bar(
            component_totals,
            figures_dir,
            metric="component_2q_ops",
            ylabel="Transpiled two-qubit gates",
            filename="fixed_case_component_twoq_stacked",
        )
    )
    outputs.extend(
        save_median_stacked_bar(
            component_totals,
            figures_dir,
            metric="component_depth",
            ylabel="Transpiled component depth",
            filename="fixed_case_component_depth_stacked",
        )
    )
    outputs.extend(
        save_share_heatmap(
            component_totals,
            figures_dir,
            metric="component_2q_ops",
            title="Component share of two-qubit gates",
            filename="fixed_case_component_twoq_share_heatmap",
        )
    )
    outputs.extend(
        save_share_heatmap(
            component_totals,
            figures_dir,
            metric="component_depth",
            title="Component share of transpiled depth",
            filename="fixed_case_component_depth_share_heatmap",
        )
    )
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot fixed-case QLBM component comparison results.",
    )
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Generate fixed-case figures and tables from the command line."""
    args = parse_args()
    outputs = generate_outputs(csv_path=args.csv_path, output_dir=args.output_dir)
    print(f"Wrote {len(outputs)} fixed-case files under {args.output_dir / 'fixed-case'}")


if __name__ == "__main__":
    main()
