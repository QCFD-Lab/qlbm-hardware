"""Generic hardware-mapped QLBM noise run."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from components.resource_estimator import QLBMResourceEstimator
from components.noise_analysis.experiment_utils import (
    build_backend,
    build_case,
    build_run_metadata,
    load_hardware_config,
    require_compatible,
    sample_counts,
    write_json,
)
from components.noise_analysis.qlbm_builders import build_full_logical_circuit


def run_hardware_noise_analysis(
    algorithm_name: str,
    hardware_name: str,
    max_timesteps: int,
    num_shots: int,
    optimization_level: int,
    seed_transpiler: int,
    seed_simulator: int,
    noise_kind: str,
    backend_method: str,
    noise_parameters: dict[str, Any],
    config_path: Path,
    output_root: Path,
) -> Path:
    """Run the full hardware-mapped noise-analysis workflow."""

    if max_timesteps < 1:
        raise ValueError("max_timesteps must be at least 1.")

    final_case = build_case(algorithm_name, max_timesteps)
    hardware_config = load_hardware_config(hardware_name, config_path)
    estimator = QLBMResourceEstimator(hardware_config)
    backend = build_backend(
        noise_kind=noise_kind,
        backend_method=backend_method,
        seed_simulator=seed_simulator,
        noise_parameters=noise_parameters,
        hardware_config=hardware_config,
    )

    output_dir = output_root / f"{final_case.label}_{hardware_name}_{noise_kind}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_metadata.json",
        build_run_metadata(
            algorithm_name=algorithm_name,
            case_label=final_case.label,
            hardware_name=hardware_name,
            max_timesteps=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            hardware_config=hardware_config,
        ),
    )

    qlbm_result = final_case.lattice.create_result(str(output_dir), "step")
    qlbm_result.visualize_geometry()

    step_reports = []
    final_report = None
    for timestep in range(0, max_timesteps + 1):
        case_timesteps = timestep if timestep > 0 else 1
        case = build_case(algorithm_name, case_timesteps)
        logical_steps = case.runner_steps if timestep > 0 else 0
        logical_circuit = build_full_logical_circuit(case, logical_steps)
        report = estimator.estimate(
            logical_circuit,
            label=f"{case.label}_step{timestep}",
            qlbm_metadata={**case.metadata, "simulated_timestep": timestep},
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            transpile_circuit=True,
            phase_polynomial_analysis=False,
        )
        require_compatible(report)

        transpiled_compact_circuit = report["transpiled_compact_circuit"]
        counts = sample_counts(backend, transpiled_compact_circuit, num_shots)

        write_json(output_dir / f"counts_step_{timestep}.json", counts)
        qlbm_result.save_timestep_counts(counts, timestep)
        write_json(output_dir / f"resource_report_step_{timestep}.json", report)
        step_reports.append({"timestep": timestep, "report": report})

        if timestep == max_timesteps:
            final_report = report

    write_json(output_dir / "resource_reports_by_timestep.json", step_reports)
    write_json(output_dir / "resource_report.json", final_report)
    return output_dir
