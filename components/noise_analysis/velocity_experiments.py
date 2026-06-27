"""Velocity-profile QLBM noise experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from components.resource_estimator import QLBMResourceEstimator
from components.noise_analysis.experiment_utils import (
    _hardware_plot_label,
    build_backend,
    build_case,
    build_run_metadata,
    load_hardware_config,
    load_hardware_configs_with_overrides,
    require_hardware_names,
    require_compatible,
    safe_path_name,
    sample_counts,
    write_lattice_json,
    write_json,
)
from components.noise_analysis.qlbm_builders import build_full_logical_circuit
from components.noise_analysis.tools.velocity_analysis import (
    analyze_velocity_counts,
    compare_velocity_fields,
)
from components.noise_analysis.velocity_outputs import (
    save_multi_hardware_velocity_metrics,
    save_multi_hardware_velocity_x_outputs,
)


def run_noiseless_vs_noise_velocity_profile_comparison(
    algorithm_name: str,
    hardware_name: str,
    final_timestep: int,
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
    """Run a final velocity-resolved measurement and compare u_x/u_y profiles."""

    if final_timestep < 1:
        raise ValueError("final_timestep must be at least 1.")
    if noise_kind == "none":
        raise ValueError(
            "Choose a non-'none' noise_kind for velocity-profile comparison."
        )

    hardware_config = load_hardware_config(hardware_name, config_path)
    estimator = QLBMResourceEstimator(hardware_config)
    case = build_case(
        algorithm_name,
        final_timestep,
        measure_velocity_qubits=True,
    )

    output_dir = (
        output_root
        / "velocity_profile_comparisons"
        / f"{case.label}_{hardware_name}_{noise_kind}_vs_none"
    )
    baseline_dir = output_dir / "baseline_none"
    noisy_dir = output_dir / f"noisy_{noise_kind}"
    analysis_dir = output_dir / "analysis"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logical_circuit = build_full_logical_circuit(case, case.runner_steps)
    report = estimator.estimate(
        logical_circuit,
        label=f"{case.label}_velocity_profile_comparison",
        qlbm_metadata={**case.metadata, "simulated_timestep": final_timestep},
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        transpile_circuit=True,
        phase_polynomial_analysis=False,
    )
    require_compatible(report)
    transpiled_compact_circuit = report["transpiled_compact_circuit"]

    write_json(
        output_dir / "comparison_metadata.json",
        {
            "analysis_kind": "final_velocity_profile_comparison",
            "algorithm_name": algorithm_name,
            "case_label": case.label,
            "hardware_name": hardware_name,
            "hardware_id": hardware_config.get("id"),
            "final_timestep": final_timestep,
            "num_shots": num_shots,
            "optimization_level": optimization_level,
            "seed_transpiler": seed_transpiler,
            "seed_simulator": seed_simulator,
            "backend_method": backend_method,
            "baseline_noise_kind": "none",
            "selected_noise_kind": noise_kind,
            "measurement": "ABGridMeasurement(measure_velocity_qubits=True)",
            "hardware_gate_fidelities": hardware_config.get("gate_fidelities", {}),
            "hardware_gate_times_s": hardware_config.get("gate_times_s", {}),
            "measurement_fidelity": hardware_config.get("measurement_fidelity"),
            "coherence": hardware_config.get("coherence", {}),
            "noise_parameters": noise_parameters
            if noise_kind == "depolarizing"
            else {},
        },
    )
    write_lattice_json(output_dir / "lattice.json", case.lattice)

    run_counts = {}
    for run_noise_kind, run_dir in {
        "none": baseline_dir,
        noise_kind: noisy_dir,
    }.items():
        backend = build_backend(
            noise_kind=run_noise_kind,
            backend_method=backend_method,
            seed_simulator=seed_simulator,
            noise_parameters=noise_parameters,
            hardware_config=hardware_config,
        )
        counts = sample_counts(backend, transpiled_compact_circuit, num_shots)
        run_counts[run_noise_kind] = counts
        write_json(run_dir / f"counts_step_{final_timestep}.json", counts)
        write_json(
            run_dir / "run_metadata.json",
            build_run_metadata(
                algorithm_name=algorithm_name,
                case_label=case.label,
                hardware_name=hardware_name,
                max_timesteps=final_timestep,
                num_shots=num_shots,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
                seed_simulator=seed_simulator,
                noise_kind=run_noise_kind,
                backend_method=backend_method,
                noise_parameters=noise_parameters,
                hardware_config=hardware_config,
                simulated_timesteps=[final_timestep],
            ),
        )

    analyze_velocity_counts(
        run_counts["none"],
        run_counts[noise_kind],
        case.lattice,
        analysis_dir,
        step=final_timestep,
        baseline_label="noiseless",
        noisy_label=noise_kind,
    )
    write_json(output_dir / "resource_report.json", report)
    return output_dir


def run_velocity_profile_comparison_batch(
    algorithm_name: str,
    hardware_names: list[str],
    final_timestep: int,
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
    """Run standard velocity-profile comparison plots for several hardware configs."""

    if not hardware_names:
        raise ValueError("hardware_names must contain at least one hardware config.")

    batch_dir = (
        output_root
        / "velocity_profile_comparison_batches"
        / (
            f"{algorithm_name}_t{final_timestep}_"
            f"{'_vs_'.join(safe_path_name(name) for name in hardware_names)}_"
            f"{noise_kind}_vs_none"
        )
    )
    batch_dir.mkdir(parents=True, exist_ok=True)

    output_dirs = []
    for hardware_name in hardware_names:
        output_dirs.append(
            run_noiseless_vs_noise_velocity_profile_comparison(
                algorithm_name=algorithm_name,
                hardware_name=hardware_name,
                final_timestep=final_timestep,
                num_shots=num_shots,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
                seed_simulator=seed_simulator,
                noise_kind=noise_kind,
                backend_method=backend_method,
                noise_parameters=noise_parameters,
                config_path=config_path,
                output_root=batch_dir,
            )
        )

    write_json(
        batch_dir / "batch_metadata.json",
        {
            "analysis_kind": "velocity_profile_comparison_batch",
            "algorithm_name": algorithm_name,
            "hardware_names": hardware_names,
            "final_timestep": final_timestep,
            "num_shots": num_shots,
            "optimization_level": optimization_level,
            "seed_transpiler": seed_transpiler,
            "seed_simulator": seed_simulator,
            "noise_kind": noise_kind,
            "backend_method": backend_method,
            "noise_parameters": noise_parameters
            if noise_kind == "depolarizing"
            else {},
            "output_dirs": [str(path) for path in output_dirs],
        },
    )
    return batch_dir


def run_multi_hardware_velocity_profile_comparison(
    algorithm_name: str,
    hardware_names: list[str],
    timesteps: list[int],
    num_shots: int,
    optimization_level: int,
    seed_transpiler: int,
    seed_simulator: int,
    noise_kind: str,
    backend_method: str,
    noise_parameters: dict[str, Any],
    config_path: Path,
    output_root: Path,
    hardware_config_overrides: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Compare velocity profiles from several hardware-mapped noisy simulations."""

    if not timesteps:
        raise ValueError("timesteps must contain at least one QLBM timestep.")
    if any(timestep < 1 for timestep in timesteps):
        raise ValueError("All timesteps must be at least 1.")
    if noise_kind == "none":
        raise ValueError("Choose a non-'none' noise_kind for hardware comparison.")

    hardware_configs = load_hardware_configs_with_overrides(
        config_path,
        hardware_config_overrides=hardware_config_overrides,
    )
    require_hardware_names(hardware_configs, hardware_names)

    max_timestep = max(timesteps)
    final_case = build_case(
        algorithm_name,
        max_timestep,
        measure_velocity_qubits=True,
    )
    case_label_base = final_case.label
    final_timestep_suffix = f"_t{max_timestep}"
    if case_label_base.endswith(final_timestep_suffix):
        case_label_base = case_label_base[: -len(final_timestep_suffix)]
    hardware_path_name = "_vs_".join(safe_path_name(name) for name in hardware_names)
    timestep_path_name = "_".join(f"t{timestep}" for timestep in sorted(timesteps))
    output_dir = (
        output_root
        / "multi_hardware_velocity_profiles"
        / f"{case_label_base}_{timestep_path_name}_{hardware_path_name}_{noise_kind}"
    )
    analysis_dir = output_dir / "analysis"
    runs_dir = output_dir / "runs"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        output_dir / "comparison_metadata.json",
        {
            "analysis_kind": "multi_hardware_velocity_profile_comparison",
            "algorithm_name": algorithm_name,
            "case_label": final_case.label,
            "hardware_names": hardware_names,
            "hardware_ids": {
                name: hardware_configs[name].get("id") for name in hardware_names
            },
            "timesteps": sorted(timesteps),
            "num_shots": num_shots,
            "optimization_level": optimization_level,
            "seed_transpiler": seed_transpiler,
            "seed_simulator": seed_simulator,
            "backend_method": backend_method,
            "baseline_noise_kind": "none",
            "selected_noise_kind": noise_kind,
            "measurement": "ABGridMeasurement(measure_velocity_qubits=True)",
            "noise_parameters": noise_parameters
            if noise_kind == "depolarizing"
            else {},
            "hardware_config_overrides": hardware_config_overrides or {},
            "hardware_configs_used": {
                name: hardware_configs[name] for name in hardware_names
            },
            "plot_note": (
                "Combined plots use the first hardware's noiseless result as the "
                "visual reference curve; per-hardware error metrics use each "
                "hardware's own noiseless sampled run as baseline."
            ),
        },
    )
    write_lattice_json(output_dir / "lattice.json", final_case.lattice)

    final_reports: dict[str, Any] = {}
    metrics_rows: list[dict[str, Any]] = []
    for timestep in sorted(timesteps):
        case = build_case(
            algorithm_name,
            timestep,
            measure_velocity_qubits=True,
        )
        step_results: list[dict[str, Any]] = []

        for hardware_name in hardware_names:
            hardware_config = hardware_configs[hardware_name]
            estimator = QLBMResourceEstimator(hardware_config)
            logical_circuit = build_full_logical_circuit(case, case.runner_steps)
            report = estimator.estimate(
                logical_circuit,
                label=f"{case.label}_{hardware_name}_velocity_step{timestep}",
                qlbm_metadata={
                    **case.metadata,
                    "simulated_timestep": timestep,
                    "hardware_name": hardware_name,
                },
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
                transpile_circuit=True,
                phase_polynomial_analysis=False,
            )
            require_compatible(report)
            transpiled_compact_circuit = report["transpiled_compact_circuit"]

            baseline_backend = build_backend(
                noise_kind="none",
                backend_method=backend_method,
                seed_simulator=seed_simulator,
                noise_parameters=noise_parameters,
                hardware_config=hardware_config,
            )
            noisy_backend = build_backend(
                noise_kind=noise_kind,
                backend_method=backend_method,
                seed_simulator=seed_simulator,
                noise_parameters=noise_parameters,
                hardware_config=hardware_config,
            )
            baseline_counts = sample_counts(
                baseline_backend, transpiled_compact_circuit, num_shots
            )
            noisy_counts = sample_counts(
                noisy_backend, transpiled_compact_circuit, num_shots
            )

            hardware_run_dir = runs_dir / f"step_{timestep:03d}" / hardware_name
            write_json(
                hardware_run_dir / f"counts_none_step_{timestep}.json",
                baseline_counts,
            )
            write_json(
                hardware_run_dir / f"counts_{noise_kind}_step_{timestep}.json",
                noisy_counts,
            )

            hardware_analysis_dir = (
                analysis_dir / f"step_{timestep:03d}" / hardware_name
            )
            metrics = analyze_velocity_counts(
                baseline_counts,
                noisy_counts,
                case.lattice,
                hardware_analysis_dir,
                step=timestep,
                baseline_label=f"{_hardware_plot_label(hardware_name, hardware_config)} noiseless",
                noisy_label=f"{_hardware_plot_label(hardware_name, hardware_config)} {noise_kind}",
            )
            comparison = compare_velocity_fields(
                baseline_counts,
                noisy_counts,
                case.lattice,
            )
            metrics_rows.append(
                {
                    "timestep": timestep,
                    "hardware_name": hardware_name,
                    "hardware_label": _hardware_plot_label(
                        hardware_name, hardware_config
                    ),
                    **metrics,
                }
            )
            step_results.append(
                {
                    "hardware_name": hardware_name,
                    "hardware_label": _hardware_plot_label(
                        hardware_name, hardware_config
                    ),
                    "comparison": comparison,
                }
            )

            if timestep == max_timestep:
                final_reports[hardware_name] = report

        save_multi_hardware_velocity_x_outputs(
            analysis_dir / f"step_{timestep:03d}",
            timestep=timestep,
            noise_kind=noise_kind,
            step_results=step_results,
        )

    save_multi_hardware_velocity_metrics(analysis_dir, metrics_rows)
    write_json(output_dir / "velocity_hardware_metrics.json", metrics_rows)
    write_json(output_dir / "resource_reports_final_timestep.json", final_reports)
    return output_dir
