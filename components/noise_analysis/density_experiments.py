"""Density-oriented QLBM noise experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qiskit import transpile

from components.resource_estimator import QLBMResourceEstimator
from components.noise_analysis.experiment_utils import (
    _hardware_plot_label,
    build_backend,
    build_case,
    build_run_metadata,
    json_safe,
    load_hardware_config,
    load_hardware_configs,
    require_hardware_names,
    require_compatible,
    safe_path_name,
    sample_counts,
    write_lattice_json,
    write_json,
)
from components.noise_analysis.density_outputs import (
    save_multi_hardware_density_outputs,
)
from components.noise_analysis.qlbm_builders import build_full_logical_circuit
from components.noise_analysis.tools.density_analysis import (
    analyze_density_counts,
    counts_to_density_field,
    density_profiles,
    normalized_density,
    save_depolarizing_probability_sweep,
    save_density_error_growth,
)


def run_noiseless_vs_noise_density_comparison(
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
    """Run only the final timestep for noiseless and noisy cases, then compare rho."""

    if final_timestep < 1:
        raise ValueError("final_timestep must be at least 1.")
    if noise_kind == "none":
        raise ValueError(
            "Choose a non-'none' noise_kind for a baseline-vs-noise comparison."
        )

    build_case(algorithm_name, final_timestep)

    hardware_config = load_hardware_config(hardware_name, config_path)
    estimator = QLBMResourceEstimator(hardware_config)
    case = build_case(algorithm_name, final_timestep)

    output_dir = (
        output_root
        / "density_comparisons"
        / f"{case.label}_{hardware_name}_{noise_kind}_vs_none"
    )
    baseline_dir = output_dir / "baseline_none"
    noisy_dir = output_dir / f"noisy_{noise_kind}"
    analysis_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    logical_circuit = build_full_logical_circuit(case, case.runner_steps)
    report = estimator.estimate(
        logical_circuit,
        label=f"{case.label}_density_comparison",
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
            "analysis_kind": "final_density_profile_comparison",
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
            "hardware_gate_fidelities": hardware_config.get("gate_fidelities", {}),
            "hardware_gate_times_s": hardware_config.get("gate_times_s", {}),
            "measurement_fidelity": hardware_config.get("measurement_fidelity"),
            "coherence": hardware_config.get("coherence", {}),
            "noise_parameters": noise_parameters
            if noise_kind == "depolarizing"
            else {},
        },
    )

    run_dirs = {
        "none": baseline_dir,
        noise_kind: noisy_dir,
    }
    run_counts = {}
    for run_noise_kind, run_dir in run_dirs.items():
        backend = build_backend(
            noise_kind=run_noise_kind,
            backend_method=backend_method,
            seed_simulator=seed_simulator,
            noise_parameters=noise_parameters,
            hardware_config=hardware_config,
        )
        qlbm_result = case.lattice.create_result(str(run_dir), "step")
        qlbm_result.visualize_geometry()

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
        qlbm_result.save_timestep_counts(counts, final_timestep)

    write_json(output_dir / "resource_report.json", report)

    analyze_density_counts(
        run_counts["none"],
        run_counts[noise_kind],
        case.lattice,
        analysis_dir,
        step=final_timestep,
        baseline_label="noiseless",
        noisy_label=noise_kind,
        normalize=True,
    )
    return output_dir


def run_density_error_growth_comparison(
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
    """Compare density errors against noiseless runs for QLBM timesteps 1..N."""

    if max_timesteps < 1:
        raise ValueError("max_timesteps must be at least 1.")
    if noise_kind == "none":
        raise ValueError("Choose a non-'none' noise_kind for density-error growth.")

    build_case(algorithm_name, max_timesteps)

    hardware_config = load_hardware_config(hardware_name, config_path)
    estimator = QLBMResourceEstimator(hardware_config)
    final_case = build_case(algorithm_name, max_timesteps)
    output_dir = (
        output_root
        / "density_error_growth"
        / f"{final_case.label}_{hardware_name}_{noise_kind}_vs_none"
    )
    analysis_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        output_dir / "comparison_metadata.json",
        {
            "analysis_kind": "density_error_growth_from_counts",
            "algorithm_name": algorithm_name,
            "case_label": final_case.label,
            "hardware_name": hardware_name,
            "hardware_id": hardware_config.get("id"),
            "max_timesteps": max_timesteps,
            "simulated_timesteps": list(range(1, max_timesteps + 1)),
            "num_shots": num_shots,
            "optimization_level": optimization_level,
            "seed_transpiler": seed_transpiler,
            "seed_simulator": seed_simulator,
            "backend_method": backend_method,
            "baseline_noise_kind": "none",
            "selected_noise_kind": noise_kind,
            "hardware_gate_fidelities": hardware_config.get("gate_fidelities", {}),
            "hardware_gate_times_s": hardware_config.get("gate_times_s", {}),
            "measurement_fidelity": hardware_config.get("measurement_fidelity"),
            "coherence": hardware_config.get("coherence", {}),
            "noise_parameters": noise_parameters
            if noise_kind == "depolarizing"
            else {},
        },
    )

    backends = {
        "none": build_backend(
            noise_kind="none",
            backend_method=backend_method,
            seed_simulator=seed_simulator,
            noise_parameters=noise_parameters,
            hardware_config=hardware_config,
        ),
        noise_kind: build_backend(
            noise_kind=noise_kind,
            backend_method=backend_method,
            seed_simulator=seed_simulator,
            noise_parameters=noise_parameters,
            hardware_config=hardware_config,
        ),
    }

    step_metrics = []
    final_report = None
    for timestep in range(1, max_timesteps + 1):
        case = build_case(algorithm_name, timestep)
        logical_circuit = build_full_logical_circuit(case, case.runner_steps)
        report = estimator.estimate(
            logical_circuit,
            label=f"{case.label}_density_growth_step{timestep}",
            qlbm_metadata={**case.metadata, "simulated_timestep": timestep},
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            transpile_circuit=True,
            phase_polynomial_analysis=False,
        )
        require_compatible(report)
        transpiled_compact_circuit = report["transpiled_compact_circuit"]

        baseline_counts = sample_counts(
            backends["none"], transpiled_compact_circuit, num_shots
        )
        noisy_counts = sample_counts(
            backends[noise_kind], transpiled_compact_circuit, num_shots
        )

        step_dir = analysis_dir / f"step_{timestep:03d}"
        metrics = analyze_density_counts(
            baseline_counts,
            noisy_counts,
            case.lattice,
            step_dir,
            step=timestep,
            baseline_label="noiseless",
            noisy_label=noise_kind,
            normalize=True,
        )
        step_metrics.append({"timestep": timestep, "metrics": metrics})

        write_json(output_dir / f"counts_none_step_{timestep}.json", baseline_counts)
        write_json(
            output_dir / f"counts_{noise_kind}_step_{timestep}.json", noisy_counts
        )

        if timestep == max_timesteps:
            final_report = report
            for run_noise_kind, counts in {
                "none": baseline_counts,
                noise_kind: noisy_counts,
            }.items():
                final_dir = output_dir / f"final_{run_noise_kind}"
                qlbm_result = case.lattice.create_result(str(final_dir), "step")
                qlbm_result.visualize_geometry()
                qlbm_result.save_timestep_counts(counts, timestep)

    save_density_error_growth(analysis_dir, step_metrics, noisy_label=noise_kind)
    write_json(output_dir / "density_error_metrics_by_timestep.json", step_metrics)
    write_json(output_dir / "resource_report.json", final_report)
    return output_dir


def run_depolarizing_probability_sweep(
    algorithm_name: str,
    hardware_name: str,
    timestep: int,
    num_shots: int,
    optimization_level: int,
    seed_transpiler: int,
    seed_simulator: int,
    two_qubit_probabilities: list[float],
    config_path: Path,
    output_root: Path,
    backend_method: str = "statevector",
    single_qubit_probability_scale: float = 0.1,
) -> Path:
    """Sweep manual depolarizing strength and compare density against noiseless."""

    if timestep < 1:
        raise ValueError("timestep must be at least 1.")
    for probability in two_qubit_probabilities:
        if probability < 0 or probability > 1:
            raise ValueError("two_qubit_probabilities must be in [0, 1].")

    hardware_config = load_hardware_config(hardware_name, config_path)
    estimator = QLBMResourceEstimator(hardware_config)
    case = build_case(algorithm_name, timestep)

    output_dir = (
        output_root
        / "depolarizing_probability_sweeps"
        / f"{case.label}_{hardware_name}_depolarizing"
    )
    analysis_dir = output_dir / "analysis"
    run_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    logical_circuit = build_full_logical_circuit(case, case.runner_steps)
    report = estimator.estimate(
        logical_circuit,
        label=f"{case.label}_depolarizing_probability_sweep",
        qlbm_metadata={**case.metadata, "simulated_timestep": timestep},
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        transpile_circuit=True,
        phase_polynomial_analysis=False,
    )
    require_compatible(report)
    transpiled_compact_circuit = report["transpiled_compact_circuit"]
    simulation_circuit = transpile(
        transpiled_compact_circuit,
        basis_gates=["id", "rz", "sx", "x", "cx"],
        optimization_level=0,
        seed_transpiler=seed_transpiler,
    )

    baseline_backend = build_backend(
        noise_kind="none",
        backend_method=backend_method,
        seed_simulator=seed_simulator,
        noise_parameters={},
        hardware_config=hardware_config,
    )
    baseline_counts = sample_counts(baseline_backend, simulation_circuit, num_shots)
    write_json(run_dir / f"counts_none_step_{timestep}.json", baseline_counts)

    write_json(
        output_dir / "sweep_metadata.json",
        {
            "analysis_kind": "depolarizing_probability_sweep",
            "algorithm_name": algorithm_name,
            "case_label": case.label,
            "hardware_name": hardware_name,
            "hardware_id": hardware_config.get("id"),
            "timestep": timestep,
            "num_shots": num_shots,
            "optimization_level": optimization_level,
            "seed_transpiler": seed_transpiler,
            "seed_simulator": seed_simulator,
            "backend_method": backend_method,
            "noise_kind": "depolarizing",
            "two_qubit_probabilities": two_qubit_probabilities,
            "single_qubit_probability_scale": single_qubit_probability_scale,
            "simulation_circuit_note": (
                "The hardware-transpiled compact circuit is decomposed to "
                "['id', 'rz', 'sx', 'x', 'cx'] for Aer execution when the "
                "selected hardware basis contains Aer-unsupported native gates."
            ),
            "simulation_circuit": json_safe(simulation_circuit),
            "hardware_gate_fidelities": hardware_config.get("gate_fidelities", {}),
            "hardware_gate_times_s": hardware_config.get("gate_times_s", {}),
            "measurement_fidelity": hardware_config.get("measurement_fidelity"),
            "coherence": hardware_config.get("coherence", {}),
        },
    )
    write_json(output_dir / "resource_report.json", report)

    sweep_metrics = []
    for two_qubit_probability in two_qubit_probabilities:
        single_qubit_probability = (
            two_qubit_probability * single_qubit_probability_scale
        )
        label = f"p2q_{two_qubit_probability:.0e}".replace("+", "")
        probability_dir = analysis_dir / label

        if two_qubit_probability == 0.0:
            noisy_counts = baseline_counts
        else:
            noisy_backend = build_backend(
                noise_kind="depolarizing",
                backend_method=backend_method,
                seed_simulator=seed_simulator,
                noise_parameters={
                    "single_qubit_probability": single_qubit_probability,
                    "two_qubit_probability": two_qubit_probability,
                },
                hardware_config=hardware_config,
            )
            noisy_counts = sample_counts(noisy_backend, simulation_circuit, num_shots)

        write_json(run_dir / f"counts_{label}_step_{timestep}.json", noisy_counts)
        metrics = analyze_density_counts(
            baseline_counts,
            noisy_counts,
            case.lattice,
            probability_dir,
            step=timestep,
            baseline_label="noiseless",
            noisy_label=f"p2={two_qubit_probability:g}",
            normalize=True,
        )
        sweep_metrics.append(
            {
                "two_qubit_probability": two_qubit_probability,
                "single_qubit_probability": single_qubit_probability,
                "metrics": metrics,
            }
        )

    save_depolarizing_probability_sweep(analysis_dir, sweep_metrics, timestep)
    write_json(
        output_dir / "depolarizing_probability_sweep_metrics.json", sweep_metrics
    )
    return output_dir


def run_multi_hardware_density_depolarizing_comparison(
    algorithm_name: str,
    hardware_names: list[str],
    max_timesteps: int,
    num_shots: int,
    optimization_level: int,
    seed_transpiler: int,
    seed_simulator: int,
    backend_method: str,
    config_path: Path,
    output_root: Path,
) -> Path:
    """Compare density error growth for hardware-derived depolarizing noise."""

    noise_kind = "hardware_depolarizing"
    if max_timesteps < 1:
        raise ValueError("max_timesteps must be at least 1.")
    hardware_configs = load_hardware_configs(config_path)
    require_hardware_names(hardware_configs, hardware_names)

    final_case = build_case(algorithm_name, max_timesteps)
    case_label_base = final_case.label
    final_timestep_suffix = f"_t{max_timesteps}"
    if case_label_base.endswith(final_timestep_suffix):
        case_label_base = case_label_base[: -len(final_timestep_suffix)]
    hardware_path_name = "_vs_".join(safe_path_name(name) for name in hardware_names)
    output_dir = (
        output_root
        / "multi_hardware_density_depolarizing"
        / f"{case_label_base}_t1_t{max_timesteps}_{hardware_path_name}_{noise_kind}"
    )
    analysis_dir = output_dir / "analysis"
    runs_dir = output_dir / "runs"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        output_dir / "comparison_metadata.json",
        {
            "analysis_kind": "multi_hardware_density_depolarizing_comparison",
            "algorithm_name": algorithm_name,
            "case_label": final_case.label,
            "hardware_names": hardware_names,
            "hardware_ids": {
                name: hardware_configs[name].get("id") for name in hardware_names
            },
            "max_timesteps": max_timesteps,
            "simulated_timesteps": list(range(1, max_timesteps + 1)),
            "num_shots": num_shots,
            "optimization_level": optimization_level,
            "seed_transpiler": seed_transpiler,
            "seed_simulator": seed_simulator,
            "backend_method": backend_method,
            "baseline_noise_kind": "none",
            "selected_noise_kind": noise_kind,
            "hardware_configs_used": {
                name: hardware_configs[name] for name in hardware_names
            },
            "plot_note": (
                "Error-growth metrics compare each noisy hardware run against "
                "that hardware's own noiseless transpiled baseline. The final "
                "density-profile overlay uses the first hardware's noiseless "
                "profile as a visual reference."
            ),
        },
    )
    write_lattice_json(output_dir / "lattice.json", final_case.lattice)

    final_reports: dict[str, Any] = {}
    metrics_rows: list[dict[str, Any]] = []
    final_profiles: list[dict[str, Any]] = []
    reference_final_profile: Any | None = None
    heatmap_step = 1
    density_heatmap_fields: list[dict[str, Any]] = []

    for timestep in range(1, max_timesteps + 1):
        case = build_case(algorithm_name, timestep)
        for hardware_name in hardware_names:
            hardware_config = hardware_configs[hardware_name]
            hardware_label = _hardware_plot_label(hardware_name, hardware_config)
            estimator = QLBMResourceEstimator(hardware_config)
            logical_circuit = build_full_logical_circuit(case, case.runner_steps)
            report = estimator.estimate(
                logical_circuit,
                label=f"{case.label}_{hardware_name}_density_depol_step{timestep}",
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
                noise_parameters={},
                hardware_config=hardware_config,
            )
            noisy_backend = build_backend(
                noise_kind=noise_kind,
                backend_method=backend_method,
                seed_simulator=seed_simulator,
                noise_parameters={},
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
            metrics = analyze_density_counts(
                baseline_counts,
                noisy_counts,
                case.lattice,
                hardware_analysis_dir,
                step=timestep,
                baseline_label=f"{hardware_label} noiseless",
                noisy_label=f"{hardware_label} {noise_kind}",
                normalize=True,
            )
            metrics_rows.append(
                {
                    "timestep": timestep,
                    "hardware_name": hardware_name,
                    "hardware_label": hardware_label,
                    **metrics,
                }
            )

            if timestep == heatmap_step:
                baseline_field = counts_to_density_field(baseline_counts, case.lattice)
                noisy_field = counts_to_density_field(noisy_counts, case.lattice)
                if not density_heatmap_fields:
                    density_heatmap_fields.append(
                        {
                            "label": "Noiseless reference",
                            "field": normalized_density(baseline_field),
                        }
                    )
                density_heatmap_fields.append(
                    {
                        "label": f"{hardware_label} ({noise_kind})",
                        "field": normalized_density(noisy_field),
                    }
                )

            if timestep == max_timesteps:
                final_reports[hardware_name] = report
                baseline_field = counts_to_density_field(baseline_counts, case.lattice)
                noisy_field = counts_to_density_field(noisy_counts, case.lattice)
                baseline_profiles = density_profiles(
                    baseline_field,
                    normalize=True,
                )
                noisy_profiles = density_profiles(
                    noisy_field,
                    normalize=True,
                )
                if reference_final_profile is None:
                    reference_final_profile = baseline_profiles["rho_x_sum"]
                final_profiles.append(
                    {
                        "hardware_name": hardware_name,
                        "hardware_label": hardware_label,
                        "baseline_rho_x_sum": baseline_profiles["rho_x_sum"],
                        "noisy_rho_x_sum": noisy_profiles["rho_x_sum"],
                    }
                )

    save_multi_hardware_density_outputs(
        analysis_dir,
        metrics_rows=metrics_rows,
        final_profiles=final_profiles,
        reference_final_profile=reference_final_profile,
        max_timesteps=max_timesteps,
        noise_kind=noise_kind,
        heatmap_step=heatmap_step,
        density_heatmap_fields=density_heatmap_fields,
    )
    write_json(output_dir / "density_hardware_metrics.json", metrics_rows)
    write_json(output_dir / "resource_reports_final_timestep.json", final_reports)
    return output_dir
