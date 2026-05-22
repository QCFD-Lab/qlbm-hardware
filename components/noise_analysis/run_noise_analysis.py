"""Run hardware-mapped QLBM noise simulations for noise analysis."""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_SOURCE_ROOT = PROJECT_ROOT / "qlbm"
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "output" / ".matplotlib"),
)

for path in (QLBM_SOURCE_ROOT, QLBM_HARDWARE_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.resource_estimator import QLBMResourceEstimator
from noise_models import build_noise_model
from qlbm_builders import CASE_BUILDERS, QLBMCase, build_full_logical_circuit
from components.noise_analysis.tools.density_analysis import (
    analyze_density_counts,
    counts_to_density_field,
    density_profiles,
    normalized_density,
    save_depolarizing_probability_sweep,
    save_density_error_growth,
)
from components.noise_analysis.tools.velocity_analysis import analyze_velocity_counts
from components.noise_analysis.tools.velocity_analysis import compare_velocity_fields


DEFAULT_CONFIG_PATH = QLBM_HARDWARE_ROOT / "components" / "resource_estimator" / "config.json"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "hardware_noise_runs"


def load_hardware_configs(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    """Load hardware configurations used by the resource estimator."""
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a nested dictionary in place."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_hardware_configs_with_overrides(
    config_path: Path = DEFAULT_CONFIG_PATH,
    hardware_config_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load hardware configs and apply experiment-local overrides."""

    hardware_configs = deepcopy(load_hardware_configs(config_path))
    for hardware_name, overrides in (hardware_config_overrides or {}).items():
        if hardware_name not in hardware_configs:
            raise ValueError(
                f"Cannot override unknown hardware_name={hardware_name!r}. "
                f"Expected one of {sorted(hardware_configs)}."
            )
        deep_update(hardware_configs[hardware_name], overrides)
    return hardware_configs


def build_backend(
    noise_kind: str,
    backend_method: str,
    seed_simulator: int,
    noise_parameters: dict[str, Any],
    hardware_config: dict[str, Any],
) -> AerSimulator:
    """Build the Aer backend for direct execution of a transpiled circuit."""

    noise_model = build_noise_model(
        kind=noise_kind,
        hardware_config=hardware_config,
        single_qubit_probability=noise_parameters.get("single_qubit_probability", 0.001),
        two_qubit_probability=noise_parameters.get("two_qubit_probability", 0.01),
    )
    options = {"method": backend_method, "seed_simulator": seed_simulator}
    if noise_model is None:
        return AerSimulator(**options)
    return AerSimulator(**options, noise_model=noise_model)


def build_run_metadata(
    algorithm_name: str,
    case_label: str,
    hardware_name: str,
    max_timesteps: int,
    num_shots: int,
    optimization_level: int,
    seed_transpiler: int,
    seed_simulator: int,
    noise_kind: str,
    backend_method: str,
    noise_parameters: dict[str, Any],
    hardware_config: dict[str, Any],
    simulated_timesteps: list[int] | None = None,
) -> dict[str, Any]:
    """Build metadata, omitting manual noise probabilities for noiseless runs."""

    metadata = {
        "algorithm_name": algorithm_name,
        "case_label": case_label,
        "hardware_name": hardware_name,
        "hardware_id": hardware_config.get("id"),
        "max_timesteps": max_timesteps,
        "simulated_timesteps": (
            simulated_timesteps
            if simulated_timesteps is not None
            else list(range(0, max_timesteps + 1))
        ),
        "num_shots": num_shots,
        "optimization_level": optimization_level,
        "seed_transpiler": seed_transpiler,
        "seed_simulator": seed_simulator,
        "noise_kind": noise_kind,
        "backend_method": backend_method,
        "hardware_gate_fidelities": hardware_config.get("gate_fidelities", {}),
        "hardware_gate_times_s": hardware_config.get("gate_times_s", {}),
        "measurement_fidelity": hardware_config.get("measurement_fidelity"),
        "coherence": hardware_config.get("coherence", {}),
    }
    if noise_kind == "depolarizing":
        metadata["noise_parameters"] = noise_parameters
    return metadata


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON file after converting non-serializable values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, indent=2, sort_keys=True)


def json_safe(value: Any) -> Any:
    """Convert report objects into JSON-serializable summaries."""

    if isinstance(value, QuantumCircuit):
        return {
            "num_qubits": value.num_qubits,
            "num_clbits": value.num_clbits,
            "depth": value.depth(),
            "size": value.size(),
            "op_counts": dict(value.count_ops()),
        }
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def require_compatible(report: dict[str, Any]) -> None:
    """Raise if the estimator reports hardware incompatibility."""

    if report.get("transpile_error"):
        raise RuntimeError(report["transpile_error"])
    compatibility = report.get("transpiled_compatibility") or {}
    if compatibility.get("compatible"):
        return
    raise RuntimeError(f"Transpiled circuit is not hardware compatible: {compatibility}")


def build_case(
    algorithm_name: str,
    num_timesteps: int,
    *,
    measure_velocity_qubits: bool = False,
) -> QLBMCase:
    """Build a QLBM case, optionally using velocity-resolved AB measurement."""

    if algorithm_name not in CASE_BUILDERS:
        raise ValueError(
            f"Unknown algorithm_name={algorithm_name!r}. "
            f"Expected one of {sorted(CASE_BUILDERS)}."
        )
    if measure_velocity_qubits:
        if algorithm_name != "abqlbm":
            raise ValueError("Velocity-profile analysis currently supports only abqlbm.")
        return CASE_BUILDERS[algorithm_name](
            num_timesteps=num_timesteps,
            measure_velocity_qubits=True,
        )
    return CASE_BUILDERS[algorithm_name](num_timesteps=num_timesteps)


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

    hardware_configs = load_hardware_configs(config_path)
    if hardware_name not in hardware_configs:
        raise ValueError(
            f"Unknown hardware_name={hardware_name!r}. "
            f"Expected one of {sorted(hardware_configs)}."
        )
    build_case(algorithm_name, max_timesteps)

    hardware_config = hardware_configs[hardware_name]
    estimator = QLBMResourceEstimator(hardware_config)
    backend = build_backend(
        noise_kind=noise_kind,
        backend_method=backend_method,
        seed_simulator=seed_simulator,
        noise_parameters=noise_parameters,
        hardware_config=hardware_config,
    )

    final_case = build_case(algorithm_name, max_timesteps)
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

    final_report = None
    for timestep in range(0, max_timesteps + 1):
        case_timesteps = timestep if timestep > 0 else 1
        case = build_case(algorithm_name, case_timesteps)
        qlbm_result = case.lattice.create_result(str(output_dir), "step")
        qlbm_result.visualize_geometry()
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
        result_run = backend.run(transpiled_compact_circuit, shots=num_shots).result()
        counts = dict(result_run.get_counts())

        write_json(output_dir / f"counts_step_{timestep}.json", counts)
        qlbm_result.save_timestep_counts(counts, timestep)

        if timestep == max_timesteps:
            final_report = report

    write_json(output_dir / "resource_report.json", final_report)
    return output_dir


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
    exclude_y_boundary: bool = False,
) -> Path:
    """Run only the final timestep for noiseless and noisy cases, then compare rho."""

    if final_timestep < 1:
        raise ValueError("final_timestep must be at least 1.")
    if noise_kind == "none":
        raise ValueError("Choose a non-'none' noise_kind for a baseline-vs-noise comparison.")

    hardware_configs = load_hardware_configs(config_path)
    if hardware_name not in hardware_configs:
        raise ValueError(
            f"Unknown hardware_name={hardware_name!r}. "
            f"Expected one of {sorted(hardware_configs)}."
        )
    build_case(algorithm_name, final_timestep)

    hardware_config = hardware_configs[hardware_name]
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
            "exclude_y_boundary": exclude_y_boundary,
            "hardware_gate_fidelities": hardware_config.get("gate_fidelities", {}),
            "hardware_gate_times_s": hardware_config.get("gate_times_s", {}),
            "measurement_fidelity": hardware_config.get("measurement_fidelity"),
            "coherence": hardware_config.get("coherence", {}),
            "noise_parameters": noise_parameters if noise_kind == "depolarizing" else {},
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

        result_run = backend.run(transpiled_compact_circuit, shots=num_shots).result()
        counts = dict(result_run.get_counts())
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
        exclude_y_boundary=exclude_y_boundary,
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
    exclude_y_boundary: bool = False,
) -> Path:
    """Compare density errors against noiseless runs for QLBM timesteps 1..N."""

    if max_timesteps < 1:
        raise ValueError("max_timesteps must be at least 1.")
    if noise_kind == "none":
        raise ValueError("Choose a non-'none' noise_kind for density-error growth.")

    hardware_configs = load_hardware_configs(config_path)
    if hardware_name not in hardware_configs:
        raise ValueError(
            f"Unknown hardware_name={hardware_name!r}. "
            f"Expected one of {sorted(hardware_configs)}."
        )
    build_case(algorithm_name, max_timesteps)

    hardware_config = hardware_configs[hardware_name]
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
            "exclude_y_boundary": exclude_y_boundary,
            "hardware_gate_fidelities": hardware_config.get("gate_fidelities", {}),
            "hardware_gate_times_s": hardware_config.get("gate_times_s", {}),
            "measurement_fidelity": hardware_config.get("measurement_fidelity"),
            "coherence": hardware_config.get("coherence", {}),
            "noise_parameters": noise_parameters if noise_kind == "depolarizing" else {},
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

        baseline_counts = dict(
            backends["none"].run(transpiled_compact_circuit, shots=num_shots).result().get_counts()
        )
        noisy_counts = dict(
            backends[noise_kind]
            .run(transpiled_compact_circuit, shots=num_shots)
            .result()
            .get_counts()
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
            exclude_y_boundary=exclude_y_boundary,
        )
        step_metrics.append({"timestep": timestep, "metrics": metrics})

        write_json(output_dir / f"counts_none_step_{timestep}.json", baseline_counts)
        write_json(output_dir / f"counts_{noise_kind}_step_{timestep}.json", noisy_counts)

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
    exclude_y_boundary: bool = False,
    exclude_x_boundary: bool = False,
) -> Path:
    """Run a final velocity-resolved measurement and compare u_x/u_y profiles."""

    if final_timestep < 1:
        raise ValueError("final_timestep must be at least 1.")
    if noise_kind == "none":
        raise ValueError("Choose a non-'none' noise_kind for velocity-profile comparison.")

    hardware_configs = load_hardware_configs(config_path)
    if hardware_name not in hardware_configs:
        raise ValueError(
            f"Unknown hardware_name={hardware_name!r}. "
            f"Expected one of {sorted(hardware_configs)}."
        )

    hardware_config = hardware_configs[hardware_name]
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
            "exclude_y_boundary": exclude_y_boundary,
            "exclude_x_boundary": exclude_x_boundary,
            "hardware_gate_fidelities": hardware_config.get("gate_fidelities", {}),
            "hardware_gate_times_s": hardware_config.get("gate_times_s", {}),
            "measurement_fidelity": hardware_config.get("measurement_fidelity"),
            "coherence": hardware_config.get("coherence", {}),
            "noise_parameters": noise_parameters if noise_kind == "depolarizing" else {},
        },
    )
    with (output_dir / "lattice.json").open("w", encoding="utf-8") as file:
        file.write(case.lattice.to_json())

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
        result_run = backend.run(transpiled_compact_circuit, shots=num_shots).result()
        counts = dict(result_run.get_counts())
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
        exclude_y_boundary=exclude_y_boundary,
        exclude_x_boundary=exclude_x_boundary,
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
    exclude_y_boundary: bool = False,
    exclude_x_boundary: bool = False,
) -> Path:
    """Run standard velocity-profile comparison plots for several hardware configs."""

    if not hardware_names:
        raise ValueError("hardware_names must contain at least one hardware config.")

    batch_dir = (
        output_root
        / "velocity_profile_comparison_batches"
        / (
            f"{algorithm_name}_t{final_timestep}_"
            f"{'_vs_'.join(_hardware_slug(name) for name in hardware_names)}_"
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
                exclude_y_boundary=exclude_y_boundary,
                exclude_x_boundary=exclude_x_boundary,
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
            "noise_parameters": noise_parameters if noise_kind == "depolarizing" else {},
            "output_dirs": [str(path) for path in output_dirs],
        },
    )
    return batch_dir


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
    exclude_y_boundary: bool = False,
) -> Path:
    """Sweep manual depolarizing strength and compare density against noiseless."""

    if timestep < 1:
        raise ValueError("timestep must be at least 1.")
    hardware_configs = load_hardware_configs(config_path)
    if hardware_name not in hardware_configs:
        raise ValueError(
            f"Unknown hardware_name={hardware_name!r}. "
            f"Expected one of {sorted(hardware_configs)}."
        )
    for probability in two_qubit_probabilities:
        if probability < 0 or probability > 1:
            raise ValueError("two_qubit_probabilities must be in [0, 1].")

    hardware_config = hardware_configs[hardware_name]
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
    baseline_counts = dict(
        baseline_backend.run(simulation_circuit, shots=num_shots)
        .result()
        .get_counts()
    )
    write_json(run_dir / "counts_none_step_1.json", baseline_counts)

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
            "exclude_y_boundary": exclude_y_boundary,
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
        single_qubit_probability = two_qubit_probability * single_qubit_probability_scale
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
            noisy_counts = dict(
                noisy_backend.run(simulation_circuit, shots=num_shots)
                .result()
                .get_counts()
            )

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
            exclude_y_boundary=exclude_y_boundary,
        )
        sweep_metrics.append(
            {
                "two_qubit_probability": two_qubit_probability,
                "single_qubit_probability": single_qubit_probability,
                "metrics": metrics,
            }
        )

    save_depolarizing_probability_sweep(analysis_dir, sweep_metrics)
    write_json(output_dir / "depolarizing_probability_sweep_metrics.json", sweep_metrics)
    return output_dir


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
    exclude_y_boundary: bool = False,
    exclude_x_boundary: bool = False,
) -> Path:
    """Compare velocity profiles from several hardware-mapped noisy simulations."""

    if not hardware_names:
        raise ValueError("hardware_names must contain at least one hardware config.")
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
    unknown_hardware = [name for name in hardware_names if name not in hardware_configs]
    if unknown_hardware:
        raise ValueError(
            f"Unknown hardware names {unknown_hardware!r}. "
            f"Expected values from {sorted(hardware_configs)}."
        )

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
    hardware_slug = "_vs_".join(_hardware_slug(name) for name in hardware_names)
    timestep_slug = "_".join(f"t{timestep}" for timestep in sorted(timesteps))
    output_dir = (
        output_root
        / "multi_hardware_velocity_profiles"
        / f"{case_label_base}_{timestep_slug}_{hardware_slug}_{noise_kind}"
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
            "exclude_y_boundary": exclude_y_boundary,
            "exclude_x_boundary": exclude_x_boundary,
            "noise_parameters": noise_parameters if noise_kind == "depolarizing" else {},
            "hardware_config_overrides": hardware_config_overrides or {},
            "hardware_configs_used": {name: hardware_configs[name] for name in hardware_names},
            "plot_note": (
                "Combined plots use the first hardware's noiseless result as the "
                "visual reference curve; per-hardware error metrics use each "
                "hardware's own noiseless sampled run as baseline."
            ),
        },
    )
    with (output_dir / "lattice.json").open("w", encoding="utf-8") as file:
        file.write(final_case.lattice.to_json())

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
            baseline_counts = dict(
                baseline_backend.run(transpiled_compact_circuit, shots=num_shots)
                .result()
                .get_counts()
            )
            noisy_counts = dict(
                noisy_backend.run(transpiled_compact_circuit, shots=num_shots)
                .result()
                .get_counts()
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
                exclude_y_boundary=exclude_y_boundary,
                exclude_x_boundary=exclude_x_boundary,
            )
            comparison = compare_velocity_fields(
                baseline_counts,
                noisy_counts,
                case.lattice,
                exclude_y_boundary=exclude_y_boundary,
                exclude_x_boundary=exclude_x_boundary,
            )
            metrics_rows.append(
                {
                    "timestep": timestep,
                    "hardware_name": hardware_name,
                    "hardware_label": _hardware_plot_label(hardware_name, hardware_config),
                    **metrics,
                }
            )
            step_results.append(
                {
                    "hardware_name": hardware_name,
                    "hardware_label": _hardware_plot_label(hardware_name, hardware_config),
                    "comparison": comparison,
                }
            )

            if timestep == max_timestep:
                final_reports[hardware_name] = report

        _save_multi_hardware_velocity_x_outputs(
            analysis_dir / f"step_{timestep:03d}",
            timestep=timestep,
            noise_kind=noise_kind,
            step_results=step_results,
        )

    _save_multi_hardware_velocity_metrics(analysis_dir, metrics_rows)
    write_json(output_dir / "velocity_hardware_metrics.json", metrics_rows)
    write_json(output_dir / "resource_reports_final_timestep.json", final_reports)
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
    exclude_y_boundary: bool = False,
) -> Path:
    """Compare density error growth for hardware-derived depolarizing noise."""

    noise_kind = "hardware_depolarizing"
    if max_timesteps < 1:
        raise ValueError("max_timesteps must be at least 1.")
    if not hardware_names:
        raise ValueError("hardware_names must contain at least one hardware config.")

    hardware_configs = load_hardware_configs(config_path)
    unknown_hardware = [name for name in hardware_names if name not in hardware_configs]
    if unknown_hardware:
        raise ValueError(
            f"Unknown hardware names {unknown_hardware!r}. "
            f"Expected values from {sorted(hardware_configs)}."
        )

    final_case = build_case(algorithm_name, max_timesteps)
    case_label_base = final_case.label
    final_timestep_suffix = f"_t{max_timesteps}"
    if case_label_base.endswith(final_timestep_suffix):
        case_label_base = case_label_base[: -len(final_timestep_suffix)]
    hardware_slug = "_vs_".join(_hardware_slug(name) for name in hardware_names)
    output_dir = (
        output_root
        / "multi_hardware_density_depolarizing"
        / f"{case_label_base}_t1_t{max_timesteps}_{hardware_slug}_{noise_kind}"
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
            "exclude_y_boundary": exclude_y_boundary,
            "hardware_configs_used": {name: hardware_configs[name] for name in hardware_names},
            "plot_note": (
                "Error-growth metrics compare each noisy hardware run against "
                "that hardware's own noiseless transpiled baseline. The final "
                "density-profile overlay uses the first hardware's noiseless "
                "profile as a visual reference."
            ),
        },
    )
    with (output_dir / "lattice.json").open("w", encoding="utf-8") as file:
        file.write(final_case.lattice.to_json())

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
            baseline_counts = dict(
                baseline_backend.run(transpiled_compact_circuit, shots=num_shots)
                .result()
                .get_counts()
            )
            noisy_counts = dict(
                noisy_backend.run(transpiled_compact_circuit, shots=num_shots)
                .result()
                .get_counts()
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
                exclude_y_boundary=exclude_y_boundary,
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
                    exclude_y_boundary=exclude_y_boundary,
                )
                noisy_profiles = density_profiles(
                    noisy_field,
                    normalize=True,
                    exclude_y_boundary=exclude_y_boundary,
                )
                if reference_final_profile is None:
                    reference_final_profile = baseline_profiles["rho_x_mean"]
                final_profiles.append(
                    {
                        "hardware_name": hardware_name,
                        "hardware_label": hardware_label,
                        "baseline_rho_x_mean": baseline_profiles["rho_x_mean"],
                        "noisy_rho_x_mean": noisy_profiles["rho_x_mean"],
                    }
                )

    _save_multi_hardware_density_outputs(
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


def _hardware_slug(hardware_name: str) -> str:
    if "ibm_nighthawk" in hardware_name:
        return "ibm_nighthawk"
    if "ibm_eagle" in hardware_name:
        return "ibm_eagle"
    if "neutral_atom" in hardware_name:
        return "neutral_atom"
    return hardware_name.replace("superconducting_", "").replace("_2024", "").replace("_2023", "")


def _hardware_plot_label(hardware_name: str, hardware_config: dict[str, Any]) -> str:
    if "ibm_nighthawk" in hardware_name:
        return "IBM Nighthawk r1"
    if "ibm_eagle" in hardware_name:
        return "IBM Eagle r3"
    if "neutral_atom" in hardware_name:
        return "Neutral atom"
    return str(hardware_config.get("device_name") or hardware_name)


def _save_multi_hardware_density_outputs(
    analysis_dir: Path,
    *,
    metrics_rows: list[dict[str, Any]],
    final_profiles: list[dict[str, Any]],
    reference_final_profile: Any,
    max_timesteps: int,
    noise_kind: str,
    heatmap_step: int | None = None,
    density_heatmap_fields: list[dict[str, Any]] | None = None,
) -> None:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    analysis_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "timestep",
        "hardware_name",
        "hardware_label",
        "field_total_variation_distance",
        "field_relative_l2_error",
        "field_rmse",
        "rho_x_sum_total_variation_distance",
        "rho_x_mean_relative_l2_error",
    ]
    with (analysis_dir / "hardware_depolarizing_density_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=metric_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics_rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    hardware_labels = []
    for row in metrics_rows:
        if row["hardware_label"] not in hardware_labels:
            hardware_labels.append(row["hardware_label"])
    for hardware_label in hardware_labels:
        hardware_rows = [row for row in metrics_rows if row["hardware_label"] == hardware_label]
        ax.plot(
            [row["timestep"] for row in hardware_rows],
            [row["field_total_variation_distance"] for row in hardware_rows],
            marker="o",
            linewidth=1.8,
            label=hardware_label,
        )
    ax.set_xlabel("QLBM timestep")
    ax.set_ylabel("density total variation distance")
    ax.set_title("Density error growth under hardware-derived depolarizing noise")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        analysis_dir / "hardware_depolarizing_density_tvd_vs_timestep.png",
        dpi=180,
    )
    plt.close(fig)

    if heatmap_step is not None and density_heatmap_fields:
        _save_multi_hardware_density_heatmap(
            analysis_dir,
            timestep=heatmap_step,
            density_heatmap_fields=density_heatmap_fields,
        )

    if reference_final_profile is None or not final_profiles:
        return

    x_values = np.arange(len(reference_final_profile))
    with (
        analysis_dir
        / f"hardware_depolarizing_final_density_profile_step_{max_timesteps}.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["x", "noiseless_reference_rho_x_mean"]
        for profile in final_profiles:
            fieldnames.append(f"{_hardware_slug(profile['hardware_name'])}_rho_x_mean")
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for x_index in range(len(reference_final_profile)):
            row = {
                "x": x_index,
                "noiseless_reference_rho_x_mean": reference_final_profile[x_index],
            }
            for profile in final_profiles:
                row[f"{_hardware_slug(profile['hardware_name'])}_rho_x_mean"] = profile[
                    "noisy_rho_x_mean"
                ][x_index]
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        x_values,
        reference_final_profile,
        color="black",
        marker="o",
        linewidth=2.0,
        label="Noiseless reference",
    )
    markers = ["s", "^", "D", "v"]
    for index, profile in enumerate(final_profiles):
        ax.plot(
            x_values,
            profile["noisy_rho_x_mean"],
            marker=markers[index % len(markers)],
            linewidth=1.8,
            label=f"{profile['hardware_label']} ({noise_kind})",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("mean normalized density rho(x)")
    ax.set_title(f"Final density profile at timestep {max_timesteps}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        analysis_dir / f"hardware_depolarizing_final_density_profile_step_{max_timesteps}.png",
        dpi=180,
    )
    plt.close(fig)


def _save_multi_hardware_density_heatmap(
    analysis_dir: Path,
    *,
    timestep: int,
    density_heatmap_fields: list[dict[str, Any]],
) -> None:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not density_heatmap_fields:
        return

    heatmap_dir = analysis_dir / f"step_{timestep:03d}"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    fields = [item["field"] for item in density_heatmap_fields]
    vmin = 0.0
    vmax = max(float(np.max(field)) for field in fields)

    fig, axes = plt.subplots(
        1,
        len(density_heatmap_fields),
        figsize=(4.2 * len(density_heatmap_fields), 3.8),
        constrained_layout=True,
    )
    if len(density_heatmap_fields) == 1:
        axes = [axes]

    image = None
    for axis, item in zip(axes, density_heatmap_fields, strict=True):
        image = axis.imshow(
            item["field"].T,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        axis.set_title(item["label"])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.88, label="normalized density")
    fig.suptitle(f"Density heatmaps after {timestep} QLBM timestep")
    fig.savefig(
        heatmap_dir / f"hardware_depolarizing_density_heatmaps_step_{timestep}.png",
        dpi=180,
    )
    plt.close(fig)

    with (
        heatmap_dir / f"hardware_depolarizing_density_heatmaps_step_{timestep}.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "x", "y", "normalized_density"])
        for item in density_heatmap_fields:
            field = item["field"]
            for x_index in range(field.shape[0]):
                for y_index in range(field.shape[1]):
                    writer.writerow(
                        [item["label"], x_index, y_index, field[x_index, y_index]]
                    )


def _save_multi_hardware_velocity_metrics(
    analysis_dir: Path,
    metrics_rows: list[dict[str, Any]],
) -> None:
    import csv

    if not metrics_rows:
        return
    columns = [
        "timestep",
        "hardware_name",
        "hardware_label",
        "ux_x_mean_relative_l2_error",
        "uy_x_mean_relative_l2_error",
        "ux_field_rmse",
        "uy_field_rmse",
        "ux_field_linf_error",
        "uy_field_linf_error",
    ]
    with (analysis_dir / "velocity_hardware_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics_rows)


def _save_multi_hardware_velocity_x_outputs(
    step_dir: Path,
    *,
    timestep: int,
    noise_kind: str,
    step_results: list[dict[str, Any]],
) -> None:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not step_results:
        return

    step_dir.mkdir(parents=True, exist_ok=True)
    reference = step_results[0]["comparison"]["baseline_profiles"]
    x_count = len(reference["ux_x_mean"])

    with (step_dir / f"multi_hardware_velocity_along_x_step_{timestep}.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = ["x", "noiseless_reference_ux", "noiseless_reference_uy"]
        for result in step_results:
            slug = _hardware_slug(result["hardware_name"])
            fieldnames.extend([f"{slug}_noisy_ux", f"{slug}_noisy_uy"])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for x_index in range(x_count):
            row = {
                "x": x_index,
                "noiseless_reference_ux": reference["ux_x_mean"][x_index],
                "noiseless_reference_uy": reference["uy_x_mean"][x_index],
            }
            for result in step_results:
                slug = _hardware_slug(result["hardware_name"])
                noisy = result["comparison"]["noisy_profiles"]
                row[f"{slug}_noisy_ux"] = noisy["ux_x_mean"][x_index]
                row[f"{slug}_noisy_uy"] = noisy["uy_x_mean"][x_index]
            writer.writerow(row)

    x_values = list(range(x_count))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        x_values,
        reference["ux_x_mean"],
        color="black",
        linewidth=2.0,
        marker="o",
        label="Noiseless reference",
    )
    markers = ["s", "^", "D", "v"]
    for index, result in enumerate(step_results):
        noisy = result["comparison"]["noisy_profiles"]
        ax.plot(
            x_values,
            noisy["ux_x_mean"],
            marker=markers[index % len(markers)],
            label=f"{result['hardware_label']} ({noise_kind})",
        )
    ax.set_title(f"Velocity Profile Along x, timestep {timestep}")
    ax.set_xlabel("x")
    ax.set_ylabel("mean u_x")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(step_dir / f"velocity_profile_along_x_step_{timestep}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        x_values,
        reference["uy_x_mean"],
        color="black",
        linewidth=2.0,
        marker="o",
        label="Noiseless reference",
    )
    for index, result in enumerate(step_results):
        noisy = result["comparison"]["noisy_profiles"]
        ax.plot(
            x_values,
            noisy["uy_x_mean"],
            marker=markers[index % len(markers)],
            label=f"{result['hardware_label']} ({noise_kind})",
        )
    ax.set_title(f"Transverse Velocity Along x, timestep {timestep}")
    ax.set_xlabel("x")
    ax.set_ylabel("mean u_y")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(step_dir / f"transverse_velocity_along_x_step_{timestep}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    algorithm_name = "abqlbm"  # Options: "abqlbm", "msqlbm", "spacetime"
    hardware_name = "superconducting_ibm_eagle_r3_2024"
    hardware_names = [
        "superconducting_ibm_nighthawk_r1_2025",
        "neutral_atom_aws_2023",
    ]
    velocity_profile_timesteps = [1, 3]
    max_timesteps = 5
    num_shots = 4096
    optimization_level = 1
    seed_transpiler = 42
    seed_simulator = 42
    backend_method = "statevector"

    # none, depolarizing, hardware_depolarizing, thermal_relaxation
    noise_kind = "hardware_depolarizing"
    noise_parameters = {
        "single_qubit_probability": 0.000001,
        "two_qubit_probability": 0.00001,
    }
    hardware_config_overrides = {
        "neutral_atom_aws_2023": {
            "coherence": {
                "t1_s": 4.0,
                "t2_s": 1.0,
            }
        }
    }
    run_density_comparison = False
    run_density_error_growth = False
    run_velocity_profile_comparison = False
    run_velocity_profile_comparison_batch_enabled = True
    run_multi_hardware_velocity_profile_comparison_enabled = False
    run_multi_hardware_density_depolarizing = False
    run_depolarizing_sweep = False
    exclude_y_boundary = False
    exclude_x_boundary = False

    if run_velocity_profile_comparison_batch_enabled:
        output_dir = run_velocity_profile_comparison_batch(
            algorithm_name=algorithm_name,
            hardware_names=hardware_names,
            final_timestep=1,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
            exclude_y_boundary=exclude_y_boundary,
            exclude_x_boundary=exclude_x_boundary,
        )
    elif run_multi_hardware_density_depolarizing:
        output_dir = run_multi_hardware_density_depolarizing_comparison(
            algorithm_name=algorithm_name,
            hardware_names=hardware_names,
            max_timesteps=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            backend_method=backend_method,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
            exclude_y_boundary=exclude_y_boundary,
        )
    elif run_multi_hardware_velocity_profile_comparison_enabled:
        output_dir = run_multi_hardware_velocity_profile_comparison(
            algorithm_name=algorithm_name,
            hardware_names=hardware_names,
            timesteps=velocity_profile_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
            hardware_config_overrides=hardware_config_overrides,
            exclude_y_boundary=exclude_y_boundary,
            exclude_x_boundary=exclude_x_boundary,
        )
    elif run_depolarizing_sweep:
        output_dir = run_depolarizing_probability_sweep(
            algorithm_name=algorithm_name,
            hardware_name=hardware_name,
            timestep=1,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            two_qubit_probabilities=[
                0.0,
                1e-5,
                3e-5,
                1e-4,
                3e-4,
                1e-3,
                3e-3,
                1e-2,
            ],
            backend_method=backend_method,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
            single_qubit_probability_scale=0.1,
            exclude_y_boundary=exclude_y_boundary,
        )
    elif run_velocity_profile_comparison:
        output_dir = run_noiseless_vs_noise_velocity_profile_comparison(
            algorithm_name=algorithm_name,
            hardware_name=hardware_name,
            final_timestep=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
            exclude_y_boundary=exclude_y_boundary,
            exclude_x_boundary=exclude_x_boundary,
        )
    elif run_density_error_growth:
        output_dir = run_density_error_growth_comparison(
            algorithm_name=algorithm_name,
            hardware_name=hardware_name,
            max_timesteps=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
            exclude_y_boundary=exclude_y_boundary,
        )
    elif run_density_comparison:
        output_dir = run_noiseless_vs_noise_density_comparison(
            algorithm_name=algorithm_name,
            hardware_name=hardware_name,
            final_timestep=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
            exclude_y_boundary=exclude_y_boundary,
        )
    else:
        output_dir = run_hardware_noise_analysis(
            algorithm_name=algorithm_name,
            hardware_name=hardware_name,
            max_timesteps=max_timesteps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
            config_path=DEFAULT_CONFIG_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
        )

    print(f"QLBM hardware noise-analysis output: {output_dir}")
    print(f"Analysis files: {output_dir / 'analysis'}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
