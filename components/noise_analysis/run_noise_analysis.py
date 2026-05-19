"""Run hardware-mapped QLBM noise simulations for noise analysis."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from qiskit import QuantumCircuit
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
    save_density_error_growth,
)
from components.noise_analysis.tools.velocity_analysis import analyze_velocity_counts


DEFAULT_CONFIG_PATH = QLBM_HARDWARE_ROOT / "components" / "resource_estimator" / "config.json"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "hardware_noise_runs"


def load_hardware_configs(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    """Load hardware configurations used by the resource estimator."""
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def main() -> None:
    algorithm_name = "abqlbm"  # Options: "abqlbm", "msqlbm", "spacetime"
    hardware_name = "superconducting_google_willow_2024"
    max_timesteps = 10
    num_shots = 4096
    optimization_level = 1
    seed_transpiler = 42
    seed_simulator = 42
    backend_method = "statevector"

    # none, depolarizing, hardware_depolarizing, thermal_relaxation
    noise_kind = "depolarizing"
    noise_parameters = {
        "single_qubit_probability": 0.000001,
        "two_qubit_probability": 0.00001,
    }
    run_density_comparison = False
    run_density_error_growth = False
    run_velocity_profile_comparison = True
    exclude_y_boundary = False
    exclude_x_boundary = False

    if run_velocity_profile_comparison:
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
