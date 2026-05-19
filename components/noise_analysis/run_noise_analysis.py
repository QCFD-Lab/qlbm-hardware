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
) -> dict[str, Any]:
    """Build metadata, omitting manual noise probabilities for noiseless runs."""

    metadata = {
        "algorithm_name": algorithm_name,
        "case_label": case_label,
        "hardware_name": hardware_name,
        "hardware_id": hardware_config.get("id"),
        "max_timesteps": max_timesteps,
        "simulated_timesteps": list(range(0, max_timesteps + 1)),
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
    if algorithm_name not in CASE_BUILDERS:
        raise ValueError(
            f"Unknown algorithm_name={algorithm_name!r}. "
            f"Expected one of {sorted(CASE_BUILDERS)}."
        )

    hardware_config = hardware_configs[hardware_name]
    estimator = QLBMResourceEstimator(hardware_config)
    backend = build_backend(
        noise_kind=noise_kind,
        backend_method=backend_method,
        seed_simulator=seed_simulator,
        noise_parameters=noise_parameters,
        hardware_config=hardware_config,
    )

    final_case = CASE_BUILDERS[algorithm_name](num_timesteps=max_timesteps)
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
        case = CASE_BUILDERS[algorithm_name](num_timesteps=case_timesteps)
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


def main() -> None:
    algorithm_name = "abqlbm"  # Options: "abqlbm", "msqlbm", "spacetime"
    hardware_name = "superconducting_google_willow_2024"
    max_timesteps = 4
    num_shots = 4096
    optimization_level = 1
    seed_transpiler = 42
    seed_simulator = 42
    backend_method = "statevector"

    # none, depolarizing, hardware_depolarizing, thermal_relaxation
    noise_kind = "depolarizing"
    noise_parameters = {
        "single_qubit_probability": 0.001,
        "two_qubit_probability": 0.001,
    }

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
    print(f"Paraview files: {output_dir / 'paraview'}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
