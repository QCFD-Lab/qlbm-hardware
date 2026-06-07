"""Shared setup and small helpers for QLBM noise experiments."""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
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

from components.noise_analysis.noise_models import build_noise_model  # noqa: E402
from components.noise_analysis.qlbm_builders import CASE_BUILDERS, QLBMCase  # noqa: E402

DEFAULT_CONFIG_PATH = (
    QLBM_HARDWARE_ROOT / "components" / "resource_estimator" / "config.json"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "hardware_noise_runs"


def load_hardware_configs(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    """Load hardware configurations used by the resource estimator."""
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_hardware_config(hardware_name: str, config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load one hardware config and fail with a useful message if it is missing."""

    hardware_configs = load_hardware_configs(config_path)
    if hardware_name not in hardware_configs:
        raise ValueError(
            f"Unknown hardware_name={hardware_name!r}. "
            f"Expected one of {sorted(hardware_configs)}."
        )
    return hardware_configs[hardware_name]


def require_hardware_names(hardware_configs: dict[str, dict[str, Any]], hardware_names: list[str]) -> None:
    """Validate a non-empty list of hardware names against loaded configs."""

    if not hardware_names:
        raise ValueError("hardware_names must contain at least one hardware config.")
    unknown_hardware = [name for name in hardware_names if name not in hardware_configs]
    if unknown_hardware:
        raise ValueError(
            f"Unknown hardware names {unknown_hardware!r}. "
            f"Expected values from {sorted(hardware_configs)}."
        )


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
        single_qubit_probability=noise_parameters.get(
            "single_qubit_probability", 0.001
        ),
        two_qubit_probability=noise_parameters.get("two_qubit_probability", 0.01),
    )
    options = {"method": backend_method, "seed_simulator": seed_simulator}
    if noise_model is None:
        return AerSimulator(**options)
    return AerSimulator(**options, noise_model=noise_model)


def sample_counts(backend: AerSimulator, circuit: QuantumCircuit, num_shots: int) -> dict[str, int]:
    """Run one sampled circuit and return a plain counts dictionary."""
    return dict(backend.run(circuit, shots=num_shots).result().get_counts())


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


def write_lattice_json(path: Path, lattice: Any) -> None:
    """Write the lattice JSON string produced by the QLBM lattice object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(lattice.to_json())


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
    raise RuntimeError(
        f"Transpiled circuit is not hardware compatible: {compatibility}"
    )


def build_case(algorithm_name: str, num_timesteps: int, measure_velocity_qubits: bool = False) -> QLBMCase:
    """Build a QLBM case, optionally using velocity-resolved AB measurement."""

    if algorithm_name not in CASE_BUILDERS:
        raise ValueError(
            f"Unknown algorithm_name={algorithm_name!r}. "
            f"Expected one of {sorted(CASE_BUILDERS)}."
        )
    if measure_velocity_qubits:
        if algorithm_name != "abqlbm":
            raise ValueError(
                "Velocity-profile analysis currently supports only abqlbm."
            )
        return CASE_BUILDERS[algorithm_name](
            num_timesteps=num_timesteps,
            measure_velocity_qubits=True,
        )
    return CASE_BUILDERS[algorithm_name](num_timesteps=num_timesteps)


def _hardware_slug(hardware_name: str) -> str:
    if "ibm_nighthawk" in hardware_name:
        return "ibm_nighthawk"
    if "ibm_eagle" in hardware_name:
        return "ibm_eagle"
    if "neutral_atom" in hardware_name:
        return "neutral_atom"
    return (
        hardware_name.replace("superconducting_", "")
        .replace("_2024", "")
        .replace("_2023", "")
    )


def _hardware_plot_label(hardware_name: str, hardware_config: dict[str, Any]) -> str:
    if "ibm_nighthawk" in hardware_name:
        return "IBM Nighthawk r1"
    if "ibm_eagle" in hardware_name:
        return "IBM Eagle r3"
    if "neutral_atom" in hardware_name:
        return "Neutral atom"
    return str(hardware_config.get("device_name") or hardware_name)
