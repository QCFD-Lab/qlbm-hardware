"""Run one simplified QLBM noise-analysis case and write Paraview outputs."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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

from qiskit_aer import AerSimulator

from noise_models import build_noise_model
from qlbm_builders import CASE_BUILDERS, create_qiskit_runner


def build_backend(noise_kind: str, backend_method: str, noise_parameters: dict):
    """Build the Aer backend used for both execution and sampling."""
    noise_model = build_noise_model(kind=noise_kind, **noise_parameters)
    if noise_kind == "none" or noise_model is None:
        return AerSimulator(method=backend_method)
    return AerSimulator(method=backend_method, noise_model=noise_model)


def write_run_metadata(output_dir: Path, metadata: dict) -> None:
    """Write a small JSON file describing the run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def build_run_metadata(
    algorithm_name: str,
    case_label: str,
    num_timesteps: int,
    runner_steps: int,
    num_shots: int,
    optimization_level: int,
    noise_kind: str,
    backend_method: str,
    noise_parameters: dict,
) -> dict:
    """Build metadata, omitting noise probabilities for noiseless runs."""

    metadata = {
        "algorithm_name": algorithm_name,
        "case_label": case_label,
        "num_timesteps": num_timesteps,
        "runner_steps": runner_steps,
        "num_shots": num_shots,
        "optimization_level": optimization_level,
        "noise_kind": noise_kind,
        "backend_method": backend_method,
    }
    if noise_kind != "none":
        metadata["noise_parameters"] = noise_parameters
    return metadata


def main() -> None:
    # Edit these variables for each experiment.
    algorithm_name = "abqlbm"  # Options: "abqlbm", "msqlbm", "spacetime"
    num_timesteps = 10
    num_shots = 4096
    optimization_level = 0
    backend_method = "statevector"

    noise_kind = "amplitude_damping"  # none, depolarizing, amplitude_damping, phase_damping, gate_error
    noise_parameters = {
        "single_qubit_probability": 0.001,
        "two_qubit_probability": 0.001,
        "damping_probability": 0.001,
        "phase_probability": 0.001,
        "gate_error_probability": 0.001,
    }

    output_root = Path(__file__).resolve().parent / "output" / "qlbm_noise_runs"

    if algorithm_name not in CASE_BUILDERS:
        raise ValueError(
            f"Unknown algorithm_name={algorithm_name!r}. "
            f"Expected one of {sorted(CASE_BUILDERS)}."
        )

    backend = build_backend(noise_kind, backend_method, noise_parameters)
    case = CASE_BUILDERS[algorithm_name](
        execution_backend=backend,
        sampling_backend=backend,
        num_timesteps=num_timesteps,
        optimization_level=optimization_level,
    )

    output_dir = output_root / f"{case.label}_{noise_kind}_shots{num_shots}"
    write_run_metadata(
        output_dir,
        build_run_metadata(
            algorithm_name=algorithm_name,
            case_label=case.label,
            num_timesteps=num_timesteps,
            runner_steps=case.runner_steps,
            num_shots=num_shots,
            optimization_level=optimization_level,
            noise_kind=noise_kind,
            backend_method=backend_method,
            noise_parameters=noise_parameters,
        ),
    )

    runner = create_qiskit_runner(case)
    runner.run(
        num_steps=case.runner_steps,
        num_shots=num_shots,
        output_directory=str(output_dir),
        output_file_name="step",
        statevector_snapshots=False,
    )

    print(f"QLBM noise-analysis output: {output_dir}")
    print(f"Paraview files: {output_dir / 'paraview'}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
