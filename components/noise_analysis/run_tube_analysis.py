"""Run a D2Q9 tube-flow noise-analysis experiment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _bootstrap_local_paths() -> None:
    project_root = Path(__file__).resolve().parents[3]
    qlbm_source = project_root / "qlbm"
    if str(qlbm_source) not in sys.path:
        sys.path.insert(0, str(qlbm_source))
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))


_bootstrap_local_paths()
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "output" / ".matplotlib"),
)

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qlbm.components import ABGridMeasurement, ABQLBM, EmptyPrimitive
from qlbm.infra import SimulationConfig
from qlbm.lattice import ABLattice

from config import (
    BeamConfig,
    NoiseConfig,
    RunConfig,
    TubeAnalysisConfig,
    TubeConfig,
    build_tube_lattice_config,
)
from initial_conditions import build_single_beam_initial_condition
from noise_models import build_noise_model, choose_backend_method
from observables import counts_to_flow_fields, save_counts, save_flow_fields, save_profile_plots


def build_lattice(config: TubeAnalysisConfig) -> ABLattice:
    """Build the configured tube lattice."""

    return ABLattice(build_tube_lattice_config(config.tube))


def build_backend(config: TubeAnalysisConfig) -> AerSimulator:
    """Build the Aer backend for the configured noise model."""

    noise_model = build_noise_model(config.noise)
    method = choose_backend_method(config.noise, config.run.backend_method)
    if noise_model is None:
        return AerSimulator(method=method)
    return AerSimulator(method=method, noise_model=noise_model)


def build_simulation_config(config: TubeAnalysisConfig, lattice: ABLattice, backend: AerSimulator) -> SimulationConfig:
    """Build and compile the QLBM simulation configuration."""

    initial_condition = build_single_beam_initial_condition(lattice, config.tube, config.beam)
    simulation_config = SimulationConfig(
        initial_conditions=initial_condition,
        algorithm=ABQLBM(lattice),
        postprocessing=EmptyPrimitive(lattice),
        measurement=ABGridMeasurement(lattice, measure_velocity_qubits=True),
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=config.run.optimization_level,
        statevector_sampling=False,
        execution_backend=backend,
        sampling_backend=backend,
    )
    simulation_config.prepare_for_simulation()
    return simulation_config


def build_step_circuit(simulation_config: SimulationConfig, step: int) -> QuantumCircuit:
    """Build a measured circuit for exactly ``step`` QLBM timesteps."""

    measurement = simulation_config.measurement
    circuit = QuantumCircuit(*(measurement.qregs + measurement.cregs))
    circuit.compose(
        simulation_config.initial_conditions,
        qubits=range(simulation_config.initial_conditions.num_qubits),
        inplace=True,
    )
    for _ in range(step):
        circuit.compose(simulation_config.algorithm.copy(), inplace=True)
    circuit.compose(simulation_config.postprocessing.copy(), inplace=True)
    circuit.compose(simulation_config.measurement.copy(), inplace=True)
    return circuit


def write_run_metadata(output_dir: Path, config: TubeAnalysisConfig, lattice: ABLattice) -> None:
    """Write the run configuration and lattice JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config.as_dict(), file, indent=2, sort_keys=True)
    with (output_dir / "lattice.json").open("w", encoding="utf-8") as file:
        file.write(lattice.to_json())


def run_analysis(config: TubeAnalysisConfig) -> Path:
    """Run the configured experiment and save all analysis outputs."""

    config.validate()
    output_dir = config.run_directory()
    lattice = build_lattice(config)
    write_run_metadata(output_dir, config, lattice)

    if config.run.config_only:
        return output_dir

    backend = build_backend(config)
    simulation_config = build_simulation_config(config, lattice, backend)

    for step in range(config.run.num_steps + 1):
        circuit = build_step_circuit(simulation_config, step)
        result = backend.run(circuit, shots=config.run.num_shots).result()
        counts = result.get_counts()
        fields = counts_to_flow_fields(counts, lattice)

        save_counts(output_dir, counts, step)
        save_flow_fields(output_dir, fields, step)
        if config.run.create_plots:
            save_profile_plots(output_dir, fields, step)

    return output_dir


def main() -> None:
    # Edit these variables for each experiment.
    width = 16
    height = 8

    beam_x = 0
    beam_y = None
    velocity_channel = 5

    steps = 1
    shots = 4096
    optimization_level = 0
    output_root = RunConfig().output_root
    backend_method = None
    create_plots = True
    config_only = False

    noise = "none"
    single_qubit_probability = 0.001
    two_qubit_probability = 0.01
    damping_probability = 0.001
    phase_probability = 0.001
    gate_error_probability = 0.001

    config = TubeAnalysisConfig(
        tube=TubeConfig(width=width, height=height),
        beam=BeamConfig(
            x=beam_x,
            y=beam_y,
            velocity_channel=velocity_channel,
        ),
        noise=NoiseConfig(
            kind=noise,
            single_qubit_probability=single_qubit_probability,
            two_qubit_probability=two_qubit_probability,
            damping_probability=damping_probability,
            phase_probability=phase_probability,
            gate_error_probability=gate_error_probability,
        ),
        run=RunConfig(
            num_steps=steps,
            num_shots=shots,
            optimization_level=optimization_level,
            output_root=output_root,
            backend_method=backend_method,
            create_plots=create_plots,
            config_only=config_only,
        ),
    )
    output_dir = run_analysis(config)
    print(f"Tube analysis output: {output_dir}")


if __name__ == "__main__":
    main()
