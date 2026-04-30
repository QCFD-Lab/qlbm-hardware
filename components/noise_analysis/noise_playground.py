import numpy as np
import json
from pathlib import Path
import time

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

from qiskit_aer import AerSimulator

from qlbm.components import ABGridMeasurement, ABInitialConditions, ABQLBM, EmptyPrimitive
from qlbm.infra import QiskitRunner, SimulationConfig
from qlbm.lattice import ABLattice
from qlbm.tools.utils import create_directory_and_parents

# Simulation settings
NUM_SHOTS = 2*12
NUM_STEPS = 1
OPTIMIZATION_LEVEL = 0
TARGET_PLATFORM = "QISKIT"
COMPILER_PLATFORM = "QISKIT"
STATEVECTOR_SAMPLING = False
SAVE_STATEVECTOR_TO_DISK = False


def build_depolarizing_noise_model() -> NoiseModel:
    """Build a simple noise model for noisy simulation."""
    noise_model = NoiseModel()

    # Simple depolarizing errors
    single_qubit_error = depolarizing_error(0.001, 1)
    two_qubit_error = depolarizing_error(0.01, 2)

    # Add gate noise
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ["h", "x", "y", "z", "sx", "rz"])
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx", "cz", "swap", "cp"])

    return noise_model


def create_simulation_config(lattice, algorithm, initial_conditions, noisy_backend):
    """Create a QLBM SimulationConfig for noisy Qiskit execution."""
    return SimulationConfig(
        initial_conditions=initial_conditions,
        algorithm=algorithm,
        postprocessing=EmptyPrimitive(lattice),
        measurement=ABGridMeasurement(lattice),
        target_platform=TARGET_PLATFORM,
        compiler_platform=COMPILER_PLATFORM,
        optimization_level=OPTIMIZATION_LEVEL,
        statevector_sampling=STATEVECTOR_SAMPLING,
        execution_backend=noisy_backend,
        sampling_backend=noisy_backend,
    )


def run_noisy_qlbm_simulation():
    """Run the unoptimized QLBM circuit with a noise model."""
    output_dir = "output/noise_playground/unoptimized"
    create_directory_and_parents(output_dir)

    lattice = ABLattice({
        "lattice": {"dim": {"x": 8, "y": 4}, "velocities": "d2q9"},
          "geometry": []
    })
    algorithm = ABQLBM(lattice)
    initial_conditions = ABInitialConditions(lattice)

    noise_model = build_depolarizing_noise_model()
    noisy_backend = AerSimulator(method="density_matrix", noise_model=noise_model)

    cfg = create_simulation_config(lattice, algorithm, initial_conditions, noisy_backend)
    cfg.prepare_for_simulation()

    runner = QiskitRunner(
        cfg,
        lattice,
        save_statevector_to_disk=SAVE_STATEVECTOR_TO_DISK,
    )

    result = runner.run(
        NUM_STEPS,
        NUM_SHOTS,
        output_dir,
        statevector_snapshots=False,
    )

    print("Noisy QLBM simulation complete.")
    print(f"Output directory: {output_dir}")
    return result


if __name__ == "__main__":
    start_time = time.time()
    run_noisy_qlbm_simulation()
    print("--- %s seconds ---" % (time.time() - start_time))
