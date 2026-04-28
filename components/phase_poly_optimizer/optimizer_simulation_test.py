#!/usr/bin/env python3
"""
Test the PhasePolyOptimizer class to verify that the all-to-all connectivity algorithm
correctly optimizes a given circuit with phase polynomials.
"""
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.transpiler import CouplingMap

from qlbm.components import ABQLBM
from components.phase_poly_optimizer import A2APhasePoly, TopologyAwarePhasePolyOptimizer, \
    ArchitectureAwarePhasePolyOptimizer
from components.resource_estimator import ResourceEstimator
from tools.simulation_statevector_comparison import compare_statevectors, compare_statevectors_ignoring_global_phase
import json
from qiskit_aer import AerSimulator
from qlbm.components import (
    ABInitialConditions,
    ABGridMeasurement,
    EmptyPrimitive,
)
from qlbm.infra import QiskitRunner, SimulationConfig
from qlbm.lattice import ABLattice
from qlbm.tools.utils import create_directory_and_parents

# Configuration Constants
NUM_SHOTS = 2 ** 12
NUM_STEPS = 5
OPTIMIZATION_LEVEL = 0
TARGET_PLATFORM = "QISKIT"
COMPILER_PLATFORM = "QISKIT"
STATEVECTOR_SAMPLING = True
SAVE_STATEVECTOR_TO_DISK = True


def load_resource_estimator_config(config_path: str) -> dict:
    """Loads the resource estimator configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def run_resource_estimation(circuit, qhw_config: dict, force_all_coupled = False, remove_idle_qubits = True):
    """Estimates resources and transpiles the circuit."""
    estimator = ResourceEstimator(qhw_config, force_no_coupling=force_all_coupled, remove_idle_qubits=remove_idle_qubits)
    result = estimator.estimate(circuit)
    return result["metrics"], result["transpiled_circuit"], estimator.coupling_map

def create_simulation_config(algorithm, latticeAB, initial_conditions):
    """Creates a QLBM SimulationConfig for Qiskit execution."""
    return SimulationConfig(
        initial_conditions=initial_conditions,
        algorithm=algorithm,
        postprocessing=EmptyPrimitive(latticeAB),
        measurement=ABGridMeasurement(latticeAB),
        target_platform=TARGET_PLATFORM,
        compiler_platform=COMPILER_PLATFORM,
        optimization_level=OPTIMIZATION_LEVEL,
        statevector_sampling=STATEVECTOR_SAMPLING,
        execution_backend=AerSimulator(method="statevector"),
        sampling_backend=AerSimulator(method="statevector"),
    )

def run_simulation_pipeline(circuit, latticeAB, initial_conditions, output_dir: str):
    """
    Executes the full simulation pipeline.
    """
    create_directory_and_parents(output_dir)
    cfg = create_simulation_config(circuit, latticeAB, initial_conditions)
    cfg.prepare_for_simulation()

    runner = QiskitRunner(cfg, latticeAB, save_statevector_to_disk=SAVE_STATEVECTOR_TO_DISK)

    # run simulation
    runner.run(
        NUM_STEPS,
        NUM_SHOTS,
        output_dir,
        statevector_snapshots=True
    )
    return runner

if __name__ == "__main__":
    # setup for simulation
    REMOVE_IDLE_QUBITS = True
    TOPOLOGY_AWARE = True

    config_path = "../resource_estimator/config.json"
    configs = load_resource_estimator_config(config_path)
    hw_config = configs["superconducting_google_willow_2024"]

    output_file_path_base = "output/ab-d2q9-8x4-1"
    output_file_path_optimized = "output/ab-d2q9-8x4-1-optimized"

    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 4}, "velocities": "d2q9"},
            "geometry": [
                {"shape": "cuboid", "x": [5, 7], "y": [2, 3], "boundary": "bounceback"},
            ],
        }
    )

    initial_conditions = ABInitialConditions(lattice)

    qlbm_algo = ABQLBM(lattice)
    base_circuit = qlbm_algo.circuit.copy()

    metrics_unopt, transpiled_circuit, coupling_map = run_resource_estimation(
        circuit=base_circuit,
        qhw_config=hw_config,
    )
    circuit_for_optimization = transpiled_circuit.copy()
    print(f"Unoptimized Metrics: {metrics_unopt}")

    optimizer = ArchitectureAwarePhasePolyOptimizer()
    optimized_circuit = optimizer.optimize(
        circuit=circuit_for_optimization, coupling_map=CouplingMap(coupling_map))

    metrics_opt, transpiled_circuit_opt, _ = run_resource_estimation(
        circuit=optimized_circuit,
        qhw_config=hw_config,
    )
    print(f"Optimized Metrics: {metrics_opt}")

    run_simulation_pipeline(transpiled_circuit, lattice, initial_conditions, output_file_path_base)
    run_simulation_pipeline(optimized_circuit, lattice, initial_conditions, output_file_path_optimized)

    sv_ref = Statevector.from_instruction(transpiled_circuit)
    sv_opt = Statevector.from_instruction(optimized_circuit)

    print("equiv:", sv_ref.equiv(sv_opt))
    print("fidelity:", state_fidelity(sv_ref, sv_opt))

    print("\n--- Comparison Complete ---")
    comparison_result = compare_statevectors(output_file_path_base, output_file_path_optimized)
    print(f"Statevectors match: {comparison_result["all_match"]}")
    print(f"Differences: {comparison_result["differences"]}")
    print(f"Results: {comparison_result["file_results"]}")
