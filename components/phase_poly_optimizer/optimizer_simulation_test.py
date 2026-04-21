#!/usr/bin/env python3
"""
Test the PhasePolyOptimizer class to verify that the all-to-all connectivity algorithm
correctly optimizes a given circuit with phase polynomials.
"""
from qiskit.quantum_info import Statevector, state_fidelity

from qlbm.components import ABQLBM
from components.phase_poly_optimizer import A2APhasePoly, TopologyAwarePhasePolyOptimizer
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
NUM_STEPS = 1
OPTIMIZATION_LEVEL = 0
TARGET_PLATFORM = "QISKIT"
COMPILER_PLATFORM = "QISKIT"
STATEVECTOR_SAMPLING = True
SAVE_STATEVECTOR_TO_DISK = True


def load_resource_estimator_config(config_path: str) -> dict:
    """Loads the resource estimator configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def create_lattice_and_algorithm(lattice_config: dict):
    """Sets up the lattice and the ABQLBM algorithm instance."""
    latticeAB = ABLattice(lattice_config)
    abqlbm = ABQLBM(latticeAB)
    return latticeAB, abqlbm

def run_resource_estimation(circuit, qhw_config: dict, force_no_coupling: bool, remove_idle_qubits: bool):
    """Estimates resources and transpiles the circuit."""
    estimator = ResourceEstimator(qhw_config, force_no_coupling=force_no_coupling, remove_idle_qubits=remove_idle_qubits)
    result = estimator.estimate(circuit)
    return result["metrics"], result["transpiled_circuit"], estimator.coupling_map

def create_simulation_config(abqlbm, latticeAB, output_dir: str):
    """Creates a QLBM SimulationConfig for Qiskit execution."""
    return SimulationConfig(
        initial_conditions=ABInitialConditions(latticeAB),
        algorithm=abqlbm,
        postprocessing=EmptyPrimitive(latticeAB),
        measurement=ABGridMeasurement(latticeAB),
        target_platform=TARGET_PLATFORM,
        compiler_platform=COMPILER_PLATFORM,
        optimization_level=OPTIMIZATION_LEVEL,
        statevector_sampling=STATEVECTOR_SAMPLING,
        execution_backend=AerSimulator(method="statevector"),
        sampling_backend=AerSimulator(method="statevector"),
    )

def run_simulation_pipeline(abqlbm, latticeAB, circuit, output_dir: str):
    """
    Executes the full simulation pipeline.
    """
    # Update the algorithm with the current circuit
    abqlbm.circuit = circuit

    create_directory_and_parents(output_dir)
    cfg = create_simulation_config(abqlbm, latticeAB, output_dir)
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

def optimize_circuit_all_to_all(circuit, debug=False):
    """Wraps the A2APhasePoly logic for all to all coupling."""
    optimizer = A2APhasePoly(circuit)
    optimized_circuit = optimizer.optimize(debug=debug)
    print(f"Optimization completed. Found {len(optimizer.blocks)} blocks.")
    return optimized_circuit

def optimize_circuit_topology_aware(circuit, coupling_map, debug=False):
    """Wraps the TopologyAwarePhasePolyOptimizer logic for all to all coupling."""
    optimizer = TopologyAwarePhasePolyOptimizer(circuit, coupling_map)

    optimized_circuit = optimizer.optimize(debug=debug)
    print(f"Optimization completed. Found {len(optimizer.blocks)} blocks.")
    return optimized_circuit


def optimizer_simulation_test(hardware_config, topology_aware=False, remove_idle_qubits=True, debug=False):
    force_all_to_all_coupling = False
    if not topology_aware:
        force_all_to_all_coupling = True

    lattice_config = {
        "lattice": {"dim": {"x": 8, "y": 4}, "velocities": "d2q9"},
        "geometry": []
    }

    latticeAB, abqlbm = create_lattice_and_algorithm(lattice_config)

    # create base circuit
    base_circuit = abqlbm.circuit

    # unoptimized simulation pipeline
    print(" --- Running Unoptimized Simulation ---")
    metrics_unopt, transpiled_circuit, re_coupling_map = run_resource_estimation(
        circuit=base_circuit,
        qhw_config=hardware_config,
        force_no_coupling=force_all_to_all_coupling,
        remove_idle_qubits=remove_idle_qubits
    )

    print(f"Unoptimized Metrics: {metrics_unopt}")

    # Create a copy for optimization to avoid changing the base simulation circuit
    circuit_for_optimization = transpiled_circuit.copy()

    # Run simulation with the transpiled unoptimized circuit
    run_simulation_pipeline(abqlbm, latticeAB, transpiled_circuit, "output/ab-d2q9-4x8-0")

    if topology_aware:
        print("\n--- Running TOPOLOGY AWARE Optimization ---")
        hw_coupling_map = hardware_config.get("coupling_map", None)
        if hw_coupling_map is None:
            hw_coupling_map = re_coupling_map
        optimized_circuit = optimize_circuit_topology_aware(circuit_for_optimization,
                                                            coupling_map=hw_coupling_map, debug=debug)


    else:
        print("\n--- Running ALL-to-ALL Optimization ---")
        optimized_circuit = optimize_circuit_all_to_all(circuit_for_optimization, debug=debug)

    # optimized simulation pipeline
    print("\n--- Running Optimized Simulation ---")
    metrics_opt, optimized_transpiled_circuit, _ = run_resource_estimation(
        circuit=optimized_circuit,
        qhw_config=hardware_config,
        force_no_coupling=force_all_to_all_coupling,
        remove_idle_qubits=remove_idle_qubits
    )
    print(f"Optimized Metrics: {metrics_opt}")

    sv_ref = Statevector.from_instruction(transpiled_circuit)
    sv_opt = Statevector.from_instruction(optimized_transpiled_circuit)

    print("equiv:", sv_ref.equiv(sv_opt))
    print("fidelity:", state_fidelity(sv_ref, sv_opt))

    # reuse the existing instance but update the circuit
    run_simulation_pipeline(abqlbm, latticeAB, optimized_transpiled_circuit, "output/ab-d2q9-4x8-0-optimized")



if __name__ == "__main__":
    # setup for simulation
    REMOVE_IDLE_QUBITS = True
    TOPOLOGY_AWARE = True

    config_path = "../resource_estimator/config.json"
    configs = load_resource_estimator_config(config_path)
    willow_config = configs["superconducting_google_willow_2024"]

    optimizer_simulation_test(
        hardware_config=willow_config,
        topology_aware=TOPOLOGY_AWARE,
        remove_idle_qubits=REMOVE_IDLE_QUBITS,
        debug=False
    )

    print("\n--- Optimizer Complete ---")

    comparison_result = compare_statevectors_ignoring_global_phase(
        "output/ab-d2q9-4x8-0/statevectors/step_1.npy",
        "output/ab-d2q9-4x8-0-optimized/statevectors/step_1.npy",)

    print(comparison_result)

    # all_match, differences = compare_statevectors("output/ab-d2q9-4x8-0", "output/ab-d2q9-4x8-0-optimized")
    # print(f"Statevectors match: {all_match}")
    # print(f"Differences: {differences}")