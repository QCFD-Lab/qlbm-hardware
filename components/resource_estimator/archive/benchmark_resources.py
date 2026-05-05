from qlbm import ABLattice, MSLattice
from qlbm.components import ABQLBM, MSQLBM
from components.phase_poly_optimizer import A2APhasePoly, TopologyAwarePhasePolyOptimizer
from components.resource_estimator import ResourceEstimator
from qiskit import QuantumCircuit
import json


def unique_gates(circuit: QuantumCircuit) -> set:
    """Return the set of unique gate names in a Qiskit circuit."""
    return set(circuit.count_ops().keys())


def make_qlbm_circuit() -> QuantumCircuit:
    lattice2d_4x8_0_obs_q4 = {"lattice": {"dim": {"x": 4, "y": 8}, "velocities": "d2q9"}, "geometry": [], }
    latticeAB = ABLattice(lattice2d_4x8_0_obs_q4)
    qc_abqlbm_4x8_0_obs_q4 = ABQLBM(latticeAB).circuit

    lattice_2d_32x32_3_obs = {"lattice": {"dim": {"x": 32, "y": 32}, "velocities": {"x": 4, "y": 4}},
                              "geometry": [{"shape": "cuboid", "x": [18, 20], "y": [6, 25], "boundary": "specular"},
                                           {"shape": "cuboid", "x": [23, 25], "y": [3, 17], "boundary": "specular"},
                                           {"shape": "cuboid", "x": [28, 29], "y": [16, 29], "boundary": "specular"}]}

    latticeMS_32x32_3_OBS = MSLattice(lattice_2d_32x32_3_obs)
    qc_msqlbm_32x32_3_OBS = MSQLBM(latticeMS_32x32_3_OBS).circuit

    return qc_abqlbm_4x8_0_obs_q4

if __name__ == "__main__":
    with open("../config.json", "r") as f:
        configs = json.load(f)

    willow_config = configs["superconducting_google_willow_2024"]
    estimator_willow = ResourceEstimator(willow_config)

    qc_msqlbm_32x32_3_OBS = make_qlbm_circuit()
    print(unique_gates(qc_msqlbm_32x32_3_OBS))
    print(qc_msqlbm_32x32_3_OBS.num_qubits)

    transpiled_result = estimator_willow.estimate(qc_msqlbm_32x32_3_OBS)
    metrics = transpiled_result["metrics"]
    transpiled_circuit = transpiled_result["transpiled_circuit"]
    print(metrics)
    print(unique_gates(transpiled_circuit))

    optimizer = TopologyAwarePhasePolyOptimizer(transpiled_circuit, estimator_willow.coupling_map)
    optimized_circuit = optimizer.optimize()
    print(len(optimizer.blocks))

    optimized_result = estimator_willow.estimate(optimized_circuit)
    metrics_optimized = optimized_result["metrics"]
    transpiled_circuit = optimized_result["transpiled_circuit"]
    print(metrics_optimized)