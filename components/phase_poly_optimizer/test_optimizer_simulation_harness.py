import sys
from pathlib import Path

from qiskit import QuantumCircuit


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.phase_poly_optimizer.optimizer_benchmark_simulation import (
    compare_normalized_counts,
    run_final_counts_simulation,
)
from components.resource_estimator.benchmark_qlbm_resources_v2 import (
    build_full_logical_circuit,
)


class FakeResult:
    def __init__(self):
        self.visualized_geometry = False
        self.saved_timesteps = []

    def visualize_geometry(self):
        self.visualized_geometry = True

    def save_timestep_counts(self, counts, timestep):
        self.saved_timesteps.append((timestep, dict(counts)))


class FakeLattice:
    def __init__(self):
        self.results = []

    def create_result(self, output_directory, output_file_name):
        result = FakeResult()
        self.results.append(
            {
                "output_directory": output_directory,
                "output_file_name": output_file_name,
                "result": result,
            }
        )
        return result


def test_full_logical_composition_uses_repeated_algorithm_and_measurement():
    initial_conditions = QuantumCircuit(2)
    initial_conditions.x(0)

    algorithm = QuantumCircuit(2)
    algorithm.cx(0, 1)

    postprocessing = QuantumCircuit(2)
    postprocessing.x(1)

    measurement = QuantumCircuit(2, 2)
    measurement.measure([0, 1], [0, 1])

    full_circuit = build_full_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        num_timesteps=2,
    )

    assert [inst.operation.name for inst in full_circuit.data] == [
        "x",
        "cx",
        "cx",
        "x",
        "measure",
        "measure",
    ]


def test_manual_final_simulation_saves_only_final_timestep(tmp_path):
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    lattice = FakeLattice()

    counts = run_final_counts_simulation(
        circuit,
        lattice,
        tmp_path,
        final_timestep=5,
        num_shots=16,
        seed_simulator=123,
    )

    assert counts == {"1": 16}
    assert len(lattice.results) == 1
    fake_result = lattice.results[0]["result"]
    assert fake_result.visualized_geometry is True
    assert fake_result.saved_timesteps == [(5, {"1": 16})]


def test_normalized_counts_comparison_passes_for_equivalent_circuits(tmp_path):
    baseline = QuantumCircuit(1, 1)
    baseline.measure(0, 0)

    optimized = QuantumCircuit(1, 1)
    optimized.x(0)
    optimized.x(0)
    optimized.measure(0, 0)

    baseline_counts = run_final_counts_simulation(
        baseline,
        FakeLattice(),
        tmp_path / "baseline",
        final_timestep=3,
        num_shots=32,
        seed_simulator=123,
    )
    optimized_counts = run_final_counts_simulation(
        optimized,
        FakeLattice(),
        tmp_path / "optimized",
        final_timestep=3,
        num_shots=32,
        seed_simulator=123,
    )

    comparison = compare_normalized_counts(
        baseline_counts,
        optimized_counts,
        max_probability_delta_tolerance=0.0,
    )

    assert comparison["match"] is True
    assert comparison["max_probability_delta"] == 0.0
