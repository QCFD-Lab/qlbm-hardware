import sys
from pathlib import Path

from qiskit import QuantumCircuit


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.phase_poly_optimizer.optimizer_benchmark_simulation import (
    build_phase_poly_intermediate_hardware_config,
    counts_simulation_skip_reason,
    compare_metric_summary,
    compare_normalized_counts,
    run_intermediate_native_phase_poly_pipeline,
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


def test_final_simulation_decomposes_non_aer_native_gates(tmp_path):
    circuit = QuantumCircuit(2, 2)
    circuit.iswap(0, 1)
    circuit.measure([0, 1], [0, 1])
    lattice = FakeLattice()

    counts = run_final_counts_simulation(
        circuit,
        lattice,
        tmp_path,
        final_timestep=1,
        num_shots=16,
        seed_simulator=123,
    )

    assert counts == {"00": 16}
    assert lattice.results[0]["result"].saved_timesteps == [(1, {"00": 16})]


def test_counts_simulation_skip_reason_for_large_compact_circuits():
    small = QuantumCircuit(3)
    large = QuantumCircuit(28)

    reason = counts_simulation_skip_reason(
        small,
        large,
        max_count_simulation_qubits=24,
    )

    assert reason["skipped"] is True
    assert reason["unoptimized_compact_qubits"] == 3
    assert reason["optimized_compact_qubits"] == 28
    assert counts_simulation_skip_reason(small, large, None) is None


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


def test_phase_poly_intermediate_config_keeps_connectivity_and_adds_cx_basis():
    hardware_config = {
        "id": "cz_backend",
        "num_qubits": 2,
        "basis_gates": ["rz", "sx", "x", "cz"],
        "coupling_map": [[0, 1], [1, 0]],
        "gate_times_s": {"rz": 0.0, "sx": 10e-9, "x": 10e-9, "cz": 80e-9},
    }

    intermediate = build_phase_poly_intermediate_hardware_config(hardware_config)

    assert intermediate["basis_gates"] == ["rz", "sx", "x", "cx"]
    assert intermediate["coupling_map"] == [[0, 1], [1, 0]]
    assert intermediate["gate_times_s"]["cx"] == 80e-9


def test_metric_summary_reports_resource_deltas():
    before = {
        "depth": 10,
        "size": 15,
        "num_1q_ops": 8,
        "num_2q_ops": 4,
        "op_counts": {"rz": 6, "cz": 4},
    }
    after = {
        "depth": 9,
        "size": 14,
        "num_1q_ops": 9,
        "num_2q_ops": 2,
        "op_counts": {"rz": 7, "cz": 2},
    }

    assert compare_metric_summary(before, after) == {
        "depth_delta": -1,
        "size_delta": -1,
        "num_1q_ops_delta": 1,
        "num_2q_ops_delta": -2,
        "op_count_deltas": {"cz": -2, "rz": 1},
    }


def test_intermediate_native_pipeline_retranspiles_to_native_basis():
    circuit = QuantumCircuit(2, 2)
    circuit.cx(0, 1)
    circuit.rz(0.2, 1)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    hardware_config = {
        "id": "cz_backend",
        "architecture": "cz test backend",
        "num_qubits": 2,
        "basis_gates": ["rz", "sx", "x", "cz"],
        "coupling_map": [[0, 1], [1, 0]],
        "gate_times_s": {"rz": 0.0, "sx": 10e-9, "x": 10e-9, "cz": 80e-9},
        "measurement_time_s": 1e-6,
    }

    result = run_intermediate_native_phase_poly_pipeline(
        circuit,
        hardware_config,
        optimizer_name="architecture_aware",
        optimization_level=0,
        seed_transpiler=42,
    )

    baseline_native = result["baseline_native_report"]["transpiled"]
    intermediate = result["intermediate_report"]["transpiled"]
    optimized_native = result["optimized_native_report"]["transpiled"]

    assert result["pipeline"] == "intermediate_native"
    assert intermediate["op_counts"].get("cx", 0) > 0
    assert baseline_native["op_counts"].get("cz", 0) > 0
    assert optimized_native["op_counts"].get("cx", 0) == 0
    assert result["optimized_native_report"]["transpiled_compatibility"]["compatible"]
