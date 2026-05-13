import sys
from pathlib import Path

import pytest
from qiskit import QuantumCircuit


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.resource_estimator import QLBMResourceEstimator


def simple_hardware_config():
    return {
        "id": "test_backend",
        "architecture": "linear test backend",
        "num_qubits": 3,
        "basis_gates": ["h", "x", "cx"],
        "coupling_map": [[0, 1], [1, 0], [1, 2], [2, 1]],
        "gate_times_s": {"h": 10e-9, "x": 10e-9, "cx": 100e-9},
        "measurement_time_s": 1e-6,
    }


def test_extract_metrics_for_simple_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    metrics = QLBMResourceEstimator(simple_hardware_config()).extract_metrics(qc)

    assert metrics["num_qubits"] == 2
    assert metrics["active_qubits"] == 2
    assert metrics["active_qubit_indices"] == [0, 1]
    assert metrics["num_1q_ops"] == 1
    assert metrics["num_2q_ops"] == 1
    assert metrics["num_measure"] == 0


def test_active_qubits_ignore_idle_classical_bits_and_barriers():
    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.barrier(1, 2)

    metrics = QLBMResourceEstimator(simple_hardware_config()).extract_metrics(qc)

    assert metrics["active_qubits"] == 1
    assert metrics["active_qubit_indices"] == [0]


def test_basis_gate_check_reports_unsupported_gate():
    qc = QuantumCircuit(1)
    qc.z(0)

    estimator = QLBMResourceEstimator(simple_hardware_config())

    assert not estimator.obeys_basis_gates(qc)
    assert estimator.unsupported_gates(qc) == ["z"]


def test_coupling_map_check_passes_and_fails():
    estimator = QLBMResourceEstimator(simple_hardware_config())

    valid = QuantumCircuit(3)
    valid.cx(0, 1)

    invalid = QuantumCircuit(3)
    invalid.cx(0, 2)

    assert estimator.obeys_coupling_map(valid)
    assert not estimator.obeys_coupling_map(invalid)
    assert estimator.coupling_violations(invalid)[0]["qubits"] == [0, 2]
    assert estimator.check_compatibility(invalid)["coupling_map_ok"] is False
    assert "num_coupling_violations" not in estimator.check_compatibility(invalid)
    assert "coupling_violations" not in estimator.check_compatibility(invalid)

    detailed = estimator.check_compatibility(
        invalid,
        include_coupling_violations=True,
    )
    assert detailed["num_coupling_violations"] == 1
    assert detailed["coupling_violations"][0]["qubits"] == [0, 2]


def test_transpilation_returns_metrics():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)

    estimator = QLBMResourceEstimator(simple_hardware_config())
    transpiled = estimator.transpile_circuit(qc, optimization_level=1, seed_transpiler=42)
    metrics = estimator.extract_metrics(transpiled)

    assert metrics["num_qubits"] == 3
    assert estimator.obeys_basis_gates(transpiled)
    assert estimator.obeys_coupling_map(transpiled)


def test_estimate_returns_compacted_simulation_circuit_after_transpilation():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    estimator = QLBMResourceEstimator(simple_hardware_config())
    report = estimator.estimate(qc, optimization_level=1, seed_transpiler=42)

    assert report["transpiled"]["num_qubits"] == 3
    assert report["transpiled"]["active_qubits"] == 2
    assert report["transpiled_compatibility"]["coupling_map_ok"] is True
    assert report["simulation"]["num_qubits"] == 2
    assert report["simulation"]["active_qubits"] == 2
    assert report["simulation_circuit"].num_qubits == 2


def test_compatibility_reports_used_qubit_capacity():
    config = simple_hardware_config()
    config["num_qubits"] = 2
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)

    estimator = QLBMResourceEstimator(config)
    compatibility = estimator.check_compatibility(qc)

    assert compatibility["used_qubits"] == 2
    assert compatibility["available_qubits"] == 2
    assert compatibility["qubit_capacity_ok"] is True
    assert compatibility["compatible"] is True


def test_hardware_validation_reports_topology_mismatch():
    estimator = QLBMResourceEstimator(
        {
            "num_qubits": 2,
            "basis_gates": ["h", "cx"],
            "coupling_map": [[0, 1], [1, 2]],
        }
    )

    validation = estimator.validate_hardware_config()

    assert validation["topology_num_qubits"] == 3
    assert validation["topology_matches_num_qubits"] is False
    assert validation["warnings"]


def test_2d_grid_disabled_qubits_match_declared_topology_size():
    estimator = QLBMResourceEstimator(
        {
            "num_qubits": 3,
            "basis_gates": ["h", "cx"],
            "coupling_type": "2d_grid",
            "coupling_params": {"rows": 2, "cols": 2, "disabled_qubits": [3]},
        }
    )

    validation = estimator.validate_hardware_config()

    assert validation["topology_num_qubits"] == 3
    assert validation["topology_matches_num_qubits"] is True
    assert validation["warnings"] == []


def test_time_estimates_serial_critical_path_and_scheduled_duration():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)

    timing = QLBMResourceEstimator(simple_hardware_config()).estimate_time(qc)

    assert timing["timing_model"] == "qiskit_asap_schedule"
    assert timing["serial_time_s"] == pytest.approx(120e-9)
    assert timing["critical_path_time_s"] == pytest.approx(110e-9)
    assert timing["scheduled_duration_s"] == pytest.approx(110e-9)
    assert timing["max_idle_time_s"] == pytest.approx(0.0)
    assert timing["warnings"] == []


def test_scheduled_duration_models_parallel_final_measurement():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    timing = QLBMResourceEstimator(simple_hardware_config()).estimate_time(qc)

    assert timing["serial_time_s"] == pytest.approx(2120e-9)
    assert timing["critical_path_time_s"] == pytest.approx(1110e-9)
    assert timing["scheduled_duration_s"] == pytest.approx(1110e-9)
    assert timing["max_idle_time_s"] == pytest.approx(0.0)


def test_scheduled_timing_reports_max_idle_time_before_and_after_activity():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(0)
    qc.x(1)

    timing = QLBMResourceEstimator(simple_hardware_config()).estimate_time(qc)

    assert timing["scheduled_duration_s"] == pytest.approx(20e-9)
    assert timing["max_idle_time_s"] == pytest.approx(10e-9)


def test_measured_circuit_includes_readout_counts_and_timing():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    estimator = QLBMResourceEstimator(simple_hardware_config())
    metrics = estimator.extract_metrics(qc)
    timing = estimator.estimate_time(qc)

    assert metrics["num_measure"] == 1
    assert timing["measurement_time_included"] is True
    assert timing["scheduled_duration_s"] == pytest.approx(1010e-9)
    assert timing["max_idle_time_s"] == pytest.approx(0.0)


def test_missing_hardware_fields_warn_without_failing():
    qc = QuantumCircuit(1)
    qc.h(0)

    estimator = QLBMResourceEstimator({"num_qubits": 1, "basis_gates": ["h"]})
    timing = estimator.estimate_time(qc)

    assert timing["unknown_gate_times"] == ["h"]
    assert timing["scheduled_duration_s"] is None
    assert timing["max_idle_time_s"] is None
    assert (
        "Scheduled duration unavailable because gate durations are missing"
        in timing["warnings"]
    )
