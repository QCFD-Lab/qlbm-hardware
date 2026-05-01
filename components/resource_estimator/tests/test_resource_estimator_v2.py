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
        "gate_fidelities": {"h": 0.999, "x": 0.999, "cx": 0.99},
        "measurement_time_s": 1e-6,
        "measurement_fidelity": 0.95,
        "coherence": {"t1_s": 50e-6, "t2_s": 40e-6},
    }


def test_extract_metrics_for_simple_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    metrics = QLBMResourceEstimator(simple_hardware_config()).extract_metrics(qc)

    assert metrics["num_qubits"] == 2
    assert metrics["active_qubits"] == 2
    assert metrics["num_1q_ops"] == 1
    assert metrics["num_2q_ops"] == 1
    assert metrics["num_measure"] == 0


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


def test_time_estimates_serial_and_critical_path():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)

    timing = QLBMResourceEstimator(simple_hardware_config()).estimate_time(qc)

    assert timing["serial_time_s"] == pytest.approx(120e-9)
    assert timing["critical_path_time_s"] == pytest.approx(110e-9)
    assert timing["warnings"] == []


def test_measurement_free_circuit_marks_readout_not_applicable():
    qc = QuantumCircuit(1)
    qc.h(0)

    fidelity = QLBMResourceEstimator(simple_hardware_config()).estimate_fidelity(qc)

    assert fidelity["readout_applicable"] is False
    assert fidelity["readout_success_probability"] is None


def test_measured_circuit_includes_readout_counts_and_fidelity():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    estimator = QLBMResourceEstimator(simple_hardware_config())
    metrics = estimator.extract_metrics(qc)
    timing = estimator.estimate_time(qc)
    fidelity = estimator.estimate_fidelity(qc)

    assert metrics["num_measure"] == 1
    assert timing["measurement_time_included"] is True
    assert fidelity["readout_applicable"] is True
    assert fidelity["readout_success_probability"] == pytest.approx(0.95)


def test_missing_hardware_fields_warn_without_failing():
    qc = QuantumCircuit(1)
    qc.h(0)

    estimator = QLBMResourceEstimator({"num_qubits": 1, "basis_gates": ["h"]})
    timing = estimator.estimate_time(qc)
    fidelity = estimator.estimate_fidelity(qc)
    coherence = estimator.estimate_coherence(qc)

    assert timing["unknown_gate_times"] == ["h"]
    assert fidelity["missing_gate_fidelities"] == ["h"]
    assert "Missing coherence.t1_s" in coherence["warnings"]


def test_compose_for_measurement_adds_measurements():
    main = QuantumCircuit(1)
    main.h(0)
    measurement = QuantumCircuit(1, 1)
    measurement.measure(0, 0)

    estimator = QLBMResourceEstimator(simple_hardware_config())
    combined = estimator.compose_for_measurement(main, measurement)

    assert combined.count_ops()["h"] == 1
    assert combined.count_ops()["measure"] == 1
