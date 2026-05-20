import sys
from pathlib import Path

import pytest
from qiskit import QuantumCircuit


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.resource_estimator import QLBMResourceEstimator
from components.resource_estimator.benchmark_qlbm_resources_v2 import (
    build_full_logical_circuit,
    build_sectioned_logical_circuit,
)


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


def phase_hardware_config():
    return {
        "id": "phase_test_backend",
        "architecture": "phase-polynomial test backend",
        "num_qubits": 4,
        "basis_gates": ["h", "x", "rz", "cx"],
        "gate_times_s": {
            "h": 10e-9,
            "x": 10e-9,
            "rz": 0.0,
            "cx": 100e-9,
        },
        "measurement_time_s": 1e-6,
    }


def build_phase_section_test_circuits():
    full = QuantumCircuit(4)
    full.cx(0, 1)
    full.rz(0.1, 1)
    full.cx(0, 1)
    full.rz(0.2, 1)
    full.h(0)
    full.cx(2, 3)
    full.rz(0.3, 3)
    full.cx(2, 3)
    full.h(2)

    sectioned = QuantumCircuit(4)
    sectioned.cx(0, 1)
    sectioned.rz(0.1, 1)
    sectioned.barrier(label="section_boundary::initial_conditions")
    sectioned.cx(0, 1)
    sectioned.rz(0.2, 1)
    sectioned.h(0)
    sectioned.cx(2, 3)
    sectioned.rz(0.3, 3)
    sectioned.cx(2, 3)
    sectioned.barrier(label="section_boundary::algorithm_step_1")
    sectioned.h(2)
    sectioned.barrier(label="section_boundary::algorithm_step_2")

    return full, sectioned, [
        "initial_conditions",
        "algorithm_step_1",
        "algorithm_step_2",
        "measurement",
    ]


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


def test_estimate_returns_compacted_transpiled_circuit_after_transpilation():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    estimator = QLBMResourceEstimator(simple_hardware_config())
    report = estimator.estimate(qc, optimization_level=1, seed_transpiler=42)

    assert report["transpiled"]["num_qubits"] == 3
    assert report["transpiled"]["active_qubits"] == 2
    assert report["transpiled_compatibility"]["coupling_map_ok"] is True
    assert report["transpiled_compact"]["num_qubits"] == 2
    assert report["transpiled_compact"]["active_qubits"] == 2
    assert report["transpiled_compact_circuit"].num_qubits == 2
    assert "phase_polynomial_analysis" not in report


def test_estimate_pretranspiled_reports_without_retranspiling(monkeypatch):
    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    estimator = QLBMResourceEstimator(simple_hardware_config())

    def fail_transpile(*args, **kwargs):
        raise AssertionError("estimate_pretranspiled must not transpile")

    monkeypatch.setattr(estimator, "transpile_circuit", fail_transpile)
    report = estimator.estimate_pretranspiled(
        qc,
        label="already_physical",
        logical_metrics={"depth": 1, "size": 1, "num_2q_ops": 1},
        phase_polynomial_analysis=True,
    )

    assert report["label"] == "already_physical"
    assert report["transpiled_circuit"] is qc
    assert report["transpiled"]["num_qubits"] == 3
    assert report["transpiled"]["active_qubits"] == 2
    assert report["transpiled_compatibility"]["compatible"] is True
    assert report["transpiled_compatibility"]["coupling_map_ok"] is True
    assert report["transpiled_compatibility"]["num_coupling_violations"] == 0
    assert report["transpiled_compact"]["num_qubits"] == 2
    assert report["transpiled_compact"]["active_qubits"] == 2
    assert report["transpiled_compact_circuit"].num_qubits == 2
    assert report["overheads"]["added_two_qubit_gates"] == 0
    assert report["phase_polynomial_analysis"]["num_blocks"] == 1
    assert "section_analysis" not in report["phase_polynomial_analysis"]


def test_estimate_pretranspiled_omits_phase_polynomial_analysis_by_default():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    report = QLBMResourceEstimator(simple_hardware_config()).estimate_pretranspiled(qc)

    assert "phase_polynomial_analysis" not in report


def test_phase_polynomial_analysis_reports_summary_and_largest_block():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.1, 1)
    qc.h(3)
    qc.cx(2, 3)
    qc.x(0)
    qc.rz(0.2, 3)
    qc.cx(2, 3)
    qc.h(1)

    analysis = QLBMResourceEstimator(simple_hardware_config()).analyze_phase_polynomial_blocks(qc)

    assert analysis["num_blocks"] == 2
    assert analysis["total_block_instructions"] == 6
    assert analysis["total_phase_instructions"] == 5
    assert analysis["total_passthrough_instructions"] == 1
    assert analysis["total_rz"] == 2
    assert analysis["total_cx"] == 3
    assert analysis["average_block_instructions"] == pytest.approx(3.0)
    assert analysis["average_phase_instructions"] == pytest.approx(2.5)
    assert analysis["average_active_qubits"] == pytest.approx(2.0)
    assert analysis["largest_block"] == {
        "start": 4,
        "end": 7,
        "instruction_count": 4,
        "phase_instruction_count": 3,
        "passthrough_instruction_count": 1,
        "active_qubits": [2, 3],
        "active_qubit_count": 2,
        "rz_count": 1,
        "cx_count": 2,
    }


def test_phase_polynomial_analysis_ignores_circuits_without_cx_or_rz_blocks():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()
    qc.x(1)

    analysis = QLBMResourceEstimator(simple_hardware_config()).analyze_phase_polynomial_blocks(qc)

    assert analysis == {
        "num_blocks": 0,
        "total_block_instructions": 0,
        "total_phase_instructions": 0,
        "total_passthrough_instructions": 0,
        "total_rz": 0,
        "total_cx": 0,
        "average_block_instructions": 0.0,
        "average_phase_instructions": 0.0,
        "average_active_qubits": 0.0,
        "largest_block": None,
    }


def test_estimate_includes_phase_polynomial_analysis_when_enabled():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    report = QLBMResourceEstimator(simple_hardware_config()).estimate(
        qc,
        optimization_level=0,
        seed_transpiler=42,
        phase_polynomial_analysis=True,
    )

    assert report["phase_polynomial_analysis"]["num_blocks"] == 1
    assert "section_analysis" not in report["phase_polynomial_analysis"]


def test_phase_polynomial_section_analysis_identifies_max_block_section():
    _, sectioned_circuit, section_names = build_phase_section_test_circuits()
    estimator = QLBMResourceEstimator(phase_hardware_config(), force_no_coupling=True)

    analysis = estimator.analyze_phase_polynomial_sections(
        sectioned_circuit,
        section_names,
        optimization_level=0,
        seed_transpiler=42,
    )

    sections = {section["section"]: section for section in analysis["sections"]}
    assert sections["initial_conditions"]["num_blocks"] == 1
    assert sections["algorithm_step_1"]["num_blocks"] == 2
    assert sections["algorithm_step_2"]["num_blocks"] == 0
    assert sections["measurement"]["num_blocks"] == 0
    assert analysis["max_block_count_section"]["section"] == "algorithm_step_1"
    assert (
        analysis["max_total_phase_instruction_section"]["section"]
        == "algorithm_step_1"
    )
    assert analysis["max_largest_block_section"]["section"] == "algorithm_step_1"


def test_estimate_nests_phase_polynomial_section_analysis_when_inputs_exist():
    full_circuit, sectioned_circuit, section_names = build_phase_section_test_circuits()
    estimator = QLBMResourceEstimator(phase_hardware_config(), force_no_coupling=True)

    report = estimator.estimate(
        full_circuit,
        sectioned_circuit=sectioned_circuit,
        section_names=section_names,
        optimization_level=0,
        seed_transpiler=42,
        phase_polynomial_analysis=True,
    )

    phase_analysis = report["phase_polynomial_analysis"]
    assert "section_analysis" in phase_analysis
    assert (
        phase_analysis["section_analysis"]["max_block_count_section"]["section"]
        == "algorithm_step_1"
    )


def test_section_analysis_reports_time_and_two_qubit_bottlenecks():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier(label="section_boundary::initial_conditions")
    qc.cx(0, 1)
    qc.x(0)
    qc.barrier(label="section_boundary::algorithm_step_1")
    qc.h(1)

    analysis = QLBMResourceEstimator(simple_hardware_config()).analyze_sections(
        qc,
        ["initial_conditions", "algorithm_step_1", "measurement"],
        optimization_level=0,
        seed_transpiler=42,
    )

    assert len(analysis["sections"]) == 3
    assert analysis["max_critical_path_time_section"]["section"] == "algorithm_step_1"
    assert analysis["max_critical_path_time_section"][
        "critical_path_time_s"
    ] == pytest.approx(110e-9)
    assert analysis["max_two_qubit_gate_section"]["section"] == "algorithm_step_1"
    assert analysis["max_two_qubit_gate_section"]["num_2q_ops"] == 1


def test_sectioned_logical_circuit_matches_full_circuit_for_multiple_timesteps():
    initial_conditions = QuantumCircuit(2)
    initial_conditions.h(0)

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
        num_timesteps=3,
    )
    sectioned_circuit, section_names = build_sectioned_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        num_timesteps=3,
    )

    full_ops = [inst.operation.name for inst in full_circuit.data]
    sectioned_ops_without_barriers = [
        inst.operation.name
        for inst in sectioned_circuit.data
        if inst.operation.name != "barrier"
    ]
    barrier_labels = [
        inst.operation.label
        for inst in sectioned_circuit.data
        if inst.operation.name == "barrier"
    ]

    assert full_ops == ["h", "cx", "cx", "cx", "x", "measure", "measure"]
    assert sectioned_ops_without_barriers == full_ops
    assert section_names == [
        "initial_conditions",
        "algorithm_step_1",
        "algorithm_step_2",
        "algorithm_step_3",
        "postprocessing",
        "measurement",
    ]
    assert barrier_labels == [
        "section_boundary::initial_conditions",
        "section_boundary::algorithm_step_1",
        "section_boundary::algorithm_step_2",
        "section_boundary::algorithm_step_3",
        "section_boundary::postprocessing",
    ]

    analysis = QLBMResourceEstimator(simple_hardware_config()).analyze_sections(
        sectioned_circuit,
        section_names,
        optimization_level=0,
        seed_transpiler=42,
    )

    assert [section["section"] for section in analysis["sections"]] == section_names
    assert [
        section["num_2q_ops"] for section in analysis["sections"]
    ] == [0, 1, 1, 1, 0, 0]


def test_full_and_sectioned_builders_match_manual_time_loop_composition():
    initial_conditions = QuantumCircuit(2)
    initial_conditions.h(0)

    algorithm = QuantumCircuit(2)
    algorithm.cx(0, 1)
    algorithm.x(0)

    postprocessing = QuantumCircuit(2)
    postprocessing.h(1)

    measurement = QuantumCircuit(2, 2)
    measurement.measure([0, 1], [0, 1])

    num_timesteps = 3
    manual = QuantumCircuit(*(measurement.qregs + measurement.cregs))
    manual.compose(
        initial_conditions.copy(),
        inplace=True,
        qubits=range(manual.num_qubits),
    )
    for _ in range(num_timesteps):
        manual.compose(algorithm.copy(), inplace=True)
    manual.compose(postprocessing.copy(), inplace=True)
    manual.compose(measurement.copy(), inplace=True)

    full_circuit = build_full_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        num_timesteps=num_timesteps,
    )
    sectioned_circuit, _ = build_sectioned_logical_circuit(
        initial_conditions,
        algorithm,
        postprocessing,
        measurement,
        num_timesteps=num_timesteps,
    )

    def operation_signature(circuit, skip_barriers=False):
        signature = []
        for inst in circuit.data:
            if skip_barriers and inst.operation.name == "barrier":
                continue
            signature.append(
                (
                    inst.operation.name,
                    [circuit.find_bit(qubit).index for qubit in inst.qubits],
                    [circuit.find_bit(clbit).index for clbit in inst.clbits],
                )
            )
        return signature

    assert operation_signature(full_circuit) == operation_signature(manual)
    assert operation_signature(
        sectioned_circuit,
        skip_barriers=True,
    ) == operation_signature(manual)


def test_logical_capacity_uses_declared_logical_qubits():
    config = simple_hardware_config()
    config["num_qubits"] = 2
    qc = QuantumCircuit(3)
    qc.h(0)

    estimator = QLBMResourceEstimator(config)
    report = estimator.estimate(qc, transpile_circuit=False)

    assert report["logical_capacity"] == {
        "logical_qubits": 3,
        "available_qubits": 2,
        "qubit_capacity_ok": False,
    }


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


def test_all_to_all_coupling_type_generates_full_topology():
    estimator = QLBMResourceEstimator(
        {
            "num_qubits": 4,
            "basis_gates": ["h", "cx"],
            "coupling_type": "all_to_all",
        }
    )

    assert len(estimator.coupling_map) == 12
    assert estimator.validate_hardware_config()["topology_matches_num_qubits"] is True


def test_fake_backend_coupling_type_uses_exact_fake_backend_topology():
    estimator = QLBMResourceEstimator(
        {
            "num_qubits": 127,
            "basis_gates": ["rz", "sx", "x", "cx"],
            "coupling_type": "fake_backend",
            "coupling_params": {"backend": "FakeKyoto"},
        }
    )

    validation = estimator.validate_hardware_config()

    assert len(estimator.coupling_map) == 144
    assert validation["topology_num_qubits"] == 127
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


def test_null_gate_times_are_reported_unknown_without_failing():
    qc = QuantumCircuit(1)
    qc.rx(0.2, 0)

    estimator = QLBMResourceEstimator(
        {
            "num_qubits": 1,
            "basis_gates": ["rx"],
            "gate_times_s": {"rx": None},
        }
    )
    timing = estimator.estimate_time(qc)

    assert timing["serial_time_s"] == 0.0
    assert timing["critical_path_time_s"] == 0.0
    assert timing["unknown_gate_times"] == ["rx"]
    assert timing["scheduled_duration_s"] is None
