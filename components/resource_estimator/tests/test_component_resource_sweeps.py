import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.resource_estimator.component_resource_sweeps import (
    build_component_case,
    build_component_circuits,
    make_component_csv_rows,
    msqlbm_component_sections,
)
from components.resource_estimator.experiment_resource_sweeps import (
    make_case_spec,
)


def test_abqlbm_component_case_repeats_streaming_and_reflection_by_timestep():
    spec = make_case_spec("timesteps", "ABQLBM", grid_size=4, num_timesteps=2)
    built_case = build_component_case(spec)

    assert [section.component_group for section in built_case.sections] == [
        "initialization",
        "streaming",
        "reflection",
        "streaming",
        "reflection",
    ]
    assert [section.logical_timestep for section in built_case.sections] == [
        None,
        1,
        1,
        2,
        2,
    ]
    assert built_case.section_names == [
        "000__initialization__ab_initial_conditions",
        "001__streaming__ab_streaming__step1",
        "002__reflection__ab_reflection__step1",
        "003__streaming__ab_streaming__step2",
        "004__reflection__ab_reflection__step2",
    ]


def test_component_circuit_builder_inserts_boundaries_without_changing_ops():
    spec = make_case_spec("grid_size", "ABQLBM", grid_size=4)
    built_case = build_component_case(spec)
    circuit, sectioned_circuit = build_component_circuits(built_case.sections)

    full_ops = [inst.operation.name for inst in circuit.data]
    sectioned_ops = [
        inst.operation.name
        for inst in sectioned_circuit.data
        if inst.operation.name != "barrier"
    ]
    barrier_labels = [
        inst.operation.label
        for inst in sectioned_circuit.data
        if inst.operation.name == "barrier"
    ]

    assert sectioned_ops == full_ops
    assert barrier_labels == [
        "section_boundary::000__initialization__ab_initial_conditions",
        "section_boundary::001__streaming__ab_streaming__step1",
    ]


def test_msqlbm_component_sections_preserve_implemented_cfl_order():
    spec = make_case_spec("grid_size", "MSQLBM", grid_size=4)
    sections = msqlbm_component_sections(spec)

    first_substep = [
        section
        for section in sections
        if section.logical_timestep == 1 and section.algorithm_substep == 1
    ]
    assert sections[0].component_group == "initialization"
    assert sections[0].component_kind == "ms_initial_conditions"
    assert [section.component_kind for section in first_substep] == [
        "ms_streaming",
        "ms_bounceback_reflection",
        "ms_bounceback_reflection",
        "ms_streaming_ancilla_preparation",
        "ms_streaming_ancilla_preparation",
    ]
    assert [section.boundary_condition for section in first_substep] == [
        None,
        "bounceback",
        "specular",
        None,
        None,
    ]


def test_spacetime_component_case_includes_initialization_before_algorithm_body():
    spec = make_case_spec("grid_size", "SpaceTimeQLBM", grid_size=4)
    built_case = build_component_case(spec)

    assert [section.component_group for section in built_case.sections] == [
        "initialization",
        "streaming",
        "reflection",
        "collision",
    ]
    assert built_case.sections[0].component_kind == (
        "spacetime_pointwise_initial_conditions"
    )


def test_make_component_csv_rows_flattens_logical_and_section_metrics():
    spec = make_case_spec("grid_size", "ABQLBM", grid_size=4)
    built_case = build_component_case(spec)
    section_entries = [
        {
            "section": section.section,
            "num_1q_ops": 10 + section.sequence_index,
            "num_2q_ops": 20 + section.sequence_index,
            "num_3q_plus_ops": 0,
            "depth": 30 + section.sequence_index,
            "size": 40 + section.sequence_index,
            "critical_path_time_s": 1e-6,
            "serial_time_s": 2e-6,
            "op_counts": {"cx": 20 + section.sequence_index},
        }
        for section in built_case.sections
    ]
    report = {
        "logical": {"depth": 100, "num_2q_ops": 50},
        "transpiled": {"depth": 200, "num_2q_ops": 80},
        "transpiled_compatibility": {"compatible": True},
        "transpiled_time": {
            "scheduled_duration_s": 3e-6,
            "critical_path_time_s": 1e-6,
            "serial_time_s": 4e-6,
            "warnings": [],
        },
        "section_analysis": {"sections": section_entries},
    }

    rows = make_component_csv_rows(
        report,
        built_case,
        "test_backend",
        {
            "id": "test",
            "device_name": "Test Backend",
            "architecture": "test architecture",
            "coupling_type": "all_to_all",
            "num_qubits": 64,
        },
    )

    assert len(rows) == 3
    assert rows[0]["component_group"] == "initialization"
    assert rows[0]["transpiled_component_2q_ops"] == 20
    assert rows[0]["algorithm_transpiled_2q_ops"] == 80
    assert rows[0]["hardware_label"] == "Test Backend"
    assert rows[1]["component_group"] == "streaming"
    assert rows[2]["component_group"] == "reflection"
