import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.resource_estimator.experiment_resource_sweeps import (
    build_error_row,
    deterministic_obstacles,
    make_case_spec,
    make_experiment_csv_row,
    routing_row_from_pair,
)


def test_deterministic_obstacles_are_stable_and_non_overlapping():
    obstacles = deterministic_obstacles(8, 8, 4)
    assert obstacles == deterministic_obstacles(8, 8, 4)
    assert len(obstacles) == 4
    assert {tuple(obstacle["x"] + obstacle["y"]) for obstacle in obstacles} == {
        (1, 1, 1, 1),
        (2, 2, 1, 1),
        (3, 3, 1, 1),
        (4, 4, 1, 1),
    }
    assert {obstacle["boundary"] for obstacle in obstacles} == {"bounceback"}


def test_csv_row_preserves_failed_transpilation():
    spec = make_case_spec("grid_size", "MSQLBM", grid_size=4)
    row = make_experiment_csv_row(
        report={
            "logical": {"num_qubits": 13, "depth": 10, "size": 20, "num_2q_ops": 3},
            "transpile_error": "CircuitTooWideForTarget('too wide')",
        },
        spec=spec,
        hardware_key="tiny_backend",
        hardware_config={
            "id": "tiny",
            "device_name": "Tiny",
            "architecture": "test",
            "coupling_type": "line",
            "num_qubits": 2,
        },
    )

    assert row["experiment"] == "grid_size"
    assert row["algorithm"] == "MSQLBM"
    assert row["logical_qubits"] == 13
    assert row["transpiled_2q_ops"] is None
    assert row["transpile_error"] == "CircuitTooWideForTarget('too wide')"


def test_build_error_row_preserves_case_and_hardware_metadata():
    spec = make_case_spec("obstacle_count", "SpaceTimeQLBM", 8, num_obstacles=1)
    row = build_error_row(
        spec,
        hardware_key="backend",
        hardware_config={"id": "Backend", "coupling_type": "all_to_all"},
        build_error="LatticeException('unsupported')",
    )

    assert row["experiment"] == "obstacle_count"
    assert row["algorithm"] == "SpaceTimeQLBM"
    assert row["num_obstacles"] == 1
    assert row["hardware"] == "Backend"
    assert row["build_error"] == "LatticeException('unsupported')"


def test_routing_row_from_pair_calculates_overhead():
    row = routing_row_from_pair(
        {"transpiled_2q_ops": 30, "transpile_error": None},
        {"transpiled_2q_ops": 10, "transpile_error": None},
    )

    assert row["routing_reference_2q_ops"] == 10
    assert row["routing_2q_overhead"] == 3.0
    assert row["routing_added_2q"] == 20
    assert row["routing_reference_transpile_error"] is None
