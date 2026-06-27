"""Small QLBM case builders for hardware noise-analysis experiments.

Edit the lattice geometry and initial conditions inside the builder functions.
The runner composes the returned components into full measured logical
circuits, then hands those circuits to the hardware resource estimator.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QLBM_SOURCE_ROOT = PROJECT_ROOT / "qlbm"
QLBM_HARDWARE_ROOT = PROJECT_ROOT / "qlbm-hardware"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "output" / ".matplotlib"),
)

for path in (QLBM_SOURCE_ROOT, QLBM_HARDWARE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qiskit import QuantumCircuit

from qlbm import ABLattice, MSLattice, SpaceTimeLattice
from qlbm.components import (
    ABGridMeasurement,
    ABInitialConditions,
    ABQLBM,
    EmptyPrimitive,
    GridMeasurement,
    MSInitialConditions,
    MSQLBM,
)
from qlbm.components.spacetime import SpaceTimeGridVelocityMeasurement, SpaceTimeQLBM
from qlbm.components.spacetime.initial.pointwise import (
    PointWiseSpaceTimeInitialConditions,
)


@dataclass(frozen=True)
class QLBMCase:
    """A hardware-transpilable QLBM case."""

    label: str
    lattice: ABLattice | MSLattice | SpaceTimeLattice
    lattice_data: dict[str, Any]
    initial_conditions: Any
    algorithm: Any
    postprocessing: Any
    measurement: Any
    algorithm_name: str
    num_timesteps: int
    runner_steps: int
    metadata: dict[str, Any]


def component_circuit(component: Any) -> QuantumCircuit:
    """Return a Qiskit circuit from either a QLBM component or circuit."""

    return component.circuit if hasattr(component, "circuit") else component


def build_full_logical_circuit(case: QLBMCase, num_steps: int) -> QuantumCircuit:
    """Build a full measured logical QLBM circuit without section boundaries."""

    initial_circuit = component_circuit(case.initial_conditions)
    algorithm_circuit = component_circuit(case.algorithm)
    postprocessing_circuit = component_circuit(case.postprocessing)
    measurement_circuit = component_circuit(case.measurement)

    circuit = QuantumCircuit(*(measurement_circuit.qregs + measurement_circuit.cregs))
    circuit.compose(initial_circuit, inplace=True, qubits=range(circuit.num_qubits))
    for _ in range(num_steps):
        circuit.compose(algorithm_circuit.copy(), inplace=True)
    circuit.compose(postprocessing_circuit.copy(), inplace=True)
    circuit.compose(measurement_circuit.copy(), inplace=True)
    return circuit


def build_abqlbm_4x8_0_case(num_timesteps: int, measure_velocity_qubits: bool = False) -> QLBMCase:
    """Build an ABQLBM circuit for 8x4 D2Q9 no obstacles
    """

    lattice_data = {
        "lattice": {"dim": {"x": 8, "y": 4}, "velocities": "D2Q9"},
        "geometry": [],
    }
    lattice = ABLattice(lattice_data)

    initial_conditions = ABInitialConditions(lattice)
    algorithm = ABQLBM(lattice)
    postprocessing = EmptyPrimitive(lattice)
    measurement = ABGridMeasurement(
        lattice,
        measure_velocity_qubits=measure_velocity_qubits,
    )

    return QLBMCase(
        label=f"abqlbm_8x4_d2q9_t{num_timesteps}",
        lattice=lattice,
        lattice_data=lattice_data,
        initial_conditions=initial_conditions,
        algorithm=algorithm,
        postprocessing=postprocessing,
        measurement=measurement,
        algorithm_name="ABQLBM",
        num_timesteps=num_timesteps,
        runner_steps=num_timesteps,
        metadata={
            "algorithm": "ABQLBM",
            "lattice": lattice_data,
            "measure_velocity_qubits": measure_velocity_qubits,
            "num_timesteps": num_timesteps,
        },
    )


def build_msqlbm_case(num_timesteps: int) -> QLBMCase:
    """Build an MSQLBM case. D2Q4 no obstacles.
    """

    lattice_data = {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
        "geometry": [],
    }
    lattice = MSLattice(lattice_data)

    initial_conditions = MSInitialConditions(lattice)
    algorithm = MSQLBM(lattice)
    postprocessing = EmptyPrimitive(lattice)
    measurement = GridMeasurement(lattice)

    return QLBMCase(
        label=f"msqlbm_4x4_v4x4_t{num_timesteps}",
        lattice=lattice,
        lattice_data=lattice_data,
        initial_conditions=initial_conditions,
        algorithm=algorithm,
        postprocessing=postprocessing,
        measurement=measurement,
        algorithm_name="MSQLBM",
        num_timesteps=num_timesteps,
        runner_steps=num_timesteps,
        metadata={
            "algorithm": "MSQLBM",
            "lattice": lattice_data,
            "num_timesteps": num_timesteps,
        },
    )


def build_spacetime_case(num_timesteps: int) -> QLBMCase:
    """Build a SpaceTimeQLBM case.

    Manually edit ``lattice_data`` and ``initial_conditions`` here for
    SpaceTimeQLBM geometry and initial-state experiments. Space-time lattices
    encode the physical timesteps in the lattice itself, so the
    runner executes one algorithm circuit.
    """

    lattice_data = {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q4"},
        "geometry": [],
    }
    lattice = SpaceTimeLattice(num_timesteps=num_timesteps, lattice_data=lattice_data)

    initial_conditions = PointWiseSpaceTimeInitialConditions(lattice)
    algorithm = SpaceTimeQLBM(lattice)
    postprocessing = EmptyPrimitive(lattice)
    measurement = SpaceTimeGridVelocityMeasurement(lattice)

    return QLBMCase(
        label=f"spacetime_4x4_d2q4_t{num_timesteps}",
        lattice=lattice,
        lattice_data=lattice_data,
        initial_conditions=initial_conditions,
        algorithm=algorithm,
        postprocessing=postprocessing,
        measurement=measurement,
        algorithm_name="SpaceTimeQLBM",
        num_timesteps=num_timesteps,
        runner_steps=1,
        metadata={
            "algorithm": "SpaceTimeQLBM",
            "lattice": lattice_data,
            "num_timesteps": num_timesteps,
        },
    )


CASE_BUILDERS: dict[str, Callable[..., QLBMCase]] = {
    "abqlbm": build_abqlbm_4x8_0_case,
    "msqlbm": build_msqlbm_case,
    "spacetime": build_spacetime_case,
}
