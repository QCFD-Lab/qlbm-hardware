"""Small QLBM case builders for noise-analysis experiments.

Edit the lattice geometry and initial conditions inside the builder functions.
The runner intentionally stays generic and only consumes the returned lattice
and ``SimulationConfig``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

from qlbm import ABLattice, MSLattice, QiskitRunner, SpaceTimeLattice
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
from qlbm.infra import SimulationConfig


@dataclass(frozen=True)
class QLBMCase:
    """A runnable QLBM case."""

    label: str
    lattice: ABLattice | MSLattice | SpaceTimeLattice
    simulation_config: SimulationConfig
    runner_steps: int


def build_abqlbm_case(
    execution_backend,
    sampling_backend,
    *,
    num_timesteps: int,
    optimization_level: int,
) -> QLBMCase:
    """Build an ABQLBM circuit for 4x8 D2Q9 no obstacles
    """

    lattice_data = {
        "lattice": {"dim": {"x": 8, "y": 4}, "velocities": "D2Q9"},
        "geometry": [],
    }
    lattice = ABLattice(lattice_data)

    initial_conditions = ABInitialConditions(lattice)
    algorithm = ABQLBM(lattice)
    postprocessing = EmptyPrimitive(lattice)
    measurement = ABGridMeasurement(lattice)

    simulation_config = SimulationConfig(
        initial_conditions=initial_conditions,
        algorithm=algorithm,
        postprocessing=postprocessing,
        measurement=measurement,
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=optimization_level,
        statevector_sampling=False,
        execution_backend=execution_backend,
        sampling_backend=sampling_backend,
    )
    simulation_config.prepare_for_simulation()

    return QLBMCase(
        label=f"abqlbm_8x4_d2q9_t{num_timesteps}",
        lattice=lattice,
        simulation_config=simulation_config,
        runner_steps=num_timesteps,
    )


def build_msqlbm_case(
    execution_backend,
    sampling_backend,
    *,
    num_timesteps: int,
    optimization_level: int,
) -> QLBMCase:
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

    simulation_config = SimulationConfig(
        initial_conditions=initial_conditions,
        algorithm=algorithm,
        postprocessing=postprocessing,
        measurement=measurement,
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=optimization_level,
        statevector_sampling=False,
        execution_backend=execution_backend,
        sampling_backend=sampling_backend,
    )
    simulation_config.prepare_for_simulation()

    return QLBMCase(
        label=f"msqlbm_4x4_v4x4_t{num_timesteps}",
        lattice=lattice,
        simulation_config=simulation_config,
        runner_steps=num_timesteps,
    )


def build_spacetime_case(
    execution_backend,
    sampling_backend,
    *,
    num_timesteps: int,
    optimization_level: int,
) -> QLBMCase:
    """Build a SpaceTimeQLBM case.

    Manually edit ``lattice_data`` and ``initial_conditions`` here for
    SpaceTimeQLBM geometry and initial-state experiments. Space-time lattices
    encode the requested physical timesteps in the lattice itself, so the
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

    simulation_config = SimulationConfig(
        initial_conditions=initial_conditions,
        algorithm=algorithm,
        postprocessing=postprocessing,
        measurement=measurement,
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=optimization_level,
        statevector_sampling=False,
        execution_backend=execution_backend,
        sampling_backend=sampling_backend,
    )
    simulation_config.prepare_for_simulation()

    return QLBMCase(
        label=f"spacetime_4x4_d2q4_t{num_timesteps}",
        lattice=lattice,
        simulation_config=simulation_config,
        runner_steps=1,
    )


CASE_BUILDERS: dict[str, Callable[..., QLBMCase]] = {
    "abqlbm": build_abqlbm_case,
    "msqlbm": build_msqlbm_case,
    "spacetime": build_spacetime_case,
}


def create_qiskit_runner(case: QLBMCase) -> QiskitRunner:
    """Create the Qiskit runner for a built case."""

    return QiskitRunner(case.simulation_config, case.lattice)

