from qlbm.components import EmptyPrimitive, ABGridMeasurement
from qlbm.components.ab import ABQLBM, ABInitialConditions
from qlbm.lattice import ABLattice, LatticeDiscretization

from qiskit_aer import AerSimulator
from qlbm.infra import QiskitRunner, SimulationConfig

# 2D lattice with periodic x-boundaries and bounce-back y-boundaries
lattice_config = {
    "lattice": {
        "dim": {"x": 16, "y": 8},
        "velocities": "d2q9"
    },
    "geometry": [
        # Top wall - bounce-back boundary
        {
            "shape": "cuboid",
            "x": [0, 15],
            "y": [7, 7],
            "boundary": "bounceback"
        },
        # Bottom wall - bounce-back
        {
            "shape": "cuboid",
            "x": [0, 15],
            "y": [0, 0],
            "boundary": "bounceback"
        }
    ]
}


lattice = ABLattice(lattice_config)

initial_conditions = ABInitialConditions(lattice)

abqlbm = ABQLBM(lattice)

cfg = SimulationConfig(
    initial_conditions=ABInitialConditions(lattice),
    algorithm=ABQLBM(lattice),
    postprocessing=EmptyPrimitive(lattice),
    measurement=ABGridMeasurement(lattice),
    target_platform="QISKIT",
    compiler_platform="QISKIT",
    optimization_level=0,
    statevector_sampling=True,
    execution_backend=AerSimulator(method="statevector"),
    sampling_backend=AerSimulator(method="statevector"),
)

runner = QiskitRunner(cfg, lattice)
runner.run(
    num_steps=20,
    num_shots=2**12,
    output_directory="tube-flow-output",
    output_file_name="test",
    statevector_snapshots=True
)

