from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from collections import Counter

from qlbm.components import MSQLBM, ABQLBM
from qlbm.lattice import MSLattice, ABLattice
from qlbm.tools.utils import create_directory_and_parents
from qlbm.qlbm import MSLattice

import pandas as pd

def make_qlbm_circuit() -> QuantumCircuit:
    lattice2d_4x8_0_obs_q4 = { "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "d2q9"}, "geometry": [], }
    lattice_2d_8x8_1_obs = { "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4 }},
            "geometry": [{"shape": "cuboid", "x": [5, 6], "y": [1, 2], "boundary": "specular" } ]}

    lattice_2d_32x32_3_obs = { "lattice": {"dim": {"x": 32, "y": 32}, "velocities": {"x": 4, "y": 4 }},
            "geometry": [{"shape": "cuboid", "x": [18, 20], "y": [6, 25], "boundary": "specular" },
                         {"shape": "cuboid", "x": [23, 25], "y": [3, 17], "boundary": "specular" },
                         {"shape": "cuboid", "x": [28, 29], "y": [16, 29], "boundary": "specular" }]}

    latticeAB = ABLattice(lattice2d_4x8_0_obs_q4)
    latticeMS = MSLattice(lattice_2d_8x8_1_obs)
    latticeMS_32x32_3_OBS = MSLattice(lattice_2d_32x32_3_obs)

    # qc_abqlbm = ABQLBM(latticeAB).circuit
    # qc_msqlbm = MSQLBM(latticeMS).circuit
    qc_msqlbm_32x32_3_OBS = MSQLBM(latticeMS_32x32_3_OBS).circuit

    return qc_msqlbm_32x32_3_OBS


def make_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """Create an n-qubit QFT circuit as a simple, dense benchmark."""
    qc = QuantumCircuit(num_qubits)
    n = num_qubits

    # Standard QFT
    for k in range(n):
        qc.h(k)
        for j in range(k + 1, n):
            qc.cp(3.141592653589793 / (2 ** (j - k)), j, k)

    # Optional final swap layer (to reverse bit order)
    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    # if with_measurements:
    #     qc.measure(range(n), range(n))

    return qc


# Platform configuration
# -------------------------------------------------------------------

# Assumed schema:
# - num_qubits  : number of qubits you will actually use for the benchmark
# - basis_gates : list of physical basis gates for that platform
# - coupling_map  : list of directed edges [control, target] or None for full connectivity
# - gate_times : dict { gate_name: gate_duration_in_seconds } (optional for timing)
#

linear_edges = []
num_qubits_spin = 12

for i in range(num_qubits_spin - 1):
    # forward edge i -> i+1
    linear_edges.append([i, i + 1])
    # backward edge i+1 -> i (so CNOT can go both ways)
    linear_edges.append([i + 1, i])

spin_qubit_coupling_map = CouplingMap(linear_edges)

def make_linear_chain_edges(num_qubits: int):
    """
    Nearest-neighbour chain, bidirectional edges.
    """
    edges = []
    for i in range(num_qubits - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return edges


def make_2d_grid_edges(rows: int, cols: int):
    """
    2D nearest-neighbour grid (rows x cols), bidirectional edges.
    Qubit index = r*cols + c.
    """
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            # right neighbour
            if c + 1 < cols:
                q_right = r * cols + (c + 1)
                edges.append([q, q_right])
                edges.append([q_right, q])
            # down neighbour
            if r + 1 < rows:
                q_down = (r + 1) * cols + c
                edges.append([q, q_down])
                edges.append([q_down, q])
    return edges

benchmark_num_qubits = 20

platforms = {
    # "spin_qubit_intel_2024": {
    #     "id": "intel 2024",
    #     "architecture": "silicon spin qubit (Si/SiGe quantum dots)",
    #     "device_name": "Tunnel Falls 12QD device",
    #     "year_reported": 2024,
    #     "num_qubits": benchmark_num_qubits,  # physical_qubit_count = 12
    #     # Single-qubit: EDSR-driven rotations -> Rx/Ry/Rz
    #     # Two-qubit: exchange-based gate -> approximate as CZ/CX for transpilation
    #     "basis_gates": ["rz", "rx", "ry", "cz", "cx"],
    #     # Linear nn array (here defined for 12 but extra nodes unused if num_qubits < 12
    #     "coupling_map": make_linear_chain_edges(num_qubits=12),
    #     # IMPORTANT NOTE!!!! very rough average, gate duration varies drastically, increasing the
    #     # number of qubits increases gate times
    #     "gate_times": {
    #         "rz": 0.0,
    #         "rx": 50e-9,  # 50 ns
    #         "ry": 50e-9,
    #         "cz": 200e-9,  # 200 ns two-qubit (or swap/entangler)
    #         "cx": 200e-9,
    #     },
    # },
    "superconducting_google_willow_2024": {
        "id": "google willow 2024",
        "architecture": "superconducting transmon",
        "device_name": "Willow",
        "year_reported": 2024,
        "num_qubits": benchmark_num_qubits,
        # Willow is transmon so Qiskit native basis is can be used
        # (Exact native gates not reported in the paper; these are the standard ones
        #  used for Google's previous hardware and are fully compatible with routing)
        "basis_gates": ["rz", "sx", "x", "cx"],

        # Willow physical layouts are 72 and 105 qubit square grids.
        # For benchmarking, embed a small 2D grid subset (2 × 5 = 10 qubits)
        # so routing behaviour matches planar nearest-neighbour constraints.
        "coupling_map": make_2d_grid_edges(rows=5, cols=20),

        # NOTE: fairly standard numbers but still averages across fixed-frequency transmon processors
        # so gate times are typical but not Willow specific
        "gate_times": {
            "rz": 0.0,  # virtual Z
            "sx": 25e-9,  # 25 ns 1q gate
            "x": 25e-9,  # 25 ns 1q gate
            "cx": 200e-9,  # 200 ns 2q CX
        },
    },
    # "trapped_ion_oxford_2024": {
    #     "id": "oxford 2024",
    #     "architecture": "trapped-ion (40Ca+), all-electronic control",
    #     "device_name": "Seven-zone microfabricated ion trap",
    #     "year_reported": 2024,
    #     "num_qubits": benchmark_num_qubits,  # physical_qubit_count = 10
    #     # Single-qubit: microwave/AC-field rotations -> Rx/Ry/Rz
    #     # Two-qubit: Mølmer–Sørensen -> model as RXX
    #     # Include CX so Qiskit can decompose via standard patterns
    #     "basis_gates": ["rx", "ry", "rz", "rxx", "cx"],
    #     # Effective connectivity: ions can be shuttled between zones,
    #     # so algorithmically close to all-to-all so use None
    #     "coupling_map": None,
    #     #  - single_qubit: 8000 ns (8 micros SK1 sequence)
    #     #  - two_qubit (MS): 120000 ns (120 micros)
    #     "gate_times": {
    #         "rx": 8000e-9,    # 8 μs
    #         "ry": 8000e-9,    # 8 μs
    #         "rz": 0.0,        # phase advance
    #         "rxx": 120000e-9  # 120 μs
    #     },
    # },
    "neutral_atom_aws_2023": {
        "id": "aws2023",
        "architecture": "neutral-atom",
        "device_name": None,  # not named in JSON
        "year_reported": 2023,
        "num_qubits": benchmark_num_qubits,  # physical_qubit_count = 60
        # Single-qubit: Raman pulses -> Rx/Ry/Rz
        # Two-qubit: Rydberg CZ
        "basis_gates": ["rx", "ry", "rz", "cz", "cx"],
        # Connectivity is effectively reconfigurable so any-to-any
        "coupling_map": None,
        # IMPORTANT NOTE!!!! very rough average, laser parameters vary the times massively
        "gate_times": {
            "rz": 0.0,  # phase-shift / virtual Z (assume negligible)
            "rx": 2e-6,  # e.g. 2 micros typical single-qubit
            "ry": 2e-6,  # same
            "cz": 6e-6,  # e.g. 6 micros for two-qubit
            "cx": 6e-6,  # if transpiled to CX
        },
    },
}


# Transpile helper
def transpile_for_platform(circuit: QuantumCircuit, basis_gates, coupling_map,
                           optimization_level: int = 3, seed_transpiler: int | None = 0,) -> QuantumCircuit:
    """
    Transpile 'circuit' for a given platform specified by basis_gates and coupling_map.

    This does not require a real backend, it uses explicit basis and connectivity.
    """
    coupling = CouplingMap(coupling_map) if coupling_map is not None else None

    tc = transpile(
        circuit,
        basis_gates=basis_gates,
        coupling_map=coupling,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    return tc


# metric extraction

ONE_Q_GATES = {
    "x", "y", "z", "h", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "sx", "sxdg", "u", "u1", "u2", "u3",
}
TWO_Q_GATES = {
    "cx", "cz", "cy", "swap", "rxx", "ryy", "rzz", "rzx", "iswap", "ecr",
}
THREE_Q_GATES = {"ccx", "cswap", "c3x"}


def classify_gate(name: str) -> str:
    """Return '1q', '2q', '3q', or 'other' based on the gate name."""
    lname = name.lower()
    if lname in ONE_Q_GATES:
        return "1q"
    if lname in TWO_Q_GATES:
        return "2q"
    if lname in THREE_Q_GATES:
        return "3q"
    return "other"


def extract_metrics( circuit: QuantumCircuit, gate_times: dict | None = None,) -> dict:
    """
    Extract basic metrics from a transpiled circuit:
      - depth, size, num_qubits
      - total counts per gate, and grouped by arity
      - naive_total_time (sum count_g * gate_times[g]) if gate_times given
    """
    ops = circuit.count_ops()
    depth = circuit.depth()
    size = circuit.size()
    num_qubits = circuit.num_qubits

    by_arity = Counter()
    for name, count in ops.items():
        arity = classify_gate(name)
        by_arity[arity] += count

    # Naive total time (no scheduling, just sum gate_time * count)
    naive_total_time = None
    if gate_times is not None:
        total_time = 0.0
        for name, count in ops.items():
            if name in gate_times:
                total_time += count * gate_times[name]
        naive_total_time = total_time

    metrics = {
        "num_qubits": num_qubits,
        "depth": depth,
        "size": size,
        "ops_per_gate": dict(ops),
        "ops_1q": by_arity["1q"],
        "ops_2q": by_arity["2q"],
        "ops_3q": by_arity["3q"],
        "ops_other": by_arity["other"],
        "naive_total_time_s": naive_total_time,
    }
    return metrics


#Main benchmark driver
# pick which circuit you want and loop over platforms

def run_benchmark(circuit_maker, platforms_dict: dict, with_measurements: bool = False,
                  optimization_level: int = 3) -> pd.DataFrame:
    """
    Run a benchmark for all platforms in 'platforms_dict' using 'circuit_maker(num_qubits)'.

    Returns a pandas DataFrame with one row per platform.
    """
    rows = []

    for name, cfg in platforms_dict.items():
        num_qubits = cfg["num_qubits"]
        basis_gates = cfg["basis_gates"]
        coupling_map = cfg.get("coupling_map", None)
        gate_times = cfg.get("gate_times", None)

        # circ = circuit_maker(num_qubits, with_measurements)
        circ = circuit_maker()

        tc = transpile_for_platform(
            circ,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=optimization_level,
        )

        metrics = extract_metrics(tc, gate_times=gate_times)
        metrics["platform"] = name

        rows.append(metrics)

    df = pd.DataFrame(rows)
    # Put platform column first
    columns = ["platform"] + [c for c in df.columns if c != "platform"]
    df = df[columns]
    return df


if __name__ == "__main__":
# Choose which circuit to benchmark:
# QFT: make_qft_circuit
# Adder: make_example_adder_circuit
    df_results = run_benchmark(
        circuit_maker=make_qlbm_circuit,
        platforms_dict=platforms,
        with_measurements=False,
        optimization_level=2,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df_results)

    output_path = "../../qlbm-hardware-output/output/msqlbm_2d_32x32_3_obs_hw_comp.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")