#!/usr/bin/env python3
"""
POC: resource estimation for QLBM on IBM QPUs.

- Builds an MSQLBM circuits (0/1/2 obstacles, 8x8, vx=vy=4)
- Transpiles to IBM targets (Fake backends)
- Runs Qiskit ResourceEstimation analysis
- Schedules with ALAP and ASAP to estimate execution duration
- Writes a CSV under qlbm-output/resource-estimates/
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ResourceEstimation,
    TimeUnitConversion,
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis
)

from qiskit_ibm_runtime.fake_provider import (
    FakeGuadalupeV2,    # 16 qubits
    FakeSherbrooke,     # 127 qubits
    FakeAlgiers,        # 27 qubits
    FakeCambridgeV2     # 28 qubits
)


from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

from pytket.passes import (
    SequencePass,
    FullPeepholeOptimise,
    DecomposeBoxes,
    DefaultMappingPass,
)
from pytket.architecture import Architecture

from qlbm.components import MSQLBM
from qlbm.lattice import MSLattice
from qlbm.tools.utils import create_directory_and_parents

from qlbm.qlbm import MSLattice


@dataclass
class EstimateRow:
    lattice: str
    backend: str
    compiler: str
    scheduler: str
    opt_level: int
    virtual_qubits: int
    physical_qubits: int
    depth: int | None
    width: int | None
    size: int | None
    tensor_factors: int | None
    twoq_gate_count: int
    swap_count: int
    op_counts_json: str
    scheduled_duration_s: float | None
    scheduled_duration_unit: str | None


def build_msqlbm_circuit(lattice_spec) -> qiskit.QuantumCircuit:
    """Construct a small MSQLBM circuit for a given lattice spec."""
    algo = MSQLBM(lattice_spec)
    return algo.circuit


def get_twoq_gate_names(backend) -> set[str]:
    """
    Infer 2-qubit gate names from a BackendV2.
    TODO: find a better implementation for this.
    """
    names = set()
    try:
        for inst, _ in backend.instructions:
            # instruction class has num_qubits attribute for standard operations
            if getattr(inst, "num_qubits", None) == 2:
                names.add(inst.name)
    except Exception:
        pass
    if not names:
        names = {"cx", "cz", "ecr", "iswap", "rzx"}
    return names


def schedule_and_duration(circ, backend, scheduler: str = "alap"):
    """
    Schedule the circuit to estimate duration
    TODO: find a better implementation for this.
    """

    if scheduler not in {"alap", "asap"}:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    pm = PassManager([
        TimeUnitConversion(target=backend.target),
        ALAPScheduleAnalysis(target=backend.target) if scheduler == "alap" else ASAPScheduleAnalysis(
            target=backend.target),
    ])
    scheduled = pm.run(circ)

    # Circuit.duration and unit exist for scheduled circuits
    duration = getattr(scheduled, "duration", None)
    unit = getattr(scheduled, "unit", None)

    if duration is None:
        return None, None
    # if unit is 'dt', convert to seconds if possible
    if unit == "dt":
        dt = getattr(backend, "dt", None)
        if dt is None and getattr(getattr(backend, "target", None), "dt", None) is not None:
            dt = backend.target.dt
        if dt:
            return duration * dt, "s"

    return float(duration), unit


def analyze_compiled(tc, backend, scheduler: str):
    """
    Given a transpiled circuit `tc`, run ResourceEstimation and scheduling.

    Returns: (property_set, op_counts, duration_s, duration_unit)
    """

    pm = PassManager([ResourceEstimation()])
    _ = pm.run(tc)
    props = dict(pm.property_set)

    op_counts = tc.count_ops()
    duration_s, unit = schedule_and_duration(tc, backend, scheduler)
    return props, op_counts, duration_s, unit

def get_coupling_edges(backend) -> list[tuple[int, int]]:
    """
    Returns coupling edges as list of (u, v) pairs.
    """
    edges = []
    try:
        if getattr(backend, "coupling_map", None):
            edges = [tuple(e) for e in backend.coupling_map]
        elif hasattr(backend, "target") and backend.target is not None:
            cm = backend.target.build_coupling_map()
            if cm is not None:
                edges = [tuple(e) for e in cm.get_edges()]
    except Exception:
        pass
    return edges

def compile_with_qiskit(circuit, backend, opt_level: int = 0):
    """
    Compiles using Qiskit.
    """
    return transpile(circuit, backend=backend, optimization_level=opt_level)


def compile_with_tket(circuit, backend):
    """
    Compiles using TKET, then returns a Qiskit circuit scheduled for the backend.
    """
    # 1) Get backend coupling
    edges = get_coupling_edges(backend)

    # 2) Fully decompose Qiskit boxes/controlled wrappers so the converter
    #    does not emit QControlBox for arbitrary controlled ops.
    #    The `reps` argument recursively decomposes up to N levels.
    qc_in = circuit.decompose(reps=10)

    # 3) Unroll to a TKET‑friendly gate set to stabilise conversion.
    #    This avoids exotic instructions that trigger QControlBox arity issues.
    tket_basis = [
        "id", "x", "y", "z", "s", "sdg", "t", "tdg", "sx",
        "rx", "ry", "rz", "p", "cx", "cz", "swap", "ccx", "cswap",
        "ecr", "rxx", "ryy", "rzz", "rzx", "measure", "barrier", "reset",
    ]
    qc_in = transpile(qc_in, basis_gates=tket_basis, optimization_level=0)

    # 4) Convert to TKET and run a light optimisation + mapping.
    tk_circ = qiskit_to_tk(qc_in)
    passes = [DecomposeBoxes(), FullPeepholeOptimise()]
    if edges:
        passes.append(DefaultMappingPass(Architecture(edges)))
    SequencePass(passes).apply(tk_circ)

    # 5) Convert back to Qiskit and check against target backend basis.
    qc = tk_to_qiskit(tk_circ)
    return transpile(qc, backend=backend, optimization_level=0)


def debug_tensor_factors_basic(backends: Dict[str, object]) -> pd.DataFrame:
    """
    Small sanity check for Qiskit's num_tensor_factors vs backend size.

    For each backend, run ResourceEstimation on a few toy circuits:
      - 2q separable (H on both qubits)
      - 2q entangled (Bell pair)
      - 3q with only one active qubit (two idle wires)

    Returns a DataFrame with one row per (circuit, backend).
    """

    tests: Dict[str, QuantumCircuit] = {}

    # 1) Two-qubit product state: |++> = H tensor H, no entanglement.
    qc_prod2 = QuantumCircuit(2, name="prod_2q_HH")
    qc_prod2.h(0)
    qc_prod2.h(1)
    tests["prod_2q_HH"] = qc_prod2

    # 2) Two-qubit Bell state: |phi+> = (|00> + |11>)/sqrt(2), entangled.
    qc_bell = QuantumCircuit(2, name="ent_2q_bell")
    qc_bell.h(0)
    qc_bell.cx(0, 1)
    tests["ent_2q_bell"] = qc_bell

    # 3) Three-qubit circuit where only q0 is used; q1 and q2 stay idle.
    qc_idle = QuantumCircuit(3, name="idle_3q_only_q0_used")
    qc_idle.h(0)
    tests["idle_3q_only_q0_used"] = qc_idle

    rows: List[Dict] = []

    for circ_name, qc in tests.items():
        logical_qubits = qc.num_qubits

        for backend_name, backend in backends.items():
            # Transpile exactly as in the main pipeline (backend-specific layout)
            tc = transpile(qc, backend=backend, optimization_level=0)

            # Run ResourceEstimation on the compiled circuit
            pm = PassManager([ResourceEstimation()])
            pm.run(tc)
            props = dict(pm.property_set)

            rows.append(
                {
                    "circuit": circ_name,
                    "backend": backend_name,
                    "logical_qubits": logical_qubits,
                    "physical_qubits": tc.num_qubits,
                    "depth": props.get("depth"),
                    "width": props.get("width"),
                    "tensor_factors": props.get("num_tensor_factors"),
                }
            )

    df = pd.DataFrame(rows)
    print("\n[debug_tensor_factors_basic] Results:")
    print(df.to_string(index=False))
    return df


def main():
    ROOT = Path("../../resource-estimates")
    create_directory_and_parents(str(ROOT))

    lattice_2d = MSLattice(
        {
            "lattice": {
                "dim": {"x": 32, "y": 32},
                "velocities": {"x": 4, "y": 4},
            },
            "geometry": [
                {"shape": "cuboid", "x": [18, 19], "y": [7, 14], "boundary": "specular"},
                {"shape": "cuboid", "x": [18, 19], "y": [19, 26], "boundary": "specular"},
                {"shape": "cuboid", "x": [26, 27], "y": [19, 26], "boundary": "specular"},
                {"shape": "cuboid", "x": [26, 27], "y": [7, 14], "boundary": "specular"},
                {"shape": "cuboid", "x": [20, 25], "y": [4, 5], "boundary": "specular"},
                {"shape": "cuboid", "x": [20, 25], "y": [16, 17], "boundary": "specular"},
                {"shape": "cuboid", "x": [20, 25], "y": [28, 29], "boundary": "specular"},
            ],
        }
    )

    lattices = [lattice_2d]

    backends = {
        # "ibm_guadalupe_v2": FakeGuadalupeV2(),
        # "ibm_sherbrooke_v2": FakeSherbrooke(),
        "ibm_algiers_v2": FakeAlgiers(),
        # "ibm_cambridge_v2": FakeCambridgeV2(),
    }

    # debug_tensor_factors_basic(backends)

    opt_level = 2  # TODO

    compilers = [
        ("qiskit", lambda c, b: compile_with_qiskit(c, b, opt_level)),
        # ("tket",   lambda c, b: compile_with_tket(c, b)),
    ]
    schedulers = ["alap", "asap"]

    rows: List[EstimateRow] = []

    for lat in lattices:
        qc = build_msqlbm_circuit(lat)
        for name, be in backends.items():
            twoq_names = get_twoq_gate_names(be)
            for compiler_name, compile_fn in compilers:
                try:
                    tc = compile_fn(qc, be)
                except Exception as e:
                    print(f"[WARN] compiler={compiler_name} backend={name} failed: {e}")
                    continue

                virt_q = qc.num_qubits
                phys_q = tc.num_qubits

                for sched in schedulers:
                    props, op_counts, dur_s, dur_unit = analyze_compiled(tc, be, sched)
                    swap_count = int(op_counts.get("swap", 0))
                    twoq_total = sum(int(op_counts.get(k, 0)) for k in twoq_names)

                    rows.append(
                        EstimateRow(
                            lattice="2D-MSQLBM",
                            backend=name,
                            compiler=compiler_name,
                            scheduler=sched,
                            opt_level=opt_level,
                            virtual_qubits=virt_q,
                            physical_qubits=phys_q,
                            depth=int(props.get("depth")) if props.get("depth") is not None else None,
                            width=int(props.get("width")) if props.get("width") is not None else None,
                            size=int(props.get("size")) if props.get("size") is not None else None,
                            tensor_factors=int(props.get("num_tensor_factors")) if props.get("num_tensor_factors") is not None else None,
                            twoq_gate_count=twoq_total,
                            swap_count=swap_count,
                            op_counts_json=json.dumps({k: int(v) for k, v in op_counts.items()}),
                            scheduled_duration_s=dur_s,
                            scheduled_duration_unit=dur_unit,
                        )
                    )

    df = pd.DataFrame([asdict(r) for r in rows])
    csv_path = ROOT / "fake_be_msqlbm.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nWrote: {csv_path}\n")
    print(df)


if __name__ == "__main__":
    main()