from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from qiskit import QuantumCircuit, transpile

from qlbm.components import MSQLBM
from qlbm.lattice import MSLattice


def build_lattice_3d_no_obstacles() -> MSLattice:
    nx = 16384
    ny = 16384
    nz = 16384

    lattice_3d = MSLattice(
        {
            "lattice": {
                "dim": {"x": nx, "y": ny, "z": nz},
                "velocities": {"x": 4, "y": 4, "z": 4},
            },
            "geometry": [],
        }
    )
    return lattice_3d

def build_msqlbm_circuit(lattice: MSLattice) -> QuantumCircuit:
    algo = MSQLBM(lattice)
    qc = algo.circuit
    if not isinstance(qc, QuantumCircuit):
        raise TypeError(f"Expected qiskit.QuantumCircuit, got {type(qc)}")
    return qc


def _count_ops_by_arity(qc: QuantumCircuit) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Returns (all_ops, ops_1q, ops_2qplus) as name->count dicts.
    Note: counts are based on instruction arity in qc.data.
    """
    all_ops: Dict[str, int] = {}
    ops_1q: Dict[str, int] = {}
    ops_2qplus: Dict[str, int] = {}

    for ci in qc.data:
        inst = ci.operation
        qargs = ci.qubits
        name = getattr(inst, "name", str(inst))

        all_ops[name] = all_ops.get(name, 0) + 1
        nq = len(qargs)
        if nq == 1:
            ops_1q[name] = ops_1q.get(name, 0) + 1
        elif nq >= 2:
            ops_2qplus[name] = ops_2qplus.get(name, 0) + 1

    return all_ops, ops_1q, ops_2qplus


@dataclass
class CircuitResources:
    label: str
    stage: str  # "logical" or "transpiled"
    num_qubits: int
    depth: int
    size: int
    op_counts: Dict[str, int]
    op_counts_1q: Dict[str, int]
    op_counts_2qplus: Dict[str, int]
    num_1q_ops: int
    num_2qplus_ops: int

    # helpful extras
    num_clbits: int
    num_measure: int
    num_reset: int
    num_barrier: int


def summarize_circuit(qc: QuantumCircuit, label: str, stage: str) -> CircuitResources:
    op_counts, ops_1q, ops_2qplus = _count_ops_by_arity(qc)

    num_measure = int(op_counts.get("measure", 0))
    num_reset = int(op_counts.get("reset", 0))
    num_barrier = int(op_counts.get("barrier", 0))

    return CircuitResources(
        label=label,
        stage=stage,
        num_qubits=qc.num_qubits,
        depth=qc.depth(),
        size=qc.size(),
        op_counts=op_counts,
        op_counts_1q=ops_1q,
        op_counts_2qplus=ops_2qplus,
        num_1q_ops=sum(ops_1q.values()),
        num_2qplus_ops=sum(ops_2qplus.values()),
        num_clbits=qc.num_clbits,
        num_measure=num_measure,
        num_reset=num_reset,
        num_barrier=num_barrier,
    )


def transpile_circuit(qc, opt_level, basis_gates: Optional[list[str]] = None) -> QuantumCircuit:
    """
    Generic transpilation with no backend.
    USed to see how metrics change under different bases and optimization levels.
    """
    if basis_gates is not None:
        basis_gates = [g for g in basis_gates if g != "barrier"]
    tc = transpile(
        qc,
        optimization_level=opt_level,
        basis_gates=basis_gates,
    )
    return tc


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def write_json_func(outpath: Path, payload: Dict[str, Any]) -> None:
    outpath.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"Wrote JSON: {outpath}")


def write_csv_func(outpath: Path, rows: list[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"Wrote CSV: {outpath}")


def flatten_resources_for_csv(r: CircuitResources) -> Dict[str, Any]:
    return {
        "label": r.label,
        "stage": r.stage,
        "num_qubits": r.num_qubits,
        "num_clbits": r.num_clbits,
        "depth": r.depth,
        "size": r.size,
        "num_1q_ops": r.num_1q_ops,
        "num_2qplus_ops": r.num_2qplus_ops,
        "num_measure": r.num_measure,
        "num_reset": r.num_reset,
        "num_barrier": r.num_barrier,
        "op_counts_json": json.dumps(r.op_counts),
        "op_counts_1q_json": json.dumps(r.op_counts_1q),
        "op_counts_2qplus_json": json.dumps(r.op_counts_2qplus),
    }


def main() -> None:
    outdir = Path("../qlbm-hardware-output/resource-estimates/test_msqlbm_resource-estimates")

    label = "MSQLBM_3D_16384x16384x16384_v4x4x4_no_obstacles"

    enable_transpile = True    # set False to skip transpilation
    optimization_level = 1    # 0,1,2,3
    basis_gates = ["rz", "sx", "x", "cx", "measure", "reset"]
    write_csv = True

    ensure_outdir(outdir)

    lattice = build_lattice_3d_no_obstacles()
    qc = build_msqlbm_circuit(lattice)

    logical = summarize_circuit(qc, label=label, stage="logical")
    results = {
        "label": label,
        "lattice": {
            "dim": {"x": 16384, "y": 16384, "z": 16384},
            "velocities": {"x": 4, "y": 4, "z": 4},
            "geometry_count": 0,
            "geometry": [],
        },
        "qiskit": {
            "transpile": {
                "enabled": bool(enable_transpile),
                "optimization_level": int(optimization_level),
                "basis_gates": basis_gates,
            }
        },
        "resources": {
            "logical": asdict(logical),
        },
    }

    rows_for_csv = [flatten_resources_for_csv(logical)]

    if enable_transpile:
        tc = transpile_circuit(qc, opt_level=optimization_level, basis_gates=basis_gates)
        transpiled = summarize_circuit(tc, label=label, stage="transpiled")
        results["resources"]["transpiled"] = asdict(transpiled)
        rows_for_csv.append(flatten_resources_for_csv(transpiled))

    json_path = outdir / f"{label}.json"
    write_json_func(json_path, results)

    if write_csv:
        csv_path = outdir / f"{label}.csv"
        write_csv_func(csv_path, rows_for_csv)

    def _print_summary(r: CircuitResources) -> None:
        print(
            f"\n[{r.stage}] qubits={r.num_qubits} depth={r.depth} size={r.size} "
            f"1q_ops={r.num_1q_ops} 2q+_ops={r.num_2qplus_ops}"
        )

    _print_summary(logical)
    if enable_transpile:
        _print_summary(transpiled)


if __name__ == "__main__":
    main()