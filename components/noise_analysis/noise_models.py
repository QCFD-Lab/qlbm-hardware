"""Qiskit Aer noise-model builders for QLBM noise experiments."""

from __future__ import annotations
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
    pauli_error,
    phase_damping_error,
)

ONE_QUBIT_GATES = ["id", "x", "y", "z", "sx", "h", "rz"]
TWO_QUBIT_GATES = ["cx", "cz", "swap", "cp"]
NOISE_KINDS = {"none", "depolarizing", "amplitude_damping", "phase_damping", "gate_error"}


def _validate_probability(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be a probability in [0, 1].")


def build_noise_model(
    kind: str,
    single_qubit_probability: float = 0.001,
    two_qubit_probability: float = 0.01,
    damping_probability: float = 0.001,
    phase_probability: float = 0.001,
    gate_error_probability: float = 0.001,
):
    """Build a Qiskit Aer ``NoiseModel`` or return ``None`` for noiseless runs."""

    if kind == "none":
        return None

    if kind not in NOISE_KINDS:
        raise ValueError(f"Unsupported noise kind {kind!r}; expected one of {sorted(NOISE_KINDS)}.")

    for name, value in {
        "single_qubit_probability": single_qubit_probability,
        "two_qubit_probability": two_qubit_probability,
        "damping_probability": damping_probability,
        "phase_probability": phase_probability,
        "gate_error_probability": gate_error_probability,
    }.items():
        _validate_probability(name, value)



    noise_model = NoiseModel()

    if kind == "depolarizing":
        one_qubit = depolarizing_error(single_qubit_probability, 1)
        two_qubit = depolarizing_error(two_qubit_probability, 2)
    elif kind == "amplitude_damping":
        one_qubit = amplitude_damping_error(damping_probability)
        two_qubit = one_qubit.tensor(one_qubit)
    elif kind == "phase_damping":
        one_qubit = phase_damping_error(phase_probability)
        two_qubit = one_qubit.tensor(one_qubit)
    elif kind == "gate_error":
        p = gate_error_probability
        one_qubit = pauli_error([("X", p), ("I", 1 - p)])
        two_qubit = one_qubit.tensor(one_qubit)
    else:
        raise ValueError(f"Unsupported noise kind {kind!r}.")

    noise_model.add_all_qubit_quantum_error(one_qubit, ONE_QUBIT_GATES)
    noise_model.add_all_qubit_quantum_error(two_qubit, TWO_QUBIT_GATES)
    return noise_model
