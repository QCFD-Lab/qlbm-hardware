"""Qiskit Aer noise-model builders for QLBM noise experiments."""

from __future__ import annotations

from typing import Any

from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
    pauli_error,
    phase_damping_error,
)

ONE_QUBIT_GATES = ["id", "x", "y", "z", "sx", "h", "rz"]
TWO_QUBIT_GATES = ["cx", "cz", "swap", "cp"]
NOISE_KINDS = {
    "none",
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "gate_error",
    "hardware_depolarizing",
}
GATE_ARITIES = {
    "id": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "sx": 1,
    "h": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "cx": 2,
    "cz": 2,
    "swap": 2,
    "cp": 2,
}


def _validate_probability(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be a probability in [0, 1].")


def fidelity_to_depolarizing_probability(fidelity: float, num_qubits: int) -> float:
    """Convert average gate fidelity to Qiskit depolarizing probability."""

    _validate_probability("fidelity", fidelity)
    if num_qubits < 1:
        raise ValueError("num_qubits must be positive.")
    dimension = 2**num_qubits
    probability = dimension / (dimension - 1) * (1 - fidelity)
    return min(1.0, max(0.0, probability))


def build_hardware_depolarizing_noise_model(
    hardware_config: dict[str, Any],
) -> NoiseModel:
    """Build gate depolarizing errors from hardware ``gate_fidelities``."""

    gate_fidelities = {
        str(name).lower(): fidelity
        for name, fidelity in hardware_config.get("gate_fidelities", {}).items()
    }
    if not gate_fidelities:
        raise ValueError("hardware_depolarizing requires gate_fidelities in hardware_config.")

    noise_model = NoiseModel()
    for gate_name, fidelity in gate_fidelities.items():
        arity = GATE_ARITIES.get(gate_name)
        if arity is None:
            continue
        probability = fidelity_to_depolarizing_probability(float(fidelity), arity)
        if probability == 0.0:
            continue
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(probability, arity),
            [gate_name],
        )
    return noise_model


def build_noise_model(
    kind: str,
    single_qubit_probability: float = 0.001,
    two_qubit_probability: float = 0.01,
    damping_probability: float = 0.001,
    phase_probability: float = 0.001,
    gate_error_probability: float = 0.001,
    hardware_config: dict[str, Any] | None = None,
):
    """Build a Qiskit Aer ``NoiseModel`` or return ``None`` for noiseless runs."""

    if kind == "none":
        return None

    if kind not in NOISE_KINDS:
        raise ValueError(f"Unsupported noise kind {kind!r}; expected one of {sorted(NOISE_KINDS)}.")

    if kind == "hardware_depolarizing":
        if hardware_config is None:
            raise ValueError("hardware_depolarizing requires hardware_config.")
        return build_hardware_depolarizing_noise_model(hardware_config)

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
