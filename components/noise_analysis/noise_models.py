"""Qiskit Aer noise-model builders for QLBM tube experiments."""

from __future__ import annotations

from config import NoiseConfig


ONE_QUBIT_GATES = ["id", "x", "y", "z", "sx", "h", "rz"]
TWO_QUBIT_GATES = ["cx", "cz", "swap", "cp"]


def build_noise_model(config: NoiseConfig):
    """Build a Qiskit Aer ``NoiseModel`` or return ``None`` for noiseless runs."""

    config.validate()
    if config.kind == "none":
        return None

    from qiskit_aer.noise import (
        NoiseModel,
        amplitude_damping_error,
        depolarizing_error,
        pauli_error,
        phase_damping_error,
    )

    noise_model = NoiseModel()

    if config.kind == "depolarizing":
        one_qubit = depolarizing_error(config.single_qubit_probability, 1)
        two_qubit = depolarizing_error(config.two_qubit_probability, 2)
    elif config.kind == "amplitude_damping":
        one_qubit = amplitude_damping_error(config.damping_probability)
        two_qubit = one_qubit.tensor(one_qubit)
    elif config.kind == "phase_damping":
        one_qubit = phase_damping_error(config.phase_probability)
        two_qubit = one_qubit.tensor(one_qubit)
    elif config.kind == "gate_error":
        p = config.gate_error_probability
        one_qubit = pauli_error([("X", p), ("I", 1 - p)])
        two_qubit = one_qubit.tensor(one_qubit)
    else:
        raise ValueError(f"Unsupported noise kind {config.kind!r}.")

    noise_model.add_all_qubit_quantum_error(one_qubit, ONE_QUBIT_GATES)
    noise_model.add_all_qubit_quantum_error(two_qubit, TWO_QUBIT_GATES)
    return noise_model


def choose_backend_method(config: NoiseConfig, explicit_method: str | None) -> str:
    """Choose a conservative Aer method for the selected noise mode."""

    if explicit_method:
        return explicit_method
    return "statevector"
