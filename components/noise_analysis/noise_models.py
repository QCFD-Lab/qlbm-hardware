"""Qiskit Aer noise-model builders for QLBM noise experiments."""

from __future__ import annotations

from typing import Any

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
)

ONE_QUBIT_GATES = ["id", "x", "y", "z", "sx", "h", "rz"]
TWO_QUBIT_GATES = ["cx", "cz", "swap", "cp"]
NOISE_KINDS = {
    "none",
    "depolarizing",
    "hardware_depolarizing",
    "thermal_relaxation",
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


def _hardware_t1_t2(hardware_config: dict[str, Any]) -> tuple[float, float]:
    coherence = hardware_config.get("coherence", {})
    try:
        t1 = float(coherence["t1_s"])
        t2 = float(coherence["t2_s"])
    except KeyError as exc:
        raise ValueError(
            "thermal_relaxation requires coherence.t1_s and coherence.t2_s in hardware_config."
        ) from exc

    if t1 <= 0:
        raise ValueError("thermal_relaxation requires t1_s > 0.")
    if t2 <= 0:
        raise ValueError("thermal_relaxation requires t2_s > 0.")
    if t2 > 2 * t1:
        raise ValueError(
            "thermal_relaxation requires T2 <= 2*T1; "
            f"received T1={t1:g}s and T2={t2:g}s."
        )
    return t1, t2


def build_thermal_relaxation_noise_model(
    hardware_config: dict[str, Any],
) -> NoiseModel:
    """Build thermal relaxation errors from hardware T1/T2 and gate times."""

    t1, t2 = _hardware_t1_t2(hardware_config)
    basis_gates = [str(gate).lower() for gate in hardware_config.get("basis_gates", [])]
    gate_times = {
        str(name).lower(): float(gate_time)
        for name, gate_time in hardware_config.get("gate_times_s", {}).items()
    }
    if not basis_gates:
        raise ValueError("thermal_relaxation requires basis_gates in hardware_config.")
    if not gate_times:
        raise ValueError("thermal_relaxation requires gate_times_s in hardware_config.")

    noise_model = NoiseModel()
    for gate_name in basis_gates:
        arity = GATE_ARITIES.get(gate_name)
        if arity is None:
            continue
        if gate_name not in gate_times:
            raise ValueError(f"thermal_relaxation requires a gate time for {gate_name!r}.")
        gate_time = gate_times[gate_name]
        if gate_time < 0:
            raise ValueError(f"thermal_relaxation gate time for {gate_name!r} must be >= 0.")
        if gate_time == 0.0:
            continue

        one_qubit_error = thermal_relaxation_error(t1, t2, gate_time)
        if arity == 1:
            quantum_error = one_qubit_error
        elif arity == 2:
            quantum_error = one_qubit_error.tensor(one_qubit_error)
        else:
            raise ValueError(f"Unsupported arity {arity} for gate {gate_name!r}.")

        noise_model.add_all_qubit_quantum_error(quantum_error, [gate_name])
    return noise_model


def build_noise_model(
    kind: str,
    single_qubit_probability: float = 0.001,
    two_qubit_probability: float = 0.01,
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

    if kind == "thermal_relaxation":
        if hardware_config is None:
            raise ValueError("thermal_relaxation requires hardware_config.")
        return build_thermal_relaxation_noise_model(hardware_config)

    for name, value in {
        "single_qubit_probability": single_qubit_probability,
        "two_qubit_probability": two_qubit_probability,
    }.items():
        _validate_probability(name, value)

    noise_model = NoiseModel()

    if kind == "depolarizing":
        one_qubit = depolarizing_error(single_qubit_probability, 1)
        two_qubit = depolarizing_error(two_qubit_probability, 2)
    else:
        raise ValueError(f"Unsupported noise kind {kind!r}.")

    noise_model.add_all_qubit_quantum_error(one_qubit, ONE_QUBIT_GATES)
    noise_model.add_all_qubit_quantum_error(two_qubit, TWO_QUBIT_GATES)
    return noise_model
