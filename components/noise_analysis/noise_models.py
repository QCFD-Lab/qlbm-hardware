"""Qiskit Aer noise-model builders for QLBM noise experiments."""

from __future__ import annotations
from typing import Any
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

GATES_BY_ARITY = {
    1: ["id", "x", "y", "z", "sx", "h", "rx", "ry", "rz", "rxy"],
    2: ["cx", "cz", "swap", "cp", "iswap", "ecr", "rzz", "zz"],
}
NOISE_KINDS = {
    "none",
    "depolarizing",
    "hardware_depolarizing",
    "thermal_relaxation",
}


def gate_arity(gate_name: str) -> int | None:
    """Return the supported gate arity, or None for gates that are not not modeled"""
    for arity, gate_names in GATES_BY_ARITY.items():
        if gate_name in gate_names:
            return arity
    return None


def _supported_gate_arity(gate_name: str, noise_kind: str) -> int:
    arity = gate_arity(gate_name)
    if arity is None:
        raise ValueError(
            f"{noise_kind} does not support gate {gate_name!r}; "
            "add it to GATES_BY_ARITY or remove it from the hardware config."
        )
    return arity


def _normalized_gate_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {str(name).lower(): value for name, value in values.items()}


def _transpiled_gate_names(hardware_config: dict[str, Any]) -> list[str]:
    gate_names = hardware_config.get(
        "transpile_basis_gates",
        hardware_config.get("basis_gates", []),
    )
    return [str(gate_name).lower() for gate_name in gate_names]


def _apply_gate_aliases(values: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    resolved_values = dict(values)
    for alias_name, source_name in aliases.items():
        alias = str(alias_name).lower()
        source = str(source_name).lower()
        if alias in resolved_values:
            continue
        if source in resolved_values:
            resolved_values[alias] = resolved_values[source]
    return resolved_values


def _float_config_value(context: str, value: Any) -> float:
    if value is None:
        raise ValueError(f"{context} must not be null.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a real number.") from exc


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


def build_hardware_depolarizing_noise_model(hardware_config: dict[str, Any]) -> NoiseModel:
    """Build gate depolarizing errors from hardware gate_fidelities."""
    gate_fidelities = _apply_gate_aliases(
        _normalized_gate_dict(hardware_config.get("gate_fidelities", {})),
        hardware_config.get("gate_time_aliases", {}),
    )
    if not gate_fidelities:
        raise ValueError(
            "hardware_depolarizing requires gate_fidelities in hardware_config."
        )

    target_gates = _transpiled_gate_names(hardware_config) or sorted(gate_fidelities)
    noise_model = NoiseModel()
    for gate_name in target_gates:
        arity = _supported_gate_arity(gate_name, "hardware_depolarizing")
        if gate_name not in gate_fidelities:
            raise ValueError(
                f"hardware_depolarizing requires a gate fidelity for {gate_name!r}."
            )
        fidelity = _float_config_value(
            f"hardware_depolarizing gate fidelity for {gate_name!r}",
            gate_fidelities[gate_name],
        )
        probability = fidelity_to_depolarizing_probability(fidelity, arity)
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
        t1_raw = coherence["t1_s"]
        t2_raw = coherence["t2_s"]
    except KeyError as exc:
        raise ValueError(
            "thermal_relaxation requires coherence.t1_s and coherence.t2_s in hardware_config."
        ) from exc
    t1 = _float_config_value("thermal_relaxation coherence.t1_s", t1_raw)
    t2 = _float_config_value("thermal_relaxation coherence.t2_s", t2_raw)

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


def build_thermal_relaxation_noise_model(hardware_config: dict[str, Any]) -> NoiseModel:
    """Build thermal relaxation errors from hardware T1/T2 and gate times."""

    t1, t2 = _hardware_t1_t2(hardware_config)
    target_gates = _transpiled_gate_names(hardware_config)
    gate_times = _apply_gate_aliases(
        _normalized_gate_dict(hardware_config.get("gate_times_s", {})),
        hardware_config.get("gate_time_aliases", {}),
    )
    if not target_gates:
        raise ValueError(
            "thermal_relaxation requires basis_gates or transpile_basis_gates in hardware_config."
        )
    if not gate_times:
        raise ValueError("thermal_relaxation requires gate_times_s in hardware_config.")

    noise_model = NoiseModel()
    for gate_name in target_gates:
        arity = _supported_gate_arity(gate_name, "thermal_relaxation")
        if gate_name not in gate_times:
            raise ValueError(
                f"thermal_relaxation requires a gate time for {gate_name!r}."
            )
        gate_time = _float_config_value(
            f"thermal_relaxation gate time for {gate_name!r}",
            gate_times[gate_name],
        )
        if gate_time < 0:
            raise ValueError(
                f"thermal_relaxation gate time for {gate_name!r} must be >= 0."
            )
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
    """Build a Qiskit Aer NoiseModel or return None for noiseless runs."""

    if kind == "none":
        return None

    if kind not in NOISE_KINDS:
        raise ValueError(
            f"Unsupported noise kind {kind!r}; expected one of {sorted(NOISE_KINDS)}."
        )

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

    noise_model.add_all_qubit_quantum_error(one_qubit, GATES_BY_ARITY[1])
    noise_model.add_all_qubit_quantum_error(two_qubit, GATES_BY_ARITY[2])
    return noise_model
