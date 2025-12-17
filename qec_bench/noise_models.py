

from __future__ import annotations

from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
    pauli_error,
)


def build_noise(noise: str, p: float) -> NoiseModel | None:
    if noise == "identity" or p <= 0:
        return None

    if noise == "depolarizing":
        err1 = depolarizing_error(p, 1)
        err2 = depolarizing_error(p, 2)
    elif noise in ("dephasing", "phase_flip"):
        # Use phase-flip (Z) channel: with prob p apply Z, else I
        err1 = pauli_error([("Z", p), ("I", 1 - p)])
        err2 = err1.tensor(err1)  # independent phase flips on each qubit of 2q gates
    elif noise == "bit_flip":
        # Bit-flip (X) channel: with prob p apply X
        err1 = pauli_error([("X", p), ("I", 1 - p)])
        err2 = err1.tensor(err1)
    elif noise == "amp_damping":
        err1 = amplitude_damping_error(p)
        err2 = err1.tensor(err1)
    else:
        raise ValueError("noise must be identity|depolarizing|dephasing|amp_damping")

    model = NoiseModel()
    # Cover common single-qubit ops that survive transpile; include virtual phases.
    one_qubit_ops = ["id", "x", "y", "z", "h", "s", "sdg", "sx", "rx", "ry", "rz", "p", "u1", "u2", "u3"]
    two_qubit_ops = ["cx"]
    model.add_all_qubit_quantum_error(err1, one_qubit_ops)
    model.add_all_qubit_quantum_error(err2, two_qubit_ops)
    return model


def build_channel_instruction(noise: str, p: float):
    """Return a single-qubit error channel as an Instruction to append once."""
    if noise == "identity" or p <= 0:
        return None
    if noise == "depolarizing":
        err1 = depolarizing_error(p, 1)
    elif noise in ("dephasing", "phase_flip"):
        err1 = pauli_error([("Z", p), ("I", 1 - p)])
    elif noise == "bit_flip":
        err1 = pauli_error([("X", p), ("I", 1 - p)])
    elif noise == "amp_damping":
        err1 = amplitude_damping_error(p)
    else:
        raise ValueError("noise must be identity|bit_flip|phase_flip|depolarizing|amp_damping")
    return err1.to_instruction()


__all__ = ["build_noise"]


