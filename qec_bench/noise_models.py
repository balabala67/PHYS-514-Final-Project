"""
Noise model helpers (separate file so we can vary channels independently).
Currently supports identity and depolarizing; easy to extend with dephasing/amp damping.
"""

from __future__ import annotations

from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error, phase_damping_error


def build_noise(noise: str, p: float) -> NoiseModel | None:
    if noise == "identity" or p <= 0:
        return None

    if noise == "depolarizing":
        err1 = depolarizing_error(p, 1)
        err2 = depolarizing_error(p, 2)
    elif noise == "dephasing":
        err1 = phase_damping_error(p)
        err2 = err1.tensor(err1)
    elif noise == "amp_damping":
        err1 = amplitude_damping_error(p)
        err2 = err1.tensor(err1)
    else:
        raise ValueError("noise must be identity|depolarizing|dephasing|amp_damping")

    model = NoiseModel()
    one_qubit_ops = ["id", "x", "y", "z", "h", "s", "sdg"]
    two_qubit_ops = ["cx"]
    model.add_all_qubit_quantum_error(err1, one_qubit_ops)
    model.add_all_qubit_quantum_error(err2, two_qubit_ops)
    return model


__all__ = ["build_noise"]


