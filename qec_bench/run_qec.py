from __future__ import annotations

"""
Simple runner: choose a code ("bit_flip" or "phase_flip"), choose noise ("identity" or "depolarizing"),
set p and initial state, and compute fidelity after encode->noise->syndrome->recover->decode.
"""

from typing import Callable

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator

from circuits import bit_flip_circuit, phase_flip_circuit
from noise_models import build_noise


def run_once(
    code: str = "bit_flip",
    initial: str = "+",
    noise: str = "depolarizing",
    p: float = 0.05,
    shots: int = 2048,
    seed: int | None = 1234,
) -> float:
    builders: dict[str, Callable[[str], tuple[QuantumCircuit, Statevector]]] = {
        "bit_flip": bit_flip_circuit,
        "phase_flip": phase_flip_circuit,
    }
    if code not in builders:
        raise ValueError(f"Unknown code '{code}'")
    qc, target = builders[code](initial)

    backend = AerSimulator(method="density_matrix")
    noise_model = build_noise(noise, p)
    job = backend.run(qc, noise_model=noise_model, shots=shots, seed_simulator=seed)
    result = job.result()
    data = result.data(0)
    rho = data["rho_logical"]
    if not isinstance(rho, DensityMatrix):
        rho = DensityMatrix(rho)
    fid = state_fidelity(rho, target)
    return fid


if __name__ == "__main__":
    fid = run_once(code="phase_flip", initial="+", noise="dephasing", p=0.05)
    print(f"bit_flip, depolarizing p=0.05 -> fidelity {fid:.4f}")


