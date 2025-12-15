from __future__ import annotations

"""
Entry point: run QECC benchmarks across codes, states, and noise channels.
Produces a CSV table of fidelities.
"""

import csv
import math
import random
from typing import Tuple

from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator

from circuits import build_code_circuit
from noise_models import build_noise


def _random_state(rng: random.Random) -> Tuple[complex, complex]:
    theta = rng.random() * math.pi
    phi = rng.random() * 2 * math.pi
    a = math.cos(theta / 2)
    b = math.sin(theta / 2) * (math.cos(phi) + 1j * math.sin(phi))
    return a, b


def run_once(
    code: str,
    initial: str,
    noise: str,
    p: float,
    shots: int = 2048,
    seed: int | None = 1234,
    rand_state: Tuple[complex, complex] | None = None,
) -> float:
    qc, target = build_code_circuit(code, initial, rand_state)
    backend = AerSimulator(method="density_matrix")
    noise_model = build_noise(noise, p)
    job = backend.run(qc, noise_model=noise_model, shots=shots, seed_simulator=seed)
    result = job.result()
    rho = result.data(0)["rho_logical"]
    if not isinstance(rho, DensityMatrix):
        rho = DensityMatrix(rho)
    return state_fidelity(rho, target)


def sweep(out_csv: str = "results.csv", p: float = 0.05, seed: int = 1234):
    rng = random.Random(seed)
    rand_ab = _random_state(rng)

    codes = ["baseline", "bit_flip", "phase_flip", "five_qubit", "steane"]
    inits = ["0", "1", "+", "-", "rand"]
    noises = ["identity", "depolarizing", "dephasing", "amp_damping"]

    rows = []
    for code in codes:
        for init in inits:
            for noise in noises:
                fid = run_once(
                    code=code,
                    initial=init,
                    noise=noise,
                    p=p if noise != "identity" else 0.0,
                    shots=2048,
                    seed=seed,
                    rand_state=rand_ab if init == "rand" else None,
                )
                rows.append(
                    {
                        "code": code,
                        "initial": init,
                        "noise": noise,
                        "p": p if noise != "identity" else 0.0,
                        "fidelity": fid,
                    }
                )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "initial", "noise", "p", "fidelity"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    # Quick example
    # fid = run_once(code="bit_flip", initial="+", noise="depolarizing", p=0.05)
    # print(f"bit_flip, depolarizing p=0.05 -> fidelity {fid:.4f}")
    # Full sweep (uncomment to run)
    sweep(out_csv="results.csv", p=0.05, seed=1234)


