from __future__ import annotations

"""
Entry point: run QECC benchmarks across codes, states, and noise channels.
Produces a CSV table of fidelities.
"""

import csv
import math
import random
from typing import Tuple

from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator

from qec_codes import build_code_circuit
from noise_models import build_channel_instruction, build_noise


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
    shots: int = 20,
    seed: int | None = 1234,
    rand_state: Tuple[complex, complex] | None = None,
    verbose: bool = False,
) -> float:
    do_syndrome = not (noise == "identity" or p == 0.0)
    noise_inst = build_channel_instruction(noise, p)
    qc, target = build_code_circuit(
        code,
        initial,
        rand_state,
        do_syndrome=do_syndrome,
        noise_inst=noise_inst,
    )
    backend = AerSimulator(method="density_matrix")

    qc = transpile(qc, backend=backend, optimization_level=3)

    if not verbose:
        job = backend.run(qc, shots=shots, seed_simulator=seed)
        result = job.result()
        rho = result.data(0)["rho_logical"]
        if not isinstance(rho, DensityMatrix):
            rho = DensityMatrix(rho)
        return state_fidelity(rho, target, validate=False)

    acc_rho = None
    for s in range(1, shots + 1):
        job = backend.run(qc, shots=1, seed_simulator=seed)
        result = job.result()
        rho = result.data(0)["rho_logical"]
        if not isinstance(rho, DensityMatrix):
            rho = DensityMatrix(rho)
        weight = 1.0 / shots
        acc_rho = rho * weight if acc_rho is None else acc_rho + rho * weight
        print(f"    shot {s}/{shots} for code={code}, init={initial}, noise={noise}")
    return state_fidelity(acc_rho, target, validate=False)


def sweep(out_csv: str = "results.csv", p: float = 0.05, seed: int = 1234, shots: int = 10):
    rng = random.Random(seed)
    rand_ab = _random_state(rng)

    codes = ["baseline", "bit_flip", "phase_flip", "five_qubit", "steane"]
    inits = ["0", "1", "+", "-", "rand"]
    noises = ["bit_flip", "phase_flip", "depolarizing", "amp_damping"]
    heavy_codes = {"five_qubit", "steane"}

    rows = []
    total = len(codes) * len(inits) * len(noises)
    done = 0
    for code in codes:
        for init in inits:
            for noise in noises:
                done += 1
                this_shots = shots if done <= 80 else shots
                verbose = (code in heavy_codes) or (done > total - 20)
                fid = run_once(
                    code=code,
                    initial=init,
                    noise=noise,
                    p=p if noise != "identity" else 0.0,
                    shots=this_shots,
                    seed=seed,
                    rand_state=rand_ab if init == "rand" else None,
                    verbose=verbose,
                )
                print(f"[{done}/{total}] code={code:10s} init={init:4s} noise={noise:12s} p={p if noise!='identity' else 0.0:.3f} fidelity={fid:.4f}")
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


def sweep_ps(
    ps: list[float],
    out_csv: str = "results_multi_p.csv",
    seed: int = 1234,
    shots: int = 20,
):
    rng = random.Random(seed)
    rand_ab = _random_state(rng)

    codes = ["baseline", "bit_flip", "phase_flip", "five_qubit", "steane"]
    inits = ["0", "1", "+", "-", "rand"]
    noises = ["bit_flip", "phase_flip", "depolarizing", "amp_damping"]
    heavy_codes = {"five_qubit", "steane"}

    total = len(ps) * len(codes) * len(inits) * len(noises)
    rows = []
    done_global = 0
    for p in ps:
        done = 0
        for code in codes:
            for init in inits:
                for noise in noises:
                    done += 1
                    done_global += 1
                    # Steane: force 1 shot for speed; otherwise use default
                    this_shots = 1 if code == "steane" else shots
                    verbose = code in heavy_codes
                    fid = run_once(
                        code=code,
                        initial=init,
                        noise=noise,
                        p=p if noise != "identity" else 0.0,
                        shots=this_shots,
                        seed=seed,
                        rand_state=rand_ab if init == "rand" else None,
                        verbose=verbose,
                    )
                    print(
                        f"[{done_global}/{total}] p={p:.3f} code={code:10s} init={init:4s} noise={noise:12s} "
                        f"p_eff={p if noise!='identity' else 0.0:.3f} fidelity={fid:.4f}"
                    )
                    rows.append(
                        {
                            "p": p if noise != "identity" else 0.0,
                            "code": code,
                            "initial": init,
                            "noise": noise,
                            "fidelity": fid,
                        }
                    )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p", "code", "initial", "noise", "fidelity"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    # fid = run_once(code="bit_flip", initial="+", noise="depolarizing", p=0.05)
    # print(f"bit_flip, depolarizing p=0.05 -> fidelity {fid:.4f}")

    ps = [0.2, 0.4, 0.6, 0.8]
    sweep_ps(ps=ps, out_csv="results_multi_p.csv", seed=1234, shots=10)


