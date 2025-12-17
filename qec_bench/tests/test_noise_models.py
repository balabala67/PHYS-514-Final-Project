from __future__ import annotations

from typing import Iterable

import os
import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector, state_fidelity

# Ensure parent is on sys.path when running from tests/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from noise_models import build_channel_instruction


def _prepare(qc: QuantumCircuit, init: str, q0):
    if init == "0":
        return Statevector([1, 0])
    if init == "1":
        qc.x(q0)
        return Statevector([0, 1])
    if init == "+":
        qc.h(q0)
        return Statevector.from_label("+")
    if init == "-":
        qc.x(q0)
        qc.h(q0)
        return Statevector.from_label("-")
    raise ValueError("init must be one of 0,1,+,-")


def run_noise_case(noise: str, p: float, init: str) -> dict:
    qc = QuantumCircuit(1)
    target = _prepare(qc, init, 0)
    # Apply the channel exactly once
    noise_inst = build_channel_instruction(noise, p)
    if noise_inst is not None:
        qc.append(noise_inst, [0])
    qc.save_density_matrix([0], label="rho")

    backend = AerSimulator(method="density_matrix")
    tqc = transpile(qc, backend, optimization_level=0)
    job = backend.run(tqc, shots=1)
    result = job.result()
    rho = result.data(0)["rho"]
    rho = rho if isinstance(rho, DensityMatrix) else DensityMatrix(rho)

    def bloch_from_state(sv: Statevector):
        bx = float((sv.expectation_value(Pauli("X"))).real)
        by = float((sv.expectation_value(Pauli("Y"))).real)
        bz = float((sv.expectation_value(Pauli("Z"))).real)
        return bx, by, bz

    def bloch_from_dm(dm: DensityMatrix):
        bx = float((dm.expectation_value(Pauli("X"))).real)
        by = float((dm.expectation_value(Pauli("Y"))).real)
        bz = float((dm.expectation_value(Pauli("Z"))).real)
        return bx, by, bz

    fid = float(state_fidelity(rho, target))
    bx0, by0, bz0 = bloch_from_state(target)
    bx1, by1, bz1 = bloch_from_dm(rho)
    data = rho.data
    p0 = float(data[0, 0].real)
    p1 = float(data[1, 1].real)
    return {
        "fidelity": fid,
        "bloch_before": (bx0, by0, bz0),
        "bloch_after": (bx1, by1, bz1),
        "p0": p0,
        "p1": p1,
        "target": target,
        "rho": rho,
    }


def sweep(noises: Iterable[str], ps: Iterable[float], inits: Iterable[str]):
    for noise in noises:
        for p in ps:
            for init in inits:
                res = run_noise_case(noise, p, init)
                bx0, by0, bz0 = res["bloch_before"]
                bx1, by1, bz1 = res["bloch_after"]
                print(f"noise={noise:12s} p={p:4.2f} init={init}")
                print(
                    f"   before: Bloch=({bx0:+.3f},{by0:+.3f},{bz0:+.3f})"
                )
                print(
                    f"   after : Bloch=({bx1:+.3f},{by1:+.3f},{bz1:+.3f}) "
                    f"fid={res['fidelity']:.4f} P0={res['p0']:.3f} P1={res['p1']:.3f}"
                )


if __name__ == "__main__":
    noises = ["bit_flip", "phase_flip", "depolarizing", "amp_damping"]
    ps = [1.0]  # use full-strength to validate channel behavior
    inits = ["0", "1", "+", "-"]
    sweep(noises, ps, inits)

