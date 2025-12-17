from __future__ import annotations

import os
import sys

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, Statevector, Pauli, state_fidelity

# Ensure parent on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qec_codes import (
    _encoder_five_qubit,
    _encoder_steane,
    _measure_stabilizers,
    _auto_correction_map,
    _apply_corrections,
)


def run_code(code: str, target_logical: str = "0"):
    if code == "five_qubit":
        stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
        encoder = _encoder_five_qubit
        n = 5
    elif code == "steane":
        stabilizers = ["IIIXXXX", "IXXIIXX", "XIXIXIX", "IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"]
        encoder = _encoder_steane
        n = 7
    else:
        raise ValueError("code must be five_qubit or steane")

    data = QuantumRegister(n, "q")
    qc = QuantumCircuit(data, name=f"{code}_single_error_check")

    # Prepare logical |0> (extend to |1> by X on first qubit)
    enc = encoder(data)
    qc.compose(enc, qubits=data, inplace=True)
    if target_logical == "1":
        qc.x(data[0])

    # Save target logical state for fidelity after decode
    target_sv = Statevector.from_label(target_logical)

    errors = ["X", "Y", "Z"]
    backend = AerSimulator(method="density_matrix")

    for idx in range(n):
        for err in errors:
            qct = qc.copy()
            # inject single Pauli error
            getattr(qct, err.lower())(data[idx])

            # measure syndrome and correct
            syn = _measure_stabilizers(qct, data, stabilizers)
            corr_map = _auto_correction_map(stabilizers)
            _apply_corrections(qct, data, syn, corr_map)

            # decode
            qct.compose(enc.inverse(), qubits=data, inplace=True)
            qct.save_density_matrix([data[0]], label="rho_logical")

            tqc = transpile(qct, backend=backend, optimization_level=0)
            result = backend.run(tqc, shots=1).result()
            syn_counts = result.get_counts(0)
            rho = DensityMatrix(result.data(0)["rho_logical"])
            fid = state_fidelity(rho, target_sv)

            print(f"--- {code} err={err} on qubit {idx} ---")
            print(f" syndrome: {syn_counts}")
            print(f" fidelity to |{target_logical}>: {fid:.4f}")
            print()


def main():
    run_code("five_qubit", "0")
    run_code("five_qubit", "1")
    run_code("steane", "0")
    run_code("steane", "1")


if __name__ == "__main__":
    main()

