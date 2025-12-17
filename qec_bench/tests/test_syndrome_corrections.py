from __future__ import annotations

import os
import sys

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, Pauli

# Ensure parent on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qec_codes import (
    _encoder_bit_flip,
    _encoder_phase_flip,
    _measure_stabilizers,
    _auto_correction_map,
    _apply_corrections,
)


def run_case(code: str, basis_state: str):
    """Prepare a specific post-noise basis state, measure syndrome, correct, and show result.
    For phase_flip, use '+/-' strings like '++-' (q2 q1 q0 order). For bit_flip, use 0/1 bitstrings."""
    if code == "bit_flip":
        stabilizers = ["ZZI", "IZZ"]
        encoder = _encoder_bit_flip
    elif code == "phase_flip":
        stabilizers = ["XXI", "IXX"]
        encoder = _encoder_phase_flip
    else:
        raise ValueError("code must be bit_flip or phase_flip")

    n = len(stabilizers[0])
    if len(basis_state) != n:
        raise ValueError(f"basis_state length {len(basis_state)} != {n}")

    data = QuantumRegister(n, "q")
    qc = QuantumCircuit(data, name=f"{code}_syndrome_check")

    # Initialize directly to the provided basis state
    # For phase_flip: allow '+/-' strings (preferred). Otherwise fallback to 0/1 -> |+>,|->
    is_pm = code == "phase_flip" and set(basis_state) <= {"+", "-"}
    for i, bit in enumerate(reversed(basis_state)):  # reversed: last char -> qubit 0
        if code == "phase_flip":
            qc.h(data[i])  # start in |+>
            if (is_pm and bit == "-") or (not is_pm and bit == "1"):
                qc.z(data[i])  # flip phase to |->
        else:
            if bit == "1":
                qc.x(data[i])

    # Save pre-correction state for stabilizer eigenvalues
    sv_pre = Statevector.from_instruction(qc)

    # Measure stabilizers
    # Use our own syn register to avoid duplicate-name issues inside helper
    # (helpers will create their own syn if none provided; we pass ours explicitly)
    syn = _measure_stabilizers(qc, data, stabilizers)
    # Apply corrections
    corr_map = _auto_correction_map(stabilizers)
    _apply_corrections(qc, data, syn, corr_map)
    qc.barrier(data)
    qc.save_density_matrix(data, label="rho_corrected")

    backend = AerSimulator(method="density_matrix")
    job = backend.run(qc, shots=1)
    result = job.result()
    syn_bits = result.get_counts(0)
    rho_corr = DensityMatrix(result.data(0)["rho_corrected"])
    sv_corr = None
    try:
        sv_corr = rho_corr.to_statevector()
    except Exception:
        pass

    print(f"=== {code} starting |{basis_state}> ===")
    print(f"syndrome counts: {syn_bits}")
    # Stabilizer eigenvalues on the starting state
    for stab in stabilizers:
        exp = sv_pre.expectation_value(Pauli(stab)).real
        print(f"  pre  <{stab}> = {exp:+.3f}")
    # Stabilizer eigenvalues
    for stab in stabilizers:
        exp = rho_corr.expectation_value(Pauli(stab)).real
        print(f"  post <{stab}> = {exp:+.3f}")
    if sv_corr:
        for i, amp in enumerate(sv_corr.data):
            if abs(amp) > 1e-8:
                print(f"  corrected |{i:0{n}b}>: {amp}")
    else:
        print("corrected state (density matrix):")
        print(rho_corr)
    print()


def main():
    # Bit-flip examples
    run_case("bit_flip", "010")  # single X error on qubit1 relative to |000>
    run_case("bit_flip", "011")  # single X error on qubit1 relative to |001>

    # Phase-flip examples: explicit +/- basis strings (q2 q1 q0)
    run_case("phase_flip", "++-")  # |++->
    run_case("phase_flip", "-+-")  # |-+->


if __name__ == "__main__":
    main()

