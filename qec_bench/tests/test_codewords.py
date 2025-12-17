"""
Inspect logical codewords: encode |0> and |1> for each code and save statevectors.
No noise, no syndrome. Prints amplitudes and writes out to text files for inspection.
"""
from __future__ import annotations

import os
import sys

from qiskit.quantum_info import Statevector, partial_trace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qec_codes import build_code_circuit


def dump_ket(prefix: str, sv: Statevector, thresh: float = 1e-10):
    for i, amp in enumerate(sv.data):
        if abs(amp) > thresh:
            print(f"  {prefix}|{i:0{sv.num_qubits}b}>: {amp}")
    print()


def main():
    codes = ["baseline", "bit_flip", "phase_flip", "five_qubit", "steane"]

    rand_state = (0.6, 0.8)

    for code in codes:
        print(f"\n{code.upper()}\n")
        for init in ("0", "1", "psi"):
            qc, _ = build_code_circuit(
                code,
                init if init != "psi" else "rand",
                rand=rand_state if init == "psi" else None,
                do_syndrome=False,
                save_logical=False,
                include_decode=False,
            )
            sv_enc = Statevector.from_instruction(qc)
            qc_dec, target = build_code_circuit(
                code,
                init if init != "psi" else "rand",
                rand=rand_state if init == "psi" else None,
                do_syndrome=False,
                save_logical=False,
                include_decode=True,
            )
            sv_dec_full = Statevector.from_instruction(qc_dec)
            rho_logical = partial_trace(sv_dec_full, list(range(1, sv_dec_full.num_qubits)))
            sv_dec = rho_logical.to_statevector()
            if init == "psi":
                a, b = rand_state
                label = f"{a}|0> + {b}|1>"
            else:
                label = f"|{init}>"
            print(f"{label} --> encoded")
            dump_ket("", sv_enc, thresh=1e-6)
            print("         --> decoded")
            dump_ket("", sv_dec, thresh=1e-6)
        print("--------------------------------")


if __name__ == "__main__":
    main()


