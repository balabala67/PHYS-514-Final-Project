"""
Check encode->(optional syndrome skip)->decode round-trip fidelity with no noise.
Uses statevector simulation; expected fidelity ~1 for all codes and initial states.
"""
from __future__ import annotations

from qiskit.quantum_info import Statevector, state_fidelity, partial_trace

from circuits import build_code_circuit


def main():
    codes = ["baseline", "bit_flip", "phase_flip", "five_qubit", "steane"]
    inits = ["0", "1", "+", "-", "rand"]
    for code in codes:
        for init in inits:
            rand = None
            if init == "rand":
                # fixed amplitudes for reproducibility
                rand = (0.6, 0.8j)  # |a|^2 + |b|^2 = 1
            qc, target = build_code_circuit(
                code,
                init,
                rand=rand,
                do_syndrome=False,  # no corrections when no noise
                save_logical=False,  # we only need final statevector
                include_decode=True,
            )
            sv = Statevector.from_instruction(qc)
            # reduce to logical qubit (assumed at index 0 after decode)
            rho = partial_trace(sv, list(range(1, sv.num_qubits)))
            fid = state_fidelity(rho, target)
            print(f"{code:10s} init={init:2s} fidelity={fid:.6f}")


if __name__ == "__main__":
    main()


