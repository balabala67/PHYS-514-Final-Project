"""
Small library of QECC circuits (focus on [[n,1,3]]): bit-flip and phase-flip.
Each builder returns a QuantumCircuit with mid-circuit syndrome measurement,
conditional recovery, and logical decode, plus the target logical state.
"""

from __future__ import annotations

from typing import Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector


def bit_flip_circuit(initial: str = "+") -> Tuple[QuantumCircuit, Statevector]:
    """3-qubit bit-flip code protecting against X errors."""
    q = QuantumRegister(3, "q")
    anc = QuantumRegister(2, "anc")
    syn = ClassicalRegister(2, "syn")
    qc = QuantumCircuit(q, anc, syn, name="bit_flip")

    # Prepare logical qubit
    if initial == "0":
        pass
    elif initial == "1":
        qc.x(q[0])
    elif initial == "+":
        qc.h(q[0])
    elif initial == "-":
        qc.x(q[0])
        qc.h(q[0])
    else:
        raise ValueError("initial must be one of 0,1,+,-")
    target = Statevector.from_label(initial)

    # Encode |ψ> -> a|000> + b|111>
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])

    # Measure ZZI (syn[0]) and IZZ (syn[1])
    qc.cx(q[0], anc[0])
    qc.cx(q[1], anc[0])
    qc.cx(q[1], anc[1])
    qc.cx(q[2], anc[1])
    qc.measure(anc, syn)

    # Conditional X corrections: 10->q0, 01->q2, 11->q1
    from qiskit.circuit.library import XGate

    for mask, idx in [(0b10, 0), (0b01, 2), (0b11, 1)]:
        g = XGate().to_mutable()
        g.condition = (syn, mask)
        qc.append(g, [q[idx]])

    # Decode
    qc.cx(q[0], q[2])
    qc.cx(q[0], q[1])

    # Save logical state
    qc.save_density_matrix([q[0]], label="rho_logical")
    return qc, target


def phase_flip_circuit(initial: str = "+") -> Tuple[QuantumCircuit, Statevector]:
    """3-qubit phase-flip code (bit-flip conjugated by H), protects against Z errors."""
    q = QuantumRegister(3, "q")
    anc = QuantumRegister(2, "anc")
    syn = ClassicalRegister(2, "syn")
    qc = QuantumCircuit(q, anc, syn, name="phase_flip")

    if initial == "0":
        pass
    elif initial == "1":
        qc.x(q[0])
    elif initial == "+":
        qc.h(q[0])
    elif initial == "-":
        qc.x(q[0])
        qc.h(q[0])
    else:
        raise ValueError("initial must be one of 0,1,+,-")
    target = Statevector.from_label(initial)

    # Conjugate bit-flip encoding with H to switch X<->Z
    qc.h(q)
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])
    qc.h(q)

    # To measure XX syndromes, conjugate by H around Z-parity check
    qc.h(q)
    qc.cx(q[0], anc[0])
    qc.cx(q[1], anc[0])
    qc.cx(q[1], anc[1])
    qc.cx(q[2], anc[1])
    qc.h(q)
    qc.measure(anc, syn)

    # Conditional Z corrections (analogous mapping)
    from qiskit.circuit.library import ZGate

    for mask, idx in [(0b10, 0), (0b01, 2), (0b11, 1)]:
        g = ZGate().to_mutable()
        g.condition = (syn, mask)
        qc.append(g, [q[idx]])

    # Decode
    qc.h(q)
    qc.cx(q[0], q[2])
    qc.cx(q[0], q[1])
    qc.h(q)

    qc.save_density_matrix([q[0]], label="rho_logical")
    return qc, target


__all__ = ["bit_flip_circuit", "phase_flip_circuit"]


