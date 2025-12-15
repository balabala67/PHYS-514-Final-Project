"""
QECC circuits for [[n,1,3]]: baseline, bit-flip, phase-flip, 5-qubit, 7-qubit Steane.
Each builder returns (QuantumCircuit with syndrome+recovery+decode, target_statevector).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli, Statevector


def _prepare_initial(qc: QuantumCircuit, q0, initial: str, rand: Tuple[complex, complex] | None) -> Statevector:
    if initial == "0":
        target = Statevector([1, 0])
    elif initial == "1":
        qc.x(q0)
        target = Statevector([0, 1])
    elif initial == "+":
        qc.h(q0)
        target = Statevector.from_label("+")
    elif initial == "-":
        qc.x(q0)
        qc.h(q0)
        target = Statevector.from_label("-")
    elif initial == "rand":
        if rand is None:
            raise ValueError("rand state requested but no amplitudes provided")
        a, b = rand
        # Map a|0> + b|1> to angles: a = cos(theta/2), b = e^{i phi} sin(theta/2)
        theta = 2 * math.acos(abs(a))
        phi = math.atan2(b.imag, b.real)
        qc.ry(theta, q0)
        qc.rz(phi, q0)
        target = Statevector([a, b])
    else:
        raise ValueError("initial must be 0,1,+,-,rand")
    return target


def _compute_syndrome(stabilizers: List[str], error: str) -> str:
    syn_bits: List[str] = []
    p_err = Pauli(error)
    for stab in stabilizers:
        p_stab = Pauli(stab)
        bit = "0" if p_err.commutes(p_stab) else "1"
        syn_bits.append(bit)
    return "".join(syn_bits)


def _auto_correction_map(stabilizers: List[str]) -> Dict[str, Tuple[str, int]]:
    mapping: Dict[str, Tuple[str, int]] = {}
    n = len(stabilizers[0])
    for q in range(n):
        for p in ("X", "Y", "Z"):
            err = ["I"] * n
            err[q] = p
            syn = _compute_syndrome(stabilizers, "".join(err))
            if syn == "0" * len(stabilizers):
                continue
            mapping.setdefault(syn, (p.lower(), q))
    return mapping


def _measure_stabilizers(qc: QuantumCircuit, data: QuantumRegister, stabilizers: List[str]) -> ClassicalRegister | None:
    if not stabilizers:
        return None
    anc = QuantumRegister(len(stabilizers), "anc")
    syn = ClassicalRegister(len(stabilizers), "syn")
    qc.add_register(anc)
    qc.add_register(syn)
    for i, stab in enumerate(stabilizers):
        a = anc[i]
        for dq, p in zip(data, stab):
            if p == "I":
                continue
            if p == "X":
                qc.h(dq)
            elif p == "Y":
                qc.sdg(dq)
                qc.h(dq)
            qc.cx(dq, a)
            if p == "X":
                qc.h(dq)
            elif p == "Y":
                qc.h(dq)
                qc.s(dq)
    qc.measure(anc, syn)
    return syn


def _apply_corrections(qc: QuantumCircuit, data: QuantumRegister, syn: ClassicalRegister | None, correction_map: Dict[str, Tuple[str, int]]):
    if syn is None:
        return
    for syn_bits, (gate_label, idx) in correction_map.items():
        mask = int(syn_bits, 2)
        gate_fn = getattr(qc, gate_label, None)
        if gate_fn is None:
            continue
        with qc.if_test((syn, mask)):
            gate_fn(data[idx])


# Encoders
def _encoder_bit_flip(data: QuantumRegister) -> QuantumCircuit:
    qc = QuantumCircuit(data, name="enc_bit_flip")
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    return qc


def _encoder_phase_flip(data: QuantumRegister) -> QuantumCircuit:
    qc = QuantumCircuit(data, name="enc_phase_flip")
    qc.h(data)
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    qc.h(data)
    return qc


def _encoder_five_qubit(data: QuantumRegister) -> QuantumCircuit:
    qc = QuantumCircuit(data, name="enc_five_qubit")
    qc.h(data[0])
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    qc.cx(data[1], data[3])
    qc.cx(data[2], data[4])
    qc.t(data[0])
    qc.t(data[1])
    qc.t(data[2])
    qc.t(data[3])
    qc.tdg(data[4])
    qc.cx(data[0], data[1])
    qc.cx(data[2], data[3])
    qc.cx(data[2], data[4])
    qc.h(data[0])
    qc.s(data[1])
    qc.s(data[2])
    qc.s(data[3])
    return qc


def _encoder_steane(data: QuantumRegister) -> QuantumCircuit:
    qc = QuantumCircuit(data, name="enc_steane")
    qc.h(data[0])
    qc.h(data[1])
    qc.h(data[2])
    qc.cx(data[0], data[3])
    qc.cx(data[0], data[4])
    qc.cx(data[1], data[3])
    qc.cx(data[1], data[5])
    qc.cx(data[2], data[4])
    qc.cx(data[2], data[5])
    qc.cx(data[0], data[6])
    qc.cx(data[1], data[6])
    qc.cx(data[2], data[6])
    return qc


def build_code_circuit(code: str, initial: str = "+", rand: Tuple[complex, complex] | None = None) -> Tuple[QuantumCircuit, Statevector]:
    code = code.lower()
    if code == "baseline":
        data = QuantumRegister(1, "q")
        qc = QuantumCircuit(data, name="baseline")
        target = _prepare_initial(qc, data[0], initial, rand)
        qc.save_density_matrix([data[0]], label="rho_logical")
        return qc, target

    if code == "bit_flip":
        stabilizers = ["ZZI", "IZZ"]
        encoder = _encoder_bit_flip
    elif code == "phase_flip":
        stabilizers = ["XXI", "IXX"]
        encoder = _encoder_phase_flip
    elif code == "five_qubit":
        stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
        encoder = _encoder_five_qubit
    elif code == "steane":
        stabilizers = ["XXXIXII", "XIXXIXI", "IXXXIIX", "ZZZIZII", "ZIZZZIZ", "IZZZIIZ"]
        encoder = _encoder_steane
    else:
        raise ValueError("code must be baseline|bit_flip|phase_flip|five_qubit|steane")

    n = len(stabilizers[0])
    data = QuantumRegister(n, "q")
    qc = QuantumCircuit(data, name=code)

    target = _prepare_initial(qc, data[0], initial, rand)

    enc = encoder(data)
    qc.compose(enc, qubits=data, inplace=True)

    syn = _measure_stabilizers(qc, data, stabilizers)
    correction_map = _auto_correction_map(stabilizers)
    _apply_corrections(qc, data, syn, correction_map)

    qc.compose(enc.inverse(), qubits=data, inplace=True)
    qc.save_density_matrix([data[0]], label="rho_logical")
    return qc, target


__all__ = ["build_code_circuit"]


