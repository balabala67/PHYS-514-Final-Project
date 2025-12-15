"""
Minimal, single-script demo: 3-qubit bit-flip code with depolarizing noise.
Prepares |+>, encodes, applies noise, measures syndrome, applies conditional
recovery, decodes, and reports fidelity of the final logical qubit.
"""

from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def build_noise_model(p: float) -> NoiseModel:
    """Depolarizing noise on 1- and 2-qubit gates."""
    model = NoiseModel()
    err1 = depolarizing_error(p, 1)
    err2 = depolarizing_error(p, 2)
    one_qubit_ops = ["id", "x", "y", "z", "h", "s", "sdg"]
    two_qubit_ops = ["cx"]
    model.add_all_qubit_quantum_error(err1, one_qubit_ops)
    model.add_all_qubit_quantum_error(err2, two_qubit_ops)
    return model


def bitflip_circuit() -> QuantumCircuit:
    q = QuantumRegister(3, "q")
    anc = QuantumRegister(2, "anc")
    syn = ClassicalRegister(2, "syn")
    qc = QuantumCircuit(q, anc, syn, name="bitflip_demo")

    # Prepare |+> on data qubit 0
    qc.h(q[0])

    # Encode |ψ> -> a|000> + b|111|
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])

    # Syndrome extraction for ZZI (s0) and IZZ (s1)
    qc.cx(q[0], anc[0])
    qc.cx(q[1], anc[0])
    qc.cx(q[1], anc[1])
    qc.cx(q[2], anc[1])
    qc.measure(anc, syn)

    # Classical corrections based on syndrome bits (syn[0] is LSB)
    # Map: 10 -> X on q0, 01 -> X on q2, 11 -> X on q1
    def append_conditional(target, mask: int):
        from qiskit.circuit import Instruction

        inst = Instruction(name="x", num_qubits=1, num_clbits=0, params=[])
        inst.condition = (syn, mask)
        qc.append(inst, [target])

    append_conditional(q[0], 0b10)
    append_conditional(q[2], 0b01)
    append_conditional(q[1], 0b11)

    # Decode (inverse of encoding)
    qc.cx(q[0], q[2])
    qc.cx(q[0], q[1])

    # Save logical state
    qc.save_density_matrix([q[0]], label="rho_logical")
    return qc


def run_demo():
    p = 0.05  # depolarizing strength
    noise_model = build_noise_model(p)
    backend = AerSimulator(method="density_matrix")
    qc = bitflip_circuit()
    job = backend.run(qc, noise_model=noise_model, shots=2048, seed_simulator=1234)
    result = job.result()
    data = result.data(0)
    rho = data["rho_logical"]
    if not isinstance(rho, DensityMatrix):
        rho = DensityMatrix(rho)
    target = Statevector.from_label("+")
    fid = state_fidelity(rho, target)
    print(f"Fidelity after correction (p={p} depol): {fid:.4f}")


if __name__ == "__main__":
    run_demo()

