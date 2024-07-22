# pylint: disable=missing-function-docstring
import cirq

from qmc.qmc import simulate


def test_feynman_path() -> None:
    qubit = cirq.GridQubit(0, 0)
    # Create a circuit
    circuit = cirq.Circuit(
        cirq.X(qubit),
        cirq.X(qubit),
        cirq.X(qubit),
        cirq.X(qubit),
        cirq.X(qubit),
    )

    result = simulate(circuit, 100)

    print(result)
