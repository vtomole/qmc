# pylint: disable=missing-function-docstring
import cirq
import numpy as np

from qmc.qmc import Result, reversible_evaluate, simulate


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
    assert result.output == cirq.ResultDict(
        params=cirq.ParamResolver({}), records={"m": np.array([[[1]]], dtype=np.dtype("int8"))}
    )


def test_and_not_nand() -> None:
    program = ["a = x and y", "b = not z", "c = not a", "d = b ^ c", "return not d"]
    assert reversible_evaluate(program, [0, 0, 1]) == Result(
        output=0, runtime=5, energy_cost=1.4244174560506876e-20
    )
