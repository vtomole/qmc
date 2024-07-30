from dataclasses import dataclass

import cirq
import networkx as nx
import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.classicalfunction import ClassicalFunction
from qiskit_aer import AerSimulator
from scipy.linalg import expm


@dataclass
class Result:
    """The Result type that stores info about
    the program that's run
    """

    output: cirq.Result
    runtime: int
    energy_cost: float


def generate_psis(n: int) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Generate the first and last counter
    :param n: Size of counter to create
    :return: 2 counters that represent the beginning and end
    """
    psi_a = np.zeros(n)
    psi_a[0] = 1

    psi_x = np.zeros(n)
    psi_x[-1] = 1

    return psi_a, psi_x


def generate_a(n: int) -> npt.NDArray[np.int_]:
    """Generate the adjacency matrix for a line
    :param n: Number of vertices
    :return: Line adjacency matrix
    """
    g = nx.Graph()
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return nx.adjacency_matrix(g).toarray()


def simulate(circuit: cirq.Circuit, total_time: int) -> Result:
    """Run the program on Feynman's simulator
    :param circuit: Program circuit
    :param total_time: Total time to run circuit
    :return: The result
    """
    num_tries = 0
    a = generate_a(len(circuit))
    for t in range(total_time):
        num_tries = num_tries + 1
        u_t = expm(1j * (-t) * a)
        psi_a, psi_x = generate_psis(len(circuit))
        amp = np.transpose(psi_x) @ u_t @ psi_a
        prob = abs(amp) ** 2
        choice = np.random.choice([True, False], p=[prob, 1 - prob])
        if choice:
            simulator = cirq.Simulator()
            circuit.append(cirq.measure(circuit.all_qubits(), key="m"))
            _result = simulator.run(circuit, repetitions=1)
            break

    enegy_cost = (num_tries * (4.11 * 10**-21) * 0.1 * len(circuit)) / t
    return Result(_result, num_tries, enegy_cost)


def input_circuit(input_list):
    circ = QuantumCircuit(4)

    for i, element in enumerate(input_list):
        if element:
            circ.x(i)
        else:
            circ.id(i)
    return circ


def reversible_evaluate(program: list[str], input_param: list[int]):
    expr_string = "def grover_oracle(x: Int1, y: Int1, z: Int1) -> Int1:\n"

    for expr in program:
        expr_string = expr_string + "    " + expr + "\n"

    expr = ClassicalFunction(expr_string)
    simulator = AerSimulator()
    quantum_circuit = expr.synth(registerless=False)
    new_creg = quantum_circuit._create_creg(1, "c")
    quantum_circuit.add_register(new_creg)
    quantum_circuit.measure(quantum_circuit.num_qubits - 1, new_creg)
    quantum_circuit = input_circuit(input_param).compose(quantum_circuit)
    qcirc = transpile(quantum_circuit, simulator)
    result = simulator.run(qcirc, shots=1).result()
    counts = result.get_counts(qcirc)
    q_result = int(list(counts.keys())[0][0])
    energy_cost = len(program) * (4.11 * 10**-21) * np.log(2)
    return Result(q_result, len(program), energy_cost)
