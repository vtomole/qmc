from dataclasses import dataclass

import cirq
import networkx as nx
import numpy as np
import numpy.typing as npt
import qiskit
from cirq.contrib.qasm_import import circuit_from_qasm
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.classicalfunction import ClassicalFunction
from scipy.linalg import expm


@dataclass
class Result:
    """The Result type that stores info about
    the program that's run
    """

    output: cirq.Result
    runtime: int
    quantum_energy_cost: float
    classical_energy_cost: float


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

    quantum_enegy_cost = (num_tries * (4.11 * 10**-21) * 0.1 * len(circuit)) / t
    classical_energy_cost = len(circuit) * (4.11 * 10**-21) * np.log(2)
    return Result(_result, num_tries, quantum_enegy_cost, classical_energy_cost)


def input_circuit(input_list: list[int]) -> qiskit.QuantumCircuit:
    """:param input_list: Inputs the data
    :return: Quantum circuit that encodes the input
    """
    circ = QuantumCircuit(4)

    for i, element in enumerate(input_list):
        if element:
            circ.x(i)
        else:
            circ.id(i)
    return circ


def qiskit_circuit(program: list[str], input_param: list[int]) -> qiskit.QuantumCircuit:
    """:param program:  Theprogram
    :param input_param: The input
    :return: The quantum circuit
    """
    expr_string = "def grover_oracle(x: Int1, y: Int1, z: Int1) -> Int1:\n"

    for expr in program:
        expr_string = expr_string + "    " + expr + "\n"

    fun = ClassicalFunction(expr_string)

    quantum_circuit = fun.synth(registerless=False)
    new_creg = quantum_circuit._create_creg(1, "c")
    quantum_circuit.add_register(new_creg)
    quantum_circuit.measure(quantum_circuit.num_qubits - 1, new_creg)
    quantum_circuit = input_circuit(input_param).compose(quantum_circuit)
    return quantum_circuit


def compare(program: list[str], input_list: list[int], total_quantum_time: int = 100) -> Result:
    """:param program: The program
    :param input_list: The input program
    :param total_quantum_time: How much to run the quantum walk
    :return: The result
    """
    the_qiskit_circuit = qiskit_circuit(program, input_list)
    cirq_circuit = circuit_from_qasm(qasm2.dumps(the_qiskit_circuit))
    return simulate(cirq_circuit, total_quantum_time)
