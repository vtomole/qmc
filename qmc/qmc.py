from dataclasses import dataclass

import cirq
import networkx as nx
import numpy as np
import numpy.typing as npt
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
