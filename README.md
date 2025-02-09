
# qmc: A python framework for Hamiltonian Quantum computation
qmc, named after the title of a [foundational paper (pdf)](www.quantum-dynamic.eu/doc/feynman85_qmc_optics_letters.pdf) in quantum computing, is a library for writing, manipulating, and optimizing Hamiltonian quantum computers.
## Installation
The qmc package is available via `pip` and can be installed in your current Python environment with the command:

```
pip install qmc
```

## Getting started
Hamiltonian quantum computers were first invented with the potential application of performing dissipasion-free computation. qmc allows one do perform such
resource estimations. Below is a circuit composed of `NOT` gates which shows that executing that circuit on a Hamiltonian quantum computer dissipates less energy than on a classical computer.  
```python
import cirq
from qmc.qmc import simulate
qubit = cirq.GridQubit(0, 0)

circuit = cirq.Circuit(
        cirq.X(qubit),
        cirq.X(qubit),
        cirq.X(qubit),
        cirq.X(qubit),
        cirq.X(qubit),
    )

result = simulate(circuit, 100)
print(result.output)
# prints
# m=1
print(result.quantum_energy_cost < result.classical_energy_cost)
# prints
# True
```

### Feature requests / Bugs / Questions
If you have questions, feature requests or you found a bug, [please file them on Github](https://github.com/vtomole/qmc/issues).
