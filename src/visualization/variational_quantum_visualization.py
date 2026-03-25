import pennylane as qml
from pennylane import numpy as np

feature_cols = ["A_scaled", "B_scaled", "C_scaled", "D_scaled", "E_scaled"]
n_qubits = len(feature_cols)

dev = qml.device("default.qubit", wires=n_qubits)

def circuit(weights, x):
    # x: shape (n_qubits,)
    # weights: shape (n_layers, n_qubits)
    qml.AngleEmbedding(x, wires=range(n_qubits))

    n_layers = weights.shape[0]

    # Explicit BasicEntanglerLayers pattern:
    # 1) single-qubit rotations on each qubit
    # 2) ring of CNOTs: 0→1, 1→2, ..., (n_qubits-1)→0
    for l in range(n_layers):
        for w in range(n_qubits):
            qml.RY(weights[l, w], wires=w)
        for w in range(n_qubits - 1):
            qml.CNOT(wires=[w, w + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])  # close the ring

    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="autograd")
def qnode(weights, x):
    return circuit(weights, x)
n_layers = 1
weights_example = np.zeros((n_layers, n_qubits))
x_example = np.zeros(n_qubits)

print(qml.draw(circuit)(weights_example, x_example))

fig, ax = qml.draw_mpl(circuit)(weights_example, x_example)
fig.savefig("vqr_circuit_expanded_5q_1layers.png", dpi=300, bbox_inches="tight")

