# EVA: Quantum Machine Learning

This repository supports the EVA course in Quantum Machine Learning at ZHAW.
It contains lecture materials, practice notebooks, and code examples used during
the semester.

## Course Format

- 14 classes
- Each class: 1.5 h lecture + 45 min practice
- Main tools: Qiskit, PennyLane, PyTorch, NumPy

## Prerequisites

- Linear algebra
- Probability and statistics
- Python programming
- Basic classical machine learning

## Learning Outcomes

By the end of the course, you should be able to:

- explain core quantum computing concepts used in QML
- build and run quantum circuits in Qiskit and PennyLane
- implement and evaluate hybrid quantum-classical ML models
- compare quantum kernels and variational approaches with classical baselines
- discuss practical limits of current QML methods with technical rigor

## Semester Roadmap

| Block | Classes | Focus |
|---|---|---|
| I. Foundations | 1-3 | Qubits, gates, measurement, observables, software stack |
| II. Data and Encoding | 4-5 | Classical ML recap, feature maps, quantum data encoding |
| III. Variational QML | 6-9 | PQCs, training, gradients, kernels, hybrid models |
| IV. Advanced Topics | 10-12 | QCNNs, quantum transformers, reservoir computing, QPINNs |
| V. Frontiers and Projects | 13-14 | Error mitigation, barren plateaus, final projects |

## Class-by-Class Plan

| Class | Topic | Practice Focus |
|---|---|---|
| 1 | Introduction to QML, qubits, Bloch sphere | First circuits in Qiskit and PennyLane |
| 2 | Gates, circuits, entanglement | Bell/GHZ states and state evolution |
| 3 | Measurement, observables, primitives | Sampler vs Estimator, expectation values |
| 4 | Classical ML through a quantum lens | XOR, SVM kernels, PyTorch baseline |
| 5 | Quantum encoding and feature maps | Basis/angle/amplitude encoding, kernels |
| 6 | PQCs and variational algorithms | VQE-style circuits and optimization |
| 7 | Training quantum circuits | Parameter-shift gradients, hybrid training |
| 8 | Quantum kernels and classification | QSVM and decision boundaries |
| 9 | Hybrid quantum-classical neural networks | Torch integration and model comparison |
| 10 | Structured QML architectures | QCNN, quanvolution, quantum attention ideas |
| 11 | Quantum reservoir and extreme learning | Time-series and financial examples |
| 12 | Generative models, QAOA, QPINNs | Optimization and physics-informed workflows |
| 13 | Training pathologies and mitigation | Barren plateaus, initialization, noise mitigation |
| 14 | Final projects and outlook | Student presentations and synthesis |

## Assessment

- Final project presentation (pass/fail)

## Repository Structure

- `lectures/` - lecture slides and plans
- `notebooks/` - practice notebooks and coding sessions
- `src/` - reusable Python code (as the project grows)
- `experiments/` - reproducible experiment scripts
- `tests/` - unit tests

## Software Setup

```bash
conda create -n qml python=3.11 -y
conda activate qml

pip install "qiskit>=2.0,<3" "qiskit-aer>=0.15" "qiskit-machine-learning>=0.8"
pip install "pennylane>=0.40" "pennylane-qiskit>=0.40"
pip install "torch>=2.5" torchvision
pip install scikit-learn matplotlib seaborn jupyter
```

Optional:

```bash
pip install qiskit-optimization
```

## Recommended Resources

- IBM Quantum Learning: https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning
- PennyLane QML Demos: https://pennylane.ai/qml/demos/
- Qiskit Documentation: https://docs.quantum.ibm.com/
- PennyLane Documentation: https://docs.pennylane.ai/
- PyTorch Documentation: https://pytorch.org/docs/

## Working Policy

- Run all notebook code before drawing conclusions.
- Report numerical claims only with runnable code.
- Keep work reproducible and clearly documented.
