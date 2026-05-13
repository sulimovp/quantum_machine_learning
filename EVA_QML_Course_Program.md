# EVA: Quantum Machine Learning — Course Program

**Instructor:** Pavel Sulimov, ZHAW  
**Format:** 14 classes × (1.5 h lecture + 45 min practice)  
**Tools:** Qiskit, PennyLane, PyTorch, NumPy  
**Prerequisites:** Linear algebra, probability, Python, basics of classical ML

---

## Course Philosophy

This course blends **quantum-inspired classical ML**, **quantum-classical hybrid ML**, **quantum physics-informed approaches (QPINNs)**, and **pure quantum ML** into a unified narrative. It covers architectures driving current research — variational circuits, quantum kernels, quantum reservoir computing, quantum transformers, and QPINNs — while grounding them in the practical constraints of NISQ-era hardware and the Swiss/IBM quantum ecosystem. Students gain both theoretical understanding and hands-on coding skills across Qiskit and PennyLane.

**Pedagogical principles:**
- From Class 2 onward, every practice session should involve either a real-world dataset (Iris, MNIST, stock data, etc.) or a concrete ML task (classification, regression, time-series forecasting), alongside any quantum-specific exercises.
- Lecture demos and practice notebooks must not be identical. If a concept appears in both, the practice version uses different data, parameters, or an extended task.
- Each practice session produces two notebooks: a solutions version and an exercises version with fill-in-the-blanks for students.

**Lecture sequencing note:** Class 7 uses the guest-lecture deck *Advanced QML: Gradients, Cost, and Barren Plateaus* (adapted from UZH). It combines parameter-shift and hybrid training with **barren plateaus**, non-gradient optimizers, and benchmarking—so barren plateaus are **not** reserved for a separate late-semester lecture. Class 13 focuses on **noise, mixed states, and error mitigation**, without repeating the full barren-plateau storyline.

**Inspiration sources:**
- [IBM Quantum Learning — QML Course](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning)
- [Yandex / ODS — QML Course (quantum-ods/qmlcourse)](https://github.com/quantum-ods/qmlcourse/tree/master)
- [YouTube QML Lecture Series](https://www.youtube.com/playlist?list=PLmRxgFnCIhaMgvot-Xuym_hn69lmzIokg)
- [Quantum Machine Learning: A Hands-on Tutorial (arXiv:2502.01146)](https://arxiv.org/abs/2502.01146)
- [PennyLane Demos](https://pennylane.ai/qml/demos/)

---

## Semester Overview

| Block | Classes | Theme |
|-------|---------|-------|
| **I. Foundations** | 1–3 | Math recap, quantum computing basics, quantum gates & circuits |
| **II. Data & Encoding** | 4–5 | Classical ML recap, quantum data encoding, feature maps |
| **III. Variational QML** | 6–9 | Parameterized circuits, training (Class 7: gradients + barren plateaus), hybrid models, kernels |
| **IV. Advanced Topics** | 10–12 | Quantum CNNs, transformers, reservoir computing, QPINNs, generative models |
| **V. Hardware realism & projects** | 13–14 | Mixed states, noise, error mitigation, final projects |

---

## Detailed Class-by-Class Program

---

### CLASS 1 — Introduction to Quantum Machine Learning

**Lecture (1.5 h):**
1. **Course overview & motivation**
   - What is QML? The five-way taxonomy: quantum-inspired, quantum-enhanced (hybrid), quantum-native, quantum physics-informed (QPINN), quantum-informed classical (QINN)
   - Clarify the QINN vs QPINN distinction: QINN uses quantum knowledge (data, constraints, regularization) to improve a *classical* NN; QPINN uses a *quantum circuit* as the function approximator with physics-informed loss
   - Where are we in the NISQ era?
   - Why QML now? Hardware progress, software ecosystem
2. **Classical ML refresher (brief)**
   - Supervised / unsupervised / reinforcement learning
   - The role of linear algebra: vectors, matrices, inner products
   - Loss functions, gradient descent (conceptual)
   - Physics-informed neural networks (PINNs) as precursor to QPINNs
3. **From bits to qubits**
   - Classical bit vs qubit: |0⟩, |1⟩, superposition |ψ⟩ = α|0⟩ + β|1⟩
   - Dirac notation, Bloch sphere visualization
   - Complex amplitudes and probabilities (Born rule)
   - Global vs local (relative) phase
4. **Quantum advantage claims for ML**
   - Speed-up (Grover, HHL — what's realistic?)
   - Expressivity of quantum models
   - Higher-dimensional feature spaces
   - Simpler architectures / fewer parameters

**Key concepts:** Qubit, superposition, amplitude, probability, Bloch sphere, Born rule

**Practice (45 min): "Hello Quantum World"**

*Math tasks:*
- **M1.1:** Given |ψ⟩ = (3/5)|0⟩ + (4i/5)|1⟩, compute P(0), P(1). Verify normalization.
- **M1.2:** Write |+⟩ = (|0⟩ + |1⟩)/√2 and |−⟩ = (|0⟩ − |1⟩)/√2 in column vector form. Verify orthogonality.
- **M1.3:** For |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩, identify θ, φ on the Bloch sphere for given states.

*Programming tasks (Qiskit + PennyLane):*
- **P1.1:** Install Qiskit and PennyLane. Create a single-qubit circuit, apply H gate, measure 1000 shots. Plot histogram.
- **P1.2:** Repeat in PennyLane: create a `default.qubit` device, apply `qml.Hadamard`, return `qml.probs()`. Compare results.
- **P1.3:** Visualize different states on the Bloch sphere using `plot_bloch_vector` (Qiskit) for |0⟩, |1⟩, |+⟩, |−⟩, |i⟩, |−i⟩.

**Resources:**
- [Quantum Soar — Phase explanation](https://www.youtube.com/@quantum-soar)
- [IBM — What is Quantum Computing?](https://www.ibm.com/think/topics/quantum-computing)
- Your slides: QI_2024_Lect_08

---

### CLASS 2 — Quantum Gates, Circuits & Entanglement

**Lecture (1.5 h):**
1. **Single-qubit gates**
   - Pauli gates: X, Y, Z (as rotations on the Bloch sphere)
   - Hadamard H, Phase S, T gates
   - Rotation gates: Rx(θ), Ry(θ), Rz(θ) — parameterized gates (crucial for QML!)
   - Matrix representations and their action on |0⟩, |1⟩
   - Universality of single-qubit gates
2. **Multi-qubit systems**
   - Tensor product: |00⟩, |01⟩, |10⟩, |11⟩
   - The 2^n scaling of Hilbert space
   - Why this matters for ML: exponential feature spaces
3. **Entanglement**
   - Bell states, CNOT gate
   - Entanglement as a resource for QML
   - Self-interference (Hadamard) vs multi-qubit interference (CNOT)
   - No-cloning theorem and implications
4. **Quantum circuits as computational graphs**
   - Circuit diagrams, depth, width
   - Analogy with neural network layers
   - Measurement and collapse

**Key concepts:** Quantum gates, rotation gates, tensor product, entanglement, CNOT, Bell states, circuit depth

**Practice (45 min): "Building Quantum Circuits"**

*Math tasks:*
- **M2.1:** Compute X|0⟩, X|1⟩, H|0⟩, H|1⟩ by matrix multiplication.
- **M2.2:** Show that HXH = Z (matrix multiplication).
- **M2.3:** Compute CNOT(H⊗I)|00⟩ step by step. Verify you get a Bell state (1/√2)(|00⟩ + |11⟩).
- **M2.4:** Prove that the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 cannot be written as a tensor product of two single-qubit states (i.e., it is entangled).

*Programming tasks:*
- **P2.1 (Qiskit):** Build a 2-qubit circuit creating all four Bell states. Run with `StatevectorSampler` (1024 shots). Display results.
- **P2.2 (PennyLane):** Implement a parameterized single-qubit rotation Ry(θ) circuit. Plot ⟨Z⟩ as a function of θ ∈ [0, 2π]. Verify it equals cos(θ).
- **P2.3:** Build a 3-qubit GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2. Measure and verify.
- **P2.4:** Visualize state evolution through a circuit using Qiskit's `Statevector` class: show the statevector after each gate.
- **P2.5 (Taste of QML):** Using scikit-learn's Iris dataset (2 classes, 2 features), encode each data point as Ry(x₁)⊗Ry(x₂) on 2 qubits and compute the resulting statevectors. Visualize how different classes map to different regions of the quantum state space. (This is a preview of quantum data encoding covered in Class 5.)

**Resources:**
- [IBM Quantum Learning — Basics of Quantum Information](https://learning.quantum.ibm.com/)
- Your slides: QI_2024_Lect_08

---

### CLASS 3 — Measurement, Observables & the Quantum Toolbox

**Lecture (1.5 h):**
1. **Measurement in quantum computing**
   - Projective measurement in computational basis
   - Why we need multiple shots — statistics of quantum measurement
   - How many shots do we need? (Hoeffding bound, confidence intervals)
   - Link to [QuantumComputing.SE discussion](https://quantumcomputing.stackexchange.com/questions/14396/)
2. **Observables and expectation values**
   - Pauli observables: I, X, Y, Z and tensor products (ZZ, XY, etc.)
   - Expectation value ⟨ψ|O|ψ⟩ — what it means physically
   - SparsePauliOp in Qiskit
   - Sampler vs Estimator primitives (your Qiskit Primitives guide!)
3. **The quantum computing software stack**
   - Qiskit: QuantumCircuit, transpilation, backends, primitives
   - PennyLane: devices, QNodes, automatic differentiation
   - Simulators vs real hardware
   - StatevectorSampler, StatevectorEstimator, AerSimulator
4. **Complex numbers in quantum & ML**
   - Why quantum naturally uses complex numbers
   - Complex-valued neural networks (classical analogue)
   - Quantum advantage #0: complex number operations for free

**Key concepts:** Measurement, shots, expectation value, Pauli operators, Sampler, Estimator, software stack

**Practice (45 min): "Measurement & Observables"**

*Math tasks:*
- **M3.1:** For |ψ⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩, compute ⟨Z⟩, ⟨X⟩, ⟨Y⟩ analytically.
- **M3.2:** For the Bell state |Φ+⟩, compute ⟨Z⊗Z⟩, ⟨Z⊗I⟩, ⟨X⊗X⟩. Interpret the results.
- **M3.3:** Using the Hoeffding bound, calculate how many shots N are needed to estimate ⟨Z⟩ within ±0.01 with 95% confidence.

*Programming tasks:*
- **P3.1 (Qiskit):** Use `StatevectorSampler` to sample a circuit with measurements. Then use `StatevectorEstimator` on the same circuit (without measurements) with `SparsePauliOp("Z")`. Compare results.
- **P3.2 (PennyLane):** Create a circuit, return `qml.expval(qml.PauliZ(0))`. Vary the rotation angle and plot the expectation value curve. Compare with analytical.
- **P3.3:** Run the same circuit with shots = [10, 100, 1000, 10000]. Plot the convergence of ⟨Z⟩ to the exact value. Compute empirical error bars.
- **P3.4:** Measure ⟨Z₀Z₁⟩ correlation on a Bell state to verify maximal entanglement.
- **P3.5 (Real-data mini task):** Use the Iris dataset (versicolor vs virginica, two features). Build a baseline logistic regression on standardized raw features, then build quantum-derived features from one-qubit angle encoding via observables (⟨X⟩, ⟨Y⟩, ⟨Z⟩). Compare test accuracy and discuss what information phase-sensitive features add.

**Resources:**
- Your note: "A Guide to Qiskit Primitives: Sampler vs. Estimator"
- [Qiskit documentation](https://docs.quantum.ibm.com/)

---

### CLASS 4 — Classical ML Refresher Through a Quantum Lens

**Lecture (1.5 h):**
1. **Supervised learning essentials**
   - Feature vectors, labels, hypothesis class
   - Linear models, logistic regression
   - Loss functions: MSE, cross-entropy
   - Gradient descent and backpropagation
2. **Kernel methods**
   - Feature maps φ(x): mapping data to higher dimensions
   - Kernel trick: K(x, x') = ⟨φ(x), φ(x')⟩
   - SVM with kernels — decision boundaries
   - Why kernels are natural for quantum: quantum states as feature vectors!
3. **Neural networks as function approximators**
   - Universal approximation theorem
   - Layers, activations, depth vs width
   - PyTorch basics: `nn.Module`, autograd, optimizers
4. **The bridge to quantum**
   - Classical NN layer ↔ quantum circuit layer
   - Parameters ↔ gate angles
   - Activation functions ↔ measurement / nonlinearity tricks
   - Feature maps ↔ data encoding circuits

**Key concepts:** Kernel trick, feature maps, gradient descent, SVM, neural network, PyTorch

**Practice (45 min): "Classical Baselines"**

*Math tasks:*
- **M4.1:** For data points {(1,1,+1), (1,-1,-1), (-1,1,-1), (-1,-1,+1)} (XOR problem), show that no linear classifier can separate them. Find a feature map φ(x₁,x₂) = (x₁, x₂, x₁x₂) that makes them linearly separable.
- **M4.2:** Compute the kernel matrix K_{ij} = ⟨φ(xᵢ), φ(xⱼ)⟩ for the above dataset with φ(x₁,x₂) = (x₁, x₂, x₁x₂).

*Programming tasks:*
- **P4.1 (PyTorch):** Implement a simple 2-layer NN for XOR classification. Train it and plot the decision boundary.
- **P4.2 (sklearn + PyTorch):** Train an SVM with RBF kernel on the Iris dataset. Visualize the kernel matrix.
- **P4.3:** Create a `torch.nn.Module` with a custom "quantum-inspired" layer that operates on complex-valued weights. Compare performance with real-valued NN on a simple task.

**Resources:**
- [Complex-valued neural networks](https://medium.com/geekculture/improve-neural-networks-by-using-complex-numbers-5e142b8931e6)
- Your note: Quantum advantage #0 — Complex numbers
- [PyTorch tutorials](https://pytorch.org/tutorials/)

---

### CLASS 5 — Quantum Data Encoding & Feature Maps

**Lecture (1.5 h):**
1. **The data encoding problem**
   - Classical data → quantum states: the bottleneck
   - Why encoding strategy matters for QML performance
   - Data re-uploading: encoding data multiple times
2. **Encoding strategies**
   - **Basis encoding**: x → |x⟩ (binary strings)
   - **Amplitude encoding**: x → amplitudes of |ψ⟩ (exponential compression!)
   - **Angle encoding**: xᵢ → Ry(xᵢ) on qubit i
   - **IQP (Instantaneous Quantum Polynomial) encoding**: with entanglement
   - **Hamiltonian encoding**: time evolution with data-dependent Hamiltonian
3. **Quantum feature maps**
   - Feature map as a unitary: U_φ(x)|0⟩ = |φ(x)⟩
   - ZZ feature map, ZFeatureMap in Qiskit
   - The quantum kernel: K(x, x') = |⟨φ(x)|φ(x')⟩|²
   - Expressivity vs complexity trade-off
4. **Quantum image encoding**
   - FRQI, NEQR — encoding images in quantum states
   - Potential advantages: high-resolution processing
   - Current limitations

**Key concepts:** Basis/amplitude/angle encoding, feature maps, quantum kernel, data re-uploading

**Practice (45 min): "Encoding Classical Data into Quantum States"**

*Math tasks:*
- **M5.1:** Encode the vector x = [0.3, 0.7, 0.5, 0.1] using amplitude encoding. Normalize it and write the quantum state.
- **M5.2:** For angle encoding of x = [π/4, π/3] on 2 qubits using Ry, write the full 2-qubit state |ψ(x)⟩ = Ry(x₁)⊗Ry(x₂)|00⟩. Compute the quantum kernel K(x, x') for x' = [π/6, π/2].
- **M5.3:** Show that the ZZ feature map with entangling layer creates a feature map in a space of dimension 2^n, larger than any polynomial classical kernel.

*Programming tasks:*
- **P5.1 (Qiskit):** Implement basis, angle, and amplitude encoding for a 4-feature dataset. Visualize the resulting statevectors.
- **P5.2 (PennyLane):** Build an `AngleEmbedding` and `AmplitudeEmbedding` template. Encode several data points and compute pairwise fidelities |⟨ψ(x)|ψ(x')⟩|².
- **P5.3:** Implement the ZZFeatureMap from Qiskit. Compute the quantum kernel matrix for the Iris dataset (2 features). Visualize it as a heatmap and compare with a classical RBF kernel.
- **P5.4:** Implement data re-uploading encoding: apply Ry(x₁)Rz(x₂)Ry(x₁)Rz(x₂) on a single qubit. Show it creates a richer feature map than single-pass encoding.

**Resources:**
- [IBM Quantum Learning — Data Encoding](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning)
- [PennyLane — Embedding templates](https://docs.pennylane.ai/en/stable/introduction/templates.html)
- Your slides: QI_2024_Lect_09

---

### CLASS 6 — Parameterized Quantum Circuits & Variational Quantum Algorithms

**Lecture (1.5 h):**
1. **Parameterized Quantum Circuits (PQCs)**
   - Ansatz design: what is a good circuit architecture?
   - Hardware-efficient ansatz vs problem-inspired ansatz
   - EfficientSU2, RealAmplitudes, TwoLocal in Qiskit
   - StronglyEntanglingLayers in PennyLane
   - Circuit expressibility and entangling capability
2. **The variational principle**
   - Variational Quantum Eigensolver (VQE) — the prototypical VQA
   - Cost function design
   - The hybrid quantum-classical optimization loop
   - Classical optimizer choices: COBYLA, SPSA, Adam, L-BFGS
3. **Nonlinearity in quantum circuits**
   - Quantum operations are linear (unitary) — so where does nonlinearity come from?
   - Measurement as nonlinearity
   - Circuit repetitions and mid-circuit measurements
   - Tracing out ancillary qubits
   - Reservoir noise as nonlinearity source
   - Hybrid approach: classical nonlinear activations between quantum layers
4. **Why does this resemble a neural network?**
   - PQC as a quantum neural network (QNN)
   - Depth ↔ number of layers, parameters ↔ weights
   - Entanglement ↔ "skip connections" / correlations

**Key concepts:** Ansatz, PQC, VQE, variational principle, hybrid optimization, expressibility

**Practice (45 min): "Your First Variational Quantum Algorithm"**

*Math tasks:*
- **M6.1:** For a single-qubit PQC: Ry(θ₁)Rz(θ₂)Ry(θ₃)|0⟩, write the output state in terms of θ₁, θ₂, θ₃. How many independent real parameters does a single qubit state have? Is this ansatz over-parameterized?
- **M6.2:** Consider the Hamiltonian H = Z. Find the ground state and ground state energy analytically. Then find θ* such that Ry(θ*)|0⟩ minimizes ⟨ψ(θ)|Z|ψ(θ)⟩.

*Programming tasks:*
- **P6.1 (PennyLane):** Implement a simple VQE to find the ground state energy of H = -Z (trivial) and H = 0.5*X + 0.3*Z (non-trivial). Use `qml.GradientDescentOptimizer`.
- **P6.2 (Qiskit):** Build a 2-qubit ansatz with `EfficientSU2`. Compute ⟨H⟩ for H = Z⊗Z + 0.5*(X⊗I + I⊗X) using `StatevectorEstimator`. Optimize with `scipy.optimize.minimize` (COBYLA).
- **P6.3:** Compare convergence of different optimizers (COBYLA, SPSA, Nelder-Mead) on the same VQE problem. Plot cost vs iteration.
- **P6.4:** Visualize the cost landscape: for a 1-parameter PQC, plot ⟨H⟩(θ) over [0, 2π]. For 2 parameters, plot a 2D surface.

**Resources:**
- [IBM — VQE tutorial](https://learning.quantum.ibm.com/)
- [PennyLane — VQE demo](https://pennylane.ai/qml/demos/tutorial_vqe/)
- Your note: Nonlinearity in quantum circuits
- Your slides: QI_2024_Lect_09

---

### CLASS 7 — Advanced QML: Gradients, Cost, and Barren Plateaus

**Lecture (1.5 h):** Guest-lecture structure (*Advanced QML: Gradients, Cost, and Barren Plateaus*; originally developed for UZH, reused here instead of a gradients-only lecture).

1. **From classical to quantum gradients**
   - Why standard deep-learning backprop scaling does not carry over unchanged
   - Hybrid pipeline: input → (optional) classical layers → PQC → expectation values → loss
   - Chain rule across classical and quantum boundaries
2. **The parameter-shift rule**
   - Gradients for Pauli rotation gates; evaluation cost (two expectations per parameter)
   - Worked hybrid examples (PennyLane + PyTorch / JAX interfaces where applicable)
3. **Barren plateaus** (moved here from an originally separate late-semester plan)
   - Definition and gradient variance vs. qubit count (McClean et al.)
   - Intuition: concentration of measure; global vs. local cost functions
   - Mitigations: local observables, structured initialization, layer-wise training, architecture choices
4. **Beyond gradients: non-gradient and robust optimisation**
   - Derivative-free methods (COBYLA, CMA-ES, Nelder-Mead); Rotosolve-style ideas (outline)
   - When shot noise or plateaus make gradient estimates unreliable
5. **Quantum advantage and honest benchmarking**
   - Where credible advantage claims exist; parameter-count fallacy
   - Practical decision framing: noise, many parameters, plateau detection → strategy choice

**Key concepts:** Parameter-shift rule, hybrid backpropagation, barren plateaus, local vs. global cost, derivative-free optimisation, benchmarking

**Practice (45 min): "Gradients, noise-aware optimisation, and plateaus"**

*Math tasks:*
- **M7.1:** For f(θ) = ⟨0|Ry(θ)†ZRy(θ)|0⟩ = cos(θ), verify the parameter-shift rule: [f(θ+π/2) − f(θ−π/2)] / 2 = −sin(θ) = df/dθ.
- **M7.2:** Hybrid chain rule: classical layer z = wx+b feeding into quantum Ry(z), loss L = (⟨Z⟩ − target)²; compute ∂L/∂w and ∂L/∂b symbolically.
- **M7.3:** For SPSA with perturbation vector Δ=(+1,−1,+1) and δ=0.1, write the gradient estimate for 3 parameters given L⁺ and L⁻.

*Programming tasks:*
- **P7.1 (PennyLane):** Implement parameter-shift gradient manually for a 1-qubit Ry circuit. Compare with PennyLane's `qml.grad()`. Plot both gradient curves.
- **P7.2 (PennyLane + PyTorch):** Hybrid model: `torch.nn.Linear(2,1)` → PennyLane QNode(Ry) → loss. Train end-to-end; verify gradients across the boundary.
- **P7.3:** SPSA gradient estimation from scratch; compare with parameter-shift on a small VQE instance; count circuit evaluations.
- **P7.4:** Benchmark wall-clock: parameter-shift vs SPSA for n = 2, 5, 10, 20 parameters on a simulator.
- **P7.5 (optional):** Gradient variance vs. qubit count for random PQCs (barren-plateau intuition); connect to Lecture 7 slides.

**Resources:**
- Your note: "Hybrid quantum-classical gradient" (detailed scenarios)
- [PennyLane — Gradients](https://pennylane.ai/qml/glossary/quantum_gradient/)
- [Mitarai et al. — Quantum Circuit Learning](https://arxiv.org/abs/1803.00745)
- [McClean et al. — Barren plateaus in quantum neural network training landscapes](https://www.nature.com/articles/s41467-018-07090-4)
- [Cerezo et al. — Cost function dependent barren plateaus](https://www.nature.com/articles/s41467-021-21728-w)
- [PennyLane — Barren Plateaus demo](https://pennylane.ai/qml/demos/tutorial_barren_plateaus/)

---

### CLASS 8 — Quantum Kernel Methods & Classification

**Lecture (1.5 h):**
1. **Quantum kernel methods**
   - Recap: classical kernel K(x,x') = ⟨φ(x), φ(x')⟩
   - Quantum kernel: K_Q(x,x') = |⟨0|U†(x')U(x)|0⟩|²
   - The quantum kernel advantage: implicit access to exponentially large feature spaces
   - Projected quantum kernel
   - Kernel alignment and trainable kernels
2. **Quantum Support Vector Machine (QSVM)**
   - Constructing the quantum kernel matrix
   - Feeding it to a classical SVM
   - When does quantum kernel beat classical?
   - Geometric differences and dataset structure
3. **Quantum classifiers with PQCs**
   - Variational Quantum Classifier (VQC)
   - Architecture: encoding → ansatz → measurement → classical post-processing
   - Binary and multi-class classification
   - Choosing cost functions for classification
4. **QML in higher dimensions**
   - Why quantum works better for high-dimensional data
   - Concentration of kernel values
   - Quantum kernel alignment

**Key concepts:** Quantum kernel, QSVM, VQC, kernel matrix, classification

**Practice (45 min): "Quantum Classification"**

*Math tasks:*
- **M8.1:** For a 1-qubit angle encoding U(x) = Ry(x)|0⟩, compute the quantum kernel K(x,x') = |⟨0|Ry(-x')Ry(x)|0⟩|² = cos²((x−x')/2). Compare with a classical cosine kernel.
- **M8.2:** Show that the ZZ feature map kernel K(x,x') involves terms like cos(xᵢ − x'ᵢ)cos(xⱼ − x'ⱼ) + cross terms, creating a richer similarity measure.

*Programming tasks:*
- **P8.1 (Qiskit):** Compute the quantum kernel matrix for the Iris dataset (2 classes, 2 features) using `ZZFeatureMap` and `FidelityStatevectorKernel`. Train a classical SVM on this kernel. Report accuracy.
- **P8.2 (PennyLane):** Implement a variational quantum classifier on the make_moons dataset. Use `AngleEmbedding` + `StronglyEntanglingLayers` + measurement. Train with Adam optimizer.
- **P8.3:** Compare quantum kernel SVM, classical RBF SVM, and VQC on the same dataset. Plot decision boundaries for each. Create a comparison table.
- **P8.4:** Implement a simple binary classifier using data re-uploading: embed features → rotate → embed → rotate → measure. Show it can solve XOR.

**Resources:**
- [IBM Quantum Learning — Quantum Kernel Methods](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning)
- [Havlíček et al. — Supervised learning with quantum-enhanced feature spaces (Nature 2019)](https://www.nature.com/articles/s41586-019-0980-2)
- [PennyLane — Quantum Kernel demo](https://pennylane.ai/qml/demos/tutorial_kernels_module/)

---

### CLASS 9 — Hybrid Quantum-Classical Neural Networks

**Lecture (1.5 h):**
1. **The hybrid architecture paradigm**
   - Why hybrid? Leverage classical strengths + quantum advantages
   - Classical pre-processing → Quantum layer → Classical post-processing
   - Quantum layers as PyTorch `nn.Module` (via PennyLane/TorchLayer)
   - Transfer learning with quantum circuits
2. **Quantum Neural Network (QNN) architectures**
   - Data-encoding vs weight layers
   - Alternating encoding-ansatz patterns
   - Dressed quantum circuits: classical layers wrapping quantum
   - Multi-qubit output interpretation
3. **Practical hybrid training**
   - End-to-end gradient flow: PyTorch autograd + PennyLane quantum gradients
   - Batch processing: how to handle batches with quantum circuits
   - Training strategies: warm-starting, curriculum learning
   - Distilling classical models into quantum
4. **Performance analysis**
   - When does quantum help? Expressivity vs generalization
   - Quantum model capacity and VC dimension
   - Overfitting in quantum models
   - Regularization techniques

**Key concepts:** Hybrid architecture, TorchLayer, transfer learning, quantum dressed circuits, model capacity

**Practice (45 min): "Building a Hybrid Quantum-Classical Network"**

*Math tasks:*
- **M9.1:** For a hybrid model: x → Linear(2,1) → Ry(z) → ⟨Z⟩ → Linear(1,1) → ŷ, count the total number of trainable parameters (classical + quantum). Write the full forward pass equation.
- **M9.2:** Derive the gradient ∂L/∂w₁ for the hybrid model above, where w₁ is a weight in the first classical layer. Show how the chain rule bridges classical and quantum parts.

*Programming tasks:*
- **P9.1 (PennyLane + PyTorch):** Build a hybrid model using `qml.qnn.TorchLayer`:
  ```
  Classical Linear(4, 2) → PennyLane QNode (2 qubits, 2 layers) → Classical Linear(2, 3)
  ```
  Train on Iris dataset. Compare accuracy with a purely classical NN of similar parameter count.
- **P9.2:** Implement transfer learning: take a pre-trained classical NN, replace the last layer with a quantum circuit layer. Fine-tune on a small dataset. Compare with classical fine-tuning.
- **P9.3:** Train the same hybrid architecture with different numbers of quantum layers (1, 2, 4, 8). Plot training/validation loss. Observe overfitting behavior.
- **P9.4:** Implement a "quantum dropout" by randomly removing entangling gates during training. Compare with standard training.

**Resources:**
- [PennyLane — Turning quantum nodes into Torch layers](https://pennylane.ai/qml/demos/tutorial_qnn_module_torch/)
- [PennyLane — Transfer Learning demo](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning/)
- [Quantum-classical hybrid neural networks (arXiv:2106.06098)](https://arxiv.org/abs/2106.06098)
- Your slides: QI_2024_Lect_09, QI_2024_Lect_10

---

### CLASS 10 — Quantum CNNs, Transformers & Structured Architectures

**Lecture (1.5 h):**
1. **Quantum Convolutional Neural Networks (QCNN)**
   - Architecture: convolutional → pooling → fully connected (quantum analogues)
   - Quantum convolution: local unitary on neighboring qubits
   - Quantum pooling: measure some qubits, conditioned operations on others
   - QCNN for phase classification (Cong et al., Nature Physics 2019)
2. **Quanvolutional Neural Networks**
   - Random quantum circuits as feature extractors
   - Quanvolution: slide a quantum circuit over image patches
   - Output: expectation values as new feature channels
   - Henderson et al. (2020): Quanvolutional Neural Networks
3. **Quantum Transformers**
   - Classical attention: Q, K, V matrices and softmax
   - Quantum self-attention: parameterized circuits for attention weights
   - Quantum Doubly Stochastic Transformer (Born et al., NeurIPS 2025): doubly stochastic attention matrices via quantum circuits
   - Orthogonal attention mechanisms from unitary circuits
   - Open question: when does quantum attention outperform classical?
4. **Shallow entangled circuits for time series**
   - IBM experiments: shallow circuits on real hardware for time series prediction (Nature Sci. Reports 2025)
   - Circuit depth vs accuracy trade-off on NISQ devices
   - Connecting structured quantum architectures to practical applications

**Key concepts:** QCNN, quanvolution, quantum attention, quantum transformer, shallow circuits

**Practice (45 min): "Quantum Structured Architectures"**

*Math tasks:*
- **M10.1:** For a 2-qubit "quantum convolution" filter U(θ₁,θ₂) = CNOT · (Ry(θ₁)⊗Ry(θ₂)), compute the output state when input is |00⟩. What are the expectation values ⟨Z₀⟩, ⟨Z₁⟩?
- **M10.2:** In classical attention, the matrix softmax(QK^T/√d) is row-stochastic. A doubly stochastic matrix has both rows and columns summing to 1. Show that any unitary matrix U satisfies: |U_{ij}|² forms a doubly stochastic matrix. (This is the key insight behind quantum attention.)

*Programming tasks:*
- **P10.1 (PennyLane + PyTorch):** Implement a quanvolutional layer: for each 2×2 patch of an MNIST digit, encode pixel values as rotation angles on 4 qubits, apply a random quantum circuit, measure ⟨Zᵢ⟩ for each qubit. This gives 4 output channels per patch. Feed into a classical CNN. Train and compare with a purely classical CNN.
- **P10.2 (PennyLane):** Build a simple QCNN for classifying 8-qubit quantum states (e.g., distinguish |GHZ⟩ from random product states). Implement quantum convolution and pooling layers.
- **P10.3:** Implement a minimal quantum attention layer: encode 4 data vectors as rotation angles, apply a parameterized unitary, extract |U_{ij}|² as attention weights. Visualize the attention matrix and verify it is doubly stochastic.

**Resources:**
- [PennyLane — Quanvolutional Neural Networks demo](https://pennylane.ai/qml/demos/tutorial_quanvolution/)
- [Cong et al. — QCNN (Nature Physics 2019)](https://www.nature.com/articles/s41567-019-0648-8)
- [Born et al. — Quantum Doubly Stochastic Transformer (NeurIPS 2025)](https://arxiv.org/abs/2504.12345)
- [Shallow entangled circuits for time series on IBM devices (Nature Sci. Reports 2025)](https://www.nature.com/articles/s41598-025-xxxx)

---

### CLASS 11 — Quantum Reservoir Computing & Extreme Learning Machines

**Lecture (1.5 h):**
1. **Reservoir computing paradigm**
   - Classical reservoir computing: fixed random network + trained readout
   - Echo state networks, liquid state machines
   - Why reservoir computing? Training only the readout layer
2. **Quantum reservoir computing (QRC)**
   - Quantum system as a reservoir: rich dynamics from quantum evolution
   - Fixed quantum circuit + classical linear readout
   - Advantages: no barren plateaus (no circuit optimization!), quantum dynamics naturally complex
   - Noise as a feature: noise-induced nonlinearity
   - QRC on real hardware: IBM 156-qubit experiments, scaling analysis
3. **Quantum Extreme Learning Machines (QELM)**
   - Random quantum circuits as feature extractors
   - Measurement statistics as features
   - Why quantum circuits are easier to invert (your note!)
   - Linear regression on measurement outcomes
4. **Time series and financial applications**
   - QRC for chaotic time series prediction
   - Temporal information processing with quantum systems
   - Financial applications: realized volatility forecasting, stock price prediction
   - Multivariate time series forecasting with gate-based QRC on NISQ hardware
   - Comparison with classical RNNs/LSTMs
5. **The quantum reservoir kernel (QuaRK)**
   - Combining reservoir computing with kernel methods
   - Quantum reservoir features as implicit kernel
   - Recent results: QuaRK for time series learning (2025)

**Key concepts:** Reservoir computing, QRC, QELM, readout layer, noise-induced nonlinearity, quantum reservoir kernel

**Practice (45 min): "Quantum Reservoir Computing"**

*Math tasks:*
- **M11.1:** For a single-qubit reservoir: input x → Rx(x) applied to the current state, measure ⟨X⟩, ⟨Y⟩, ⟨Z⟩ → features. For a time series [x₁, x₂, ..., xₜ], describe the sequential encoding process. How does the reservoir "remember" past inputs?
- **M11.2:** In a QELM with 3 qubits and measurements of all Pauli strings up to weight 2 (I, X, Y, Z, XX, XY, ...), how many features do you obtain?

*Programming tasks:*
- **P11.1 (PennyLane):** Implement a quantum reservoir: fixed random circuit on 4 qubits, encode time series values as Ry rotations, measure all single-qubit Pauli expectations. Train a linear readout on the Mackey-Glass time series. Compare with a classical echo state network.
- **P11.2:** Implement a QELM for the Iris dataset: random 4-qubit circuit with angle encoding, measure all 2-qubit Pauli expectations (15 features), train a ridge regression classifier.
- **P11.3:** Financial time series: apply the quantum reservoir to a simple stock return prediction task. Compare with a classical LSTM baseline. Analyze: does noise on the quantum reservoir help or hurt?
- **P11.4:** Study the effect of reservoir size: vary the number of qubits (2, 4, 6, 8) and the circuit depth. Plot prediction accuracy vs reservoir dimension.

**Resources:**
- [IBM — Quantum noise-induced reservoir computing (2023)](https://research.ibm.com/publications/optimizing-quantum-noise-induced-reservoir-computing-for-nonlinear-and-chaotic-time-series-prediction)
- [QuaRK: Quantum Reservoir Kernel for Time Series (arXiv:2602.13531)](https://arxiv.org/abs/2602.13531)
- [Quantum reservoir computing for realized volatility forecasting (2024)](https://scholar.google.com/scholar?q=quantum+reservoir+computing+realized+volatility)
- [QRC scaling on 156-qubit IBM hardware (arXiv:2508.xxx)](https://arxiv.org/search/?query=%22quantum+reservoir+computing%22+IBM+hardware)
- Your note: reservoir noise as nonlinearity source

---

### CLASS 12 — Generative Models, Optimization & Quantum Physics-Informed NNs

**Lecture (1.5 h):**
1. **Quantum Approximate Optimization Algorithm (QAOA)**
   - Combinatorial optimization problems (MaxCut, TSP, portfolio optimization)
   - Problem Hamiltonian and mixer Hamiltonian
   - QAOA circuit: alternating problem/mixer unitaries
   - QAOA as a variational algorithm
   - Applications to ML: clustering, feature selection
2. **Quantum generative models**
   - Quantum Born Machine: sampling from |ψ(θ)|² distribution
   - Quantum Circuit Born Machine (QCBM)
   - Quantum GANs: quantum generator vs classical discriminator
   - Physics-informed generative ML for quantum-centric supercomputing (IBM, arXiv:2512.06858)
3. **Quantum Physics-Informed Neural Networks (QPINNs)**
   - Recap: classical PINNs encode PDE residuals in the loss
   - QPINNs: replace the classical NN with a parameterized quantum circuit
   - The loss function: L = L_data + lambda * L_PDE, where L_PDE enforces the differential equation
   - Applications: Schrodinger equation, Maxwell's equations, fluid dynamics
   - Challenges: barren plateaus in QPINN training, circuit depth requirements
   - Recent results: hybrid QPINNs for high-speed flow, GPU-accelerated QPINN training
4. **Quantum utility and practical quantum advantage**
   - IBM quantum utility experiments: 127-qubit Ising simulation beyond classical brute force (Kim et al., Nature 2023)
   - What "utility" means vs "supremacy" vs "advantage"
   - Error mitigation as enabler of utility-scale computation
   - Where QML fits: the road from utility to practical advantage

**Key concepts:** QAOA, Born machine, quantum GAN, QPINN, quantum utility, error mitigation

**Practice (45 min): "Generative, Optimization & Physics-Informed Tasks"**

*Math tasks:*
- **M12.1:** For the MaxCut problem on a triangle graph (3 nodes, 3 edges), write the problem Hamiltonian C = Σ (1 - ZᵢZⱼ)/2. What is the maximum cut value? Find the optimal bitstrings.
- **M12.2:** For a QPINN solving the 1D heat equation du/dt = k * d²u/dx², write the PDE residual loss term. How would you evaluate d²u/dx² if u(x) is the output of a parameterized quantum circuit?

*Programming tasks:*
- **P12.1 (Qiskit):** Implement QAOA for MaxCut on a 4-node random graph. Use p=1 and p=2 QAOA layers. Compare the approximation ratio.
- **P12.2 (PennyLane):** Implement a Quantum Circuit Born Machine: train a PQC to reproduce a given target distribution (e.g., a discretized Gaussian). Use KL divergence as loss.
- **P12.3 (PennyLane):** Implement a minimal QPINN: use a parameterized quantum circuit to approximate the solution of du/dx = -u (exponential decay). The loss is purely physics-informed — no training data. Compare the learned solution with the analytical u(x) = e^{-x}.
- **P12.4:** Implement a simple quantum GAN: quantum generator (PQC producing a distribution) + classical discriminator (small NN). Train adversarially on a 1D distribution.

**Resources:**
- [PennyLane — QAOA demo](https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut/)
- [PennyLane — Quantum GANs demo](https://pennylane.ai/qml/demos/tutorial_quantum_gans/)
- [Kyriienko et al. — Solving nonlinear differential equations with differentiable quantum circuits (Phys. Rev. A 2021)](https://arxiv.org/abs/2011.10395)
- [Kim et al. — Evidence for the utility of quantum computing before fault tolerance (Nature 2023)](https://www.nature.com/articles/s41586-023-06096-3)
- [Physics-Informed Generative ML for Quantum-centric Supercomputing (arXiv:2512.06858)](https://arxiv.org/abs/2512.06858)
- Your slides: QI_2024_Lect_10

---

### CLASS 13 — Noise, Mixed States, and Error Mitigation

**Lecture (1.5 h):** Noise, mixed states, and error mitigation (barren plateaus and trainability: **Class 7**).

1. **Why pure statevectors are not enough on hardware**
   - Simulators vs. devices; statistical and coherent errors
2. **Density matrices**
   - Pure states: ρ = |ψ⟩⟨ψ|; classical mixtures
   - Expectation values: ⟨O⟩ = Tr(ρ O)
3. **Quantum channels (CPTP maps) and simple noise models**
   - Kraus form; depolarizing, dephasing, amplitude damping (qualitative effect on observables)
4. **Consequences for QML**
   - Measured features under noise: bias, attenuation, and shot variance
5. **Error mitigation (toolkit level)**
   - Measurement error mitigation (calibration / confusion matrix)
   - Zero-noise extrapolation (ZNE); pointers to probabilistic cancellation and twirling as extensions
6. **Forward pointer**
   - Class 14: projects and research outlook

**Key concepts:** Density matrix, quantum channel, noise models, error mitigation, NISQ measurement statistics

**Practice (45 min): "Noise models and mitigation hooks"**

*Math tasks:*
- **M13.1:** For a one-qubit state |ψ⟩ = R_y(θ)|0⟩, write ρ = |ψ⟩⟨ψ| and compute Tr(ρ Z) analytically.
- **M13.2:** Depolarizing channel: ρ' = (1−p)ρ + p I/2. Show how ⟨Z⟩ scales with p for a given ρ.

*Programming tasks:*
- **P13.1 (Qiskit Aer or similar):** Run a small circuit with a noise model; compare ⟨Z⟩ or ⟨ZZ⟩ with the noiseless simulator.
- **P13.2:** Implement a minimal ZNE workflow: execute at scaled noise levels, extrapolate to zero; compare to noiseless reference on a toy observable.
- **P13.3 (optional):** Measurement error mitigation on a few-qubit line: apply a calibration matrix to observed bitstring probabilities.

**Resources:**
- [Qiskit — Building noise models](https://docs.quantum.ibm.com/guides/build-noise-models) (check current API for your Qiskit version)
- [Temme et al. — Error mitigation for short-depth quantum circuits](https://arxiv.org/abs/1612.02058) (ZNE context)
- Barren-plateau references remain relevant for **Class 7**; see Class 7 resources above.

---

### CLASS 14 — Final Projects & Frontiers of QML

**Lecture (1.5 h):**
1. **Student project presentations** (30-40 min)
   - Each team presents their project (5-7 min each)
   - Q&A and discussion
2. **Frontiers of QML** (30 min)
   - Quantum utility experiments and what they mean for QML (Kim et al., Nature 2023; IBM roadmap)
   - Quantum transformers and attention mechanisms (Born et al., NeurIPS 2025)
   - QPINNs for scientific computing: Schrodinger, Maxwell, Navier-Stokes
   - Quantum reservoir computing at scale: recent experiments on 100+ qubits
   - Quantum machine learning for quantum data (quantum chemistry, materials science)
   - Fault-tolerant QML: what changes with error correction?
3. **Quantum-inspired classical algorithms** (20 min)
   - Dequantization results (Tang, 2019)
   - Tensor network methods for ML
   - Classical simulation of quantum circuits — the moving target
   - What this means for QML advantage claims
4. **The Swiss quantum ecosystem & industry outlook** (10 min)
   - QuantumBasel: industry hub, D-Wave partnership, applied QML
   - QC2 (University of Basel + QuantumBasel): QML research, hardware-agnostic approaches
   - ZHAW quantum activities
   - Career paths in QML: academia, startups, industry
5. **Course recap** (10 min)
   - Summary of quantum advantages in ML: what's proven, what's promising, what's hype
   - When to use quantum: a decision framework
   - Open research problems

**Final Project Options (assigned in Class 10, due in Class 14):**

Students choose one of the following projects (teams of 2-3):

1. **Quantum Kernel Comparison Study:** Compare quantum kernels (ZZ, ZFeature, custom) with classical kernels (RBF, polynomial) on 3+ real-world datasets. Analyze when quantum helps and why.

2. **Hybrid Quantum-Classical Image Classifier:** Build and train a quanvolutional + classical CNN pipeline for Fashion-MNIST. Compare with pure CNN. Analyze quantum feature maps.

3. **Quantum Time Series Forecasting:** Implement quantum reservoir computing or a variational QNN for a real-world time series (financial data, weather, sensor data). Compare with LSTM baseline.

4. **QAOA for Portfolio Optimization:** Formulate portfolio optimization as QUBO, implement QAOA in Qiskit/PennyLane. Compare with classical solvers for small instances.

5. **Barren Plateau Analysis:** Systematically study barren plateaus across different ansatz architectures, initialization strategies, and cost functions. Propose and test a mitigation strategy.

6. **Quantum Physics-Informed NN:** Implement a QPINN for a simple ODE or PDE (e.g., harmonic oscillator, 1D diffusion equation). Compare convergence with a classical PINN of similar parameter count.

7. **Custom Project:** Students propose their own QML project (must be approved by instructor).

**Deliverables:**
- Jupyter notebook with code, results, and analysis
- 5-7 min presentation
- 2-page report summarizing findings

---

## Assessment

Final project presentation (pass/fail).

---

## AI Tools Policy

Using AI coding assistants (ChatGPT, Claude, Cursor, GitHub Copilot, etc.) is welcome in this course. Quantum computing has a steep learning curve, and these tools can accelerate your understanding if used critically.

**Why caution is needed.** Qiskit and PennyLane evolve rapidly. Large language models are typically trained on data that is 6--18 months old, which means they often generate code for deprecated APIs. Specific pitfalls:

- **Qiskit 0.x/1.x vs 2.x:** The execution model changed dramatically across major versions. Functions like `execute(circuit, backend)`, the `BasicAer` and `Aer.get_backend()` interfaces, and the `qiskit.opflow` module were removed in Qiskit 1.0 (January 2024). Qiskit 2.0 (2025) further cleaned up the API. AI tools frequently generate pre-1.0 code. The current API uses *primitives*: `StatevectorSampler`, `StatevectorEstimator`.
- **PennyLane:** The device and gradient APIs evolve across minor versions. Always check that the syntax matches the version you have installed.
- **Conceptual errors:** AI models sometimes produce plausible-looking but physically wrong quantum circuits or incorrect mathematical derivations. Quantum computing is counterintuitive enough that errors are not always obvious.

**Practical rules:**

1. **State your versions** when prompting an AI: "I am using Qiskit 2.x and PennyLane 0.40+."
2. **Run every generated snippet.** If it throws a deprecation warning or error, do not patch blindly; check the official migration guides.
3. **Cross-reference with official documentation:**
   - Qiskit: [docs.quantum.ibm.com](https://docs.quantum.ibm.com/)
   - PennyLane: [docs.pennylane.ai](https://docs.pennylane.ai/)
4. **If using Cursor or similar IDE assistants,** set up project-level rules (`.cursor/rules/`) that pin framework versions and preferred API patterns. This dramatically improves code generation quality.
5. **Cite your tools.** If an AI helped you write a substantial part of an assignment, mention it briefly. This is not penalized; hiding it is.

---

## Recommended Textbooks & Resources

### Primary
- **Schuld, M. & Petruccione, F.** *Machine Learning with Quantum Computers* (Springer, 2021)
- **Hidary, J.D.** *Quantum Computing: An Applied Approach* (Springer, 2021)

### Online Courses & Tutorials
- [IBM Quantum Learning — QML Course](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning)
- [PennyLane QML Demos](https://pennylane.ai/qml/demos/)
- [Qiskit Textbook — QML Chapter](https://github.com/Qiskit/textbook/tree/main/notebooks/quantum-machine-learning)
- [Yandex ODS QML Course](https://github.com/quantum-ods/qmlcourse/tree/master)
- [YouTube QML Lectures](https://www.youtube.com/playlist?list=PLmRxgFnCIhaMgvot-Xuym_hn69lmzIokg)

### Key Papers
- Havlíček et al. *Supervised learning with quantum-enhanced feature spaces* (Nature, 2019)
- McClean et al. *Barren plateaus in quantum neural network training landscapes* (Nat. Commun., 2018)
- Cerezo et al. *Variational quantum algorithms* (Nat. Rev. Phys., 2021)
- Mitarai et al. *Quantum circuit learning* (Phys. Rev. A, 2018)
- Schuld et al. *Effect of data encoding on the expressive power of variational quantum ML models* (Phys. Rev. A, 2021)
- Kübler et al. *The inductive bias of quantum kernels* (NeurIPS, 2021)
- Kim et al. *Evidence for the utility of quantum computing before fault tolerance* (Nature, 2023)
- Born et al. *Quantum Doubly Stochastic Transformer* (NeurIPS, 2025)
- Kyriienko et al. *Solving nonlinear differential equations with differentiable quantum circuits* (Phys. Rev. A, 2021)
- Agliardi et al. *Mitigating exponential concentration in covariant quantum kernels* (npj Quantum Inf., 2025)

### Software Documentation
- [Qiskit Documentation](https://docs.quantum.ibm.com/)
- [PennyLane Documentation](https://docs.pennylane.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## Software Setup

```bash
# Create conda environment
conda create -n qml python=3.11 -y
conda activate qml

# Core packages (pinned ranges for reproducibility)
pip install "qiskit>=2.0,<3" "qiskit-aer>=0.15" "qiskit-machine-learning>=0.8"
pip install "pennylane>=0.40" "pennylane-qiskit>=0.40"
pip install "torch>=2.5" torchvision
pip install scikit-learn matplotlib seaborn jupyter

# Optional
pip install qiskit-optimization  # for QAOA (Class 12)
```

---

## Week-by-Week Schedule Summary

| Week | Class | Topic | Key Practice |
|------|-------|-------|-------------|
| 1 | 1 | Intro to QML, qubits, Bloch sphere | Hello Quantum World (Qiskit + PennyLane) |
| 2 | 2 | Gates, circuits, entanglement | Bell states, parameterized circuits |
| 3 | 3 | Measurement, observables, toolbox | Sampler vs Estimator, expectation values, Iris mini task |
| 4 | 4 | Classical ML refresher (quantum lens) | PyTorch NN, SVM kernels, XOR |
| 5 | 5 | Data encoding & feature maps | Encoding strategies, kernel matrices |
| 6 | 6 | PQCs & variational algorithms | VQE implementation |
| 7 | 7 | Gradients, barren plateaus & optimisation strategy | Parameter-shift, hybrid backprop, optional BP demo |
| 8 | 8 | Quantum kernels & classification | QSVM, VQC, decision boundaries |
| 9 | 9 | Hybrid quantum-classical NNs | TorchLayer, transfer learning |
| 10 | 10 | Quantum CNNs, transformers & structured arch. | Quanvolution, QCNN, quantum attention |
| 11 | 11 | Quantum reservoir computing | QRC, QELM, time series, finance |
| 12 | 12 | Generative models, QAOA & QPINNs | QAOA, Born machines, QPINN for ODEs |
| 13 | 13 | Mixed states, noise & error mitigation | Noise models, ZNE, measurement mitigation |
| 14 | 14 | Final projects & frontiers | Presentations, course recap |
