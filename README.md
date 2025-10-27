# NumPy Neural Networks

A from-scratch implementation of a standard feed-forward Neural Network (NN) and a simple Recurrent Neural Network (RNN) using only NumPy. This project is intended for educational purposes to demonstrate the core mechanics of forward propagation, backpropagation, and modern optimization algorithms.

---

## ✨ Features

This project provides two main classes:

### `neuralNetwork` (Feed-Forward NN)
* Standard multi-layer perceptron (MLP) architecture.
* **Activations:** ReLU for hidden layers and Softmax for the output layer.
* **Initialization:** He initialization for weights.
* **Training:**
    * Standard backpropagation.
    * Early stopping based on loss change (`cutoff`).
    * Divergence detection (stops if loss increases for `diverge_cutoff` epochs).

### `recurrentNeuralNetwork` (Simple RNN)
* Simple (Elman-style) RNN architecture with hidden-to-hidden connections.
* **Activations:** ReLU for hidden states and Softmax for the output layer.
* **Training:**
    * Backpropagation Through Time (BPTT) for sequences.
    * Hidden state is reset at the beginning of each new training sequence.

### Common Features
* **Optimizers:** Includes implementations for:
    * **Lion** (EvoLved Sign Momentum)
    * **AdamW** (Adam with Weight Decay)
* **Regularization:**
    * Weight Decay (implemented within the optimizer update steps).
    * Gradient Clipping (hard-coded to `[-1.0, 1.0]`).
* **Core Logic:** Built purely on `numpy` for all matrix operations.

---

## 📦 Requirements

* Python 3.x
* NumPy

You can install NumPy using pip:
```bash
pip install numpy