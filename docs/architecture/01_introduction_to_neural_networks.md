# Introduction to Neural Networks

## Overview

Neural networks are computational models inspired by the biological neural networks in the human brain. They consist of interconnected nodes (neurons) that process information using connectionist approaches to computation.

```
Input Layer      Hidden Layer(s)     Output Layer
    O                  O                  O
    O ----------->     O  ----------->    O
    O                  O                  O
    O                  O
```

## What is NeuraCommand?

NeuraCommand is a Python library and CLI tool designed for streamlined creation, training, and deployment of neural networks. It provides a numpy-based foundation for building various neural network architectures from scratch.

### Key Features

- **Flexible Network Creation**: Build custom network architectures using the Sequential API
- **Multiple Layer Types**: Dense, Activation, Dropout, BatchNorm, Flatten
- **Various Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU
- **CLI Interface**: Create, train, and predict using command-line tools

## The Building Blocks of Neural Networks

### 1. Neurons (Nodes)

A neuron is the fundamental unit of a neural network. It takes inputs, applies weights, adds a bias, and produces an output through an activation function.

```
                    ┌─────────────────────────────┐
Input x₁ ──w₁──────>│                             │
                    │   z = Σ(wᵢxᵢ) + b           │
Input x₂ ──w₂──────>│                             ├──> Output: f(z)
                    │   f = activation function   │
Input x₃ ──w₃──────>│                             │
                    └─────────────────────────────┘
                              Bias b
```

**Mathematical Representation:**

$$z = \sum_{i=1}^{n} w_i x_i + b$$

$$\text{output} = f(z)$$

Where:
- $x_i$ are the inputs
- $w_i$ are the weights
- $b$ is the bias
- $f$ is the activation function

### 2. Layers

Neurons are organized into layers:

| Layer Type | Description |
|------------|-------------|
| **Input Layer** | Receives the raw input data |
| **Hidden Layers** | Perform intermediate computations |
| **Output Layer** | Produces the final prediction |

### 3. Weights and Biases

- **Weights (w)**: Parameters that determine the strength of connection between neurons
- **Biases (b)**: Allow the activation function to be shifted left or right

## How Neural Networks Learn

Neural networks learn through an iterative process:

1. **Forward Propagation**: Input data flows through the network to produce predictions
2. **Loss Calculation**: The difference between predictions and actual values is measured
3. **Backward Propagation**: Gradients are calculated to determine how to adjust weights
4. **Parameter Update**: Weights and biases are updated to minimize the loss

```
                Forward Pass
    Input ──────────────────────> Output
                                    │
                                    v
                            Calculate Loss
                                    │
               Backward Pass        │
    Gradients <─────────────────────┘
        │
        v
    Update Weights
```

## Your First Neural Network in NeuraCommand

Here's a simple example of creating a neural network using NeuraCommand:

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU, Softmax

# Create a simple neural network
model = Sequential([
    Dense(2, 4),      # Input: 2 features, Output: 4 neurons
    ReLU(),           # Activation function
    Dense(4, 2),      # Hidden layer: 4 inputs, 2 outputs
    Softmax()         # Output activation for classification
])

# Prepare sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# Train the model
model.fit(X, y, epochs=100, learning_rate=0.1)

# Make predictions
predictions = model.predict(X)
print("Predictions:", predictions)
```

### Using the CLI

You can also create networks using the command-line interface:

```bash
# Create a network from JSON architecture
python main.py --action create \
    --network-architecture-json architecture.json \
    --model-pkl model.pkl

# Train the network
python main.py --action train \
    --model-pkl model.pkl \
    --features-csv features.csv \
    --targets-csv targets.csv \
    --epochs 100

# Make predictions
python main.py --action predict \
    --model-pkl model.pkl \
    --input-csv input.csv \
    --output-mode display
```

## Types of Neural Networks

This documentation series will cover progressively more complex architectures:

| Architecture | Use Case | Documentation |
|--------------|----------|---------------|
| Perceptron | Binary classification | [02_perceptron_fundamentals.md](02_perceptron_fundamentals.md) |
| Multi-Layer Perceptron (MLP) | General classification/regression | [06_multilayer_perceptrons.md](06_multilayer_perceptrons.md) |
| Convolutional Neural Networks (CNN) | Image processing | [07_convolutional_neural_networks.md](07_convolutional_neural_networks.md) |
| Recurrent Neural Networks (RNN) | Sequential data | [08_recurrent_neural_networks.md](08_recurrent_neural_networks.md) |
| Transformers | NLP, attention-based tasks | [09_transformer_architecture.md](09_transformer_architecture.md) |

## Prerequisites

Before diving deeper into this documentation, you should have:

1. **Python basics**: Variables, functions, classes
2. **NumPy fundamentals**: Array operations, broadcasting
3. **Linear algebra**: Vectors, matrices, dot products
4. **Calculus basics**: Derivatives, chain rule

## Roadmap

This documentation is organized progressively from simple to complex:

1. **Perceptron Fundamentals** - Understanding single neurons
2. **Activation Functions** - Non-linearity in networks
3. **Gradient Descent** - Optimization basics
4. **Backpropagation** - Learning algorithm
5. **Multi-Layer Perceptrons** - Deep networks
6. **CNNs** - Computer vision
7. **RNNs** - Sequential data
8. **Transformers** - Attention mechanisms
9. **Reinforcement Learning** - Decision-making agents
10. **Regularization** - Preventing overfitting
11. **Practical Tips** - Best practices

## Next Steps

Continue to [02_perceptron_fundamentals.md](02_perceptron_fundamentals.md) to learn about the perceptron, the simplest form of a neural network.
