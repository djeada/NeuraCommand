# Activation Functions

## Introduction

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Without activation functions, no matter how many layers a network has, it would behave like a single linear transformation.

## Why Activation Functions Matter

Consider a network without activation functions:

$$\text{Layer 1: } z_1 = W_1 x + b_1$$
$$\text{Layer 2: } z_2 = W_2 z_1 + b_2 = W_2(W_1 x + b_1) + b_2$$
$$= W_2 W_1 x + W_2 b_1 + b_2 = W' x + b'$$

This collapses to a single linear transformation! Activation functions break this linearity.

## Activation Functions in NeuraCommand

NeuraCommand provides several activation functions:

```python
from neura_command.layers.activation import (
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
    ELU
)

# Using as standalone layers
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense

model = Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
])
```

## Common Activation Functions

### 1. Sigmoid (Logistic)

The sigmoid function squashes values between 0 and 1.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Derivative:**
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

```
Output
  1.0 │            ──────────
      │         ──'
      │       ─'
  0.5 │     ─'
      │   ─'
      │  ─'
  0.0 │──────────────────────
      └─────────┼─────────→ z
              0
```

**Properties:**
| Pros | Cons |
|------|------|
| Smooth gradient | Vanishing gradient for large \|z\| |
| Output bounded (0,1) | Not zero-centered |
| Interpretable as probability | Computationally expensive |

**NeuraCommand Usage:**
```python
from neura_command.layers.activation import Sigmoid
from neura_command.network_utils.activation_functions import SigmoidActivation

# As a layer
layer = Sigmoid()

# Direct computation
import numpy as np
activation = SigmoidActivation()
z = np.array([-2, -1, 0, 1, 2])
output = activation.compute(z)
# output: [0.119, 0.269, 0.5, 0.731, 0.881]
```

### 2. Hyperbolic Tangent (Tanh)

Similar to sigmoid but outputs values between -1 and 1.

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Derivative:**
$$\tanh'(z) = 1 - \tanh^2(z)$$

```
Output
  1.0 │            ──────────
      │         ──'
      │       ─'
  0.0 │─────────────────────
      │       .
      │   . '
 -1.0 │───────'──────────────
      └─────────┼─────────→ z
              0
```

**Properties:**
| Pros | Cons |
|------|------|
| Zero-centered output | Vanishing gradient for large \|z\| |
| Stronger gradients than sigmoid | Still computationally expensive |

**NeuraCommand Usage:**
```python
from neura_command.layers.activation import Tanh

model = Sequential([
    Dense(10, 20),
    Tanh(),
    Dense(20, 10)
])
```

### 3. ReLU (Rectified Linear Unit)

The most popular activation function in modern deep learning.

$$\text{ReLU}(z) = \max(0, z)$$

**Derivative:**
$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

```
Output
      │              /
      │            /
      │          /
      │        /
  0.0 │──────/──────────────
      └─────────┼─────────→ z
              0
```

**Properties:**
| Pros | Cons |
|------|------|
| Computationally efficient | "Dying ReLU" problem |
| Sparse activation | Not zero-centered |
| Reduces vanishing gradient | Unbounded output |

**NeuraCommand Usage:**
```python
from neura_command.layers.activation import ReLU

model = Sequential([
    Dense(784, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 10)
])
```

### 4. Leaky ReLU

Addresses the "dying ReLU" problem by allowing small negative values.

$$\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

Where α is typically 0.01.

```
Output
      │              /
      │            /
      │          /
      │        /
  0.0 │──────/──────────────
      │    /'
      │  /'  (small slope α)
      │/'
      └─────────┼─────────→ z
              0
```

**NeuraCommand Usage:**
```python
from neura_command.layers.activation import LeakyReLU

model = Sequential([
    Dense(784, 256),
    LeakyReLU(alpha=0.01),  # Specify the negative slope
    Dense(256, 10)
])
```

### 5. ELU (Exponential Linear Unit)

Combines benefits of ReLU with smoother negative values.

$$\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$

**NeuraCommand Usage:**
```python
from neura_command.layers.activation import ELU

model = Sequential([
    Dense(784, 256),
    ELU(alpha=1.0),
    Dense(256, 10)
])
```

### 6. Softmax

Used for multi-class classification; converts logits to probabilities that sum to 1.

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Example:**
```
Input logits: [2.0, 1.0, 0.1]

e^2.0 = 7.389
e^1.0 = 2.718
e^0.1 = 1.105

Sum = 11.212

Softmax output: [0.659, 0.242, 0.099]
               (sums to 1.0)
```

**NeuraCommand Usage:**
```python
from neura_command.layers.activation import Softmax

# Classification network
model = Sequential([
    Dense(784, 256),
    ReLU(),
    Dense(256, 10),
    Softmax()  # Output layer for 10-class classification
])
```

## Comparison Chart

| Activation | Range | Zero-Centered | Computation | Best For |
|------------|-------|---------------|-------------|----------|
| Sigmoid | (0, 1) | No | Expensive | Binary output, gates |
| Tanh | (-1, 1) | Yes | Expensive | RNNs, hidden layers |
| ReLU | [0, ∞) | No | Fast | Hidden layers (default) |
| Leaky ReLU | (-∞, ∞) | No | Fast | When ReLU dies |
| ELU | (-α, ∞) | ~Yes | Moderate | Better than ReLU |
| Softmax | (0, 1) | No | Moderate | Multi-class output |

## Hand Calculation Example

Let's compute activations for input z = [-2, -1, 0, 1, 2]:

### Sigmoid:
$$\sigma(-2) = \frac{1}{1+e^2} = \frac{1}{8.389} = 0.119$$
$$\sigma(0) = \frac{1}{1+e^0} = \frac{1}{2} = 0.5$$
$$\sigma(2) = \frac{1}{1+e^{-2}} = \frac{1}{1.135} = 0.881$$

### ReLU:
$$\text{ReLU}(-2) = \max(0, -2) = 0$$
$$\text{ReLU}(0) = \max(0, 0) = 0$$
$$\text{ReLU}(2) = \max(0, 2) = 2$$

### Tanh:
$$\tanh(0) = \frac{e^0 - e^0}{e^0 + e^0} = \frac{0}{2} = 0$$
$$\tanh(1) = \frac{e^1 - e^{-1}}{e^1 + e^{-1}} = \frac{2.718 - 0.368}{2.718 + 0.368} = 0.762$$

## Derivatives for Backpropagation

Understanding derivatives is crucial for training:

| Activation | Derivative |
|------------|------------|
| Sigmoid | $\sigma(z)(1-\sigma(z))$ |
| Tanh | $1 - \tanh^2(z)$ |
| ReLU | 1 if z > 0, else 0 |
| Leaky ReLU | 1 if z > 0, else α |
| ELU | 1 if z > 0, else $\alpha e^z$ |

## Choosing the Right Activation

```
                      Start
                        │
                        ▼
               ┌─────────────────┐
               │ Output Layer?   │
               └────────┬────────┘
                   │    │
              Yes  │    │ No
                   ▼    ▼
           ┌────────┐  ┌────────────┐
           │Classification│ │ Hidden Layer │
           │or Regression?│ └──────┬───────┘
           └────┬───┘              │
      Binary    │    Multi-class   │
         │      │         │        ▼
         ▼      │         ▼    ┌───────┐
    ┌────────┐  │   ┌─────────┐│ ReLU  │
    │Sigmoid │  │   │ Softmax ││ First │
    └────────┘  │   └─────────┘└───┬───┘
                │                  │
         Regression                │ Issues?
                │                  │
                ▼             Yes  │
           ┌────────┐              ▼
           │Linear  │    ┌─────────────────┐
           │(None)  │    │ Try LeakyReLU   │
           └────────┘    │ or ELU          │
                         └─────────────────┘
```

## Practical Example: Comparing Activations

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU, Sigmoid, Tanh

# Create sample data
np.random.seed(42)
X = np.random.randn(100, 10)
y = np.random.randn(100, 1)

# Test different activations
activations = {
    'ReLU': ReLU(),
    'Sigmoid': Sigmoid(),
    'Tanh': Tanh()
}

for name, activation in activations.items():
    model = Sequential([
        Dense(10, 20),
        activation,
        Dense(20, 1)
    ])
    
    # Train and evaluate
    model.fit(X, y, epochs=100, learning_rate=0.01, verbose=False)
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    print(f"{name}: MSE = {mse:.4f}")
```

## Key Takeaways

1. **Sigmoid**: Best for binary classification outputs and gates
2. **Tanh**: Zero-centered, good for RNNs
3. **ReLU**: Default choice for hidden layers
4. **LeakyReLU/ELU**: When ReLU neurons are dying
5. **Softmax**: Multi-class classification output

## Exercises

1. Compute sigmoid, ReLU, and tanh for z = [-3, -1.5, 0, 1.5, 3] by hand
2. Implement a custom activation function that combines ReLU and sigmoid
3. Train the same network with different activations and compare convergence

## Next Steps

- [04_gradient_descent_optimization.md](04_gradient_descent_optimization.md) - Learn optimization techniques
- [05_backpropagation.md](05_backpropagation.md) - Understand how gradients flow through activations
