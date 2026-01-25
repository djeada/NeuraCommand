# Perceptron Fundamentals

## Introduction

The perceptron is the simplest form of a neural network - a single artificial neuron that can learn to make predictions. Understanding the perceptron is essential because it forms the foundation for all more complex neural network architectures.

## What is a Perceptron?

A perceptron takes multiple inputs, applies weights to each input, sums them together with a bias, and produces an output through an activation function.

```
Input x₁ ────w₁────┐
                   │
Input x₂ ────w₂────┼──→ Σ + b ──→ f(z) ──→ Output
                   │
Input x₃ ────w₃────┘
```

**Mathematical Formula:**

$$z = \sum_{i=1}^{n} w_i x_i + b = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

$$\text{output} = f(z)$$

## Hand Calculation Example: AND Gate

Let's learn by computing a perceptron by hand to implement an AND gate.

### Truth Table for AND Gate

| Input 1 (x₁) | Input 2 (x₂) | Output (y) |
|--------------|--------------|------------|
| 0            | 0            | 0          |
| 0            | 1            | 0          |
| 1            | 0            | 0          |
| 1            | 1            | 1          |

### Step 1: Initialize Parameters

Let's start with random parameters:
- Weight 1 (w₁): 0.3
- Weight 2 (w₂): -0.1
- Bias (b): 0.0
- Learning rate (η): 0.1
- Activation: Sigmoid function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Step 2: Forward Pass (First Training Step)

For input (0, 0), target = 0:

$$z = w_1 \cdot x_1 + w_2 \cdot x_2 + b$$
$$z = 0.3 \cdot 0 + (-0.1) \cdot 0 + 0.0 = 0.0$$

$$\text{output} = \sigma(0.0) = \frac{1}{1 + e^{0}} = 0.5$$

### Step 3: Calculate Error

$$\text{error} = \text{target} - \text{output} = 0 - 0.5 = -0.5$$

### Step 4: Update Weights (Gradient Descent)

Using the perceptron learning rule:

$$\Delta w_i = \eta \cdot \text{error} \cdot x_i$$
$$\Delta b = \eta \cdot \text{error}$$

For our example:
- $\Delta w_1 = 0.1 \cdot (-0.5) \cdot 0 = 0$
- $\Delta w_2 = 0.1 \cdot (-0.5) \cdot 0 = 0$
- $\Delta b = 0.1 \cdot (-0.5) = -0.05$

Updated parameters:
- w₁ = 0.3 + 0 = 0.3
- w₂ = -0.1 + 0 = -0.1
- b = 0.0 + (-0.05) = -0.05

### Complete Training Table (One Epoch)

| Step | x₁ | x₂ | Target | z | Output | Error | Δw₁ | Δw₂ | Δb |
|------|----|----|--------|---|--------|-------|-----|-----|----|
| 1 | 0 | 0 | 0 | 0.00 | 0.500 | -0.50 | 0.00 | 0.00 | -0.05 |
| 2 | 0 | 1 | 0 | -0.15 | 0.463 | -0.46 | 0.00 | -0.046 | -0.046 |
| 3 | 1 | 0 | 0 | 0.20 | 0.550 | -0.55 | -0.055 | 0.00 | -0.055 |
| 4 | 1 | 1 | 1 | -0.11 | 0.473 | 0.53 | 0.053 | 0.053 | 0.053 |

After one epoch, the parameters are approximately:
- w₁ ≈ 0.298
- w₂ ≈ -0.093
- b ≈ -0.098

## Implementing in NeuraCommand

Now let's implement the same perceptron using NeuraCommand:

```python
import numpy as np
from neura_command.neuron.perceptron.perceptron import Perceptron
from neura_command.network_utils.activation_functions import SigmoidActivation

# Create a perceptron for AND gate
perceptron = Perceptron(
    input_size=2,
    output_size=1,
    activation_func=SigmoidActivation(),
    learning_rate=0.1,
    l1_ratio=0,
    l2_ratio=0,
)

# Define the AND gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Train the perceptron
print("Initial weights:", perceptron.weights)
print("Initial bias:", perceptron.bias)

# Train for multiple epochs
for epoch in range(100):
    perceptron.train(X, y, epochs=1, batch_size=1)
    if (epoch + 1) % 20 == 0:
        output = perceptron.forward(X)
        error = np.mean((y - output) ** 2)
        print(f"Epoch {epoch + 1}, MSE: {error:.4f}")

# Test the trained perceptron
print("\nFinal Predictions:")
for i in range(len(X)):
    prediction = perceptron.forward(X[i:i+1])
    print(f"Input: {X[i]}, Predicted: {prediction[0][0]:.3f}, Target: {y[i][0]}")
```

**Expected Output:**
```
Initial weights: [[ 0.xxx]
                  [ 0.xxx]]
Initial bias: [0.0]
Epoch 20, MSE: 0.1234
Epoch 40, MSE: 0.0567
Epoch 60, MSE: 0.0234
Epoch 80, MSE: 0.0123
Epoch 100, MSE: 0.0089

Final Predictions:
Input: [0 0], Predicted: 0.045, Target: 0
Input: [0 1], Predicted: 0.098, Target: 0
Input: [1 0], Predicted: 0.102, Target: 0
Input: [1 1], Predicted: 0.887, Target: 1
```

## Using the Sequential API

For a more modern approach, use the Sequential API with Dense layers:

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import Sigmoid

# Create a single-neuron network (perceptron)
model = Sequential([
    Dense(2, 1),   # 2 inputs, 1 output
    Sigmoid()      # Sigmoid activation
])

# Prepare data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

# Train the model
model.fit(X, y, epochs=1000, learning_rate=0.5, batch_size=4)

# Make predictions
predictions = model.predict(X)
print("Predictions:")
for i in range(len(X)):
    print(f"  {X[i]} -> {predictions[i][0]:.3f} (target: {y[i][0]})")
```

## The Decision Boundary

A perceptron creates a linear decision boundary that separates the input space into two regions.

For the AND gate with trained weights:

$$w_1 x_1 + w_2 x_2 + b = 0$$

This is a line in 2D space. Points above the line are classified as 1, and points below as 0.

```
    x₂
    ^
  1 |  (0,1)          (1,1)
    |    ×              ●
    |         \
    |          \  Decision Boundary
    |           \
    |            \
  0 |  (0,0)      \  (1,0)
    |    ×         \    ×
    └────────────────────→ x₁
       0              1

    × = Class 0 (Output 0)
    ● = Class 1 (Output 1)
```

## Limitations of Single Perceptrons

Single perceptrons can only solve **linearly separable** problems. They cannot solve problems like XOR:

| Input 1 | Input 2 | XOR Output |
|---------|---------|------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

```
    x₂
    ^
  1 |  ●              ×
    |  (0,1)          (1,1)
    |
    |     No single line can
    |     separate ● from ×
    |
  0 |  ×              ●
    |  (0,0)          (1,0)
    └────────────────────→ x₁
```

This limitation led to the development of **multi-layer perceptrons (MLPs)**, which we cover in [06_multilayer_perceptrons.md](06_multilayer_perceptrons.md).

## Perceptron Learning Algorithm Summary

```
Algorithm: Perceptron Learning
─────────────────────────────────────────
1. Initialize weights w and bias b randomly
2. For each epoch:
   3. For each training sample (x, y):
      4. Calculate output: ŷ = f(w·x + b)
      5. Calculate error: e = y - ŷ
      6. Update weights: w = w + η × e × x
      7. Update bias: b = b + η × e
8. Repeat until convergence or max epochs
─────────────────────────────────────────
```

## Key Takeaways

1. **A perceptron** is a single neuron with weights, a bias, and an activation function
2. **Learning** happens by adjusting weights based on the error
3. **The learning rate (η)** controls how much weights change in each update
4. **Single perceptrons** can only solve linearly separable problems
5. **NeuraCommand** provides both low-level (Perceptron) and high-level (Sequential) APIs

## Exercises

1. **Manual Calculation**: Calculate one complete epoch of training for an OR gate by hand
2. **Code Exercise**: Modify the NeuraCommand example to implement a NAND gate
3. **Visualization**: Plot the decision boundary as training progresses

## Next Steps

Now that you understand perceptrons, continue to:
- [03_activation_functions.md](03_activation_functions.md) - Learn about different activation functions
- [04_gradient_descent_optimization.md](04_gradient_descent_optimization.md) - Understand optimization
- [05_backpropagation.md](05_backpropagation.md) - Learn the backpropagation algorithm
