# Multi-Layer Perceptrons

## Introduction

Multi-Layer Perceptrons (MLPs) are feedforward neural networks with one or more hidden layers. They can learn complex, non-linear relationships that single perceptrons cannot, making them the foundation for deep learning.

## From Perceptron to MLP

A single perceptron can only solve linearly separable problems. MLPs overcome this limitation by stacking multiple layers.

```
Single Perceptron:            Multi-Layer Perceptron:
                              
x₁ ──┐                        x₁ ──┐   ┌──●──┐
     ├──●──→ ŷ                     ├──●┤    ├──●──→ ŷ
x₂ ──┘                        x₂ ──┘   └──●──┘
                              
(Linear boundary only)        (Complex boundaries possible)
```

## The XOR Problem

XOR is not linearly separable - no single line can separate 1s from 0s:

```
    x₂
    1 │  ●              ×          ● = Output 1
      │  (0,1)          (1,1)      × = Output 0
      │
      │     No single line works!
      │
    0 │  ×              ●
      │  (0,0)          (1,0)
      └────────────────────→ x₁
         0              1
```

But an MLP can solve it using multiple boundaries:

```
    x₂
    1 │  ●    \    /    ×         Region between lines = 1
      │       \  /                Outside = 0
      │        \/
      │        /\
      │       /  \
    0 │  ×   /    \     ●
      └────────────────────→ x₁
```

## MLP Architecture

```
Input Layer        Hidden Layer 1      Hidden Layer 2       Output Layer
(n features)       (h₁ neurons)        (h₂ neurons)        (o outputs)

    ○                   ●                   ●                   ○
    ○ ──────────────→   ●  ──────────────→  ●  ──────────────→  ○
    ○                   ●                   ●                   ○
    ○                   ●                   ●
    
    Receives           Learns low-level    Learns high-level   Produces
    raw data           features            features            predictions
```

## Mathematical Representation

For an MLP with L layers:

**Layer 1 (First Hidden):**
$$z^{(1)} = W^{(1)} x + b^{(1)}$$
$$a^{(1)} = f(z^{(1)})$$

**Layer l (General Hidden):**
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = f(z^{(l)})$$

**Layer L (Output):**
$$z^{(L)} = W^{(L)} a^{(L-1)} + b^{(L)}$$
$$\hat{y} = f_{output}(z^{(L)})$$

Where:
- $W^{(l)}$ = weight matrix for layer l
- $b^{(l)}$ = bias vector for layer l
- $f$ = activation function
- $a^{(l)}$ = activation output of layer l

## NeuraCommand Implementation

### Creating an MLP

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU, Sigmoid, Softmax

# MLP for XOR problem
xor_model = Sequential([
    Dense(2, 8),    # 2 inputs → 8 hidden neurons
    ReLU(),
    Dense(8, 4),    # 8 → 4 hidden neurons
    ReLU(),
    Dense(4, 1),    # 4 → 1 output
    Sigmoid()       # Binary classification
])

# MLP for multi-class classification (e.g., digits 0-9)
classifier = Sequential([
    Dense(784, 256),   # 28x28 pixels → 256 neurons
    ReLU(),
    Dense(256, 128),   # 256 → 128 neurons
    ReLU(),
    Dense(128, 10),    # 128 → 10 classes
    Softmax()          # Multi-class output
])

# MLP for regression
regressor = Sequential([
    Dense(10, 64),     # 10 features → 64 neurons
    ReLU(),
    Dense(64, 32),     # 64 → 32 neurons
    ReLU(),
    Dense(32, 1)       # 32 → 1 output (no activation for regression)
])
```

### Training the XOR Model

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU, Sigmoid

# Create XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Build model
model = Sequential([
    Dense(2, 8),
    ReLU(),
    Dense(8, 1),
    Sigmoid()
])

# Train
print("Training XOR model...")
losses = []
for epoch in range(2000):
    model.fit(X, y, epochs=1, learning_rate=0.5, batch_size=4)
    
    if (epoch + 1) % 500 == 0:
        predictions = model.predict(X)
        loss = np.mean((predictions - y) ** 2)
        losses.append(loss)
        print(f"Epoch {epoch+1}: Loss = {loss:.6f}")

# Test predictions
print("\nFinal Predictions:")
predictions = model.predict(X)
for i in range(len(X)):
    pred = predictions[i, 0]
    target = y[i, 0]
    result = "✓" if (pred > 0.5) == target else "✗"
    print(f"  XOR({X[i,0]:.0f}, {X[i,1]:.0f}) = {pred:.4f} → {pred > 0.5:.0f} (target: {target:.0f}) {result}")
```

**Expected Output:**
```
Training XOR model...
Epoch 500: Loss = 0.125678
Epoch 1000: Loss = 0.023456
Epoch 1500: Loss = 0.005678
Epoch 2000: Loss = 0.001234

Final Predictions:
  XOR(0, 0) = 0.0234 → 0 (target: 0) ✓
  XOR(0, 1) = 0.9812 → 1 (target: 1) ✓
  XOR(1, 0) = 0.9756 → 1 (target: 1) ✓
  XOR(1, 1) = 0.0189 → 0 (target: 0) ✓
```

## Universal Approximation Theorem

MLPs with a single hidden layer containing enough neurons can approximate any continuous function to arbitrary precision.

```
Any Function f(x):              MLP Approximation:

    │   ╭───╮                       │   ┌─┬─┐
    │  ╱     ╲                      │  ┌┘ └┐│
    │ ╱       ╲   ────────→         │ ┌┘   └┤
    │╱         ╲                    │┌┘     └┐
    └──────────────                 └──────────────
    
    True function              Approximation with
                               many hidden neurons
```

However, deeper networks often learn better with fewer total parameters:
- Shallow: 1 hidden layer with 1000 neurons
- Deep: 3 hidden layers with 50 neurons each (more efficient!)

## Designing MLP Architectures

### Layer Width (Number of Neurons)

```
Too Narrow:           Just Right:           Too Wide:
    ●                    ●●●                  ●●●●●●●●
    │                    │││                  ││││││││
    ●                    ●●●                  ●●●●●●●●
    
Underfits:           Learns well:          Overfits:
Can't learn          Generalizes           Memorizes
complex patterns     properly              training data
```

### Layer Depth (Number of Layers)

| Depth | Pros | Cons | Use Cases |
|-------|------|------|-----------|
| Shallow (1-2 layers) | Fast, less overfitting | Limited capacity | Simple problems |
| Medium (3-5 layers) | Good balance | Needs regularization | Most problems |
| Deep (6+ layers) | High capacity | Training challenges | Complex problems |

### Common Architectures

```python
# Binary Classification
binary_classifier = Sequential([
    Dense(input_size, 64),
    ReLU(),
    Dense(64, 32),
    ReLU(),
    Dense(32, 1),
    Sigmoid()
])

# Multi-class Classification
multi_classifier = Sequential([
    Dense(input_size, 128),
    ReLU(),
    Dense(128, 64),
    ReLU(),
    Dense(64, num_classes),
    Softmax()
])

# Regression
regressor = Sequential([
    Dense(input_size, 64),
    ReLU(),
    Dense(64, 32),
    ReLU(),
    Dense(32, 1)  # No activation for regression
])

# Funnel Architecture (gradually reduce)
funnel = Sequential([
    Dense(1000, 512),
    ReLU(),
    Dense(512, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 10)
])
```

## Hand Calculation: Forward Pass

Let's trace through a small MLP: 2→3→1

**Input:** x = [1, 0]

**Hidden Layer (3 neurons):**
```
Weights W₁:          Bias b₁:
[0.1, 0.2]           [0.01]
[0.3, 0.4]           [0.02]
[0.5, 0.6]           [0.03]

z₁ = W₁ @ x + b₁
   = [0.1×1 + 0.2×0 + 0.01]   [0.11]
     [0.3×1 + 0.4×0 + 0.02] = [0.32]
     [0.5×1 + 0.6×0 + 0.03]   [0.53]

a₁ = ReLU(z₁) = [0.11, 0.32, 0.53]  (all positive, unchanged)
```

**Output Layer (1 neuron):**
```
Weights W₂:           Bias b₂:
[0.7, 0.8, 0.9]       [0.04]

z₂ = W₂ @ a₁ + b₂
   = 0.7×0.11 + 0.8×0.32 + 0.9×0.53 + 0.04
   = 0.077 + 0.256 + 0.477 + 0.04
   = 0.85

a₂ = sigmoid(0.85) = 1/(1 + e^(-0.85)) = 0.70
```

Final output: **ŷ = 0.70**

## Training Tips

### 1. Weight Initialization

NeuraCommand uses He initialization by default for ReLU:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

```python
# NeuraCommand handles this automatically
layer = Dense(784, 256, weight_init='he')  # Default
layer = Dense(784, 256, weight_init='xavier')  # For sigmoid/tanh
```

### 2. Batch Size Selection

| Batch Size | Memory | Speed | Generalization |
|------------|--------|-------|----------------|
| Small (16-32) | Low | Slower | Better |
| Medium (64-128) | Medium | Balanced | Balanced |
| Large (256-512) | High | Faster | May overfit |

### 3. Learning Rate

```python
# Start with these defaults
learning_rate = 0.001  # For Adam
learning_rate = 0.01   # For SGD

# Reduce if training is unstable
# Increase if training is too slow
```

## Complete Example: Digit Classification

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU, Softmax

# Simulated digit data (in practice, load from MNIST)
np.random.seed(42)
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.eye(10)[np.random.randint(0, 10, 1000)].astype(np.float32)

X_test = np.random.randn(100, 784).astype(np.float32)
y_test = np.eye(10)[np.random.randint(0, 10, 100)].astype(np.float32)

# Build MLP
model = Sequential([
    Dense(784, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
])

# Training loop
epochs = 50
batch_size = 32
learning_rate = 0.01

print("Training digit classifier...")
for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Train on batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        model.fit(X_batch, y_batch, epochs=1, learning_rate=learning_rate, batch_size=batch_size)
    
    if (epoch + 1) % 10 == 0:
        # Evaluate
        predictions = model.predict(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        print(f"Epoch {epoch+1}: Accuracy = {accuracy:.2%}")

print("Training complete!")
```

## Key Takeaways

1. **MLPs** can solve non-linear problems by stacking layers
2. **Hidden layers** learn hierarchical representations
3. **Width and depth** must be balanced to avoid under/overfitting
4. **Universal approximation** - MLPs can approximate any function
5. **Architecture design** requires experimentation

## Exercises

1. Solve the XOR problem with the minimum number of hidden neurons
2. Experiment with different architectures for a classification task
3. Visualize the decision boundary at each hidden layer

## Next Steps

- [07_convolutional_neural_networks.md](07_convolutional_neural_networks.md) - Learn about CNNs for images
- [11_regularization_techniques.md](11_regularization_techniques.md) - Prevent overfitting
- [12_practical_training_tips.md](12_practical_training_tips.md) - Best practices
