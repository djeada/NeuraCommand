# Backpropagation

## Introduction

Backpropagation (backward propagation of errors) is the algorithm used to calculate gradients in neural networks. It's the engine that makes learning possible by efficiently computing how each weight contributes to the total error.

## The Chain Rule Foundation

Backpropagation is based on the chain rule from calculus:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

For a composition of functions $f(g(h(x)))$:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

## The Forward-Backward Process

```
Forward Pass (Computing Output):
─────────────────────────────────────────────────────────→

Input ──→ Layer 1 ──→ Layer 2 ──→ Layer 3 ──→ Output ──→ Loss
  x         z₁,a₁      z₂,a₂      z₃,a₃        ŷ         J

←─────────────────────────────────────────────────────────
Backward Pass (Computing Gradients):

∂J/∂w₁ ←── ∂J/∂a₁ ←── ∂J/∂a₂ ←── ∂J/∂a₃ ←── ∂J/∂ŷ ←── ∂J/∂J
```

## Complete Hand Calculation Example

Let's work through backpropagation for a small network solving XOR.

### Network Architecture

```
       Input Layer     Hidden Layer      Output Layer
       
x₁ ────────●─────────────●h₁──────────────●───────→ ŷ
           w₁           w₃              w₅
                  ↗       ↘        ↗
x₂ ────────●─────────────●h₂─────────────
           w₂           w₄              w₆
           
       (2 neurons)    (2 neurons)     (1 neuron)
```

### Step 1: Initialize Parameters

Weights:
- $w_1 = 0.5, w_2 = 0.6$ (input to hidden)
- $w_3 = 0.3, w_4 = 0.4$ (input to hidden)
- $w_5 = 0.7, w_6 = 0.8$ (hidden to output)

Biases:
- $b_1 = 0.1, b_2 = 0.2$ (hidden layer)
- $b_3 = 0.3$ (output layer)

Activation: Sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$

### Step 2: Forward Pass

For input $(x_1, x_2) = (0, 1)$ with target $y = 1$:

**Hidden Layer Calculations:**

$$z_1 = w_1 \cdot x_1 + w_3 \cdot x_2 + b_1$$
$$z_1 = 0.5 \cdot 0 + 0.3 \cdot 1 + 0.1 = 0.4$$

$$h_1 = \sigma(z_1) = \sigma(0.4) = \frac{1}{1+e^{-0.4}} = 0.598$$

$$z_2 = w_2 \cdot x_1 + w_4 \cdot x_2 + b_2$$
$$z_2 = 0.6 \cdot 0 + 0.4 \cdot 1 + 0.2 = 0.6$$

$$h_2 = \sigma(z_2) = \sigma(0.6) = \frac{1}{1+e^{-0.6}} = 0.646$$

**Output Layer Calculations:**

$$z_3 = w_5 \cdot h_1 + w_6 \cdot h_2 + b_3$$
$$z_3 = 0.7 \cdot 0.598 + 0.8 \cdot 0.646 + 0.3 = 1.136$$

$$\hat{y} = \sigma(z_3) = \sigma(1.136) = 0.757$$

### Step 3: Calculate Loss

Using Mean Squared Error (MSE):

$$J = \frac{1}{2}(y - \hat{y})^2 = \frac{1}{2}(1 - 0.757)^2 = 0.0295$$

### Step 4: Backward Pass

Now we calculate gradients, starting from the output and working backward.

**Output Layer Gradient:**

$$\frac{\partial J}{\partial \hat{y}} = -(y - \hat{y}) = -(1 - 0.757) = -0.243$$

$$\frac{\partial \hat{y}}{\partial z_3} = \sigma'(z_3) = \hat{y}(1-\hat{y}) = 0.757 \cdot 0.243 = 0.184$$

$$\delta_3 = \frac{\partial J}{\partial z_3} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} = -0.243 \cdot 0.184 = -0.045$$

**Gradients for Output Weights:**

$$\frac{\partial J}{\partial w_5} = \delta_3 \cdot h_1 = -0.045 \cdot 0.598 = -0.027$$

$$\frac{\partial J}{\partial w_6} = \delta_3 \cdot h_2 = -0.045 \cdot 0.646 = -0.029$$

$$\frac{\partial J}{\partial b_3} = \delta_3 = -0.045$$

**Hidden Layer Gradients:**

$$\delta_1 = \delta_3 \cdot w_5 \cdot h_1(1-h_1) = -0.045 \cdot 0.7 \cdot 0.598 \cdot 0.402 = -0.0076$$

$$\delta_2 = \delta_3 \cdot w_6 \cdot h_2(1-h_2) = -0.045 \cdot 0.8 \cdot 0.646 \cdot 0.354 = -0.0082$$

**Gradients for Hidden Weights:**

$$\frac{\partial J}{\partial w_1} = \delta_1 \cdot x_1 = -0.0076 \cdot 0 = 0$$

$$\frac{\partial J}{\partial w_3} = \delta_1 \cdot x_2 = -0.0076 \cdot 1 = -0.0076$$

$$\frac{\partial J}{\partial w_2} = \delta_2 \cdot x_1 = -0.0082 \cdot 0 = 0$$

$$\frac{\partial J}{\partial w_4} = \delta_2 \cdot x_2 = -0.0082 \cdot 1 = -0.0082$$

### Step 5: Update Weights

With learning rate $\eta = 0.1$:

| Weight | Old Value | Gradient | New Value |
|--------|-----------|----------|-----------|
| $w_1$ | 0.500 | 0.000 | 0.500 |
| $w_2$ | 0.600 | 0.000 | 0.600 |
| $w_3$ | 0.300 | -0.0076 | 0.301 |
| $w_4$ | 0.400 | -0.0082 | 0.401 |
| $w_5$ | 0.700 | -0.027 | 0.703 |
| $w_6$ | 0.800 | -0.029 | 0.803 |
| $b_1$ | 0.100 | -0.0076 | 0.101 |
| $b_2$ | 0.200 | -0.0082 | 0.201 |
| $b_3$ | 0.300 | -0.045 | 0.305 |

## The Backpropagation Algorithm

```
Algorithm: Backpropagation
──────────────────────────────────────────────────────────────
Input: Training sample (x, y), network weights W, learning rate η

1. FORWARD PASS:
   For each layer l = 1 to L:
       z[l] = W[l] @ a[l-1] + b[l]
       a[l] = activation(z[l])
   
   Compute loss: J = loss_function(a[L], y)

2. BACKWARD PASS:
   Compute output layer delta:
       δ[L] = ∂J/∂a[L] ⊙ activation'(z[L])
   
   For each layer l = L-1 down to 1:
       δ[l] = (W[l+1].T @ δ[l+1]) ⊙ activation'(z[l])

3. COMPUTE GRADIENTS:
   For each layer l = 1 to L:
       ∂J/∂W[l] = δ[l] @ a[l-1].T
       ∂J/∂b[l] = δ[l]

4. UPDATE WEIGHTS:
   For each layer l = 1 to L:
       W[l] = W[l] - η × ∂J/∂W[l]
       b[l] = b[l] - η × ∂J/∂b[l]
──────────────────────────────────────────────────────────────
```

## NeuraCommand Implementation

In NeuraCommand, backpropagation is handled automatically by the Sequential model:

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import Sigmoid

# Create network matching our hand calculation
model = Sequential([
    Dense(2, 2),    # Input to hidden
    Sigmoid(),
    Dense(2, 1),    # Hidden to output  
    Sigmoid()
])

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# The fit method performs forward pass, loss calculation, 
# backward pass, and weight updates
model.fit(X, y, epochs=1000, learning_rate=0.5)

# Verify learning
predictions = model.predict(X)
for i in range(len(X)):
    print(f"Input: {X[i]}, Prediction: {predictions[i,0]:.3f}, Target: {y[i,0]}")
```

## Understanding Layer-by-Layer Gradients

### Dense Layer Backward Pass

For a dense layer: $z = Wx + b$, $a = \sigma(z)$

```python
# Pseudocode for Dense layer backward
def backward(self, grad_output):
    # grad_output: ∂L/∂a (gradient from next layer)
    
    # Gradient through activation
    if self.activation:
        grad = grad_output * self.activation.derivative(self.pre_activation)
    else:
        grad = grad_output
    
    # Gradient w.r.t. weights: ∂L/∂W = input.T @ grad
    self.grad_weights = self.input_cache.T @ grad / batch_size
    
    # Gradient w.r.t. bias: ∂L/∂b = sum of grad
    self.grad_bias = np.mean(grad, axis=0, keepdims=True)
    
    # Gradient to pass backward: ∂L/∂input = grad @ W.T
    return grad @ self.weights.T
```

### Activation Layer Backward Pass

For activation layer: $a = f(z)$

```python
# Pseudocode for Activation layer backward
def backward(self, grad_output):
    # Simply multiply by derivative of activation
    return grad_output * self.activation.derivative(self.input_cache)
```

## Gradient Flow Visualization

```
Layer 3 (Output)           Layer 2           Layer 1 (Input)
    ↓ δ₃                      ↓ δ₂              ↓ δ₁
    │                         │                 │
    └──→ ∂J/∂w₅,w₆            └──→ ∂J/∂w₃,w₄    └──→ ∂J/∂w₁,w₂
    │                         │                 │
    │    Gradient info        │                 │
    ├────────────────────────→├────────────────→│
         backpropagates            through chain rule
```

## Common Issues in Backpropagation

### 1. Vanishing Gradients

When gradients become very small, early layers learn slowly.

**Cause:** Repeated multiplication of small derivatives (e.g., sigmoid's max derivative is 0.25)

$$\delta_1 = \delta_3 \cdot w_3 \cdot \sigma'(z_2) \cdot w_1 \cdot \sigma'(z_1)$$

If each $\sigma'$ ≈ 0.25, gradients shrink: $0.25^n$ becomes tiny!

**Solutions:**
- Use ReLU instead of sigmoid (derivative = 1 for positive values)
- Use batch normalization
- Use skip connections (residual networks)

### 2. Exploding Gradients

When gradients become very large, weights update wildly.

**Cause:** Large weights and repeated multiplication

**Solutions:**
- Gradient clipping: `grad = np.clip(grad, -1.0, 1.0)`
- Proper weight initialization
- Batch normalization

### 3. Dead Neurons

ReLU neurons that always output 0 and never update.

**Cause:** Negative inputs causing ReLU derivative = 0

**Solutions:**
- Use Leaky ReLU or ELU
- Careful learning rate selection
- Better weight initialization

## Numerical Gradient Checking

Verify your backpropagation implementation using numerical gradients:

```python
def numerical_gradient(model, X, y, param, epsilon=1e-5):
    """Compute numerical gradient for verification"""
    original = param.copy()
    
    gradients = np.zeros_like(param)
    for i in range(param.size):
        # Compute f(x + epsilon)
        param.flat[i] = original.flat[i] + epsilon
        loss_plus = compute_loss(model, X, y)
        
        # Compute f(x - epsilon)
        param.flat[i] = original.flat[i] - epsilon
        loss_minus = compute_loss(model, X, y)
        
        # Numerical gradient
        gradients.flat[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Restore original
        param.flat[i] = original.flat[i]
    
    return gradients

# Compare with analytical gradient
analytical_grad = model.layers[0].grad_weights
numerical_grad = numerical_gradient(model, X, y, model.layers[0].weights)

# Should be very close (< 1e-5 difference)
difference = np.max(np.abs(analytical_grad - numerical_grad))
print(f"Gradient check: max difference = {difference}")
```

## Summary Table

| Forward Pass | Backward Pass |
|--------------|---------------|
| Input → Output | Output → Input |
| Compute activations | Compute gradients |
| z = Wx + b | ∂L/∂W = δ @ input.T |
| a = f(z) | ∂L/∂input = W.T @ δ |
| Loss = L(ŷ, y) | δ = ∂L/∂a ⊙ f'(z) |

## Key Takeaways

1. **Backpropagation** uses the chain rule to compute gradients efficiently
2. **Forward pass** computes predictions; **backward pass** computes gradients
3. **Gradients flow backward** from output layer to input layer
4. **Vanishing/exploding gradients** are common issues with solutions
5. **Numerical gradient checking** validates your implementation

## Exercises

1. Complete the hand calculation for input (1, 1) with target 0
2. Implement backpropagation from scratch for a 2-layer network
3. Add gradient checking to verify your implementation

## Next Steps

- [06_multilayer_perceptrons.md](06_multilayer_perceptrons.md) - Build deeper networks
- [11_regularization_techniques.md](11_regularization_techniques.md) - Learn about regularization
