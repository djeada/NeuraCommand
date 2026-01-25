# Gradient Descent Optimization

## Introduction

Gradient descent is the fundamental optimization algorithm used to train neural networks. It works by iteratively adjusting parameters in the direction that minimizes the loss function.

## The Core Idea

Imagine you're blindfolded on a hilly terrain and want to reach the lowest point. You would:
1. Feel the slope under your feet
2. Take a step in the downhill direction
3. Repeat until you can't go any lower

This is exactly what gradient descent does mathematically!

```
High Loss
    │                 ★ Start
    │                 /\
    │                /  \
    │               /    \ 
    │              /      \ 
    │         /\  /        \
    │        /  \/          \
    │       /    ↓ Steps     \ 
    │      /                  \ 
    │   ● /                    \ ●
Low Loss ─────────────────────────→ Parameters (Weights)
    
    ★ = Starting point
    ● = Local/Global minima
    ↓ = Gradient descent steps
```

## Mathematical Formulation

The gradient descent update rule is:

$$\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$$

Where:
- $\theta$ = parameters (weights and biases)
- $\alpha$ = learning rate
- $\nabla J(\theta)$ = gradient of loss function with respect to parameters
- $t$ = iteration number

## Hand Calculation Example

Let's minimize a simple function: $J(\theta) = \theta^2$

The gradient is: $\frac{dJ}{d\theta} = 2\theta$

Starting with $\theta_0 = 4$ and learning rate $\alpha = 0.1$:

| Step | θ | J(θ) = θ² | Gradient = 2θ | Update: θ - α×gradient |
|------|---|-----------|---------------|------------------------|
| 0 | 4.000 | 16.000 | 8.000 | 4.0 - 0.1×8.0 = 3.200 |
| 1 | 3.200 | 10.240 | 6.400 | 3.2 - 0.1×6.4 = 2.560 |
| 2 | 2.560 | 6.554 | 5.120 | 2.56 - 0.1×5.12 = 2.048 |
| 3 | 2.048 | 4.194 | 4.096 | 2.048 - 0.1×4.096 = 1.638 |
| 4 | 1.638 | 2.684 | 3.277 | 1.638 - 0.1×3.277 = 1.311 |
| 5 | 1.311 | 1.719 | 2.621 | ... converging to 0 |

## Variants of Gradient Descent

### 1. Batch Gradient Descent

Uses the entire dataset to compute the gradient at each step.

```
for each epoch:
    gradient = compute_gradient(entire_dataset)
    parameters = parameters - learning_rate * gradient
```

| Pros | Cons |
|------|------|
| Stable convergence | Slow for large datasets |
| Guaranteed direction to minimum | High memory usage |
| Easy to implement | Can get stuck in local minima |

### 2. Stochastic Gradient Descent (SGD)

Uses a single random sample to compute the gradient.

```
for each epoch:
    shuffle(dataset)
    for each sample (x, y):
        gradient = compute_gradient(x, y)
        parameters = parameters - learning_rate * gradient
```

| Pros | Cons |
|------|------|
| Fast updates | Noisy gradient estimates |
| Can escape local minima | May not converge smoothly |
| Low memory usage | Requires more iterations |

### 3. Mini-Batch Gradient Descent

Uses small batches of samples (best of both worlds).

```
for each epoch:
    shuffle(dataset)
    for each batch in batches(dataset, batch_size):
        gradient = compute_gradient(batch)
        parameters = parameters - learning_rate * gradient
```

**Common batch sizes**: 32, 64, 128, 256

| Pros | Cons |
|------|------|
| Balanced speed/stability | Batch size is a hyperparameter |
| GPU-friendly | Some noise in gradients |
| Reduces overfitting | Requires tuning |

## NeuraCommand Implementation

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU

# Create a simple network
model = Sequential([
    Dense(2, 10),
    ReLU(),
    Dense(10, 1)
])

# Sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Train with different batch sizes
# Batch GD (batch_size = all samples)
model.fit(X, y, epochs=100, learning_rate=0.1, batch_size=4)

# Mini-batch GD
model.fit(X, y, epochs=100, learning_rate=0.1, batch_size=2)

# SGD (batch_size = 1)
model.fit(X, y, epochs=100, learning_rate=0.1, batch_size=1)
```

## Learning Rate

The learning rate $\alpha$ is one of the most important hyperparameters.

### Too Small Learning Rate
```
Loss
  │★
  │ ↘
  │   ↘
  │     ↘         Very slow convergence
  │       ↘
  │         ↘
  │           ●
  └──────────────→ Iterations (many needed)
```

### Too Large Learning Rate
```
Loss
  │     ★
  │    ↙ ↘       
  │   ↗   ↘      Oscillates, may diverge!
  │  ↙     ↘
  │ ↗       ●
  └──────────────→ Iterations
```

### Just Right
```
Loss
  │★
  │ ↘
  │   ↘
  │     ●        Smooth convergence
  └──────────────→ Iterations
```

### Learning Rate Schedules

Reduce learning rate during training for better convergence:

**Step Decay:**
```python
def step_decay(epoch, initial_lr=0.1, drop=0.5, epochs_drop=10):
    return initial_lr * (drop ** (epoch // epochs_drop))
```

**Exponential Decay:**
```python
def exponential_decay(epoch, initial_lr=0.1, decay_rate=0.95):
    return initial_lr * (decay_rate ** epoch)
```

## Advanced Optimizers

### Momentum

Adds "velocity" to help accelerate through flat regions and dampen oscillations.

$$v_{t+1} = \beta v_t + \nabla J(\theta_t)$$
$$\theta_{t+1} = \theta_t - \alpha v_{t+1}$$

Where $\beta$ is the momentum coefficient (typically 0.9).

```
Without Momentum:        With Momentum:
    ↓ ↓ ↓ ↓               ↓
      ↓ ↓                   ↓
        ↓                     ↓
        ●                       ●
        
(zigzag path)           (smoother path)
```

### RMSprop

Adapts learning rates based on recent gradient magnitudes.

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

### Adam (Adaptive Moment Estimation)

Combines momentum and RMSprop - the most popular optimizer.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Default hyperparameters:**
- $\alpha = 0.001$
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

## Optimizer Comparison

| Optimizer | Pros | Cons | When to Use |
|-----------|------|------|-------------|
| SGD | Simple, generalizes well | Slow, needs tuning | Simple problems |
| SGD+Momentum | Faster convergence | One more hyperparameter | Most cases |
| RMSprop | Adaptive LR | Can be unstable | RNNs |
| Adam | Fast, adaptive | Memory heavy | Default choice |

## Practical Tips

### 1. Learning Rate Finding

Start with a high learning rate and reduce:

```python
learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
for lr in learning_rates:
    model = create_model()
    history = model.fit(X, y, epochs=10, learning_rate=lr)
    print(f"LR: {lr}, Final Loss: {history[-1]}")
```

### 2. Gradient Clipping

Prevent exploding gradients:

```python
def clip_gradients(gradients, max_norm=1.0):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        gradients = [g * scale for g in gradients]
    return gradients
```

### 3. Early Stopping

Stop training when validation loss stops improving:

```python
best_val_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_data)
    val_loss = evaluate(model, val_data)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        save_model(model)
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

## Convergence Visualization

```
Loss over Training:

Loss
  │
  │ ★                                   Overfitting →
  │   \                              /\  /\
  │     \                           /  \/
  │       \  ___   Training    ___/
  │         \   \___          /      
  │          \______\________/ Validation
  │                  \_____/  
  │                          
  └────────────────────────────────────→ Epochs
        Good fit zone   ↑
                    Best checkpoint
```

## Complete Training Loop Example

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU, Sigmoid

# Create model
model = Sequential([
    Dense(2, 8),
    ReLU(),
    Dense(8, 4),
    ReLU(),
    Dense(4, 1),
    Sigmoid()
])

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Training parameters
learning_rate = 0.5
batch_size = 4
epochs = 1000

# Train with loss tracking
losses = []
for epoch in range(epochs):
    # Forward pass
    predictions = model.predict(X)
    
    # Calculate loss (MSE)
    loss = np.mean((predictions - y) ** 2)
    losses.append(loss)
    
    # Backward pass and update (handled internally by fit)
    model.fit(X, y, epochs=1, learning_rate=learning_rate, batch_size=batch_size)
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Final predictions
print("\nFinal Predictions:")
for i in range(len(X)):
    pred = model.predict(X[i:i+1])[0, 0]
    print(f"  {X[i]} -> {pred:.3f} (target: {y[i, 0]})")
```

## Key Takeaways

1. **Gradient descent** minimizes loss by following the negative gradient
2. **Learning rate** controls step size - critical hyperparameter
3. **Mini-batch** offers the best balance of speed and stability
4. **Adam** is a good default optimizer for most tasks
5. **Monitor both training and validation loss** to detect overfitting

## Exercises

1. Implement gradient descent by hand for $J(\theta) = (\theta - 3)^2$
2. Compare convergence with different learning rates: 0.01, 0.1, 1.0
3. Implement momentum from scratch and visualize the difference

## Next Steps

- [05_backpropagation.md](05_backpropagation.md) - Learn how gradients are calculated
- [06_multilayer_perceptrons.md](06_multilayer_perceptrons.md) - Apply optimization to deep networks
