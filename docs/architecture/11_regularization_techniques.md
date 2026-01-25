# Regularization Techniques

## Introduction

Regularization techniques prevent overfitting - when a model learns to memorize training data rather than learning general patterns. This document covers essential regularization methods used in neural networks.

## The Overfitting Problem

```
Training vs Validation Loss:

Loss
  │
  │ ★                                   Overfitting →
  │   \                              /\  /\
  │     \                           /  \/
  │       \  ___   Training    ___/
  │         \   \___          /      
  │          \______\________/ Validation
  │                  \_____/  
  │           ↑              
  │       Best point         
  └────────────────────────────────────→ Epochs
```

**Signs of Overfitting:**
- Low training error, high validation error
- Model performs well on training data but poorly on new data
- High model complexity relative to data size

## L1 and L2 Regularization (Weight Decay)

### L2 Regularization (Ridge)

Add penalty proportional to sum of squared weights:

$$J_{regularized} = J_{original} + \lambda \sum_{i} w_i^2$$

**Effect:** Encourages smaller weights, smoother decision boundaries

```python
import numpy as np

# L2 regularization during training
lambda_l2 = 0.01

# Add to gradient
grad_weights = original_grad + lambda_l2 * weights

# Or add to loss
loss = original_loss + lambda_l2 * np.sum(weights ** 2)
```

### L1 Regularization (Lasso)

Add penalty proportional to sum of absolute weights:

$$J_{regularized} = J_{original} + \lambda \sum_{i} |w_i|$$

**Effect:** Encourages sparsity - some weights become exactly zero

```python
# L1 regularization
lambda_l1 = 0.01

# Add to gradient
grad_weights = original_grad + lambda_l1 * np.sign(weights)

# Add to loss
loss = original_loss + lambda_l1 * np.sum(np.abs(weights))
```

### Comparison

| Aspect | L1 | L2 |
|--------|----|----|
| Weight effect | Sparse (many zeros) | Small but non-zero |
| Feature selection | Yes | No |
| Computation | Harder (non-differentiable at 0) | Easier |
| Use case | Feature selection | General regularization |

### Combined: Elastic Net

$$J = J_{original} + \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2$$

## Dropout

Randomly "drop" neurons during training, forcing the network to be robust.

```
Training:                     Inference:
                             
○ ─── ● ─── ○                ○ ─── ● ─── ○
      │    ╲                       │    ╲
      ✗      ○    ×0.5 →           ●      ○
      ╲    ╱                       ╲    ╱
○ ─── ● ─── ○                ○ ─── ● ─── ○
                             
(✗ = dropped, ● = kept)      (All neurons × 0.5)
```

### How Dropout Works

During training:
1. For each neuron, randomly "drop" it with probability p (typically 0.5)
2. Dropped neurons output zero
3. Remaining neurons are scaled by 1/(1-p)

During inference:
- Use all neurons
- Scale outputs by (1-p) OR use inverted dropout (scale during training)

### NeuraCommand Dropout

```python
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU
from neura_command.layers.dropout import Dropout

model = Sequential([
    Dense(784, 256),
    ReLU(),
    Dropout(rate=0.5),  # 50% dropout
    Dense(256, 128),
    ReLU(),
    Dropout(rate=0.3),  # 30% dropout
    Dense(128, 10)
])

# During training, dropout is active
output = model.forward(X, training=True)

# During inference, dropout is disabled
predictions = model.forward(X, training=False)
```

### Implementation Concept

```python
class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None
    
    def forward(self, x, training=True):
        if not training:
            return x
        
        # Create random mask
        self.mask = np.random.binomial(1, 1-self.rate, x.shape)
        
        # Apply mask and scale
        return x * self.mask / (1 - self.rate)
    
    def backward(self, grad):
        # Gradient only flows through kept neurons
        return grad * self.mask / (1 - self.rate)
```

## Batch Normalization

Normalize activations within each mini-batch to stabilize and accelerate training.

### The Problem

Deep networks suffer from "internal covariate shift" - the distribution of layer inputs changes during training.

### How BatchNorm Works

For each mini-batch:

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

Where γ and β are learnable parameters.

### NeuraCommand BatchNorm

```python
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU
from neura_command.layers.batchnorm import BatchNorm

model = Sequential([
    Dense(784, 256),
    BatchNorm(),      # Normalize before activation
    ReLU(),
    Dense(256, 128),
    BatchNorm(),
    ReLU(),
    Dense(128, 10)
])
```

### BatchNorm Benefits

| Benefit | Description |
|---------|-------------|
| Faster training | Allows higher learning rates |
| Regularization | Slight regularization effect (noise from batch stats) |
| Reduces sensitivity | Less sensitive to weight initialization |
| Smoother optimization | More stable gradients |

## Early Stopping

Stop training when validation loss stops improving.

```
Loss
  │
  │\
  │ \
  │  \    Training         Best model
  │   \_______________         ↓
  │      \            \       ★
  │       \__Validation \___/  \____
  │           \_____/           ↑
  │                          Overfitting begins
  └────────────────────────────────────→ Epochs
```

### Implementation

```python
def train_with_early_stopping(model, X_train, y_train, X_val, y_val,
                               max_epochs=1000, patience=10):
    best_val_loss = float('inf')
    best_weights = None
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        # Train one epoch
        model.fit(X_train, y_train, epochs=1, learning_rate=0.01)
        
        # Evaluate on validation set
        val_predictions = model.predict(X_val)
        val_loss = np.mean((val_predictions - y_val) ** 2)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [layer.parameters() for layer in model.layers 
                           if hasattr(layer, 'parameters')]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            # Restore best weights
            restore_weights(model, best_weights)
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
    
    return model
```

## Data Augmentation

Artificially increase training data variety by applying transformations.

### For Images

```python
def augment_image(image):
    """Apply random augmentations to an image"""
    
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
    
    # Random rotation (-15 to +15 degrees)
    angle = np.random.uniform(-15, 15)
    image = rotate(image, angle)
    
    # Random zoom (90% to 110%)
    scale = np.random.uniform(0.9, 1.1)
    image = zoom(image, scale)
    
    # Random brightness adjustment
    brightness = np.random.uniform(0.8, 1.2)
    image = image * brightness
    
    # Add noise
    noise = np.random.normal(0, 0.05, image.shape)
    image = image + noise
    
    return np.clip(image, 0, 1)
```

### For Text/Sequences

```python
def augment_text(text):
    """Apply random augmentations to text"""
    words = text.split()
    
    # Random word swap
    if len(words) > 2 and np.random.rand() > 0.5:
        idx = np.random.randint(len(words) - 1)
        words[idx], words[idx+1] = words[idx+1], words[idx]
    
    # Random word deletion
    if len(words) > 3 and np.random.rand() > 0.7:
        idx = np.random.randint(len(words))
        words.pop(idx)
    
    # Synonym replacement (with external dictionary)
    # ...
    
    return ' '.join(words)
```

## Weight Initialization

Proper initialization prevents vanishing/exploding gradients at the start of training.

### Xavier/Glorot Initialization

For sigmoid/tanh activations:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

### He Initialization

For ReLU activations:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

### In NeuraCommand

```python
from neura_command.layers.dense import Dense

# He initialization (default)
layer = Dense(784, 256, weight_init='he')

# Xavier initialization
layer = Dense(784, 256, weight_init='xavier')
```

## Gradient Clipping

Prevent exploding gradients by limiting gradient magnitude.

### Clip by Value

```python
def clip_by_value(gradients, clip_value=1.0):
    """Clip each gradient element to [-clip_value, clip_value]"""
    return [np.clip(g, -clip_value, clip_value) for g in gradients]
```

### Clip by Norm

```python
def clip_by_norm(gradients, max_norm=1.0):
    """Clip gradient if its norm exceeds max_norm"""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        gradients = [g * scale for g in gradients]
    
    return gradients
```

## Label Smoothing

Instead of hard labels (0 or 1), use soft labels to reduce overconfidence.

```python
def smooth_labels(labels, smoothing=0.1):
    """Apply label smoothing"""
    num_classes = labels.shape[1]
    return labels * (1 - smoothing) + smoothing / num_classes

# Example:
# Hard label:   [0, 0, 1, 0, 0]
# Soft label:   [0.02, 0.02, 0.92, 0.02, 0.02]
```

## Noise Injection

Add noise to inputs, weights, or gradients for regularization.

### Input Noise

```python
def add_input_noise(x, std=0.1):
    """Add Gaussian noise to inputs during training"""
    noise = np.random.normal(0, std, x.shape)
    return x + noise
```

### Weight Noise

```python
def add_weight_noise(weights, std=0.01):
    """Add noise to weights during training"""
    noise = np.random.normal(0, std, weights.shape)
    return weights + noise
```

## Regularization Comparison

| Technique | Prevents | Overhead | When to Use |
|-----------|----------|----------|-------------|
| L2 | Large weights | Low | Default choice |
| Dropout | Co-adaptation | Medium | Deep networks |
| BatchNorm | Covariate shift | Medium | Deep networks |
| Early Stopping | Overtraining | Low | Always |
| Data Augmentation | Data scarcity | Variable | When data is limited |
| Gradient Clipping | Exploding gradients | Low | RNNs, deep networks |

## Combining Regularization Techniques

```python
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU
from neura_command.layers.dropout import Dropout
from neura_command.layers.batchnorm import BatchNorm

# Modern regularized network
model = Sequential([
    Dense(784, 256, weight_init='he'),
    BatchNorm(),
    ReLU(),
    Dropout(0.3),
    
    Dense(256, 128, weight_init='he'),
    BatchNorm(),
    ReLU(),
    Dropout(0.3),
    
    Dense(128, 10)
])

# Training with L2 regularization, early stopping, and data augmentation
def train_regularized(model, X_train, y_train, X_val, y_val):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(100):
        # Data augmentation
        X_aug = augment_batch(X_train)
        
        # Train (L2 included in optimizer)
        model.fit(X_aug, y_train, epochs=1, 
                 learning_rate=0.001, batch_size=32)
        
        # Validate
        val_predictions = model.predict(X_val)
        val_loss = compute_loss(val_predictions, y_val)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping!")
                break
```

## Key Takeaways

1. **L2 regularization** shrinks weights; L1 creates sparsity
2. **Dropout** prevents co-adaptation by randomly dropping neurons
3. **Batch normalization** stabilizes training and adds regularization
4. **Early stopping** prevents overtraining
5. **Combine techniques** for best results

## Exercises

1. Train a network with and without dropout, compare results
2. Implement early stopping for a classification task
3. Experiment with different L2 regularization strengths

## Next Steps

- [12_practical_training_tips.md](12_practical_training_tips.md) - Best practices for training
