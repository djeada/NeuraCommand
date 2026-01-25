# Practical Training Tips

## Introduction

This document provides practical guidance for training neural networks effectively. These tips come from practical experience and can save you significant time in debugging and improving your models.

## Training Pipeline Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Data Prep  │ ──→ │   Model     │ ──→ │  Training   │
│             │     │   Design    │     │   Loop      │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      ▼                   ▼                   ▼
• Clean data         • Architecture      • Learning rate
• Normalize          • Layer sizes       • Batch size
• Split data         • Activations       • Epochs
• Augment            • Regularization    • Monitoring
```

## 1. Data Preparation

### Always Normalize Your Data

Neural networks work best with normalized inputs:

```python
import numpy as np

# Method 1: Min-Max Scaling (0 to 1)
def min_max_scale(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8)

# Method 2: Standardization (mean=0, std=1)
def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

# Example
X_train = np.random.randn(1000, 10) * 100 + 50  # Unnormalized
X_train_normalized = standardize(X_train)

print(f"Before: mean={X_train.mean():.2f}, std={X_train.std():.2f}")
print(f"After: mean={X_train_normalized.mean():.2f}, std={X_train_normalized.std():.2f}")
```

### Proper Train/Validation/Test Split

```python
def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15):
    """Split data into train, validation, and test sets"""
    n = len(X)
    indices = np.random.permutation(n)
    
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    
    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]
    
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])

# Usage
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
```

### Check for Data Issues

```python
def validate_data(X, y):
    """Check common data issues"""
    issues = []
    
    # Check for NaN
    if np.isnan(X).any():
        issues.append(f"Found {np.isnan(X).sum()} NaN values in X")
    
    # Check for Inf
    if np.isinf(X).any():
        issues.append(f"Found {np.isinf(X).sum()} Inf values in X")
    
    # Check for constant features
    constant_cols = np.where(X.std(axis=0) == 0)[0]
    if len(constant_cols) > 0:
        issues.append(f"Constant features at indices: {constant_cols}")
    
    # Check for class imbalance (classification)
    if len(y.shape) == 1 or y.shape[1] == 1:
        unique, counts = np.unique(y, return_counts=True)
        imbalance = counts.max() / counts.min()
        if imbalance > 5:
            issues.append(f"Class imbalance ratio: {imbalance:.1f}")
    
    if issues:
        print("⚠️ Data issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Data looks good!")
    
    return len(issues) == 0
```

## 2. Model Architecture Guidelines

### Starting Architecture

Start simple and add complexity:

```python
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU, Softmax

# Start simple
simple_model = Sequential([
    Dense(input_size, 64),
    ReLU(),
    Dense(64, output_size)
])

# If underfitting, add capacity
medium_model = Sequential([
    Dense(input_size, 128),
    ReLU(),
    Dense(128, 64),
    ReLU(),
    Dense(64, output_size)
])

# If still underfitting, go deeper
complex_model = Sequential([
    Dense(input_size, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 64),
    ReLU(),
    Dense(64, output_size)
])
```

### Layer Sizing Rules of Thumb

```
Input ──→ Larger ──→ Smaller ──→ Output

Examples:
• Classification (10 classes):
  784 → 256 → 128 → 64 → 10

• Regression:
  50 → 128 → 64 → 32 → 1

• General rule: Each layer 0.5x to 2x previous
```

### Activation Function Selection

| Task | Hidden Layers | Output Layer |
|------|---------------|--------------|
| Classification (binary) | ReLU | Sigmoid |
| Classification (multi) | ReLU | Softmax |
| Regression | ReLU | None (Linear) |
| Deep networks | ReLU or LeakyReLU | Task-dependent |

## 3. Hyperparameter Selection

### Learning Rate

The most important hyperparameter:

```python
# Learning rate finder
def find_learning_rate(model, X, y, min_lr=1e-7, max_lr=10, steps=100):
    """Find optimal learning rate using LR range test"""
    lr_mult = (max_lr / min_lr) ** (1 / steps)
    lr = min_lr
    losses = []
    lrs = []
    
    for step in range(steps):
        # Train one step
        predictions = model.predict(X[:32])
        loss = np.mean((predictions - y[:32]) ** 2)
        
        model.fit(X[:32], y[:32], epochs=1, learning_rate=lr, batch_size=32)
        
        losses.append(loss)
        lrs.append(lr)
        lr *= lr_mult
        
        # Stop if loss explodes
        if len(losses) > 1 and losses[-1] > losses[0] * 10:
            break
    
    # Find LR with steepest decrease
    gradients = np.gradient(losses)
    best_idx = np.argmin(gradients)
    
    print(f"Suggested learning rate: {lrs[best_idx]:.2e}")
    return lrs[best_idx]
```

### Common Learning Rates

| Optimizer | Typical Range |
|-----------|---------------|
| SGD | 0.01 - 0.1 |
| SGD + Momentum | 0.001 - 0.01 |
| Adam | 0.0001 - 0.001 |

### Batch Size

```
Small batch (16-32):
  ✓ Better generalization
  ✓ More regularization
  ✗ Slower training
  ✗ Noisier gradients

Large batch (256-1024):
  ✓ Faster training
  ✓ More stable gradients
  ✗ May need LR warmup
  ✗ May generalize worse
```

**Rule of thumb:** Start with 32 or 64, adjust based on GPU memory and convergence.

## 4. Training Loop Best Practices

### Complete Training Loop

```python
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, learning_rate=0.001, batch_size=32,
                patience=10):
    """Full training loop with best practices"""
    
    best_val_loss = float('inf')
    best_weights = None
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Training loop
        train_losses = []
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            predictions = model.predict(X_batch)
            batch_loss = np.mean((predictions - y_batch) ** 2)
            train_losses.append(batch_loss)
            
            # Backward pass and update
            model.fit(X_batch, y_batch, epochs=1, 
                     learning_rate=learning_rate, batch_size=len(X_batch))
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        
        # Validation
        val_predictions = model.predict(X_val)
        val_loss = np.mean((val_predictions - y_val) ** 2)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best weights (conceptual)
            best_weights = save_weights(model)
        else:
            epochs_without_improvement += 1
        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            restore_weights(model, best_weights)
            break
    
    return history
```

### Learning Rate Scheduling

```python
def learning_rate_schedule(epoch, initial_lr=0.001, schedule='step'):
    """Compute learning rate for current epoch"""
    
    if schedule == 'step':
        # Reduce by 10x every 30 epochs
        return initial_lr * (0.1 ** (epoch // 30))
    
    elif schedule == 'exponential':
        # Reduce by 5% each epoch
        return initial_lr * (0.95 ** epoch)
    
    elif schedule == 'cosine':
        # Cosine annealing
        total_epochs = 100
        return initial_lr * (1 + np.cos(np.pi * epoch / total_epochs)) / 2
    
    elif schedule == 'warmup':
        # Linear warmup for 5 epochs, then constant
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        return initial_lr
    
    return initial_lr
```

## 5. Debugging Training

### Common Issues and Solutions

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Loss = NaN | Learning rate too high | Reduce LR by 10x |
| Loss = NaN | Bad data (NaN/Inf) | Clean data |
| Loss doesn't decrease | LR too low | Increase LR |
| Loss doesn't decrease | Bug in implementation | Check gradients |
| Validation loss increases | Overfitting | Add regularization |
| Training is unstable | LR too high | Reduce LR, add batch norm |

### Gradient Checking

```python
def check_gradients(model, X, y, epsilon=1e-5):
    """Verify gradients using numerical approximation"""
    
    # Get analytical gradients
    predictions = model.predict(X)
    loss = np.mean((predictions - y) ** 2)
    # Assume model stores gradients after backward pass
    
    # Get first layer weights
    layer = model.layers[0]
    weights = layer.weights.copy()
    analytical_grad = layer._grad_weights
    
    # Compute numerical gradients
    numerical_grad = np.zeros_like(weights)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            # f(w + epsilon)
            weights[i, j] += epsilon
            layer.weights = weights
            loss_plus = np.mean((model.predict(X) - y) ** 2)
            
            # f(w - epsilon)
            weights[i, j] -= 2 * epsilon
            layer.weights = weights
            loss_minus = np.mean((model.predict(X) - y) ** 2)
            
            # Numerical gradient
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore
            weights[i, j] += epsilon
            layer.weights = weights
    
    # Compare
    difference = np.abs(analytical_grad - numerical_grad).max()
    relative_error = difference / (np.abs(analytical_grad).max() + 1e-8)
    
    print(f"Max absolute difference: {difference:.2e}")
    print(f"Relative error: {relative_error:.2e}")
    
    if relative_error < 1e-5:
        print("✓ Gradients look correct!")
    else:
        print("⚠️ Gradient mismatch detected!")
```

### Sanity Checks

```python
def sanity_checks(model, X, y):
    """Quick sanity checks before full training"""
    
    print("Running sanity checks...")
    
    # 1. Can model overfit a single batch?
    print("\n1. Single batch overfit test:")
    X_batch, y_batch = X[:10], y[:10]
    for i in range(100):
        model.fit(X_batch, y_batch, epochs=1, learning_rate=0.1)
    predictions = model.predict(X_batch)
    loss = np.mean((predictions - y_batch) ** 2)
    print(f"   Loss after 100 steps: {loss:.4f}")
    if loss < 0.01:
        print("   ✓ Model can overfit single batch")
    else:
        print("   ⚠️ Model cannot overfit - check architecture/LR")
    
    # 2. Does loss decrease with training?
    print("\n2. Loss decrease test:")
    initial_loss = np.mean((model.predict(X[:100]) - y[:100]) ** 2)
    for i in range(10):
        model.fit(X[:100], y[:100], epochs=1, learning_rate=0.01)
    final_loss = np.mean((model.predict(X[:100]) - y[:100]) ** 2)
    print(f"   Initial: {initial_loss:.4f} → Final: {final_loss:.4f}")
    if final_loss < initial_loss:
        print("   ✓ Loss decreases with training")
    else:
        print("   ⚠️ Loss not decreasing - check gradients")
    
    # 3. Check output range
    print("\n3. Output range check:")
    outputs = model.predict(X[:100])
    print(f"   Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
    print(f"   Target range: [{y[:100].min():.2f}, {y[:100].max():.2f}]")
```

## 6. Monitoring and Visualization

### Training Metrics to Track

```python
class TrainingMonitor:
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'gradient_norm': []
        }
    
    def log(self, epoch, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def plot(self):
        """Plot training curves"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.metrics['train_loss'], label='Train')
        axes[0].plot(self.metrics['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Loss Curves')
        
        # Accuracy
        if self.metrics['train_acc']:
            axes[1].plot(self.metrics['train_acc'], label='Train')
            axes[1].plot(self.metrics['val_acc'], label='Validation')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].set_title('Accuracy Curves')
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        print("\n=== Training Summary ===")
        print(f"Final train loss: {self.metrics['train_loss'][-1]:.4f}")
        print(f"Final val loss: {self.metrics['val_loss'][-1]:.4f}")
        print(f"Best val loss: {min(self.metrics['val_loss']):.4f}")
        print(f"Total epochs: {len(self.metrics['train_loss'])}")
```

## 7. Quick Reference Cheat Sheet

### Before Training

- [ ] Data is normalized
- [ ] No NaN/Inf values
- [ ] Train/val/test split done
- [ ] Class balance checked

### During Training

- [ ] Monitor train AND validation loss
- [ ] Use early stopping
- [ ] Save best model checkpoint
- [ ] Log learning rate if scheduling

### Hyperparameter Tuning Order

1. **Learning rate** (most important)
2. **Batch size**
3. **Network depth/width**
4. **Regularization strength**
5. **Other hyperparameters**

### If Things Go Wrong

```
Loss = NaN?
  → Lower learning rate 10x
  → Check for NaN in data
  → Use gradient clipping

Loss not decreasing?
  → Increase learning rate
  → Check for bugs (gradient check)
  → Increase model capacity

Validation loss increasing?
  → Add regularization (dropout, L2)
  → Reduce model capacity
  → Get more data / augment
```

## Key Takeaways

1. **Data preparation** is crucial - normalize, clean, split properly
2. **Start simple** and add complexity as needed
3. **Learning rate** is the most important hyperparameter
4. **Monitor both training and validation** metrics
5. **Debug systematically** using sanity checks and gradient verification
6. **Save checkpoints** and use early stopping

## Exercises

1. Implement the sanity checks for your own model
2. Train a model with and without learning rate scheduling
3. Create a training monitoring dashboard for your experiments

## Conclusion

This completes the NeuraCommand advanced architecture documentation series. We covered:

1. Introduction to Neural Networks
2. Perceptron Fundamentals
3. Activation Functions
4. Gradient Descent Optimization
5. Backpropagation
6. Multi-Layer Perceptrons
7. Convolutional Neural Networks
8. Recurrent Neural Networks
9. Transformer Architecture
10. Reinforcement Learning
11. Regularization Techniques
12. Practical Training Tips (this document)

Happy training! 🚀
