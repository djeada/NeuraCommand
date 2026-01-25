# Non-Linear Classification with NeuraCommand

## Introduction

This tutorial demonstrates how to classify non-linearly separable datasets using neural networks with NeuraCommand. We use the `make_moons` dataset from scikit-learn, which generates two interleaving half circles that cannot be separated by a linear decision boundary.

![Decision Boundary Example](https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png)

## Problem Overview

The moons dataset is a classic example of a non-linearly separable classification problem. It consists of two classes arranged in interleaving half-moon shapes. A simple linear classifier (like a perceptron) cannot accurately classify this data because no straight line can separate the two classes.

### Why Neural Networks?

Neural networks with hidden layers and non-linear activation functions can learn complex, non-linear decision boundaries. By stacking multiple layers with activation functions like ReLU, the network can approximate any continuous function, including the curved boundary needed to separate the moons.

**Key Concepts:**
- **Non-linear activation functions**: Functions like ReLU and Sigmoid introduce non-linearity, allowing the network to learn curved decision boundaries.
- **Hidden layers**: Additional layers between input and output enable the network to learn hierarchical representations.
- **Universal approximation**: A neural network with at least one hidden layer can approximate any continuous function.

## Dataset

The `make_moons` dataset generates:
- **Features**: 2D coordinates (x, y)
- **Labels**: Binary (0 or 1) indicating which moon the point belongs to
- **Noise**: Gaussian noise can be added to make the problem more challenging

### Sample Data Points

| Feature 1 | Feature 2 | Label |
|-----------|-----------|-------|
| -0.75     | 0.50      | 0     |
| 1.25      | 0.25      | 1     |
| 0.50      | 0.85      | 0     |
| 0.00      | -0.35     | 1     |

## Neural Network Architecture

We use a simple but effective architecture:

```
Input (2 features)
    ↓
Dense(2 → 16) + ReLU
    ↓
Dense(16 → 8) + ReLU
    ↓
Dense(8 → 1) + Sigmoid
    ↓
Output (probability 0-1)
```

### Layer Breakdown

| Layer | Input Size | Output Size | Activation | Parameters |
|-------|------------|-------------|------------|------------|
| Dense 1 | 2 | 16 | ReLU | 48 |
| Dense 2 | 16 | 8 | ReLU | 136 |
| Dense 3 | 8 | 1 | Sigmoid | 9 |
| **Total** | - | - | - | **193** |

## How to Run

### Prerequisites

Make sure you have the necessary dependencies installed:

```bash
pip install numpy matplotlib scikit-learn
```

Also ensure that NeuraCommand is installed or that the `src` directory is in your Python path.

### Running the Demo

Navigate to the demo directory and run:

```bash
cd demo/practical_demos/non_linear_classification
python main.py
```

### Expected Output

1. **Dataset Visualization**: A scatter plot showing the two moons
2. **Model Summary**: Architecture overview with parameter counts
3. **Training Progress**: Loss values per epoch
4. **Training History Plot**: Graph of training and validation loss
5. **Evaluation Metrics**: Test accuracy and classification report
6. **Decision Boundary Plot**: Visualization of how the model separates the classes

## Understanding the Code

### 1. Data Preparation

```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# Standardize features (important for neural networks!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
```

**Why standardize?** Neural networks train faster and more reliably when features have zero mean and unit variance.

### 2. Model Building

```python
from neura_command.network.sequential import Sequential
from neura_command.layers import Dense, ReLU, Sigmoid

model = Sequential(
    layers=[
        Dense(2, 16),
        ReLU(),
        Dense(16, 8),
        ReLU(),
        Dense(8, 1),
        Sigmoid(),
    ],
    optimizer=Adam(learning_rate=0.01),
    loss=MSELoss(),
)
```

### 3. Training

```python
history = model.train(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
)
```

### 4. Evaluation

```python
predictions_prob = model.predict(X_test)
predictions = (predictions_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, predictions)
```

## Key Takeaways

1. **Linear vs Non-Linear Problems**: Some classification problems require non-linear decision boundaries that simple linear models cannot learn.

2. **Hidden Layers Matter**: The hidden layers with non-linear activations enable the network to learn complex patterns.

3. **ReLU Activation**: Rectified Linear Unit (ReLU) is a popular choice for hidden layers because it:
   - Introduces non-linearity
   - Avoids vanishing gradient problem
   - Is computationally efficient

4. **Sigmoid for Binary Classification**: The sigmoid activation in the output layer squashes predictions to [0, 1], making it suitable for binary classification.

5. **Feature Scaling**: Always standardize your features before training neural networks for better convergence.

## Troubleshooting

### Low Accuracy
- Increase the number of epochs
- Try a smaller learning rate
- Add more neurons to hidden layers

### Slow Training
- Reduce batch size for more frequent updates
- Increase learning rate (carefully)
- Check if data is properly normalized

### Overfitting
- Add dropout layers
- Reduce model complexity
- Use early stopping

## Next Steps

After mastering this tutorial, you can:

1. **Experiment with different datasets**: Try `make_circles`, `make_blobs`, or real-world datasets
2. **Modify the architecture**: Add more layers, change hidden layer sizes
3. **Try different optimizers**: SGD with momentum, RMSprop, AdaGrad
4. **Add regularization**: Implement dropout or L2 regularization
5. **Explore multi-class classification**: Extend to problems with more than 2 classes

## References

- [make_moons Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
- [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
