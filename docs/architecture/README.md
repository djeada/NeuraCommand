# NeuraCommand Architecture Documentation

This documentation series provides comprehensive guides on neural network architectures, from fundamental concepts to advanced topics. Each document builds upon previous ones, creating a progressive learning path.

## Documentation Overview

```
Start Here ─────────────────────────────────────────────────────────────────→
│
├──→ 01 Introduction ─→ 02 Perceptron ─→ 03 Activations ─→ 04 Gradient ─→ │
│                                                                           │
│    05 Backprop ─→ 06 MLPs ─→ 07 CNNs ─→ 08 RNNs ─→ 09 Transformers ─→   │
│                                                                           │
│    10 Reinforcement ─→ 11 Regularization ─→ 12 Practical Tips          ←─│
│                                                         │
└────────────────────────────────────────────────────────→ Complete! 🎉
```

## Table of Contents

### Fundamentals

| # | Topic | Description |
|---|-------|-------------|
| [01](01_introduction_to_neural_networks.md) | **Introduction to Neural Networks** | Overview of neural networks, NeuraCommand features, and the building blocks |
| [02](02_perceptron_fundamentals.md) | **Perceptron Fundamentals** | Single neurons, hand calculations vs NeuraCommand implementation |
| [03](03_activation_functions.md) | **Activation Functions** | Sigmoid, ReLU, Tanh, Softmax, and when to use each |
| [04](04_gradient_descent_optimization.md) | **Gradient Descent Optimization** | Optimization algorithms, learning rates, and advanced optimizers |
| [05](05_backpropagation.md) | **Backpropagation** | The learning algorithm with complete hand calculations |

### Network Architectures

| # | Topic | Description |
|---|-------|-------------|
| [06](06_multilayer_perceptrons.md) | **Multi-Layer Perceptrons** | Deep feedforward networks, XOR problem, architecture design |
| [07](07_convolutional_neural_networks.md) | **Convolutional Neural Networks** | CNNs for image processing, convolution, pooling, famous architectures |
| [08](08_recurrent_neural_networks.md) | **Recurrent Neural Networks** | RNNs, LSTMs, GRUs for sequential data |
| [09](09_transformer_architecture.md) | **Transformer Architecture** | Attention mechanisms, self-attention, modern NLP architectures |

### Advanced Topics

| # | Topic | Description |
|---|-------|-------------|
| [10](10_reinforcement_learning.md) | **Reinforcement Learning** | Q-learning, DQN, policy gradients, actor-critic methods |
| [11](11_regularization_techniques.md) | **Regularization Techniques** | Dropout, batch normalization, L1/L2, early stopping |
| [12](12_practical_training_tips.md) | **Practical Training Tips** | Data preparation, debugging, hyperparameter tuning |

## Quick Start

### For Beginners

Start with the fundamentals in order:

1. Read [01_introduction_to_neural_networks.md](01_introduction_to_neural_networks.md) to understand the basics
2. Work through [02_perceptron_fundamentals.md](02_perceptron_fundamentals.md) and try the hand calculations
3. Continue sequentially through documents 03-06

### For Practitioners

Jump to the topic you need:

- **Image Processing**: [07_convolutional_neural_networks.md](07_convolutional_neural_networks.md)
- **Sequential Data**: [08_recurrent_neural_networks.md](08_recurrent_neural_networks.md)
- **NLP/Attention**: [09_transformer_architecture.md](09_transformer_architecture.md)
- **Game AI/Agents**: [10_reinforcement_learning.md](10_reinforcement_learning.md)
- **Overfitting Issues**: [11_regularization_techniques.md](11_regularization_techniques.md)
- **Training Problems**: [12_practical_training_tips.md](12_practical_training_tips.md)

## Learning Approach

Each document follows a consistent structure:

1. **Introduction**: Why this topic matters
2. **Concepts**: Theory with ASCII diagrams and formulas
3. **Hand Calculations**: Work through examples step by step
4. **NeuraCommand Examples**: Implement using the library
5. **Key Takeaways**: Summary of important points
6. **Exercises**: Practice problems
7. **Next Steps**: Links to related documents

## Example: Your First Neural Network

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU, Softmax

# Create a neural network
model = Sequential([
    Dense(784, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
])

# Train on your data
model.fit(X_train, y_train, epochs=100, learning_rate=0.01)

# Make predictions
predictions = model.predict(X_test)
```

## Prerequisites

To get the most from this documentation:

- **Python**: Basic programming knowledge
- **NumPy**: Array operations and broadcasting
- **Mathematics**: Linear algebra and calculus basics

## Related Resources

- [Main NeuraCommand README](../../README.md)
- [Concept Demos](../../demo/concept_demos/README.md)
- [Practical Demos](../../demo/practical_demos/README.md)

## Contributing

We welcome contributions to improve this documentation! Please:

1. Follow the existing document structure
2. Include both theory and practical examples
3. Add ASCII diagrams for visual concepts
4. Provide hand calculations where appropriate
5. Include NeuraCommand code examples

## References

1. Nielsen, M. (2015). *Neural Networks and Deep Learning*
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*
3. Vaswani, A. et al. (2017). *Attention Is All You Need*
4. Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*
