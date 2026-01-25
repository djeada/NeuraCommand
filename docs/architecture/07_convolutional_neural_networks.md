# Convolutional Neural Networks (CNNs)

## Introduction

Convolutional Neural Networks are specialized neural networks designed for processing structured grid data, particularly images. They automatically learn spatial hierarchies of features through the use of convolutional layers.

## Why CNNs for Images?

### The Problem with MLPs for Images

Using a standard MLP for a 224×224 RGB image:
- Input size: 224 × 224 × 3 = 150,528 neurons
- First hidden layer (1024 neurons): 150,528 × 1024 = 154 million parameters!

This leads to:
- Huge memory requirements
- Overfitting due to too many parameters
- No understanding of spatial relationships

### How CNNs Solve This

CNNs use:
1. **Local connectivity**: Neurons connect only to nearby regions
2. **Parameter sharing**: Same weights used across the image
3. **Spatial invariance**: Features detected anywhere in the image

```
MLP:                          CNN:
Every pixel →                 Local regions →
Every neuron                  Shared filters

150,528 connections           3×3 filter = 9 weights
per neuron                    shared across image!
```

## CNN Building Blocks

### 1. Convolutional Layer

A convolutional layer slides a small filter (kernel) across the input, computing dot products.

```
Input Image (5×5):          Filter (3×3):         Feature Map:
┌───┬───┬───┬───┬───┐       ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 2 │ 1 │       │ 1 │ 0 │ 1 │       ┌───┬───┬───┐
├───┼───┼───┼───┼───┤       ├───┼───┼───┤       │ 4 │ 3 │ 4 │
│ 0 │ 1 │ 0 │ 1 │ 0 │   *   │ 0 │ 1 │ 0 │   =   ├───┼───┼───┤
├───┼───┼───┼───┼───┤       ├───┼───┼───┤       │ 2 │ 4 │ 3 │
│ 1 │ 0 │ 1 │ 0 │ 1 │       │ 1 │ 0 │ 1 │       ├───┼───┼───┤
├───┼───┼───┼───┼───┤       └───┴───┴───┘       │ 4 │ 3 │ 4 │
│ 2 │ 1 │ 0 │ 1 │ 2 │                           └───┴───┴───┘
├───┼───┼───┼───┼───┤       
│ 1 │ 0 │ 1 │ 2 │ 1 │       
└───┴───┴───┴───┴───┘       
```

**Convolution Calculation:**

$$\text{Output}[i,j] = \sum_{m}\sum_{n} \text{Input}[i+m, j+n] \times \text{Filter}[m, n]$$

**Example (top-left corner):**
```
Input region:     Filter:
┌───┬───┬───┐     ┌───┬───┬───┐
│ 1 │ 0 │ 1 │     │ 1 │ 0 │ 1 │
├───┼───┼───┤  *  ├───┼───┼───┤  = 1×1 + 0×0 + 1×1 + 
│ 0 │ 1 │ 0 │     │ 0 │ 1 │ 0 │    0×0 + 1×1 + 0×0 +
├───┼───┼───┤     ├───┼───┼───┤    1×1 + 0×0 + 1×1 = 4
│ 1 │ 0 │ 1 │     │ 1 │ 0 │ 1 │
└───┴───┴───┘     └───┴───┴───┘
```

### 2. Pooling Layer

Pooling reduces spatial dimensions while keeping important features.

**Max Pooling (2×2):**
```
Input (4×4):              Output (2×2):
┌───┬───┬───┬───┐         ┌───┬───┐
│ 1 │ 3 │ 2 │ 4 │         │ 3 │ 4 │  (max of each 2×2 region)
├───┼───┼───┼───┤    →    ├───┼───┤
│ 5 │ 2 │ 1 │ 8 │         │ 6 │ 8 │
├───┼───┼───┼───┤         └───┴───┘
│ 6 │ 1 │ 4 │ 2 │
├───┼───┼───┼───┤
│ 3 │ 4 │ 5 │ 1 │
└───┴───┴───┴───┘
```

**Average Pooling (2×2):**
```
┌───┬───┐           (1+3+5+2)/4 = 2.75
│ 1 │ 3 │
├───┼───┤     →     2.75
│ 5 │ 2 │
└───┴───┘
```

### 3. Flatten Layer

Converts 2D/3D feature maps to 1D vector for fully connected layers.

```
Feature Maps (2×2×3):      Flattened (12 elements):
                           
Channel 1:  Channel 2:     ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
┌───┬───┐   ┌───┬───┐      │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │
│ 1 │ 2 │   │ 5 │ 6 │  →   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
├───┼───┤   ├───┼───┤
│ 3 │ 4 │   │ 7 │ 8 │
└───┴───┘   └───┴───┘
```

## CNN Architecture Patterns

### Basic CNN Pattern

```
Input → [Conv → ReLU → Pool] × N → Flatten → Dense → Output

Example for MNIST (28×28 grayscale):

Input: 28×28×1
    ↓
Conv2D(32, 3×3) → 26×26×32
    ↓
ReLU
    ↓
MaxPool(2×2) → 13×13×32
    ↓
Conv2D(64, 3×3) → 11×11×64
    ↓
ReLU
    ↓
MaxPool(2×2) → 5×5×64
    ↓
Flatten → 1600
    ↓
Dense(128) → 128
    ↓
ReLU
    ↓
Dense(10) → 10
    ↓
Softmax → 10 classes
```

### Conceptual NeuraCommand CNN

While NeuraCommand focuses on dense layers, you can conceptually build CNN-like structures:

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.flatten import Flatten
from neura_command.layers.activation import ReLU, Softmax

# After flattening image features, use MLP classifier
# (Actual convolution would be done with separate preprocessing)
classifier = Sequential([
    Flatten(),           # Flatten input if needed
    Dense(784, 256),     # 28×28 = 784 flattened
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
])
```

## Convolution Parameters

### Stride

How many pixels the filter moves at each step.

```
Stride = 1:                    Stride = 2:
┌─┬─┬─┬─┬─┐                    ┌─┬─┬─┬─┬─┐
│■│■│■│ │ │  →  Output: 3×3    │■│■│■│ │ │  →  Output: 2×2
├─┼─┼─┼─┼─┤                    ├─┼─┼─┼─┼─┤
│■│■│■│ │ │                    │■│■│■│ │ │
├─┼─┼─┼─┼─┤                    ├─┼─┼─┼─┼─┤
│■│■│■│ │ │                    │ │ │■│■│■│
├─┼─┼─┼─┼─┤                    ├─┼─┼─┼─┼─┤
│ │ │ │ │ │                    │ │ │■│■│■│
├─┼─┼─┼─┼─┤                    ├─┼─┼─┼─┼─┤
│ │ │ │ │ │                    │ │ │■│■│■│
└─┴─┴─┴─┴─┘                    └─┴─┴─┴─┴─┘
```

### Padding

Adding zeros around the input to control output size.

```
Valid (no padding):           Same (zero padding):
Input: 5×5                    Input: 5×5 → Padded: 7×7
Filter: 3×3                   Filter: 3×3
Output: 3×3                   Output: 5×5

┌─┬─┬─┬─┬─┐                   ┌─┬─┬─┬─┬─┬─┬─┐
│ │ │ │ │ │                   │0│0│0│0│0│0│0│
├─┼─┼─┼─┼─┤                   ├─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │  →  3×3           │0│ │ │ │ │ │0│  →  5×5
├─┼─┼─┼─┼─┤                   ├─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │                   │0│ │ │ │ │ │0│
├─┼─┼─┼─┼─┤                   ...
│ │ │ │ │ │
├─┼─┼─┼─┼─┤
│ │ │ │ │ │
└─┴─┴─┴─┴─┘
```

**Output Size Formula:**

$$\text{Output Size} = \frac{\text{Input Size} - \text{Filter Size} + 2 \times \text{Padding}}{\text{Stride}} + 1$$

## Feature Hierarchy

CNNs learn hierarchical features:

```
Layer 1 (Early):     Layer 2 (Middle):    Layer 3 (Deep):
Edges, corners       Textures, shapes     Objects, faces

    ╱  ╲  │          ╭──╮   ◠◡◠         🐱  🏠  👤
   ─  ╲  ╱           │  │   ≋≋≋         
   │  ─  ╲           ╰──╯   ○ ○         

Simple patterns  →   Combinations   →   Complex objects
```

## Famous CNN Architectures

### LeNet-5 (1998)
The original CNN for digit recognition:
```
Input → Conv → Pool → Conv → Pool → FC → FC → Output
28×28    20     10     50     5      120   84   10
```

### AlexNet (2012)
Deep CNN that won ImageNet:
- 5 convolutional layers
- 3 fully connected layers
- ReLU activation (key innovation)
- Dropout for regularization

### VGG (2014)
Deeper networks with small (3×3) filters:
- VGG-16: 16 layers
- VGG-19: 19 layers
- Principle: Deeper is better (with 3×3 filters)

### ResNet (2015)
Skip connections to train very deep networks:
```
        ┌─────────────────┐
        │                 │
x ──→ Conv → Conv ──(+)──→ output
        │         ↑
        └─────────┘
          Identity
          shortcut
```

## Hand Calculation: Convolution

**Input (4×4):**
```
1  2  3  0
0  1  2  3
3  0  1  2
2  3  0  1
```

**Filter (2×2):**
```
1  0
0  1
```

**Convolution (stride=1, no padding):**

```
Position (0,0): 1×1 + 2×0 + 0×0 + 1×1 = 2
Position (0,1): 2×1 + 3×0 + 1×0 + 2×1 = 4
Position (0,2): 3×1 + 0×0 + 2×0 + 3×1 = 6
Position (1,0): 0×1 + 1×0 + 3×0 + 0×1 = 0
Position (1,1): 1×1 + 2×0 + 0×0 + 1×1 = 2
Position (1,2): 2×1 + 3×0 + 1×0 + 2×1 = 4
Position (2,0): 3×1 + 0×0 + 2×0 + 3×1 = 6
Position (2,1): 0×1 + 1×0 + 3×0 + 0×1 = 0
Position (2,2): 1×1 + 2×0 + 0×0 + 1×1 = 2

Output (3×3):
2  4  6
0  2  4
6  0  2
```

## Using CNNs in Practice

### Preprocessing Images

```python
import numpy as np

def preprocess_image(image):
    """Preprocess image for CNN input"""
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Standardize (zero mean, unit variance)
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / (std + 1e-7)
    
    return image

# For batch processing
def preprocess_batch(images):
    return np.array([preprocess_image(img) for img in images])
```

### Data Augmentation

Increase training data variety:

```python
def augment_image(image):
    """Apply random augmentations"""
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
    
    # Random rotation (small angle)
    # Random crop
    # Color jittering
    
    return image
```

## CNN vs MLP Comparison

| Aspect | MLP | CNN |
|--------|-----|-----|
| Parameters | Many (all-to-all) | Few (weight sharing) |
| Spatial awareness | None | Built-in |
| Translation invariance | None | Yes (pooling) |
| Best for | Tabular data | Images, grids |
| Memory | High | Lower |

## Key Takeaways

1. **Convolution** extracts local features with shared weights
2. **Pooling** reduces dimensions and adds translation invariance
3. **Deep CNNs** learn hierarchical features (edges → shapes → objects)
4. **Architecture patterns**: Conv-ReLU-Pool stacks followed by dense layers
5. **Parameter efficiency**: CNNs need far fewer parameters than MLPs for images

## Exercises

1. Compute a 3×3 convolution by hand on a 5×5 input
2. Calculate the output size for different stride and padding values
3. Research and summarize one modern CNN architecture (ResNet, EfficientNet, etc.)

## Next Steps

- [08_recurrent_neural_networks.md](08_recurrent_neural_networks.md) - Learn about RNNs for sequences
- [09_transformer_architecture.md](09_transformer_architecture.md) - Modern attention-based architectures
