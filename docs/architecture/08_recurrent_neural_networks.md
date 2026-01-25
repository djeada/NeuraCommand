# Recurrent Neural Networks (RNNs)

## Introduction

Recurrent Neural Networks are designed to process sequential data by maintaining a "memory" of previous inputs. Unlike feedforward networks where information flows in one direction, RNNs have loops that allow information to persist.

## Why RNNs for Sequences?

### The Limitation of Feedforward Networks

Standard MLPs treat each input independently - they have no concept of order or sequence.

```
MLP for Text:           Problem:
                        
"I love this" → MLP →   Same output as "this love I"?
                        Order matters in language!
```

### How RNNs Solve This

RNNs process inputs one at a time while maintaining a hidden state that captures information from previous steps.

```
                    ┌──────┐
         ┌────────→│  h₃  │──────→ y₃
         │         └──────┘
         │              ↑
┌──────┐ │         ┌──────┐
│  h₁  │─┴────────→│  h₂  │
└──────┘           └──────┘
    ↑                   ↑
    │                   │
   x₁                  x₂                 x₃
  "I"               "love"             "this"
```

## RNN Architecture

### Basic RNN Cell

```
                    ┌─────────────┐
                    │             │
                    │    h_{t}    │←──┐
                    │             │   │
                    └──────┬──────┘   │
                           │          │
          ┌────────────────┼──────────┘
          │                │
          ↓                ↓
     ┌─────────────────────────┐
     │                         │
     │   h_t = f(W_h·h_{t-1}   │
     │          + W_x·x_t + b) │
     │                         │
     └────────────┬────────────┘
                  │
                  ↓
                 y_t
```

### Mathematical Formulation

At each time step t:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)$$

$$y_t = W_y h_t + b_y$$

Where:
- $h_t$ = hidden state at time t
- $x_t$ = input at time t
- $y_t$ = output at time t
- $W_h, W_x, W_y$ = weight matrices
- $b_h, b_y$ = bias vectors

## Hand Calculation Example

Let's trace through a simple RNN processing the sequence [1, 2, 3].

### Setup
- Input dimension: 1
- Hidden dimension: 2
- $W_x = [[0.5], [0.3]]$ (2×1)
- $W_h = [[0.1, 0.2], [0.3, 0.1]]$ (2×2)
- $b = [0, 0]$
- Initial hidden state: $h_0 = [0, 0]$
- Activation: tanh

### Step 1: Process x₁ = 1

$$h_1 = \tanh(W_h \cdot h_0 + W_x \cdot x_1 + b)$$

$$h_1 = \tanh\left(\begin{bmatrix}0.1 & 0.2\\0.3 & 0.1\end{bmatrix}\begin{bmatrix}0\\0\end{bmatrix} + \begin{bmatrix}0.5\\0.3\end{bmatrix} \cdot 1\right)$$

$$h_1 = \tanh\left(\begin{bmatrix}0.5\\0.3\end{bmatrix}\right) = \begin{bmatrix}0.462\\0.291\end{bmatrix}$$

### Step 2: Process x₂ = 2

$$h_2 = \tanh(W_h \cdot h_1 + W_x \cdot x_2)$$

$$h_2 = \tanh\left(\begin{bmatrix}0.1 & 0.2\\0.3 & 0.1\end{bmatrix}\begin{bmatrix}0.462\\0.291\end{bmatrix} + \begin{bmatrix}0.5\\0.3\end{bmatrix} \cdot 2\right)$$

$$h_2 = \tanh\left(\begin{bmatrix}0.046 + 0.058\\0.139 + 0.029\end{bmatrix} + \begin{bmatrix}1.0\\0.6\end{bmatrix}\right)$$

$$h_2 = \tanh\left(\begin{bmatrix}1.104\\0.768\end{bmatrix}\right) = \begin{bmatrix}0.802\\0.645\end{bmatrix}$$

### Step 3: Process x₃ = 3

Following the same process:

$$h_3 = \tanh\left(\begin{bmatrix}0.080 + 0.129\\0.241 + 0.065\end{bmatrix} + \begin{bmatrix}1.5\\0.9\end{bmatrix}\right)$$

$$h_3 = \tanh\left(\begin{bmatrix}1.709\\1.206\end{bmatrix}\right) = \begin{bmatrix}0.936\\0.835\end{bmatrix}$$

The final hidden state [0.936, 0.835] encodes information from the entire sequence.

## Types of RNN Architectures

### 1. One-to-One (Standard Neural Network)
```
x → [Network] → y

Use: Single input, single output
Example: Image classification
```

### 2. One-to-Many
```
x → [RNN] → y₁ → y₂ → y₃ → ...

Use: Single input, sequence output
Example: Image captioning
```

### 3. Many-to-One
```
x₁ → x₂ → x₃ → [RNN] → y

Use: Sequence input, single output
Example: Sentiment analysis
```

### 4. Many-to-Many (Same Length)
```
x₁ → x₂ → x₃
 ↓    ↓    ↓
y₁   y₂   y₃

Use: Sequence input, sequence output
Example: POS tagging, video classification
```

### 5. Many-to-Many (Different Lengths)
```
x₁ → x₂ → [Encoder] → [Decoder] → y₁ → y₂ → y₃ → y₄

Use: Sequence-to-sequence
Example: Machine translation
```

## The Vanishing Gradient Problem

RNNs struggle to learn long-range dependencies because gradients become very small over many time steps.

```
Gradient Flow:

t=10     t=9      t=8      t=7      ...     t=1
 ↓        ↓        ↓        ↓                ↓
δ₁₀ → ×0.9 → ×0.9 → ×0.9 → ... → 0.9¹⁰ ≈ 0.35

After 100 steps: 0.9¹⁰⁰ ≈ 0.00003 (vanished!)
```

## Long Short-Term Memory (LSTM)

LSTMs solve the vanishing gradient problem with a gating mechanism.

### LSTM Cell Structure

```
                    ┌─────────────────────────────────────┐
                    │              LSTM Cell              │
                    │                                     │
        ┌───────────┼──────────────────────┐             │
        │           │      c_{t-1}         │             │
        │           │         │            │             │
        │           │    ┌────┴────┐       │             │
        │           │    │    ×    │←─ f_t (forget)      │
        │           │    └────┬────┘       │             │
        │           │         │            │             │
        │           │    ┌────┴────┐       │             │
        │           │    │    +    │←─ i_t × c̃_t        │
        │           │    └────┬────┘       │             │
        │           │         │            │             │
        │           │        c_t ──────────┼─────────────→ c_t
        │           │         │            │
        │  h_{t-1} ─┼─────────┼────────────┤
        │           │    tanh(c_t) × o_t ──┼─────────────→ h_t
        └───────────┼──────────────────────┘
                    │
                    └─────────────────────────────────────┘
                              ↑
                             x_t
```

### LSTM Equations

**Forget Gate**: What to remove from memory
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate**: What new information to store
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Memory**:
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**Cell State Update**:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output Gate**: What to output
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State**:
$$h_t = o_t \odot \tanh(c_t)$$

## Gated Recurrent Unit (GRU)

GRU is a simplified version of LSTM with fewer parameters.

```
GRU Cell:
         
z_t = σ(W_z·[h_{t-1}, x_t])     (Update gate)
r_t = σ(W_r·[h_{t-1}, x_t])     (Reset gate)
h̃_t = tanh(W·[r_t⊙h_{t-1}, x_t])  (Candidate)
h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t   (Output)
```

### LSTM vs GRU Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters | More | Fewer |
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Memory | Separate cell state | Combined |
| Training | Slower | Faster |
| Performance | Often better for long sequences | Often comparable |

## Conceptual Implementation in NeuraCommand

While NeuraCommand focuses on dense layers, here's how you might structure RNN-like processing:

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import Tanh, Softmax

class SimpleRNNCell:
    """Conceptual RNN cell using NeuraCommand components"""
    
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Weights for input and hidden state
        self.W_x = np.random.randn(input_size, hidden_size) * 0.1
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b = np.zeros(hidden_size)
        
    def forward(self, x, h_prev):
        """Single step forward pass"""
        z = np.dot(x, self.W_x) + np.dot(h_prev, self.W_h) + self.b
        h = np.tanh(z)
        return h
    
    def process_sequence(self, sequence):
        """Process entire sequence"""
        h = np.zeros(self.hidden_size)
        hidden_states = []
        
        for x in sequence:
            h = self.forward(x, h)
            hidden_states.append(h.copy())
            
        return np.array(hidden_states)

# Example usage
rnn = SimpleRNNCell(input_size=10, hidden_size=20)
sequence = np.random.randn(5, 10)  # 5 time steps, 10 features
hidden_states = rnn.process_sequence(sequence)
print(f"Output shape: {hidden_states.shape}")  # (5, 20)

# Final classification using Sequential
classifier = Sequential([
    Dense(20, 10),
    Softmax()
])

# Use final hidden state for classification
final_state = hidden_states[-1:]
predictions = classifier.predict(final_state)
```

## RNN Applications

### 1. Sentiment Analysis (Many-to-One)

```python
# Process review word by word, classify at the end
sequence = ["This", "movie", "is", "amazing"]

for word in sequence:
    embedding = get_embedding(word)
    hidden = rnn.forward(embedding, hidden)

sentiment = classifier.predict(hidden)  # Positive/Negative
```

### 2. Text Generation (One-to-Many)

```python
# Generate text character by character
seed = "The"
generated = seed

for _ in range(100):
    input_char = encode(generated[-1])
    hidden = rnn.forward(input_char, hidden)
    next_char = sample(classifier.predict(hidden))
    generated += next_char
```

### 3. Sequence Labeling (Many-to-Many)

```python
# Tag each word in a sentence
sentence = ["The", "cat", "sat"]
tags = []

for word in sentence:
    embedding = get_embedding(word)
    hidden = rnn.forward(embedding, hidden)
    tag = classifier.predict(hidden)  # Noun, Verb, etc.
    tags.append(tag)
```

## Bidirectional RNNs

Process sequences in both directions to capture context from past AND future.

```
Forward:   x₁ → x₂ → x₃ → x₄
            ↓    ↓    ↓    ↓
           h₁→  h₂→  h₃→  h₄→

Backward:  x₁ ← x₂ ← x₃ ← x₄
            ↓    ↓    ↓    ↓
          ←h₁  ←h₂  ←h₃  ←h₄

Combined: [h₁→,←h₁] [h₂→,←h₂] [h₃→,←h₃] [h₄→,←h₄]
```

## Training RNNs: Backpropagation Through Time (BPTT)

RNNs are trained by unfolding the network through time and applying backpropagation.

```
Unfolded RNN:

t=1        t=2        t=3
 x₁         x₂         x₃
  ↓          ↓          ↓
[RNN] ──→ [RNN] ──→ [RNN]
  ↓          ↓          ↓
 y₁         y₂         y₃
  ↓          ↓          ↓
 L₁         L₂         L₃
  
Total Loss: L = L₁ + L₂ + L₃

Gradients flow back through all time steps
```

## Key Takeaways

1. **RNNs** process sequences by maintaining hidden state across time steps
2. **Vanilla RNNs** suffer from vanishing gradients for long sequences
3. **LSTMs and GRUs** use gating mechanisms to learn long-range dependencies
4. **Architecture choice** depends on task: many-to-one, one-to-many, etc.
5. **Bidirectional RNNs** capture context from both directions

## Exercises

1. Compute 3 time steps of an RNN by hand with given weights
2. Implement a simple RNN cell and process a sequence
3. Compare LSTM and GRU for a sequence classification task

## Next Steps

- [09_transformer_architecture.md](09_transformer_architecture.md) - Learn about attention mechanisms
- [10_reinforcement_learning.md](10_reinforcement_learning.md) - Sequential decision making
