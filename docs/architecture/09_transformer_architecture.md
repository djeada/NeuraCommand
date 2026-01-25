# Transformer Architecture

## Introduction

Transformers have revolutionized deep learning, particularly in natural language processing and beyond. Introduced in the "Attention Is All You Need" paper (2017), they replace recurrence with self-attention, enabling parallelization and capturing long-range dependencies more effectively.

## The Core Innovation: Attention

### Why Attention?

RNNs process sequences sequentially, making them:
1. Slow to train (no parallelization)
2. Prone to forgetting long-range dependencies

Attention allows every position to directly connect to every other position in one step.

```
RNN:  x₁ → x₂ → x₃ → x₄ → x₅    (Sequential, O(n) steps)
                                 Long path from x₁ to x₅

Attention:
      x₁ ←→ x₂ ←→ x₃ ←→ x₄ ←→ x₅  (All-to-all, O(1) steps)
      ↑↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↔↑  Direct connection everywhere
```

## Self-Attention Mechanism

Self-attention allows each position in a sequence to attend to all other positions.

### The Query-Key-Value Framework

For each input, we create three vectors:
- **Query (Q)**: What am I looking for?
- **Key (K)**: What do I contain?
- **Value (V)**: What information do I provide?

```
Input: "The cat sat"

For "sat":
  Query:  "I'm looking for subjects/actors"
  Key:    "I'm an action/verb"
  Value:  "Information about sitting"

Attention connects "sat" to "cat" because
"cat"'s Key matches "sat"'s Query (who did the action?)
```

### Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q, K, V$ = Query, Key, Value matrices
- $d_k$ = dimension of keys (for scaling)

## Hand Calculation: Self-Attention

Let's compute attention for a simple sequence with 3 tokens.

### Step 1: Input Embeddings

```
Tokens: [A, B, C]
Embeddings (dimension 4):
  A: [1, 0, 1, 0]
  B: [0, 1, 0, 1]
  C: [1, 1, 0, 0]

X = [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 1, 0, 0]]
```

### Step 2: Create Q, K, V

Weight matrices (4×3 for simplicity, normally 4×4):

```
W_Q = [[1, 0, 1],      W_K = [[0, 1, 0],      W_V = [[1, 0, 0],
       [0, 1, 0],             [1, 0, 1],             [0, 1, 0],
       [1, 0, 0],             [0, 0, 1],             [0, 0, 1],
       [0, 0, 1]]             [1, 1, 0]]             [1, 1, 0]]
```

**Compute Q:**
```
Q = X @ W_Q
A: [1,0,1,0] @ W_Q = [2, 0, 1]
B: [0,1,0,1] @ W_Q = [0, 1, 1]
C: [1,1,0,0] @ W_Q = [1, 1, 1]

Q = [[2, 0, 1],
     [0, 1, 1],
     [1, 1, 1]]
```

**Similarly for K and V:**
```
K = [[0, 1, 1],
     [2, 1, 1],
     [1, 1, 1]]

V = [[1, 0, 1],
     [1, 2, 0],
     [1, 1, 0]]
```

### Step 3: Compute Attention Scores

$$\text{Scores} = QK^T$$

```
Score(A,A) = Q_A · K_A = [2,0,1]·[0,1,1] = 0+0+1 = 1
Score(A,B) = Q_A · K_B = [2,0,1]·[2,1,1] = 4+0+1 = 5
Score(A,C) = Q_A · K_C = [2,0,1]·[1,1,1] = 2+0+1 = 3

Scores = [[1, 5, 3],
          [1, 3, 2],
          [2, 4, 3]]
```

### Step 4: Scale and Softmax

Scale by $\sqrt{d_k} = \sqrt{3} \approx 1.73$:

```
Scaled = [[0.58, 2.89, 1.73],
          [0.58, 1.73, 1.16],
          [1.16, 2.31, 1.73]]
```

Apply softmax (per row):
```
Weights = [[0.09, 0.65, 0.26],
           [0.15, 0.47, 0.38],
           [0.17, 0.53, 0.30]]
```

### Step 5: Weighted Sum of Values

```
Output_A = 0.09×V_A + 0.65×V_B + 0.26×V_C
         = 0.09×[1,0,1] + 0.65×[1,2,0] + 0.26×[1,1,0]
         = [0.09,0,0.09] + [0.65,1.30,0] + [0.26,0.26,0]
         = [1.00, 1.56, 0.09]

Final output is weighted combination of all values,
where weights indicate relevance/attention.
```

## Multi-Head Attention

Instead of one attention function, use multiple "heads" that attend to different aspects.

```
                    ┌─── Head 1 ──→ Concat ──→ Linear ──→ Output
                    │
Input ──→ Split ────┼─── Head 2 ──→
                    │
                    └─── Head 3 ──→

Each head learns different attention patterns:
- Head 1: Syntax (subject-verb agreement)
- Head 2: Semantics (word meanings)
- Head 3: Position relationships
```

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

## The Transformer Architecture

### Full Architecture

```
                     ┌─────────────────────────┐
                     │     Output Probs        │
                     └───────────┬─────────────┘
                                 │
                     ┌───────────┴─────────────┐
                     │        Softmax          │
                     └───────────┬─────────────┘
                                 │
                     ┌───────────┴─────────────┐
                     │        Linear           │
                     └───────────┬─────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            │                DECODER                   │
            │  ┌────────────────────────────────────┐ │
            │  │   Add & Norm                       │ │
            │  │        ↑                           │ │
            │  │   Feed Forward                     │ │
            │  │        ↑                           │ │
            │  │   Add & Norm                       │ │
            │  │        ↑                           │ │
            │  │   Encoder-Decoder Attention ←──────┼─┼──── From Encoder
            │  │        ↑                           │ │
            │  │   Add & Norm                       │ │
            │  │        ↑                           │ │
            │  │   Masked Self-Attention            │ │
            │  └────────────────────────────────────┘ │
            │                  × N                    │
            └────────────────────┬────────────────────┘
                                 │
                     ┌───────────┴─────────────┐
                     │  Output Embedding +     │
                     │  Positional Encoding    │
                     └───────────┬─────────────┘
                                 │
                              Output
                             (shifted)


            ┌────────────────────┴────────────────────┐
            │                ENCODER                   │
            │  ┌────────────────────────────────────┐ │
            │  │   Add & Norm                       │ │
            │  │        ↑                           │ │
            │  │   Feed Forward                     │ │
            │  │        ↑                           │ │
            │  │   Add & Norm                       │ │
            │  │        ↑                           │ │
            │  │   Self-Attention                   │ │
            │  └────────────────────────────────────┘ │
            │                  × N                    │
            └────────────────────┬────────────────────┘
                                 │
                     ┌───────────┴─────────────┐
                     │  Input Embedding +      │
                     │  Positional Encoding    │
                     └───────────┬─────────────┘
                                 │
                              Input
```

### Key Components

#### 1. Positional Encoding

Since attention is order-invariant, we add position information:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

```python
import numpy as np

def positional_encoding(max_len, d_model):
    """Generate positional encodings"""
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    
    div_term = np.exp(np.arange(0, d_model, 2) * 
                      -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

# Example
pe = positional_encoding(100, 512)
# pe[0] is position 0's encoding, pe[1] is position 1's, etc.
```

#### 2. Feed-Forward Network

After attention, each position passes through a feed-forward network:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

```python
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU

# Transformer FFN layer
ffn = Sequential([
    Dense(512, 2048),  # Expand
    ReLU(),
    Dense(2048, 512)   # Contract back
])
```

#### 3. Layer Normalization

```python
def layer_norm(x, epsilon=1e-6):
    """Layer normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)
```

#### 4. Residual Connections

```
Output = LayerNorm(x + Sublayer(x))

     x ─────────────────────┐
     │                      │
     ↓                      │
  [Sublayer]                │
     │                      │
     ↓                      │
    (+) ←───────────────────┘
     │
     ↓
  [LayerNorm]
     │
     ↓
   Output
```

## Conceptual Implementation

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU, Softmax

class SimpleAttention:
    """Simplified attention mechanism"""
    
    def __init__(self, d_model):
        self.d_model = d_model
        # Linear projections
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        
    def forward(self, x):
        """Compute self-attention"""
        # Create Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Compute attention scores
        scores = Q @ K.T / np.sqrt(self.d_model)
        
        # Softmax to get weights
        weights = self.softmax(scores)
        
        # Weighted sum of values
        output = weights @ V
        return output
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SimpleTransformerBlock:
    """Simplified transformer encoder block"""
    
    def __init__(self, d_model, d_ff):
        self.attention = SimpleAttention(d_model)
        self.ffn = Sequential([
            Dense(d_model, d_ff),
            ReLU(),
            Dense(d_ff, d_model)
        ])
        
    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention.forward(x)
        x = layer_norm(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn.predict(x)
        x = layer_norm(x + ffn_out)
        
        return x


def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)


# Example usage
d_model = 64
seq_len = 10

block = SimpleTransformerBlock(d_model, d_ff=256)
x = np.random.randn(seq_len, d_model)
output = block.forward(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
```

## Transformer Variants

### Encoder-Only Models (e.g., BERT)
```
Input → [Encoder] → Contextual Representations

Use: Classification, NER, embeddings
```

### Decoder-Only Models (e.g., GPT)
```
Input → [Decoder with causal mask] → Next Token

Use: Text generation, completion
```

### Encoder-Decoder Models (e.g., T5, BART)
```
Source → [Encoder] → [Decoder] → Target

Use: Translation, summarization
```

## Attention Visualization

```
Sentence: "The cat sat on the mat"

Attention weights for "sat":

      The   cat   sat   on    the   mat
sat   0.05  0.70  0.10  0.05  0.02  0.08
      ─     ███   ▓     ─     ─     ▓

"sat" attends strongly to "cat" (the subject)
and somewhat to "mat" (related noun)
```

## Key Innovations

| Feature | Benefit |
|---------|---------|
| Self-Attention | O(1) path length for any positions |
| Multi-Head | Attend to different representation subspaces |
| Positional Encoding | Capture sequence order without recurrence |
| Residual + LayerNorm | Stable training for deep networks |
| Parallelization | Faster training than RNNs |

## Computational Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Self-Attention | O(n²d) | O(n²) |
| RNN | O(nd²) | O(1) |

Where n = sequence length, d = model dimension.

For long sequences (n > d), attention is slower, leading to efficient variants:
- Sparse Attention
- Linear Attention
- Longformer

## Key Takeaways

1. **Attention** allows direct connections between any positions
2. **Multi-head attention** captures different relationship types
3. **Positional encoding** adds sequence order information
4. **Residual connections** enable training deep models
5. **Transformers** are the foundation of modern NLP (BERT, GPT, T5)

## Exercises

1. Compute attention weights by hand for a 3-token sequence
2. Implement positional encoding and visualize the patterns
3. Research one transformer variant (BERT, GPT, T5) and summarize its architecture

## Next Steps

- [10_reinforcement_learning.md](10_reinforcement_learning.md) - Decision-making with neural networks
- [11_regularization_techniques.md](11_regularization_techniques.md) - Training techniques for transformers
