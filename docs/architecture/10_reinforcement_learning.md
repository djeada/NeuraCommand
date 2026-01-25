# Reinforcement Learning

## Introduction

Reinforcement Learning (RL) is a paradigm where an agent learns to make decisions by interacting with an environment. Unlike supervised learning, there's no explicit teacher - the agent learns from rewards and penalties based on its actions.

## The RL Framework

```
                    ┌─────────────────────────────────┐
                    │                                 │
                    │         ENVIRONMENT             │
                    │                                 │
                    └──────────┬──────────────────────┘
                               │
                     State s_t │
                     Reward r_t│
                               │
                    ┌──────────▼──────────────────────┐
                    │                                 │
                    │           AGENT                 │
                    │    (Neural Network Policy)      │
                    │                                 │
                    └──────────┬──────────────────────┘
                               │
                    Action a_t │
                               │
                               ▼
                        Execute action
                               │
                               └───────────────────────→
```

### Key Components

| Component | Description | Example |
|-----------|-------------|---------|
| **State (s)** | Current observation | Game screen, robot sensors |
| **Action (a)** | Decision made by agent | Move left, jump, accelerate |
| **Reward (r)** | Feedback signal | +1 for win, -1 for loss |
| **Policy (π)** | Strategy for choosing actions | Neural network |
| **Value (V)** | Expected future reward | How good is this state? |

## Markov Decision Process (MDP)

RL problems are formalized as MDPs:

$$\text{MDP} = (S, A, P, R, \gamma)$$

- **S**: Set of states
- **A**: Set of actions
- **P**: Transition probability P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor (0 < γ ≤ 1)

### The Markov Property

The future depends only on the current state, not the history:

$$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1}|s_t, a_t)$$

## Hand Calculation Example: Gridworld

Consider a simple 3×3 gridworld:

```
┌───┬───┬───┐
│ S │   │   │
├───┼───┼───┤
│   │ X │   │
├───┼───┼───┤
│   │   │ G │
└───┴───┴───┘

S = Start (reward: 0)
X = Obstacle (reward: -1)
G = Goal (reward: +10)
Actions: Up, Down, Left, Right
γ = 0.9
```

### Value Iteration

We compute the value of each state:

$$V(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$$

**Iteration 0:** All values = 0

**Iteration 1:**
- V(G) = 10 (terminal state)
- V(right of G) = 0 + 0.9 × 10 = 9
- V(above G) = 0 + 0.9 × 10 = 9

**After convergence:**
```
┌─────┬─────┬─────┐
│ 6.56│ 7.29│ 8.1 │
├─────┼─────┼─────┤
│ 7.29│  X  │ 9.0 │
├─────┼─────┼─────┤
│ 8.1 │ 9.0 │ 10  │
└─────┴─────┴─────┘
```

The agent follows the gradient of values toward the goal!

## Q-Learning

Q-Learning learns action-values Q(s,a) - the value of taking action a in state s.

### The Bellman Equation

$$Q(s,a) = R(s,a) + \gamma \max_{a'} Q(s', a')$$

### Q-Learning Update Rule

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

Where α is the learning rate.

### Hand Calculation: Q-Learning Step

```
State: s = (1,0)  (middle-left cell)
Action: a = Right
Reward: r = 0 (no special reward)
Next state: s' = (1,1)  (but it's an obstacle!)
Actually next state: s' = (1,0) (stayed in place, r = -1)

Current Q-table:
        Up    Down   Left   Right
(0,0)   0     0      0      0
(1,0)   0     0      0      0.5
...

Update with α=0.1, γ=0.9:
Q(1,0, Right) = 0.5 + 0.1 × [-1 + 0.9 × max(Q(1,0)) - 0.5]
              = 0.5 + 0.1 × [-1 + 0.9 × 0.5 - 0.5]
              = 0.5 + 0.1 × [-1.05]
              = 0.395
```

## Deep Q-Network (DQN)

For complex state spaces (like images), use a neural network to approximate Q:

```
State s ──→ [Neural Network] ──→ Q(s, a₁)
         (CNN or MLP)           Q(s, a₂)
                                Q(s, a₃)
                                ...
```

### DQN Implementation Concept

```python
import numpy as np
from neura_command.network.sequential import Sequential
from neura_command.layers.dense import Dense
from neura_command.layers.activation import ReLU

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99       # Discount factor
        self.epsilon = 1.0      # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Q-Network
        self.model = Sequential([
            Dense(state_size, 64),
            ReLU(),
            Dense(64, 64),
            ReLU(),
            Dense(64, action_size)
        ])
        
        # Experience replay buffer
        self.memory = []
        self.batch_size = 32
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def replay(self):
        """Train on random batch from memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch_idx = np.random.choice(len(self.memory), self.batch_size)
        batch = [self.memory[i] for i in batch_idx]
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                # Bellman equation
                next_q = self.model.predict(next_state.reshape(1, -1))
                target = reward + self.gamma * np.max(next_q)
            
            # Get current Q values
            current_q = self.model.predict(state.reshape(1, -1))
            current_q[0, action] = target
            
            # Train
            self.model.fit(
                state.reshape(1, -1), 
                current_q,
                epochs=1,
                learning_rate=self.learning_rate
            )
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### Training Loop

```python
def train_dqn(env, agent, episodes=1000):
    """Train DQN agent on environment"""
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Choose action
            action = agent.choose_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from experience
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Reward: {total_reward}, "
                  f"Epsilon: {agent.epsilon:.2f}")
```

## Policy Gradient Methods

Instead of learning Q-values, directly learn a policy π(a|s).

### REINFORCE Algorithm

$$\nabla J(\theta) = \mathbb{E}_\tau \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right]$$

```python
class PolicyGradientAgent:
    def __init__(self, state_size, action_size):
        self.model = Sequential([
            Dense(state_size, 64),
            ReLU(),
            Dense(64, action_size),
            Softmax()  # Output action probabilities
        ])
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def choose_action(self, state):
        """Sample action from policy"""
        probs = self.model.predict(state.reshape(1, -1))[0]
        action = np.random.choice(len(probs), p=probs)
        return action
    
    def store_transition(self, state, action, reward):
        """Store trajectory"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def compute_returns(self, gamma=0.99):
        """Compute discounted returns"""
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def train(self, learning_rate=0.01):
        """Update policy using collected trajectory"""
        returns = self.compute_returns()
        
        for state, action, G in zip(self.episode_states, 
                                     self.episode_actions, 
                                     returns):
            # Create target that encourages the taken action
            probs = self.model.predict(state.reshape(1, -1))
            target = probs.copy()
            target[0, action] += learning_rate * G
            target = target / target.sum()  # Renormalize
            
            self.model.fit(
                state.reshape(1, -1),
                target,
                epochs=1,
                learning_rate=learning_rate
            )
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
```

## Actor-Critic Methods

Combine policy gradient (actor) with value estimation (critic):

```
State s
    │
    ├────→ [Actor Network] ────→ π(a|s) (Policy)
    │
    └────→ [Critic Network] ───→ V(s) (Value)
    
Actor: Learns which actions to take
Critic: Evaluates how good the actions are
```

### Advantage Actor-Critic (A2C)

$$A(s,a) = Q(s,a) - V(s) = r + \gamma V(s') - V(s)$$

The advantage tells us how much better an action is compared to average.

## Exploration vs Exploitation

### The Dilemma

```
Exploit:                    Explore:
Go to known good             Try new restaurants,
restaurant                   might find better one
                            
Safe, suboptimal             Risky, potential upside
```

### Strategies

| Strategy | Description |
|----------|-------------|
| ε-greedy | Random action with probability ε |
| Softmax | Actions proportional to Q-values |
| UCB | Upper confidence bound |
| Noise | Add noise to actions or parameters |

```python
# Epsilon-greedy
def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))
    return np.argmax(q_values)

# Softmax (Boltzmann)
def softmax_action(q_values, temperature=1.0):
    exp_q = np.exp(q_values / temperature)
    probs = exp_q / exp_q.sum()
    return np.random.choice(len(q_values), p=probs)
```

## RL in NeuraCommand: Snake AI Example

The repository includes a Snake AI demo that uses RL concepts:

```python
# From demo/practical_demos/snake_ai/
# The snake learns to:
# - Navigate toward food
# - Avoid walls and itself
# - Maximize score

# State representation:
# - Distance to food (4 directions)
# - Danger detection (3 directions)
# - Current direction

# Actions:
# - Go straight
# - Turn left
# - Turn right

# Reward:
# - +10 for eating food
# - -10 for collision
# - Small penalty for each step (encourages efficiency)
```

## Reward Shaping

Designing good rewards is crucial:

### Bad Reward Design
```
Reward: +1 only when reaching goal

Problem: Agent receives no signal for progress
         Random exploration takes forever
```

### Good Reward Design
```
Reward:
  +10 for reaching goal
  -0.1 for each step (encourages speed)
  +distance_improvement (shaped reward)
  -1 for hitting walls

Agent gets continuous feedback
```

## Key RL Algorithms Summary

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| Q-Learning | Value-based | Learn Q(s,a), act greedily |
| DQN | Deep Value | Neural network for Q |
| REINFORCE | Policy Gradient | Directly optimize policy |
| A2C/A3C | Actor-Critic | Combine policy + value |
| PPO | Policy Gradient | Stable policy updates |
| SAC | Actor-Critic | Maximum entropy RL |

## Key Takeaways

1. **RL** learns from interaction, not labeled data
2. **Exploration vs exploitation** is fundamental tradeoff
3. **Q-learning** learns action values; **policy gradient** learns policies directly
4. **Deep RL** uses neural networks for function approximation
5. **Reward design** is critical for success

## Exercises

1. Implement Q-learning for a simple gridworld by hand
2. Train a DQN agent on a simple environment
3. Compare ε-greedy and softmax exploration

## Next Steps

- [11_regularization_techniques.md](11_regularization_techniques.md) - Prevent overfitting in RL
- [12_practical_training_tips.md](12_practical_training_tips.md) - Best practices for training
