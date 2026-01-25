from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

import numpy as np


class Optimizer(ABC):
    """Base class for all optimizers.
    
    Optimizers update network parameters based on computed gradients.
    They maintain per-parameter state (momentum, squared gradients, etc.)
    using parameter IDs as keys.
    """

    @property
    def learning_rate(self) -> float:
        """Current learning rate."""
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set learning rate."""
        self._learning_rate = value

    @abstractmethod
    def update(self, param_id: int, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Update weights based on gradient.
        
        Args:
            param_id: Unique identifier for this parameter (use id(weights))
            weights: Current parameter values
            gradient: Gradient of loss w.r.t. weights
            
        Returns:
            Updated weights
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset optimizer state (e.g., for new training run)."""
        pass
    
    def step(self, layers) -> None:
        """Update all parameters in all layers.
        
        Args:
            layers: List of Layer objects with parameters() and gradients() methods
        """
        for layer in layers:
            params = layer.parameters()
            grads = layer.gradients()
            
            for i, (param, grad) in enumerate(zip(params, grads)):
                if grad is not None:
                    param_id = id(param)
                    updated = self.update(param_id, param, grad)
                    # Update in-place
                    param[...] = updated


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with optional momentum.
    
    Args:
        learning_rate: Learning rate (step size)
        momentum: Momentum factor (0 = no momentum)
        nesterov: Whether to use Nesterov momentum
        weight_decay: L2 regularization factor
        
    Example:
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        self._learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self._velocity: Dict[int, np.ndarray] = {}

    def update(self, param_id: int, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        # Apply weight decay
        if self.weight_decay > 0:
            gradient = gradient + self.weight_decay * weights
        
        if self.momentum > 0:
            if param_id not in self._velocity:
                self._velocity[param_id] = np.zeros_like(weights)
            
            v = self._velocity[param_id]
            v = self.momentum * v - self._learning_rate * gradient
            self._velocity[param_id] = v
            
            if self.nesterov:
                return weights + self.momentum * v - self._learning_rate * gradient
            else:
                return weights + v
        else:
            return weights - self._learning_rate * gradient

    def reset(self) -> None:
        self._velocity = {}


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation).
    
    Combines momentum with adaptive learning rates per parameter.
    
    Args:
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment estimates
        beta2: Exponential decay rate for second moment estimates
        epsilon: Small constant for numerical stability
        weight_decay: L2 regularization factor (AdamW style)
        
    Reference:
        Kingma & Ba, 2014: "Adam: A Method for Stochastic Optimization"
        
    Example:
        optimizer = Adam(learning_rate=0.001)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self._learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._m: Dict[int, np.ndarray] = {}  # First moment
        self._v: Dict[int, np.ndarray] = {}  # Second moment
        self._t: int = 0  # Timestep

    def update(self, param_id: int, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        # Initialize state for this parameter if needed
        if param_id not in self._m:
            self._m[param_id] = np.zeros_like(weights)
            self._v[param_id] = np.zeros_like(weights)

        # Increment timestep (once per step, not per parameter)
        # This is handled in step() method, but for single calls:
        if not hasattr(self, '_step_called'):
            self._t += 1
        
        m = self._m[param_id]
        v = self._v[param_id]

        # Update biased first moment estimate
        m = self.beta1 * m + (1 - self.beta1) * gradient
        self._m[param_id] = m

        # Update biased second raw moment estimate
        v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)
        self._v[param_id] = v

        # Bias correction
        m_hat = m / (1 - self.beta1 ** self._t)
        v_hat = v / (1 - self.beta2 ** self._t)

        # AdamW-style weight decay (applied to weights, not gradient)
        if self.weight_decay > 0:
            weights = weights - self._learning_rate * self.weight_decay * weights

        # Update weights
        return weights - self._learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def step(self, layers) -> None:
        """Update all parameters, incrementing timestep once."""
        self._t += 1
        self._step_called = True
        super().step(layers)
        del self._step_called

    def reset(self) -> None:
        self._m = {}
        self._v = {}
        self._t = 0


class RMSprop(Optimizer):
    """RMSprop optimizer.
    
    Maintains a moving average of squared gradients to normalize updates.
    
    Args:
        learning_rate: Learning rate
        rho: Decay rate for moving average
        epsilon: Small constant for numerical stability
        weight_decay: L2 regularization factor
        
    Example:
        optimizer = RMSprop(learning_rate=0.001)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self._learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._avg_sq_grad: Dict[int, np.ndarray] = {}

    def update(self, param_id: int, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if param_id not in self._avg_sq_grad:
            self._avg_sq_grad[param_id] = np.zeros_like(weights)
        
        # Apply weight decay
        if self.weight_decay > 0:
            gradient = gradient + self.weight_decay * weights

        # Update running average of squared gradients
        avg = self._avg_sq_grad[param_id]
        avg = self.rho * avg + (1 - self.rho) * (gradient ** 2)
        self._avg_sq_grad[param_id] = avg

        # Update weights
        return weights - self._learning_rate * gradient / (np.sqrt(avg) + self.epsilon)

    def reset(self) -> None:
        self._avg_sq_grad = {}


class AdaGrad(Optimizer):
    """AdaGrad optimizer.
    
    Adapts learning rate per-parameter based on accumulated gradients.
    Works well for sparse gradients.
    
    Args:
        learning_rate: Initial learning rate
        epsilon: Small constant for numerical stability
        
    Note:
        Learning rate effectively decreases over time as gradients accumulate.
        
    Example:
        optimizer = AdaGrad(learning_rate=0.01)
    """

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        self._learning_rate = learning_rate
        self.epsilon = epsilon
        self._accumulated_grad: Dict[int, np.ndarray] = {}

    def update(self, param_id: int, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if param_id not in self._accumulated_grad:
            self._accumulated_grad[param_id] = np.zeros_like(weights)

        acc = self._accumulated_grad[param_id]
        acc = acc + gradient ** 2
        self._accumulated_grad[param_id] = acc
        
        return weights - self._learning_rate * gradient / (np.sqrt(acc) + self.epsilon)

    def reset(self) -> None:
        self._accumulated_grad = {}
