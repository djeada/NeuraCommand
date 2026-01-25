"""
Dense (fully-connected) layer implementation.
"""

from typing import List, Optional, Tuple, Literal

import numpy as np

from neura_command.layers.base import Layer, ParameterMixin, DeviceType
from neura_command.backend import get_backend, get_default_device, to_device


class Dense(Layer, ParameterMixin):
    """Fully-connected (dense) layer.
    
    Implements: output = activation(input @ weights + bias)
    
    Args:
        in_features: Number of input features
        out_features: Number of output features  
        activation: Activation function instance (optional, can be None for linear)
        use_bias: Whether to include a bias term (default: True)
        weight_init: Weight initialization method ('he', 'xavier', or callable)
        device: Device to create layer on ('cpu' or 'cuda')
        
    Example:
        layer = Dense(784, 128, activation=ReLU())
        output = layer.forward(input_data)
        
        # GPU usage
        layer = Dense(784, 128, device='cuda')
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=None,
        use_bias: bool = True,
        weight_init: str = 'he',
        device: DeviceType = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_bias = use_bias
        
        # Use default device if not specified
        self._device = device if device is not None else get_default_device()
        xp = get_backend(self._device)
        
        # Initialize weights
        if weight_init == 'he':
            self.weights = self.he_init((in_features, out_features), self._device)
        elif weight_init == 'xavier':
            self.weights = self.xavier_init((in_features, out_features), self._device)
        else:
            self.weights = (xp.random.randn(in_features, out_features) * 0.01).astype(xp.float32)
        
        # Initialize bias
        if use_bias:
            self.bias = self.zeros_init((1, out_features), self._device)
        else:
            self.bias = None
        
        # Gradient storage
        self._grad_weights: Optional[np.ndarray] = None
        self._grad_bias: Optional[np.ndarray] = None
        
        # Cache for backward pass
        self._input_cache: Optional[np.ndarray] = None
        self._pre_activation_cache: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Input of shape (batch_size, in_features)
            training: Whether in training mode
            
        Returns:
            Output of shape (batch_size, out_features)
        """
        xp = get_backend(self._device)
        self._training = training
        self._input_cache = x
        
        # Linear transformation
        z = x @ self.weights
        if self.use_bias:
            z = z + self.bias
        
        self._pre_activation_cache = z
        
        # Apply activation if present
        if self.activation is not None:
            return self.activation.compute(z)
        return z
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass.
        
        Args:
            grad: Gradient from next layer, shape (batch_size, out_features)
            
        Returns:
            Gradient to pass to previous layer, shape (batch_size, in_features)
        """
        xp = get_backend(self._device)
        batch_size = grad.shape[0]
        
        # Gradient through activation
        if self.activation is not None:
            grad = grad * self.activation.derivative(self._pre_activation_cache)
        
        # Gradient w.r.t. weights: input.T @ grad
        self._grad_weights = self._input_cache.T @ grad / batch_size
        
        # Gradient w.r.t. bias: sum over batch
        if self.use_bias:
            self._grad_bias = xp.mean(grad, axis=0, keepdims=True)
        
        # Gradient w.r.t. input: grad @ weights.T
        grad_input = grad @ self.weights.T
        
        return grad_input
    
    def parameters(self) -> List[np.ndarray]:
        """Return trainable parameters."""
        if self.use_bias:
            return [self.weights, self.bias]
        return [self.weights]
    
    def gradients(self) -> List[np.ndarray]:
        """Return gradients for parameters."""
        if self.use_bias:
            return [self._grad_weights, self._grad_bias]
        return [self._grad_weights]
    
    def to(self, device: DeviceType) -> 'Dense':
        """Move layer to specified device.
        
        Args:
            device: Target device ('cpu' or 'cuda')
            
        Returns:
            self (for chaining)
        """
        if device == self._device:
            return self
        
        self._device = device
        self.weights = to_device(self.weights, device)
        if self.use_bias:
            self.bias = to_device(self.bias, device)
        
        # Clear caches
        self._input_cache = None
        self._pre_activation_cache = None
        self._grad_weights = None
        self._grad_bias = None
        
        return self
    
    def __repr__(self) -> str:
        act_name = self.activation.__class__.__name__ if self.activation else 'None'
        return (
            f"Dense(in_features={self.in_features}, out_features={self.out_features}, "
            f"activation={act_name}, use_bias={self.use_bias}, device='{self._device}')"
        )
