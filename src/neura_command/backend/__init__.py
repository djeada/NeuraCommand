"""
Backend abstraction for NeuraCommand.

Provides a unified interface for NumPy and CuPy, enabling transparent
CPU/GPU execution. CuPy is optional - falls back to NumPy if not installed.

Usage:
    from neura_command.backend import get_backend, set_default_device
    
    # Get backend for specific device
    xp = get_backend('cuda')  # Returns cupy if available, else raises
    xp = get_backend('cpu')   # Returns numpy
    
    # Set default device
    set_default_device('cuda')
    xp = get_backend()  # Uses default
    
    # Use like numpy
    x = xp.array([1, 2, 3])
    y = xp.dot(x, x)
"""

from neura_command.backend.array_backend import (
    get_backend,
    get_array_module,
    set_default_device,
    get_default_device,
    to_device,
    to_numpy,
    is_cupy_available,
    Backend,
)

__all__ = [
    "get_backend",
    "get_array_module",
    "set_default_device",
    "get_default_device",
    "to_device",
    "to_numpy",
    "is_cupy_available",
    "Backend",
]
