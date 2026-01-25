"""
Array backend abstraction for NumPy/CuPy interoperability.

This module provides a clean abstraction over NumPy and CuPy, allowing
NeuraCommand to run on CPU or GPU with minimal code changes.

Design goals:
- Zero overhead when using NumPy only
- CuPy is optional (not a required dependency)
- Easy device transfers (CPU <-> GPU)
- Thread-safe default device setting
"""

import threading
from typing import Literal, Union, Any
from dataclasses import dataclass

import numpy as np

# Try to import cupy, but don't fail if not available
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


DeviceType = Literal["cpu", "cuda", "gpu"]

# Thread-local storage for default device
_thread_local = threading.local()
_global_default_device: DeviceType = "cpu"


@dataclass
class Backend:
    """Container for backend information."""

    name: str
    device: DeviceType
    module: Any  # numpy or cupy module

    def __repr__(self) -> str:
        return f"Backend(name='{self.name}', device='{self.device}')"


def is_cupy_available() -> bool:
    """Check if CuPy is installed and CUDA is available.

    Returns:
        True if CuPy can be used, False otherwise
    """
    if not CUPY_AVAILABLE:
        return False
    try:
        # Check if CUDA is actually available
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def get_default_device() -> DeviceType:
    """Get the default device for the current thread.

    Returns:
        Current default device ('cpu' or 'cuda')
    """
    return getattr(_thread_local, "device", _global_default_device)


def set_default_device(device: DeviceType) -> None:
    """Set the default device for the current thread.

    Args:
        device: Device to use ('cpu', 'cuda', or 'gpu')

    Raises:
        ValueError: If device is 'cuda'/'gpu' but CuPy is not available

    Example:
        set_default_device('cuda')
        model = Sequential([...])  # Will use GPU
    """
    global _global_default_device

    device = _normalize_device(device)

    if device == "cuda" and not is_cupy_available():
        raise ValueError(
            "Cannot set device to 'cuda': CuPy is not available. "
            "Install with: pip install cupy-cuda11x (or appropriate version)"
        )

    _thread_local.device = device
    _global_default_device = device


def _normalize_device(device: DeviceType) -> DeviceType:
    """Normalize device name ('gpu' -> 'cuda')."""
    if device == "gpu":
        return "cuda"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unknown device: {device}. Use 'cpu' or 'cuda'.")
    return device


def get_backend(device: DeviceType = None) -> Any:
    """Get the array module (numpy or cupy) for a device.

    Args:
        device: Device to get backend for. If None, uses default device.

    Returns:
        numpy or cupy module

    Raises:
        ValueError: If device is 'cuda' but CuPy is not available

    Example:
        xp = get_backend('cuda')
        x = xp.array([1, 2, 3])  # Creates CuPy array on GPU
    """
    if device is None:
        device = get_default_device()
    else:
        device = _normalize_device(device)

    if device == "cuda":
        if not is_cupy_available():
            raise ValueError(
                "CuPy is not available. Install with: pip install cupy-cuda11x"
            )
        return cp

    return np


def get_array_module(array) -> Any:
    """Get the array module for an existing array.

    Works with both NumPy and CuPy arrays, returning the appropriate module.

    Args:
        array: NumPy or CuPy array

    Returns:
        numpy or cupy module

    Example:
        x = cp.array([1, 2, 3])
        xp = get_array_module(x)  # Returns cupy
    """
    if CUPY_AVAILABLE:
        return cp.get_array_module(array)
    return np


def to_device(array, device: DeviceType) -> Any:
    """Transfer array to specified device.

    Args:
        array: NumPy or CuPy array
        device: Target device ('cpu' or 'cuda')

    Returns:
        Array on the target device

    Example:
        x_gpu = to_device(x_cpu, 'cuda')
        x_cpu = to_device(x_gpu, 'cpu')
    """
    device = _normalize_device(device)

    if device == "cpu":
        return to_numpy(array)

    # Transfer to GPU
    if not is_cupy_available():
        raise ValueError("CuPy is not available for GPU transfer")

    if isinstance(array, np.ndarray):
        return cp.asarray(array)

    # Already a CuPy array
    return array


def to_numpy(array) -> np.ndarray:
    """Convert array to NumPy (transfer from GPU if needed).

    Args:
        array: NumPy or CuPy array

    Returns:
        NumPy array (on CPU)

    Example:
        x_cpu = to_numpy(x_gpu)
    """
    if isinstance(array, np.ndarray):
        return array

    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)

    # Fallback: try to convert via numpy
    return np.asarray(array)


def get_device(array) -> DeviceType:
    """Get the device an array is on.

    Args:
        array: NumPy or CuPy array

    Returns:
        'cpu' or 'cuda'
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return "cuda"
    return "cpu"


class DeviceContext:
    """Context manager for temporarily changing the default device.

    Example:
        with DeviceContext('cuda'):
            model = Sequential([...])  # Uses GPU
        # Back to previous default device
    """

    def __init__(self, device: DeviceType):
        self.device = _normalize_device(device)
        self.previous_device = None

    def __enter__(self):
        self.previous_device = get_default_device()
        set_default_device(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_device is not None:
            set_default_device(self.previous_device)
        return False
