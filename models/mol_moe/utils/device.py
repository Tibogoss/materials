"""
Device Management Utilities for MoL-MoE Training

This module provides centralized device handling utilities to ensure
consistent tensor placement across GPU/CPU in the MoE training system.
"""

import torch
import torch.nn as nn
from typing import Union, Optional, Any, List, Dict


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get a torch device, with automatic CUDA detection if not specified.

    Args:
        device: Device specification (None, 'cuda', 'cpu', torch.device)
                If None, auto-detects CUDA availability

    Returns:
        torch.device: The resolved device

    Examples:
        >>> get_device()  # Auto-detect
        device(type='cuda', index=0)
        >>> get_device('cpu')
        device(type='cpu')
    """
    if device is None:
        # Auto-detect: use CUDA if available, else CPU
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        return torch.device(device)

    raise TypeError(f"device must be None, str, or torch.device, got {type(device)}")


def to_device(obj: Any, device: torch.device) -> Any:
    """
    Move an object (tensor, module, dict, list) to the specified device.

    Handles:
    - torch.Tensor: moved to device
    - nn.Module: moved to device
    - dict: recursively move values
    - list/tuple: recursively move elements
    - None: returns None
    - other types: returned unchanged

    Args:
        obj: Object to move
        device: Target device

    Returns:
        Object moved to device (or original if not moveable)

    Examples:
        >>> tensor = torch.randn(3, 4)
        >>> tensor_gpu = to_device(tensor, torch.device('cuda'))
        >>> model = to_device(nn.Linear(10, 5), torch.device('cuda'))
    """
    if obj is None:
        return None

    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    if isinstance(obj, nn.Module):
        return obj.to(device)

    if isinstance(obj, dict):
        return {key: to_device(val, device) for key, val in obj.items()}

    if isinstance(obj, (list, tuple)):
        moved = [to_device(item, device) for item in obj]
        return type(obj)(moved)  # Preserve list/tuple type

    # For other types (int, float, str, etc.), return unchanged
    return obj


def get_model_device(model: nn.Module) -> torch.device:
    """
    Extract the device that a model's parameters are on.

    Args:
        model: PyTorch model

    Returns:
        torch.device: Device of first parameter, or CPU if model has no parameters

    Examples:
        >>> model = nn.Linear(10, 5).cuda()
        >>> get_model_device(model)
        device(type='cuda', index=0)
    """
    try:
        # Get device from first parameter
        return next(model.parameters()).device
    except StopIteration:
        # Model has no parameters, default to CPU
        return torch.device('cpu')


def ensure_device(tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    """
    Ensure a tensor is on the target device, moving only if necessary.

    This is more efficient than always calling .to(device) because it
    checks if the tensor is already on the correct device first.

    Args:
        tensor: Input tensor
        target_device: Desired device

    Returns:
        Tensor on target device (same object if already correct, moved copy otherwise)

    Examples:
        >>> tensor_cpu = torch.randn(3, 4)
        >>> tensor_gpu = ensure_device(tensor_cpu, torch.device('cuda'))
        >>> # No-op if already on correct device:
        >>> tensor_gpu2 = ensure_device(tensor_gpu, torch.device('cuda'))
        >>> assert tensor_gpu is tensor_gpu2  # Same object, not copied
    """
    if tensor.device == target_device:
        return tensor  # Already on correct device, no-op
    return tensor.to(target_device)


def validate_device_consistency(*tensors_or_modules: Union[torch.Tensor, nn.Module]) -> bool:
    """
    Validate that all provided tensors/modules are on the same device.

    Raises informative error if devices don't match.
    Useful for debugging device mismatch issues.

    Args:
        *tensors_or_modules: Variable number of tensors or modules to check

    Returns:
        bool: True if all on same device (or if empty input)

    Raises:
        RuntimeError: If tensors/modules are on different devices

    Examples:
        >>> t1 = torch.randn(3, 4).cuda()
        >>> t2 = torch.randn(3, 4).cuda()
        >>> validate_device_consistency(t1, t2)  # OK
        True
        >>> t3 = torch.randn(3, 4)  # CPU
        >>> validate_device_consistency(t1, t3)  # Raises error
        RuntimeError: Device mismatch detected...
    """
    if len(tensors_or_modules) == 0:
        return True

    devices = []
    for i, obj in enumerate(tensors_or_modules):
        if isinstance(obj, torch.Tensor):
            devices.append((f"tensor_{i}", obj.device))
        elif isinstance(obj, nn.Module):
            try:
                device = next(obj.parameters()).device
                devices.append((f"module_{i}", device))
            except StopIteration:
                # Module has no parameters, skip
                continue
        else:
            raise TypeError(f"Argument {i} must be Tensor or Module, got {type(obj)}")

    if len(devices) == 0:
        return True  # No devices to check

    # Check all devices match the first one
    first_name, first_device = devices[0]
    for name, device in devices[1:]:
        if device != first_device:
            raise RuntimeError(
                f"Device mismatch detected:\n"
                f"  {first_name}: {first_device}\n"
                f"  {name}: {device}\n"
                f"All tensors/modules must be on the same device."
            )

    return True
