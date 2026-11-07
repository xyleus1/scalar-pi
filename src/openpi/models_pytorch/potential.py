"""Scalar-potential operators independent of OpenPI's model and data dependencies.

The model supplies the complete action-dependent field. This module neither
detaches actions nor changes the conditioning graph. The caller owns the action
leaf and must detach between *inference* integration steps.
"""

from collections.abc import Callable

import torch


def dot_potential(actions: torch.Tensor, raw_field: torch.Tensor) -> torch.Tensor:
    """Contract a field with its input, returning one scalar per example.

    Both tensors have shape ``[batch, horizon, action_dim]``. The contraction is
    a sum, not an average: averaging would rescale the deployed velocity. Half
    precision inputs are multiplied and accumulated in FP32; FP64 is preserved
    for numerical derivative tests. No finite-value check synchronizes the GPU
    on this hot path; the training/inference wrappers check their outputs.
    """
    if actions.ndim != 3 or raw_field.shape != actions.shape:
        raise ValueError("actions and raw_field must share shape [batch, horizon, action_dim]")
    if not actions.is_floating_point() or not raw_field.is_floating_point():
        raise TypeError("actions and raw_field must be real floating-point tensors")
    if actions.device != raw_field.device:
        raise ValueError("actions and raw_field must be on the same device")
    accumulation_dtype = torch.promote_types(actions.dtype, raw_field.dtype)
    if accumulation_dtype in (torch.float16, torch.bfloat16):
        accumulation_dtype = torch.float32
    return (actions.to(accumulation_dtype) * raw_field.to(accumulation_dtype)).sum(dim=(-2, -1))


def potential_velocity(
    raw_field_fn: Callable[[torch.Tensor], torch.Tensor],
    actions: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Return ``grad_x sum(x * raw_field_fn(x))`` without detaching ``x``.

    Use ``create_graph=True`` for flow-matching training, then ordinary outer
    ``loss.backward()``. Use ``False`` for serving and detach each Euler state.
    The supplied field must not couple distinct batch examples; within-chunk
    interactions are unrestricted. Frozen context can be captured in a closure.

    Gradients include both ``f(x)`` and ``J_f(x).T @ x``. Detaching the field,
    or returning it directly, implements a different architecture.
    """
    if not actions.requires_grad:
        raise ValueError("actions must require gradients; the caller must create the action leaf")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("potential_velocity cannot run inside torch.inference_mode()")
    if not torch.is_grad_enabled():
        raise RuntimeError("potential_velocity requires a torch.enable_grad() region")
    potential = dot_potential(actions, raw_field_fn(actions))
    return torch.autograd.grad(potential.sum(), actions, create_graph=create_graph)[0]
