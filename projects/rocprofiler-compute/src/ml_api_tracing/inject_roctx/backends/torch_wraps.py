# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""The PyTorch functions and tensor methods that the torch backend wraps.

This module does nothing when imported, so tests can read the tables without
loading the backend.
"""

# (module path, attribute name, marker name) of the functions that are wrapped.
EXTRA_STRUCTURAL_WRAPS = (
    ("torch.autograd", "backward", "torch.autograd.backward"),
    ("torch.autograd", "grad", "torch.autograd.grad"),
    ("torch.autograd.functional", "hessian", "torch.autograd.functional.hessian"),
    ("torch.autograd.functional", "jacobian", "torch.autograd.functional.jacobian"),
    ("torch.autograd.functional", "jvp", "torch.autograd.functional.jvp"),
    ("torch.autograd.functional", "vjp", "torch.autograd.functional.vjp"),
    ("torch.autograd.functional", "hvp", "torch.autograd.functional.hvp"),
    ("torch.autograd.functional", "vhp", "torch.autograd.functional.vhp"),
    ("torch.cuda", "synchronize", "torch.cuda.synchronize"),
    ("torch.cuda", "current_device", "torch.cuda.current_device"),
    ("torch.cuda", "device_count", "torch.cuda.device_count"),
    ("torch.cuda", "empty_cache", "torch.cuda.empty_cache"),
    ("torch.cuda", "manual_seed", "torch.cuda.manual_seed"),
    ("torch.cuda", "memory_allocated", "torch.cuda.memory_allocated"),
    ("torch.cuda", "reset_peak_memory_stats", "torch.cuda.reset_peak_memory_stats"),
    ("torch.cuda", "set_device", "torch.cuda.set_device"),
    ("torch.jit", "script", "torch.jit.script"),
    ("torch.jit", "trace", "torch.jit.trace"),
    # Top-level ATen wrappers also reachable as Tensor methods; both routes bracketed.
    ("torch", "argmax", "torch.argmax"),
    ("torch", "sum", "torch.sum"),
    ("torch", "eq", "torch.eq"),
    ("torch", "mean", "torch.mean"),
    ("torch", "max", "torch.max"),
    # Functional losses: outer Python facade gets user-frame attribution.
    ("torch.nn.functional", "nll_loss", "torch.nn.functional.nll_loss"),
    ("torch.nn.functional", "cross_entropy", "torch.nn.functional.cross_entropy"),
    ("torch.nn.functional", "mse_loss", "torch.nn.functional.mse_loss"),
    ("torch.nn.functional", "log_softmax", "torch.nn.functional.log_softmax"),
    ("torch.nn.functional", "softmax", "torch.nn.functional.softmax"),
    ("torch.nn.functional", "relu", "torch.nn.functional.relu"),
    ("torch", "randn", "torch.randn"),
    ("torch", "rand", "torch.rand"),
    ("torch", "zeros", "torch.zeros"),
    ("torch", "ones", "torch.ones"),
    ("torch", "empty", "torch.empty"),
    ("torch", "tensor", "torch.tensor"),
)


# Tensor methods often used as entry points (e.g. output.argmax(dim=1)).
TENSOR_METHOD_WRAPS = (
    "item",
    "argmax",
    "sum",
    "mean",
    "max",
    "eq",
    "numpy",
    "tolist",
)

DEEP_TENSOR_METHOD_WRAPS = (
    "to",
    "cpu",
    "cuda",
    "contiguous",
)
