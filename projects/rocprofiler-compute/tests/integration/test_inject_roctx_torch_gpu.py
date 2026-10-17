# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""GPU checks for TorchBackend tensor method wraps."""

import pytest

from utils.inject_roctx._backends.torch import (
    DEEP_TENSOR_METHOD_WRAPS,
    TENSOR_METHOD_WRAPS,
)


def _install_tensor_method_wrappers(monkeypatch, torch_backend, torch_module):
    for method_name in TENSOR_METHOD_WRAPS + DEEP_TENSOR_METHOD_WRAPS:
        method = getattr(torch_module.Tensor, method_name, None)
        if method is not None:
            monkeypatch.setattr(torch_module.Tensor, method_name, method)
    torch_backend.install_tensor_method_wrappers()


def test_lazy_batch_norm_to_cuda_keeps_uninitialized_buffer(monkeypatch):
    from tests.integration.common import require_torch
    from utils.inject_roctx._backends import torch as torch_backend

    require_torch(gpu=True)
    import torch
    from torch.nn.parameter import UninitializedBuffer

    if not torch_backend._resolve_torch():
        pytest.skip("torch could not be resolved for inject_roctx backend")

    monkeypatch.setattr(torch_backend, "_push_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: None)
    _install_tensor_method_wrappers(monkeypatch, torch_backend, torch)

    model = torch.nn.LazyBatchNorm1d().to("cuda")
    assert isinstance(model.running_mean, UninitializedBuffer)
    assert isinstance(model.running_var, UninitializedBuffer)
    assert model.running_mean.device.type == "cuda"
    model(torch.randn(2, 4, device="cuda"))
