# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from types import SimpleNamespace

import pytest

from utils.inject_roctx._backends.torch import (
    DEEP_TENSOR_METHOD_WRAPS,
    TENSOR_METHOD_WRAPS,
    format_wrap_args,
)


class FakeDType:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name


def fake_tensor(shape=(2, 4), dtype_name="torch.float32"):
    return SimpleNamespace(shape=shape, dtype=FakeDType(dtype_name))


def noop_torch_structural_wraps(monkeypatch, torch_backend):
    for name in (
        "patch_distributed_collectives",
        "patch_process_group_methods",
        "patch_cuda_graph",
        "patch_compile_callable",
        "install_tensor_backward_wrapper",
        "inject_roctx_into_optimizer",
        "install_function_apply_wrappers",
        "install_tensor_method_wrappers",
        "install_extra_structural_wrappers",
        "inject_roctx_into_model",
        "inject_roctx_into_module_methods",
    ):
        monkeypatch.setattr(torch_backend, name, lambda *args, **kwargs: None)


def install_tensor_method_wrappers_for_test(monkeypatch, torch_backend, torch_module):
    for method_name in TENSOR_METHOD_WRAPS + DEEP_TENSOR_METHOD_WRAPS:
        method = getattr(torch_module.Tensor, method_name, None)
        if method is not None:
            monkeypatch.setattr(torch_module.Tensor, method_name, method)
    torch_backend.install_tensor_method_wrappers()


def test_format_wrap_args_renders_tensors_and_skips_non_tensors():
    tensor = fake_tensor()
    rendered = format_wrap_args((tensor, "skip"), {"bias": tensor, "flag": True})
    assert rendered == "(float32[2x4], bias=float32[2x4])"
    assert format_wrap_args((), {}) == "n/a"


def test_dispatcher_marker_name_for_aten_packet():
    from utils.inject_roctx._backends.torch import dispatcher_marker_name_for

    packet = SimpleNamespace(_qualified_op_name="aten::addmm")
    func = SimpleNamespace(overloadpacket=packet)
    assert dispatcher_marker_name_for(func) == "torch.ops.aten.addmm"


def test_roctx_wrapper_idempotent_and_push_pop(monkeypatch):
    from utils.inject_roctx._backends import torch as torch_backend

    pushes = []
    pops = []
    monkeypatch.setattr(
        torch_backend,
        "_push_scope",
        lambda *args, **kwargs: pushes.append(1),
    )
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: pops.append(1))
    wrapped = torch_backend.roctx_wrapper(lambda value: value + 1)
    assert wrapped(1) == 2
    assert pushes == [1]
    assert pops == [1]
    assert torch_backend.roctx_wrapper(wrapped) is wrapped


def test_inject_roctx_into_model_pushes_class_forward(monkeypatch):
    from tests.integration.common import require_torch
    from utils.inject_roctx._backends import torch as torch_backend

    require_torch()
    import torch

    torch_backend._resolve_torch()
    pushes = []
    monkeypatch.setattr(
        torch_backend,
        "_push_scope",
        lambda name, location, backend="", args="n/a": pushes.append((name, backend)),
    )
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: None)
    original_call = torch.nn.Module.__call__
    try:
        torch_backend.inject_roctx_into_model()
        torch.nn.Linear(2, 2)(torch.zeros(1, 2))
        assert ("nn.Module.Linear.forward", "torch") in pushes
    finally:
        torch.nn.Module.__call__ = original_call


def test_torch_backend_install_skips_dispatcher_when_collector_loads(monkeypatch):
    from utils.inject_roctx._backends import torch as torch_backend

    noop_torch_structural_wraps(monkeypatch, torch_backend)
    monkeypatch.setattr(torch_backend.torch_trace_collector, "install", lambda: True)
    hook_calls = []
    monkeypatch.setattr(
        torch_backend,
        "install_dispatcher_hook",
        lambda: hook_calls.append(1),
    )
    monkeypatch.setattr(torch_backend, "_ROCTX_AVAILABLE", True)
    monkeypatch.setattr(torch_backend, "_resolve_torch", lambda: True)
    torch_backend.TorchBackend().install()
    assert hook_calls == []


def test_torch_backend_install_falls_back_to_dispatch_mode(monkeypatch):
    from utils.inject_roctx._backends import torch as torch_backend

    noop_torch_structural_wraps(monkeypatch, torch_backend)
    monkeypatch.setattr(torch_backend.torch_trace_collector, "install", lambda: False)
    hook_calls = []
    monkeypatch.setattr(
        torch_backend,
        "install_dispatcher_hook",
        lambda: hook_calls.append(1),
    )
    monkeypatch.setattr(torch_backend, "_ROCTX_AVAILABLE", True)
    monkeypatch.setattr(torch_backend, "_resolve_torch", lambda: True)
    torch_backend.TorchBackend().install()
    assert hook_calls == [1]


def test_deep_tensor_method_wraps_disabled_by_env(monkeypatch):
    from utils.inject_roctx._backends.torch import (
        TENSOR_METHOD_WRAPS,
        _selected_tensor_method_wraps,
    )

    monkeypatch.setenv("ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS", "0")
    assert _selected_tensor_method_wraps() == TENSOR_METHOD_WRAPS


def test_function_apply_wrappers_idempotent(monkeypatch):
    from tests.integration.common import require_torch
    from utils.inject_roctx._backends import torch as torch_backend

    require_torch()
    import torch

    if not torch_backend._resolve_torch():
        pytest.skip("torch could not be resolved for inject_roctx backend")

    push_counter = {"count": 0}

    def _count_push(*_args, **_kwargs):
        push_counter["count"] += 1

    monkeypatch.setattr(torch_backend, "_push_scope", _count_push)
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: None)

    class Foo(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x + 1

        @staticmethod
        def backward(ctx, grad_out):
            return grad_out

    class Bar(Foo):
        pass

    assert torch_backend.install_function_apply_wrappers() is True
    assert getattr(
        getattr(Foo.__dict__.get("apply"), "__func__", None),
        "_roctx_wrapped",
        False,
    )
    assert "apply" not in Bar.__dict__

    x = torch.tensor(1.0, requires_grad=True)
    y = Bar.apply(x)
    y.backward()
    assert push_counter["count"] == 1


def test_lazy_linear_to_under_tensor_wraps(monkeypatch):
    from tests.integration.common import require_torch
    from utils.inject_roctx._backends import torch as torch_backend

    require_torch()
    import torch
    from torch.nn.parameter import UninitializedParameter

    if not torch_backend._resolve_torch():
        pytest.skip("torch could not be resolved for inject_roctx backend")

    pushes = []
    monkeypatch.setattr(
        torch_backend,
        "_push_scope",
        lambda name, location, backend="", args="n/a": pushes.append(name),
    )
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: None)
    install_tensor_method_wrappers_for_test(monkeypatch, torch_backend, torch)

    model = torch.nn.LazyLinear(10).to("cpu")
    assert isinstance(model.weight, UninitializedParameter)
    torch.zeros(1).to("cpu")
    assert "torch.Tensor.to" in pushes
    assert torch.Tensor.to.__name__ == "to"


def test_uninitialized_tensor_item_still_raises(monkeypatch):
    from tests.integration.common import require_torch
    from utils.inject_roctx._backends import torch as torch_backend

    require_torch()
    import torch
    from torch.nn.parameter import UninitializedParameter

    if not torch_backend._resolve_torch():
        pytest.skip("torch could not be resolved for inject_roctx backend")

    param = UninitializedParameter()
    with pytest.raises(ValueError, match="uninitialized"):
        param.item()
    with pytest.raises(ValueError, match="uninitialized"):
        param.contiguous()

    monkeypatch.setattr(torch_backend, "_push_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: None)
    install_tensor_method_wrappers_for_test(monkeypatch, torch_backend, torch)

    wrapped = UninitializedParameter()
    with pytest.raises(ValueError, match="uninitialized"):
        wrapped.item()
    with pytest.raises(ValueError, match="uninitialized"):
        wrapped.contiguous()


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
    install_tensor_method_wrappers_for_test(monkeypatch, torch_backend, torch)

    model = torch.nn.LazyBatchNorm1d().to("cuda")
    assert isinstance(model.running_mean, UninitializedBuffer)
    assert isinstance(model.running_var, UninitializedBuffer)
    assert model.running_mean.device.type == "cuda"
    model(torch.randn(2, 4, device="cuda"))


def test_lazy_batch_norm_to_dtype_keeps_uninitialized_buffer(monkeypatch):
    from tests.integration.common import require_torch
    from utils.inject_roctx._backends import torch as torch_backend

    require_torch()
    import torch
    from torch.nn.parameter import UninitializedBuffer

    if not torch_backend._resolve_torch():
        pytest.skip("torch could not be resolved for inject_roctx backend")

    monkeypatch.setattr(torch_backend, "_push_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: None)
    install_tensor_method_wrappers_for_test(monkeypatch, torch_backend, torch)

    model = torch.nn.LazyBatchNorm1d().to(dtype=torch.float64)
    assert isinstance(model.running_mean, UninitializedBuffer)
    assert isinstance(model.running_var, UninitializedBuffer)
    assert model.running_mean.dtype == torch.float64
    model(torch.randn(2, 4, dtype=torch.float64))


def test_uninitialized_buffer_keeps_persistent_on_to(monkeypatch):
    from tests.integration.common import require_torch
    from utils.inject_roctx._backends import torch as torch_backend

    require_torch()
    import torch
    from torch.nn.parameter import UninitializedBuffer

    if not torch_backend._resolve_torch():
        pytest.skip("torch could not be resolved for inject_roctx backend")

    monkeypatch.setattr(torch_backend, "_push_scope", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: None)
    install_tensor_method_wrappers_for_test(monkeypatch, torch_backend, torch)

    moved = UninitializedBuffer(persistent=False).to(dtype=torch.float64)
    assert isinstance(moved, UninitializedBuffer)
    assert moved.persistent is False

