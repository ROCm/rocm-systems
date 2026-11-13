# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for Torch backend wrappers, metadata, and lazy tensors."""

from types import SimpleNamespace

import pytest

from tests.integration.common import require_torch
from utils.inject_roctx import core
from utils.inject_roctx._backends import torch as torch_backend
from utils.inject_roctx._backends.torch import (
    DEEP_TENSOR_METHOD_WRAPS,
    TENSOR_METHOD_WRAPS,
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


class UnreadableShape:
    dtype = "torch.float32"

    @property
    def shape(self):
        raise RuntimeError("shape is not available before initialization")


class UnreadableDtype:
    shape = (2, 3)

    @property
    def dtype(self):
        raise ValueError("dtype is unavailable")


@pytest.fixture
def torch_module():
    require_torch()
    import torch

    return torch


@pytest.fixture
def recorded_ranges(monkeypatch):
    markers = []
    active = []

    def push(marker):
        markers.append(marker)
        active.append(marker)

    monkeypatch.setattr(core._STATE, "range_push", push)
    monkeypatch.setattr(core._STATE, "range_pop", active.pop)
    monkeypatch.setattr(core._thread_local, "depth", 0, raising=False)
    yield markers
    assert active == []


def test_format_wrap_args_renders_tensors_and_skips_non_tensors():
    tensor = fake_tensor()
    rendered = torch_backend.format_wrap_args(
        (tensor, "skip"), {"bias": tensor, "flag": True}
    )
    assert rendered == "(float32[2x4], bias=float32[2x4])"
    assert torch_backend.format_wrap_args((), {}) == "n/a"


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


def test_lazy_linear_to_under_tensor_wraps(monkeypatch):
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


def test_lazy_batch_norm_to_dtype_keeps_uninitialized_buffer(monkeypatch):
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


@pytest.mark.parametrize(
    ("value", "argument_kind"),
    [
        pytest.param(UnreadableShape(), "positional", id="shape-positional"),
        pytest.param(UnreadableDtype(), "keyword", id="dtype-keyword"),
    ],
)
def test_format_wrap_args_omits_unreadable_metadata(value, argument_kind):
    args = (value,) if argument_kind == "positional" else ()
    kwargs = {"tensor": value} if argument_kind == "keyword" else {}

    assert torch_backend.format_wrap_args(args, kwargs) == "n/a"


def test_format_wrap_args_preserves_readable_tensors(torch_module):
    tensor = torch_module.ones(2, 3)
    scalar = torch_module.tensor(1.0, dtype=torch_module.float64)

    assert (
        torch_backend.format_wrap_args(
            (UnreadableShape(), tensor, 42),
            {"unreadable": UnreadableDtype(), "scalar": scalar},
        )
        == "(float32[2x3], scalar=float64[])"
    )


@pytest.mark.parametrize("conversion", ["device", "dtype"])
def test_direct_tensor_to_wrapper_converts_uninitialized_parameter(
    torch_module, recorded_ranges, conversion
):
    parameter = torch_module.nn.parameter.UninitializedParameter(
        device="cpu", dtype=torch_module.float32
    )
    kwargs = (
        {"device": "cpu"} if conversion == "device" else {"dtype": torch_module.float64}
    )
    original_to = torch_module.Tensor.to
    wrapped_to = torch_backend.roctx_wrapper(
        original_to, "torch.Tensor.to", backend="torch"
    )

    expected = original_to(parameter, **kwargs)
    converted = wrapped_to(parameter, **kwargs)

    assert isinstance(converted, torch_module.Tensor)
    assert converted.device == expected.device
    assert converted.dtype == expected.dtype
    assert len(recorded_ranges) == 1
    assert recorded_ranges[0].startswith("torch.Tensor.to:")
    assert "|args=n/a|torch" in recorded_ranges[0]


@pytest.mark.parametrize(
    "created_after_install", [False, True], ids=["existing-child", "new-child"]
)
@pytest.mark.parametrize("override_methods", [False, True], ids=["inherit", "override"])
def test_inherited_function_apply_is_wrapped_once(
    torch_module, monkeypatch, recorded_ranges, created_after_install, override_methods
):
    class FunctionBase(torch_module.autograd.Function):
        pass

    class Scale(FunctionBase):
        @staticmethod
        def forward(ctx, tensor):
            return tensor * 2

        @staticmethod
        def backward(ctx, gradient):
            return gradient * 2

    child_methods = {}
    if override_methods:

        def forward(ctx, tensor):
            return tensor * 3

        def backward(ctx, gradient):
            return gradient * 3

        child_methods = {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        }

    inherited = None
    if not created_after_install:
        inherited = type("InheritedScale", (Scale,), child_methods)

    monkeypatch.setattr(torch_backend._STATE, "function", FunctionBase)
    assert torch_backend.install_function_apply_wrappers()
    assert torch_backend.install_function_apply_wrappers()
    if created_after_install:
        inherited = type("InheritedScale", (Scale,), child_methods)

    assert getattr(
        getattr(Scale.__dict__.get("apply"), "__func__", None),
        "_roctx_wrapped",
        False,
    )
    assert "apply" not in inherited.__dict__

    tensor = torch_module.tensor(3.0, requires_grad=True)
    output = inherited.apply(tensor)
    output.backward()

    multiplier = 3 if override_methods else 2
    assert output.item() == 3.0 * multiplier
    assert tensor.grad.item() == multiplier
    assert len(recorded_ranges) == 1
    assert recorded_ranges[0].startswith("torch.autograd.Function.apply:")
    assert recorded_ranges[0].endswith("|args=n/a|torch")


@pytest.mark.parametrize("apply_kind", ["classmethod", "staticmethod"])
def test_custom_function_apply_preserves_binding(
    torch_module, monkeypatch, recorded_ranges, apply_kind
):
    class FunctionBase(torch_module.autograd.Function):
        pass

    def class_apply(cls, tensor):
        return tensor * cls.multiplier

    def static_apply(tensor):
        return tensor * 3

    class Scale(FunctionBase):
        multiplier = 2
        apply = (
            classmethod(class_apply)
            if apply_kind == "classmethod"
            else staticmethod(static_apply)
        )

    monkeypatch.setattr(torch_backend._STATE, "function", FunctionBase)
    assert torch_backend.install_function_apply_wrappers()

    class Child(Scale):
        multiplier = 3

    tensor = torch_module.tensor(3.0, requires_grad=True)
    output = Child.apply(tensor=tensor)
    output.backward()

    assert output.item() == 9.0
    assert tensor.grad.item() == 3.0
    assert len(recorded_ranges) == 1


@pytest.mark.parametrize("push_succeeds", [False, True])
@pytest.mark.parametrize("workload_raises", [False, True])
def test_torch_wrapper_only_pops_successful_launcher_push(
    monkeypatch, push_succeeds, workload_raises
):
    events = []
    monkeypatch.setattr(
        torch_backend, "_push_scope", lambda *_a, **_k: events.append("range")
    )
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: events.append("end range"))
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "push_launcher_tid",
        lambda: events.append("launcher push") or push_succeeds,
    )
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "pop_launcher_tid",
        lambda: events.append("launcher pop"),
    )

    def workload():
        events.append("workload")
        if workload_raises:
            raise ValueError("workload failed")
        return "result"

    wrapped = torch_backend.roctx_wrapper(workload, publish_launcher_tid=True)
    if workload_raises:
        with pytest.raises(ValueError, match="workload failed"):
            wrapped()
    else:
        assert wrapped() == "result"
    expected = ["range", "launcher push", "workload"]
    if push_succeeds:
        expected.append("launcher pop")
    assert events == expected + ["end range"]


def test_failed_nested_launcher_push_preserves_outer_launcher(monkeypatch):
    launcher_stack = []
    push_results = iter([True, False])
    monkeypatch.setattr(torch_backend, "_push_scope", lambda *_a, **_k: None)
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: None)

    def push_launcher():
        pushed = next(push_results)
        if pushed:
            launcher_stack.append("outer")
        return pushed

    monkeypatch.setattr(
        torch_backend.torch_trace_collector, "push_launcher_tid", push_launcher
    )
    monkeypatch.setattr(
        torch_backend.torch_trace_collector, "pop_launcher_tid", launcher_stack.pop
    )
    nested = torch_backend.roctx_wrapper(lambda: None, publish_launcher_tid=True)

    def outer():
        nested()
        assert launcher_stack == ["outer"]

    torch_backend.roctx_wrapper(outer, publish_launcher_tid=True)()
    assert launcher_stack == []
