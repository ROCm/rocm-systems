# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CPU regressions for Torch metadata and inherited Function.apply wrappers."""

import pytest
from integration.common import require_torch

from utils.inject_roctx import core
from utils.inject_roctx._backends import torch as torch_backend


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


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(UnreadableShape(), id="shape"),
        pytest.param(UnreadableDtype(), id="dtype"),
    ],
)
@pytest.mark.parametrize("argument_kind", ["positional", "keyword"])
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
    torch_module, monkeypatch, conversion
):
    markers = []
    monkeypatch.setattr(core._STATE, "range_push", markers.append)
    monkeypatch.setattr(core._STATE, "range_pop", lambda: None)
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
    assert len(markers) == 1
    assert markers[0].startswith("torch.Tensor.to:")
    assert "|args=n/a|torch" in markers[0]


@pytest.mark.parametrize(
    "created_after_install", [False, True], ids=["existing-child", "new-child"]
)
def test_inherited_function_apply_is_wrapped_once(
    torch_module, monkeypatch, recorded_ranges, created_after_install
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

    inherited = None
    if not created_after_install:
        inherited = type("InheritedScale", (Scale,), {})

    monkeypatch.setattr(torch_backend._STATE, "function", FunctionBase)
    assert torch_backend.install_function_apply_wrappers()
    assert torch_backend.install_function_apply_wrappers()
    if created_after_install:
        inherited = type("InheritedScale", (Scale,), {})

    tensor = torch_module.tensor(3.0, requires_grad=True)
    output = inherited.apply(tensor)
    output.backward()

    assert output.item() == 6.0
    assert tensor.grad.item() == 2.0
    assert len(recorded_ranges) == 1
    assert recorded_ranges[0].startswith("torch.autograd.Function.apply:")
    assert recorded_ranges[0].endswith("|args=n/a|torch")
