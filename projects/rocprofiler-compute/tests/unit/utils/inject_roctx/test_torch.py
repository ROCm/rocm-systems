# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from types import SimpleNamespace

from utils.inject_roctx._backends.torch import format_wrap_args


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
