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
