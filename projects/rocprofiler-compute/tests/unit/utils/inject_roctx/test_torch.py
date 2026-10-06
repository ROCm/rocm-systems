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
