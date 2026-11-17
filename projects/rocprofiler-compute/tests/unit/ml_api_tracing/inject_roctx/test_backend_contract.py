# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for registration and lookup of recording backends."""

import types


def make_backend(name, install_fn=None):
    backend = types.SimpleNamespace()
    backend.name = name
    backend.install = install_fn or (lambda: None)
    return backend


class FalseBackend(types.SimpleNamespace):
    def __bool__(self):
        return False


def test_register_overwrites_by_name_and_keeps_false_backend(fresh_registry):
    original = make_backend("alpha")
    replacement = FalseBackend(name="alpha", install=lambda: None)
    fresh_registry.register(original)
    fresh_registry.register(replacement)
    assert fresh_registry.get_backend("alpha") is replacement
    assert fresh_registry.get_backend("missing") is None
