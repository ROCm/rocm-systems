# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Header/library version mismatch checking (NCCL 2.32.6, upstream 4821370).

``Communicator.properties`` and ``DevCommResource._allocate`` hand NCCL
versioned structs stamped with the bindings' header version, which an older
``librccl.so`` rejects. Both must raise :class:`RuntimeError` before reaching
the library when it reports a version older than the bindings'.

Standard library only: the nccl.core sources are loaded from this tree with
the compiled bindings, numpy, packaging and cuda.core replaced by stubs, so
neither a wheel build nor a GPU is needed. Run directly with ``python3``
(3.8 or newer).
"""

import dataclasses
import functools
import importlib
import os
import sys
import types
import typing
import unittest

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nccl")
_HEADER_VERSION = "2.32.6"


class _ReachedLibrary(Exception):
    """A call into the (stubbed) compiled bindings."""


class _StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stub_class(name)

    def __or__(cls, other):
        return cls

    __ror__ = __or__

    def __call__(cls, *args, **kwargs):
        # Stub decorators hand back what they wrap; anything else is a stub.
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return cls

    def __getitem__(cls, item):
        return cls


def _stub_class(name):
    return _StubMeta(name, (), {})


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stub_class(name)


class _Version(tuple):
    """packaging.version.Version for plain X.Y.Z strings."""

    def __new__(cls, text):
        return super().__new__(cls, (int(p) for p in str(text).split(".")))

    def __str__(self):
        return ".".join(map(str, self))


class _FakeBindings(_StubModule):
    """nccl.bindings.nccl: every function call is a call into the library."""

    lib_version = 23206

    def __getattr__(self, name):
        if name[:1].islower():
            def _call(*args, **kwargs):
                raise _ReachedLibrary(name)
            return _call
        return super().__getattr__(name)

    def get_version(self):
        return self.lib_version


def _dataclass_compat(dataclass):
    def compat(cls=None, **kwargs):
        kwargs.pop("slots", None)
        kwargs.pop("kw_only", None)
        return dataclass(**kwargs) if cls is None else dataclass(cls, **kwargs)
    return compat


def _load():
    bindings = _FakeBindings("nccl.bindings.nccl")
    bindings.__version__ = _HEADER_VERSION
    bindings.NCCLError = type("NCCLError", (Exception,), {})
    stubs = {"nccl.bindings.nccl": bindings}
    for name in ("numpy", "cuda", "cuda.core", "cuda.core.typing", "cuda.core.utils",
                 "cuda.core.experimental", "nccl.bindings", "packaging"):
        stubs[name] = _StubModule(name)
    stubs["nccl.bindings"].nccl = bindings
    stubs["packaging.version"] = _StubModule("packaging.version")
    stubs["packaging.version"].Version = _Version
    for pkg, path in (("nccl", _SRC), ("nccl.core", os.path.join(_SRC, "core"))):
        mod = types.ModuleType(pkg)
        mod.__path__ = [os.path.abspath(path)]
        stubs[pkg] = mod
    # Python < 3.10 lacks names and dataclass options the sources use at
    # import time; lend stand-ins for the import only.
    patches = [(typing, "TypeAlias", object), (functools, "cache", functools.lru_cache(None))]
    if sys.version_info < (3, 10):
        patches.append((dataclasses, "dataclass", _dataclass_compat(dataclasses.dataclass)))
    patches = [(m, n, getattr(m, n, None), v) for m, n, v in patches
               if n == "dataclass" or not hasattr(m, n)]
    for mod, name, _, value in patches:
        setattr(mod, name, value)
    # The loaded modules keep their stubbed globals; sys.modules is restored
    # so a shared pytest session still sees the real packages.
    saved = dict(sys.modules)
    sys.modules.update(stubs)
    try:
        return (bindings, importlib.import_module("nccl.core.communicator"),
                importlib.import_module("nccl.core.resources"))
    finally:
        for mod, name, orig, _ in patches:
            if orig is None:
                delattr(mod, name)
            else:
                setattr(mod, name, orig)
        for name in [k for k in sys.modules if saved.get(k) is not sys.modules[k]]:
            if name in saved:
                sys.modules[name] = saved[name]
            else:
                del sys.modules[name]


BINDINGS, COMMUNICATOR, RESOURCES = _load()
MISMATCH = "no older than the nccl4py bindings"


def _communicator():
    comm = COMMUNICATOR.Communicator.__new__(COMMUNICATOR.Communicator)
    comm._check_valid = lambda *args: None
    comm._comm = object()
    comm._comm_properties = None
    return comm


def _dev_comm_resource():
    res = RESOURCES.DevCommResource.__new__(RESOURCES.DevCommResource)
    res._comm = object()
    res._reqs_lowpp = types.SimpleNamespace(ptr=0)
    res._dev_comm = None
    return res


class VersionCheck(unittest.TestCase):
    def setUp(self):
        BINDINGS.lib_version = 23206

    def test_properties_rejects_older_library(self):
        BINDINGS.lib_version = 23100
        with self.assertRaisesRegex(RuntimeError, MISMATCH):
            _communicator().properties

    def test_dev_comm_allocate_rejects_older_library(self):
        BINDINGS.lib_version = 22900
        with self.assertRaisesRegex(RuntimeError, MISMATCH):
            _dev_comm_resource()._allocate()


class Control(unittest.TestCase):
    """Matching library: both calls go through to ncclCommQueryProperties /
    ncclDevCommCreate on either tree."""

    def setUp(self):
        BINDINGS.lib_version = 23206

    def test_properties_reaches_library(self):
        with self.assertRaisesRegex(_ReachedLibrary, "comm_query_properties"):
            _communicator().properties

    def test_dev_comm_allocate_reaches_library(self):
        with self.assertRaisesRegex(_ReachedLibrary, "dev_comm_create"):
            _dev_comm_resource()._allocate()


if __name__ == "__main__":
    unittest.main()
