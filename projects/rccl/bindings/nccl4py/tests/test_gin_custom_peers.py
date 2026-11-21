# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the ``nccl.core`` GIN custom peer array API.

``NcclGinConnectionType.CUSTOM_ARRAY`` and
``NCCLDevCommRequirements.gin_custom_peers`` feed ``ginCustomArray`` /
``ginCustomArrayCount`` of ``ncclDevCommRequirements`` through
``Communicator.create_dev_comm``. Importing ``nccl.core`` needs the compiled
bindings and hip-python, so these tests load the pieces they drive straight
from the package sources and need only the standard library (a small numpy
stand-in drives ``_materialize_gin_custom_peers``). Run with pytest or
``python3 tests/test_gin_custom_peers.py [TestCase]``.
"""

import __future__
import ast
import enum
import os
import unittest

_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nccl", "core")


def _parse(name):
    with open(os.path.join(_CORE, name)) as fh:
        return ast.parse(fh.read())


def _node(tree, kind, name):
    for node in ast.walk(tree):
        if isinstance(node, kind) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in nccl.core")


def _load(node, namespace):
    code = compile(ast.Module(body=[node], type_ignores=[]), "<nccl.core>", "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, namespace)
    return namespace[node.name]


def _fields(cls_name):
    reqs = _node(_parse("communicator.py"), ast.ClassDef, cls_name)
    return {n.target.id: n for n in reqs.body if isinstance(n, ast.AnnAssign)}


class _Array(list):
    """Just enough of ``numpy.ndarray`` for ``_materialize_gin_custom_peers``."""

    def __init__(self, data, dtype):
        super().__init__(data)
        self.dtype = dtype
        self.ndim = 2 if any(isinstance(x, (list, tuple)) for x in data) else 1
        self.size = len(data)


class _Np:
    int32 = "int32"
    ndarray = _Array

    @staticmethod
    def asarray(data, dtype=None):
        return _Array(data, dtype)


class GinCustomArrayTests(unittest.TestCase):
    def test_custom_array_connection_type(self):
        cls = _load(_node(_parse("typing.py"), ast.ClassDef, "NcclGinConnectionType"),
                    {"IntEnum": enum.IntEnum})
        self.assertEqual(int(cls.CUSTOM_ARRAY), 4)  # NCCL_GIN_CONNECTION_CUSTOM_ARRAY

    def test_materialize_gin_custom_peers(self):
        fn = _load(_node(_parse("communicator.py"), ast.FunctionDef,
                         "_materialize_gin_custom_peers"),
                   {"_np": _Np, "Sequence": list})
        self.assertIsNone(fn(None))
        self.assertIsNone(fn([]))
        arr = fn((5, 1, 3))
        self.assertEqual((arr.dtype, list(arr)), ("int32", [5, 1, 3]))
        with self.assertRaises(ValueError):
            fn([[0, 1], [2, 3]])

    def test_requirements_expose_gin_custom_peers(self):
        default = _fields("NCCLDevCommRequirements").get("gin_custom_peers")
        self.assertIsNotNone(default, "NCCLDevCommRequirements lacks gin_custom_peers")
        self.assertIsInstance(default.value, ast.Constant)
        self.assertIsNone(default.value.value)

    def test_create_dev_comm_sets_custom_array(self):
        fn = _node(_parse("communicator.py"), ast.FunctionDef, "create_dev_comm")
        guarded = [n for n in ast.walk(fn) if isinstance(n, ast.If) and any(
            isinstance(a, ast.Attribute) and a.attr == "CUSTOM_ARRAY" for a in ast.walk(n.test))]
        self.assertTrue(guarded, "create_dev_comm does not handle CUSTOM_ARRAY")
        body = guarded[0].body
        calls = {c.func.id for n in body for c in ast.walk(n)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        self.assertIn("_materialize_gin_custom_peers", calls)
        targets = {t.attr for n in body if isinstance(n, ast.Assign) for t in n.targets
                   if isinstance(t, ast.Attribute)}
        self.assertLessEqual({"gin_custom_array", "gin_custom_array_count"}, targets)


class ControlGinCustomStrideTests(unittest.TestCase):
    def test_control_custom_stride_connection_type(self):
        cls = _load(_node(_parse("typing.py"), ast.ClassDef, "NcclGinConnectionType"),
                    {"IntEnum": enum.IntEnum})
        self.assertEqual(int(cls.CUSTOM_STRIDE), 3)
        self.assertIn("gin_custom_stride", _fields("NCCLDevCommRequirements"))
        _node(_parse("communicator.py"), ast.FunctionDef, "create_dev_comm")


if __name__ == "__main__":
    unittest.main()
