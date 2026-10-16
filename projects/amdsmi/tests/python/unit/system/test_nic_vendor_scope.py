#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NIC vendor statistics tier: chosen with a setter, read with the previous 4-argument getter.

``amdsmi_get_nic_vendor_statistics`` keeps its previous signature, so the Python function selects
the tier through ``amdsmi_set_nic_stat_scope`` before it queries. The shipped
``amdsmi_interface.py`` needs a built ``libamd_smi.so`` to import, so the function is compiled
from the real source and run against a fake wrapper that records the order of the calls.
"""

import ast
import ctypes
import types
import unittest
from typing import Any, Dict, List, Tuple

from common.common import amdsmi_interface_path

_INTERFACE_PATH = amdsmi_interface_path(__file__)
_FUNCTION_NAME = "amdsmi_get_nic_vendor_statistics"

_STATUS_SUCCESS = 0
_STATUS_FAILURE = 1
_STATUS_RESET_FAILURE = 2
_SCOPE_DEFAULT = 0
_SCOPE_EXTENDED = 1
_PORT_INDEX = 2
_NAME_BYTES = 256
_COUNTER_NAME = b"tx_packets"
_COUNTER_VALUE = 9


class _FakeParameterException(Exception):
    """Stands in for AmdSmiParameterException."""


class _FakeHandle:
    """Stands in for the ctypes amdsmi_processor_handle type."""


class _Stat(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * _NAME_BYTES), ("value", ctypes.c_uint64)]


def _fake_check_res(status: int) -> None:
    if status != _STATUS_SUCCESS:
        raise AssertionError(f"unexpected status {status}")


def _load_function(wrapper: types.SimpleNamespace) -> Any:
    tree = ast.parse(_INTERFACE_PATH.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == _FUNCTION_NAME]
    if len(nodes) != 1:
        raise AssertionError(f"{_FUNCTION_NAME} not found in amdsmi_interface.py")
    namespace: Dict[str, Any] = {
        "amdsmi_wrapper": wrapper,
        "ctypes": ctypes,
        "Dict": Dict,
        "AmdSmiParameterException": _FakeParameterException,
        "_check_res": _fake_check_res,
    }
    exec(compile(ast.Module(nodes, []), str(_INTERFACE_PATH), "exec"), namespace)
    return namespace[_FUNCTION_NAME]


def _fake_wrapper() -> Tuple[types.SimpleNamespace, List[Tuple[str, tuple]]]:
    """A wrapper that reports one counter and logs every set and get call in order."""
    calls: List[Tuple[str, tuple]] = []

    def fake_set_scope(handle: object, scope: int) -> int:
        calls.append(("set", (handle, scope)))
        return _STATUS_SUCCESS

    def fake_get_stats(*args: Any) -> int:
        calls.append(("get", args))
        num_stats, stats = args[-2], args[-1]
        num_stats._obj.value = 1
        if stats is not None:
            stats[0].name = _COUNTER_NAME
            stats[0].value = _COUNTER_VALUE
        return _STATUS_SUCCESS

    wrapper = types.SimpleNamespace(
        amdsmi_processor_handle=_FakeHandle,
        amdsmi_nic_stat_t=_Stat,
        AMDSMI_NIC_STAT_SCOPE_DEFAULT=_SCOPE_DEFAULT,
        amdsmi_set_nic_stat_scope=fake_set_scope,
        amdsmi_get_nic_vendor_statistics=fake_get_stats,
    )
    return wrapper, calls


class TestNicVendorStatisticsScope(unittest.TestCase):
    def test_the_requested_scope_is_set_before_the_first_query(self) -> None:
        wrapper, calls = _fake_wrapper()
        handle = _FakeHandle()

        result = _load_function(wrapper)(handle, _PORT_INDEX, scope=_SCOPE_EXTENDED)

        self.assertEqual(result, {_COUNTER_NAME.decode(): _COUNTER_VALUE})
        self.assertEqual([kind for kind, _ in calls][:3], ["set", "get", "get"])
        self.assertEqual(calls[0][1], (handle, _SCOPE_EXTENDED))

    def test_an_omitted_scope_asks_for_the_default_tier(self) -> None:
        wrapper, calls = _fake_wrapper()

        _load_function(wrapper)(_FakeHandle(), _PORT_INDEX)

        self.assertEqual(calls[0][1][1], _SCOPE_DEFAULT)

    def test_the_handle_is_left_on_the_default_tier(self) -> None:
        wrapper, calls = _fake_wrapper()
        handle = _FakeHandle()

        _load_function(wrapper)(handle, _PORT_INDEX, scope=_SCOPE_EXTENDED)

        self.assertEqual(calls[-1], ("set", (handle, _SCOPE_DEFAULT)))

    def test_the_handle_is_left_on_the_default_tier_when_the_query_fails(self) -> None:
        wrapper, calls = _fake_wrapper()
        handle = _FakeHandle()
        wrapper.amdsmi_get_nic_vendor_statistics = lambda *args: _STATUS_FAILURE

        with self.assertRaises(AssertionError):
            _load_function(wrapper)(handle, _PORT_INDEX, scope=_SCOPE_EXTENDED)

        self.assertEqual(calls[-1], ("set", (handle, _SCOPE_DEFAULT)))

    def test_a_failed_reset_is_reported_when_the_query_succeeded(self) -> None:
        wrapper, calls = _fake_wrapper()
        reset_calls: List[int] = []

        def fake_set_scope(handle: object, scope: int) -> int:
            calls.append(("set", (handle, scope)))
            reset_calls.append(scope)
            return _STATUS_SUCCESS if len(reset_calls) == 1 else _STATUS_RESET_FAILURE

        wrapper.amdsmi_set_nic_stat_scope = fake_set_scope

        with self.assertRaisesRegex(AssertionError, f"status {_STATUS_RESET_FAILURE}"):
            _load_function(wrapper)(_FakeHandle(), _PORT_INDEX)

    def test_a_failed_reset_does_not_hide_the_query_error(self) -> None:
        wrapper, _ = _fake_wrapper()
        reset_calls: List[int] = []

        def fake_set_scope(handle: object, scope: int) -> int:
            reset_calls.append(scope)
            return _STATUS_SUCCESS if len(reset_calls) == 1 else _STATUS_RESET_FAILURE

        wrapper.amdsmi_set_nic_stat_scope = fake_set_scope
        wrapper.amdsmi_get_nic_vendor_statistics = lambda *args: _STATUS_FAILURE

        with self.assertRaisesRegex(AssertionError, f"status {_STATUS_FAILURE}"):
            _load_function(wrapper)(_FakeHandle(), _PORT_INDEX)

    def test_a_port_with_no_counters_still_resets_the_handle(self) -> None:
        wrapper, calls = _fake_wrapper()
        handle = _FakeHandle()

        def fake_get_stats(*args: Any) -> int:
            calls.append(("get", args))
            args[-2]._obj.value = 0
            return _STATUS_SUCCESS

        wrapper.amdsmi_get_nic_vendor_statistics = fake_get_stats

        self.assertEqual(_load_function(wrapper)(handle, _PORT_INDEX, scope=_SCOPE_EXTENDED), {})
        self.assertEqual(calls[-1], ("set", (handle, _SCOPE_DEFAULT)))

    def test_the_query_uses_the_four_argument_signature(self) -> None:
        wrapper, calls = _fake_wrapper()
        handle = _FakeHandle()

        _load_function(wrapper)(handle, _PORT_INDEX)

        get_calls = [args for kind, args in calls if kind == "get"]
        self.assertTrue(all(len(args) == 4 for args in get_calls))
        self.assertTrue(all(args[:2] == (handle, _PORT_INDEX) for args in get_calls))
