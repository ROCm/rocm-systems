#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""``amdsmi_get_nic_rdma_port_statistics`` follows the two-call pattern and returns a dict.

The checkout's ``py-interface`` is loaded as a package of its own, because the installed ``amdsmi``
can predate the NIC functions. Only the C call is patched.
"""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve()


def _find_py_interface():
    """Return the checkout's py-interface directory, or None (installed test tree)."""
    for directory in _HERE.parents:
        candidate = directory / "py-interface"
        if (candidate / "amdsmi_interface.py").is_file():
            return candidate
    return None


_PY_INTERFACE = _find_py_interface()
_PACKAGE = "amdsmi_checkout"
iface = None
wrapper = None
exceptions = None
_LOAD_ERROR = None

if _PY_INTERFACE is not None:
    try:
        _package = types.ModuleType(_PACKAGE)
        _package.__path__ = [str(_PY_INTERFACE)]
        sys.modules[_PACKAGE] = _package
        iface = importlib.import_module(f"{_PACKAGE}.amdsmi_interface")
        wrapper = importlib.import_module(f"{_PACKAGE}.amdsmi_wrapper")
        exceptions = importlib.import_module(f"{_PACKAGE}.amdsmi_exception")
    except (ImportError, OSError) as error:
        _LOAD_ERROR = error


def setUpModule():
    if _PY_INTERFACE is None:
        raise unittest.SkipTest("py-interface source not found (installed test tree)")
    if _LOAD_ERROR is not None:
        raise unittest.SkipTest(f"py-interface not importable: {_LOAD_ERROR}")


STATS = {"rx_rdma_ucast_pkts": 7, "tx_rdma_ucast_pkts": 9, "qps_created": 0}


def _fake_c_call(stats_by_name, status=0):
    """A stand-in for the C function: count on the first call, fill the array on the second."""
    calls = []

    def _call(handle, rdma_port_index, num_stats_ref, stats):
        calls.append((rdma_port_index, stats is None))
        if status != 0:
            return status
        if stats is None:
            num_stats_ref._obj.value = len(stats_by_name)
            return 0
        for index, (name, value) in enumerate(stats_by_name.items()):
            stats[index].name = name.encode("utf-8")
            stats[index].value = value
        num_stats_ref._obj.value = len(stats_by_name)
        return 0

    return _call, calls


class TestNicRdmaPortStatistics(unittest.TestCase):
    def setUp(self):
        self.handle = wrapper.amdsmi_processor_handle()

    def _patched(self, c_call):
        return mock.patch.object(
            wrapper, "amdsmi_get_nic_rdma_port_statistics", side_effect=c_call, create=True
        )

    def test_returns_name_to_value_dict(self):
        c_call, _ = _fake_c_call(STATS)
        with self._patched(c_call):
            result = iface.amdsmi_get_nic_rdma_port_statistics(self.handle, 0)
        self.assertEqual(result, STATS)

    def test_counts_first_then_fetches_for_the_requested_port(self):
        c_call, calls = _fake_c_call(STATS)
        with self._patched(c_call):
            iface.amdsmi_get_nic_rdma_port_statistics(self.handle, 2)
        self.assertEqual(calls, [(2, True), (2, False)])

    def test_zero_counters_returns_empty_without_a_second_call(self):
        c_call, calls = _fake_c_call({})
        with self._patched(c_call):
            result = iface.amdsmi_get_nic_rdma_port_statistics(self.handle, 0)
        self.assertEqual(result, {})
        self.assertEqual(calls, [(0, True)])

    def test_library_error_is_raised(self):
        c_call, _ = _fake_c_call(STATS, status=wrapper.AMDSMI_STATUS_NOT_SUPPORTED)
        with self._patched(c_call):
            with self.assertRaises(exceptions.AmdSmiLibraryException):
                iface.amdsmi_get_nic_rdma_port_statistics(self.handle, 5)

    def test_non_handle_is_rejected(self):
        with self.assertRaises(exceptions.AmdSmiParameterException):
            iface.amdsmi_get_nic_rdma_port_statistics("not-a-handle", 0)


if __name__ == "__main__":
    unittest.main()
