#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amdsmi_get_gpu_process_info_self() binding tests."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


_PACKAGE_DIR = Path(__file__).resolve().parents[4] / "py-interface"


def _load_source_package() -> types.ModuleType:
    package = types.ModuleType("amdsmi_process_info_self_under_test")
    package.__path__ = [str(_PACKAGE_DIR)]
    sys.modules[package.__name__] = package
    package.amdsmi_wrapper = importlib.import_module(f"{package.__name__}.amdsmi_wrapper")
    package.amdsmi_interface = importlib.import_module(f"{package.__name__}.amdsmi_interface")
    return package


# The installed test tree has no py-interface beside it; test the installed package there.
if _PACKAGE_DIR.is_dir():
    amdsmi = _load_source_package()
else:
    from common.common import amdsmi


class TestProcessInfoSelf(unittest.TestCase):
    def test_reports_usage(self) -> None:
        wrapper = amdsmi.amdsmi_wrapper
        handle = wrapper.amdsmi_processor_handle()

        def _stub(_handle: object, info_ptr: object) -> int:
            info = info_ptr._obj
            info.pid = 42
            info.memory_usage.vram_mem = 64 << 20
            info.mem = 64 << 20
            info.cu_occupancy = 0xFFFFFFFF
            info.sdma_usage = 0xFFFFFFFFFFFFFFFF
            info.evicted_time = 0xFFFFFFFF
            return wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(wrapper, "amdsmi_get_gpu_process_info_self", _stub, create=True):
            info = amdsmi.amdsmi_interface.amdsmi_get_gpu_process_info_self(handle)

        self.assertEqual(info["pid"], 42)
        self.assertEqual(info["memory_usage"]["vram_mem"], 64 << 20)
        self.assertEqual(info["cu_occupancy"], "N/A")

    # A libamd_smi.so older than this API leaves the wrapper attribute unbound.
    def test_older_library_raises_not_supported(self) -> None:
        wrapper = amdsmi.amdsmi_wrapper
        iface = amdsmi.amdsmi_interface
        handle = wrapper.amdsmi_processor_handle()
        old_wrapper = types.ModuleType(wrapper.__name__)
        old_wrapper.__dict__.update(
            {k: v for k, v in vars(wrapper).items() if k != "amdsmi_get_gpu_process_info_self"}
        )
        with mock.patch.object(iface, "amdsmi_wrapper", old_wrapper):
            with self.assertRaises(iface.AmdSmiLibraryException) as ctx:
                iface.amdsmi_get_gpu_process_info_self(handle)
        self.assertEqual(ctx.exception.get_error_code(), wrapper.AMDSMI_STATUS_NOT_SUPPORTED)
