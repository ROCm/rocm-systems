#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`amd-smi static --cuid` output shape and the CUID component binding.

Reporting tests stub the C library and use the real CLI and logger across two
GPUs. Binding tests load this tree's py-interface and skip only when
libamd_smi.so cannot be loaded.
"""

import argparse
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = find_cli_dir(*cli_search_order(_THIS_DIR))
STATIC_PATH = os.path.join(_CLI_DIR, "subcommands", "static.py") if _CLI_DIR else None
_SOURCE_PY_INTERFACE_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..", "py-interface")
)
_PY_INTERFACE_PACKAGE = "amdsmi_py_interface_under_test"


def _fake_amdsmi_modules():
    """Stub ``amdsmi`` package so the CLI modules import cleanly."""
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    interface.amdsmi_wrapper = wrapper
    wrapper.AMDSMI_STATUS_INVAL = 1
    wrapper.AMDSMI_STATUS_NOT_SUPPORTED = 2
    wrapper.AMDSMI_STATUS_NO_PERM = 10
    # amdsmi_helpers reads these while its class body runs.
    wrapper.AMDSMI_STATUS_UNEXPECTED_SIZE = 42
    wrapper.AMDSMI_STATUS_UNEXPECTED_DATA = 43

    class _StubLibraryException(Exception):
        """Carries an error code, which the CLI branches on."""

        def __init__(self, err_code=1):
            super().__init__(f"stub amdsmi error {err_code}")
            self.err_code = err_code

        def get_error_code(self):
            return self.err_code

        def get_error_info(self, detailed=True):
            return str(self)

    exception.AmdSmiLibraryException = _StubLibraryException
    interface.AmdSmiLibraryException = _StubLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    modules = {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_wrapper": wrapper,
    }
    return interface, modules


class _BindingTestBase(unittest.TestCase):
    """This tree's Python binding, loaded against the built libamd_smi.so."""

    @classmethod
    def setUpClass(cls):
        # This tree's binding under a private package name, so neither an
        # installed amdsmi nor the stub the CLI classes register can stand in.
        # Loaded without the package __init__, which needs the generated
        # _version.py.
        if not os.path.isfile(os.path.join(_SOURCE_PY_INTERFACE_DIR, "amdsmi_interface.py")):
            raise unittest.SkipTest(f"no py-interface at {_SOURCE_PY_INTERFACE_DIR}")
        package = types.ModuleType(_PY_INTERFACE_PACKAGE)
        package.__path__ = [_SOURCE_PY_INTERFACE_DIR]
        sys.modules[_PY_INTERFACE_PACKAGE] = package
        try:
            interface = importlib.import_module(_PY_INTERFACE_PACKAGE + ".amdsmi_interface")
            exception = importlib.import_module(_PY_INTERFACE_PACKAGE + ".amdsmi_exception")
        except Exception:
            cls._forget_package()
            raise
        if interface.amdsmi_wrapper._loaded_lib_path is None:
            cls._forget_package()
            raise unittest.SkipTest("libamd_smi.so could not be loaded")
        cls.interface = interface
        cls.parameter_exception = exception.AmdSmiParameterException

    @classmethod
    def tearDownClass(cls):
        cls._forget_package()

    @staticmethod
    def _forget_package():
        for name in [n for n in sys.modules if n.split(".")[0] == _PY_INTERFACE_PACKAGE]:
            del sys.modules[name]


class TestCuidComponentsBinding(_BindingTestBase):
    """amdsmi_get_cuid_components decodes every entry, and lists a component
    that appears between its count call and its fill call."""

    def _fake(self, totals):
        """Stand in for the C call; totals[k] is the node's size at call k."""
        wrapper = self.interface.amdsmi_wrapper
        self.calls = []

        def _entry(count_ref, components):
            total = totals[min(len(self.calls), len(totals) - 1)]
            self.calls.append(bool(components))
            count = count_ref._obj
            capacity = count.value if components else 0
            for i in range(min(capacity, total)):
                entry = components[i]
                entry.info.derived = f"0000000{i}-0000-8000-8000-000000000000".encode()
                entry.info.component_type = wrapper.AMDSMI_CUID_COMPONENT_GPU
                entry.info.source = wrapper.AMDSMI_CUID_SOURCE_DRIVER
                entry.bdf = f"0000:0{i}:00.0".encode()
                entry.vendor_id = 0x1002
            count.value = total
            if components and capacity < total:
                return wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE
            return wrapper.AMDSMI_STATUS_SUCCESS

        patcher = mock.patch.object(wrapper, "amdsmi_get_cuid_components", _entry, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_entries_are_decoded(self):
        self._fake([2])
        components = self.interface.amdsmi_get_cuid_components()
        self.assertEqual([c["bdf"] for c in components], ["0000:00:00.0", "0000:01:00.0"])
        self.assertEqual(components[1]["derived"], "00000001-0000-8000-8000-000000000000")
        self.assertEqual(components[0]["component_type"], "GPU")
        self.assertEqual(components[0]["source"], "DRIVER")
        self.assertIs(components[0]["auxiliary"], False)
        self.assertEqual(components[0]["vendor_id"], 0x1002)
        self.assertEqual(components[0]["primary"], "")

    def test_a_component_appearing_between_calls_is_listed(self):
        self._fake([1, 2])
        components = self.interface.amdsmi_get_cuid_components()
        self.assertEqual(self.calls, [False, True, False, True])
        self.assertEqual(len(components), 2)


class _FakeHelpers:
    """The slice of ``AMDSMIHelpers`` that `static` and the logger reach for.

    ``handle_gpus`` delegates to the real implementation: it is the loop that
    turns one invocation into one ``static_gpu`` call per device, and a
    reimplementation of it here would be the very thing under test.
    """

    def __init__(self, helpers_module):
        self._real = helpers_module.AMDSMIHelpers

    def handle_gpus(self, args, logger, subcommand):
        return self._real.handle_gpus(self, args, logger, subcommand)

    def run_device_subcommand(self, subcommand, args, **device_kwarg):
        return self._real.run_device_subcommand(self, subcommand, args, **device_kwarg)

    def get_gpu_cuid_info(self, device_handle, include_primary=False):
        return self._real.get_gpu_cuid_info(self, device_handle, include_primary)

    def get_gpu_id_from_device_handle(self, device_handle):
        return device_handle

    def os_info(self):
        return ("linux", "x86_64")

    def check_required_groups(self):
        pass

    def is_linux(self):
        return True

    def is_baremetal(self):
        return True

    def is_virtual_os(self):
        return False

    def is_hypervisor(self):
        return False

    def is_amd_hsmp_initialized(self):
        return False

    def is_amdgpu_initialized(self):
        return True


class TestStaticCuidOutputShape(unittest.TestCase):
    """`amd-smi static --cuid`: the fixed field names, per GPU."""

    # Neither is 0: static_gpu treats a falsy device handle as "no device
    # selected".
    GPU_HANDLES = [1, 2]

    @classmethod
    def setUpClass(cls):
        cls.interface, modules = _fake_amdsmi_modules()

        # amdsmi_helpers pulls in amdsmi_init, which initialises the real
        # library at import time and exits the interpreter when no driver is
        # loaded. Stand in for it with the names amdsmi_helpers reads.
        modules["amdsmi_init"] = fake_module(
            "amdsmi_init",
            AMDSMI_INIT_FLAG=0,
            AMD_VENDOR_ID=0x1002,
            amdsmi_interface=cls.interface,
            amdsmi_exception=modules["amdsmi.amdsmi_exception"],
        )
        # Dropped so they are imported afresh against the stub amdsmi, and put
        # back when the class finishes.
        for name in ("amdsmi_helpers", "amdsmi_logger", "amdsmi_cli_exceptions", "BDF"):
            modules[name] = None
        stub_modules(cls, modules)

        cls.module = load_cli_module("static_cuid_under_test", STATIC_PATH, _CLI_DIR)
        import amdsmi_helpers
        import amdsmi_logger

        cls.helpers_module = amdsmi_helpers
        cls.logger_module = amdsmi_logger

    def setUp(self):
        self.interface.amdsmi_get_gpu_cuid_info = lambda handle: {
            "primary": "",
            "derived": f"deadbeef-0000-8000-0000-00000000000{handle}",
            "component_type": "GPU",
            "auxiliary": True,
            "source": "LIBRARY",
        }
        self.interface.amdsmi_get_gpu_asic_info = lambda handle: {}

    def _run(self, output_format, **arg_overrides):
        """Run one `amd-smi static` invocation and return what it printed."""
        helpers = _FakeHelpers(self.helpers_module)
        command = self.module.StaticCommands()
        command.helpers = helpers
        command.logger = self.logger_module.AMDSMILogger(
            format=output_format, destination="stdout", helpers=helpers
        )
        command.group_check_printed = True
        command.device_handles = list(self.GPU_HANDLES)
        command.cpu_handles = []
        # NIC discovery probes the host; this invocation is about GPUs.
        command._static_nics = lambda *args, **kwargs: False

        selectors = dict.fromkeys(
            (
                "asic",
                "bus",
                "vbios",
                "driver",
                "ras",
                "vram",
                "cache",
                "board",
                "process_isolation",
                "clock",
                "mem_carveout",
                "partition",
                "limit",
                "soc_pstate",
                "xgmi_plpd",
                "profile",
                "numa",
                "dfc_ucode",
                "fb_info",
                "num_vf",
                "cuid",
                "cuid_primary",
            ),
            False,
        )
        selectors.update(gpu=None, cpu=None, nic=None)
        selectors.update(arg_overrides)

        captured = io.StringIO()
        with redirect_stdout(captured):
            command.static(argparse.Namespace(**selectors))
        return captured.getvalue()

    def test_json_gpu_blocks_carry_identity_and_observation_metadata(self):
        document = json.loads(self._run("json", cuid=True))

        self.assertEqual(list(document), ["gpu_data"])
        self.assertEqual(len(document["gpu_data"]), len(self.GPU_HANDLES))
        for gpu_block in document["gpu_data"]:
            self.assertEqual(
                sorted(gpu_block["cuid"]),
                [
                    "auxiliary",
                    "component_type",
                    "cuid_metadata_status",
                    "derived_cuid",
                    "identifier_kind",
                    "primary_cuid",
                    "source",
                ],
            )

    def test_human_readable_prints_one_block_per_gpu(self):
        printed = self._run("human_readable", cuid=True)

        self.assertEqual(printed.count("DERIVED_CUID"), len(self.GPU_HANDLES))
        self.assertNotIn("SEED", printed)

    def test_csv_keeps_the_per_device_names(self):
        printed = self._run("csv", cuid=True)

        self.assertIn("gpu,derived_cuid,primary_cuid,component_type,auxiliary,source", printed)
        self.assertNotIn("seed", printed)

    def test_cuid_primary_alone_selects_the_block(self):
        document = json.loads(self._run("json", cuid_primary=True))

        for gpu_block in document["gpu_data"]:
            self.assertIn("cuid", gpu_block)

    def test_no_cuid_block_unless_asked(self):
        document = json.loads(self._run("json", asic=True))

        for gpu_block in document["gpu_data"]:
            self.assertNotIn("cuid", gpu_block)
