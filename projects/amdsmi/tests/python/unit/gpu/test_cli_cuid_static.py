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

# ``common.common`` bootstraps the real amdsmi package at import time, which
# fails on a stale or mismatched install. The CLI classes below fully stub
# ``amdsmi`` and only need ``amdsmi_path`` to locate the *installed* CLI
# fallback, so degrade gracefully, as test_cli_set_clk_limit.py does. Exception
# rather than ImportError: a stale install raises AttributeError out of
# build_type_lists() rather than failing to import.
try:
    from common.common import amdsmi_path
except Exception:  # pragma: no cover - harness/install unavailable or stale
    amdsmi_path = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
_INSTALLED_CLI_DIR = (
    os.path.join(os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli")
    if amdsmi_path
    else ""
)


def _resolve_cli_dir():
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        if cli_dir and os.path.isfile(os.path.join(cli_dir, "subcommands", "static.py")):
            return cli_dir
    return None


_CLI_DIR = _resolve_cli_dir()
_SOURCE_PY_INTERFACE_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..", "py-interface")
)
_PY_INTERFACE_PACKAGE = "amdsmi_py_interface_under_test"


def _install_fake_amdsmi():
    """Register a stub ``amdsmi`` package so the CLI modules import cleanly."""
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    interface.amdsmi_wrapper = wrapper

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

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    sys.modules["amdsmi.amdsmi_wrapper"] = wrapper
    return interface


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

    _SAVED_MODULE_NAMES = (
        "amdsmi",
        "amdsmi.amdsmi_interface",
        "amdsmi.amdsmi_exception",
        "amdsmi.amdsmi_wrapper",
        "amdsmi_init",
        "amdsmi_helpers",
        "amdsmi_logger",
        "amdsmi_cli_exceptions",
        "BDF",
    )

    @classmethod
    def setUpClass(cls):
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI not found (source or installed)")
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        cls.interface = _install_fake_amdsmi()

        # amdsmi_helpers pulls in amdsmi_init, which initialises the real
        # library at import time and exits the interpreter when no driver is
        # loaded. Stand in for it with the two names amdsmi_helpers reads.
        fake_init = types.ModuleType("amdsmi_init")
        fake_init.AMDSMI_INIT_FLAG = 0
        fake_init.AMD_VENDOR_ID = 0x1002
        fake_init.amdsmi_interface = cls.interface
        fake_init.amdsmi_exception = sys.modules["amdsmi.amdsmi_exception"]
        sys.modules["amdsmi_init"] = fake_init

        if _CLI_DIR not in sys.path:
            sys.path.insert(0, _CLI_DIR)
        import amdsmi_helpers
        import amdsmi_logger

        cls.helpers_module = amdsmi_helpers
        cls.logger_module = amdsmi_logger

        spec = importlib.util.spec_from_file_location(
            "static_cuid_under_test", os.path.join(_CLI_DIR, "subcommands", "static.py")
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

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


if __name__ == "__main__":
    unittest.main()
