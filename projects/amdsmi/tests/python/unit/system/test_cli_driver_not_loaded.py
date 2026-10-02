#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for amdgpu-gated subcommands invoked while amdgpu is not loaded.

When another driver (e.g. an AI NIC) lets the CLI initialize without amdgpu, the
subparsers that need amdgpu are never registered. On a host with an AMD GPU,
invoking one must report that amdgpu is not loaded instead of claiming the
command is unsupported. Loads the installed ``amdsmi_parser`` and
``amdsmi_helpers`` with their init-time dependencies stubbed, so no GPU
hardware is needed.
"""

import argparse
import csv
import functools
import importlib.util
import inspect
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from common.common import amdsmi_path

_CLI_DIR = Path(amdsmi_path).parents[1] / "libexec" / "amdsmi_cli"
PARSER_PATH = _CLI_DIR / "amdsmi_parser.py"
HELPERS_PATH = _CLI_DIR / "amdsmi_helpers.py"
_SWAPPED_MODULES = ("amdsmi_init", "amdsmi_helpers", "amdsmi_cli_exceptions", "_version")


class _FakeHelpers:
    """Helper surface the parser reads while building subparsers without amdgpu.

    Defaults model the reported system: Linux baremetal with an AMD GPU present,
    the AI NIC driver live and amdgpu not loaded.
    """

    def __init__(
        self,
        amdgpu: bool = False,
        gpu_present: bool = True,
        baremetal: bool = True,
        hypervisor: bool = False,
        output_format: str = "human_readable",
    ) -> None:
        self._amdgpu = amdgpu
        self._gpu_present = gpu_present
        self._baremetal = baremetal
        self._hypervisor = hypervisor
        self._output_format = output_format

    def is_amdgpu_initialized(self) -> bool:
        return self._amdgpu

    def is_amd_gpu_present(self) -> bool:
        return self._gpu_present

    def is_ainic_initialized(self) -> bool:
        return True

    def is_brcm_nic_initialized(self) -> bool:
        return False

    def is_brcm_switch_initialized(self) -> bool:
        return False

    def is_amd_hsmp_initialized(self) -> bool:
        return False

    def get_gpu_choices(self) -> tuple:
        return {}, ""

    def get_nic_choices(self) -> tuple:
        return {}, ""

    def get_clock_types(self) -> tuple:
        # static builds its --clock help from these even without amdgpu
        return ["SYS", "MEM", "PCIE"], [0, 1, 2]

    def is_linux(self) -> bool:
        return True

    def is_windows(self) -> bool:
        return False

    def is_hypervisor(self) -> bool:
        return self._hypervisor

    def is_baremetal(self) -> bool:
        return self._baremetal

    def is_virtual_os(self) -> bool:
        return not self._baremetal

    def os_info(self) -> str:
        return "Linux Baremetal" if self._baremetal else "Linux Guest"

    def get_rocm_version(self) -> str:
        return "N/A"

    def get_output_format(self) -> str:
        return self._output_format


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None)
def _import_cli_modules() -> tuple:
    """Import the installed helpers and parser against the real exceptions module.

    ``amdsmi_init`` initializes the library on import, so it is stubbed with the
    names the helpers bind. Sibling suites may leave stubs in ``sys.modules``, so
    these entries and ``sys.path`` are changed only for the imports.
    """
    saved = {name: sys.modules.pop(name, None) for name in _SWAPPED_MODULES}
    added_path = str(_CLI_DIR) not in sys.path
    try:
        init_stub = types.ModuleType("amdsmi_init")
        init_stub.AMD_VENDOR_ID = 0x1002
        init_stub.amdsmi_interface = types.SimpleNamespace()
        init_stub.amdsmi_exception = types.SimpleNamespace()
        sys.modules["amdsmi_init"] = init_stub
        if not (_CLI_DIR / "_version.py").is_file():  # generated at build time
            version_stub = types.ModuleType("_version")
            version_stub.__version__ = "0.0.0-test"
            sys.modules["_version"] = version_stub
        if added_path:
            sys.path.insert(0, str(_CLI_DIR))
        helpers_mod = _load_module("amdsmi_helpers", HELPERS_PATH)
        sys.modules["amdsmi_helpers"] = helpers_mod
        parser_mod = _load_module("amdsmi_parser_under_test", PARSER_PATH)
    finally:
        if added_path:
            sys.path.remove(str(_CLI_DIR))
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return helpers_mod, parser_mod


class TestCliDriverNotLoaded(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not PARSER_PATH.is_file():
            raise unittest.SkipTest(f"amdsmi_parser not installed at {PARSER_PATH}")
        _, cls.parser_mod = _import_cli_modules()
        cls.exceptions = cls.parser_mod.amdsmi_cli_exceptions

    def _make_parser(self, command: str, **helper_kwargs: object) -> argparse.ArgumentParser:
        """Build the parser as amdsmi_cli.py does for ``amd-smi <command>``."""
        parser_cls = self.parser_mod.AMDSMIParser
        callbacks = {
            name: (lambda _args: None)
            for name in inspect.signature(parser_cls.__init__).parameters
            if name not in ("self", "sys_argv", "helpers")
        }
        return parser_cls(
            **callbacks, sys_argv=["amd-smi", command], helpers=_FakeHelpers(**helper_kwargs)
        )

    def _parse_error(self, command: str, **helper_kwargs: object) -> Exception:
        parser = self._make_parser(command, **helper_kwargs)
        with self.assertRaises(self.exceptions.AmdSmiException) as ctx:
            parser.parse_args([command])
        return ctx.exception

    def test_every_parser_skipped_without_amdgpu_is_recorded(self) -> None:
        # -h builds every subparser; without amdgpu the gated ones return early.
        # profile is Windows-hypervisor only and default is not built for -h.
        parser = self._make_parser("-h")
        skipped = set(parser.possible_commands) - set(parser.subparsers.choices)
        self.assertEqual(
            skipped - {"profile", "default"},
            parser.amdgpu_skipped_commands,
            "gate the parser with _skip_without_amdgpu, or exclude a non-amdgpu gate here",
        )

    def test_amdgpu_skipped_commands_report_driver_not_loaded(self) -> None:
        for command in self._make_parser("-h").amdgpu_skipped_commands:
            with self.subTest(command=command):
                exc = self._parse_error(command)
                self.assertIsInstance(exc, self.exceptions.AmdSmiGpuDriverNotLoadedException)
                self.assertEqual(exc.value, int(self.exceptions.AmdSmiExitCode.GPU_DRIVER_NOT_LOADED))
                message = str(exc)
                self.assertIn(f"Command '{command}' requires the amdgpu driver", message)
                self.assertIn("sudo modprobe amdgpu", message)

    def test_driver_not_loaded_json_and_csv_are_well_formed(self) -> None:
        payload = json.loads(str(self._parse_error("list", output_format="json")))
        self.assertEqual(
            payload["code"], int(self.exceptions.AmdSmiExitCode.GPU_DRIVER_NOT_LOADED)
        )
        self.assertIn("sudo modprobe amdgpu", payload["error"])

        rows = list(csv.reader(io.StringIO(str(self._parse_error("list", output_format="csv")))))
        self.assertEqual(rows[0], ["error", "code", "error_type"])
        self.assertEqual(len(rows[1]), 3, rows[1])
        self.assertEqual(
            rows[1][1].strip(), str(int(self.exceptions.AmdSmiExitCode.GPU_DRIVER_NOT_LOADED))
        )

    def test_host_without_amd_gpu_still_reports_not_supported(self) -> None:
        # e.g. a CPU-only host: loading amdgpu would not enable GPU subcommands.
        exc = self._parse_error("list", gpu_present=False)
        self.assertIsInstance(exc, self.exceptions.AmdSmiCommandNotSupportedException)
        self.assertEqual(exc.value, int(self.exceptions.AmdSmiExitCode.COMMAND_NOT_SUPPORTED))

    def test_platform_unsupported_command_still_reports_not_supported(self) -> None:
        # profile is Windows-hypervisor only, so loading amdgpu would not enable it.
        exc = self._parse_error("profile")
        self.assertIsInstance(exc, self.exceptions.AmdSmiCommandNotSupportedException)
        self.assertEqual(exc.value, int(self.exceptions.AmdSmiExitCode.COMMAND_NOT_SUPPORTED))

    def test_platform_gate_before_amdgpu_keeps_not_supported(self) -> None:
        # A platform check skips these parsers before their amdgpu gate runs, so they
        # report "not supported" rather than a missing driver.
        for command, platform in (
            ("process", {"hypervisor": True}),
            ("bad-pages", {"baremetal": False}),
        ):
            with self.subTest(command=command):
                exc = self._parse_error(command, **platform)
                self.assertIsInstance(exc, self.exceptions.AmdSmiCommandNotSupportedException)
                self.assertEqual(
                    exc.value, int(self.exceptions.AmdSmiExitCode.COMMAND_NOT_SUPPORTED)
                )


class TestAmdGpuPresence(unittest.TestCase):
    """Fixtures mirror sysfs: /sys/bus/pci/devices holds symlinks to device dirs."""

    @classmethod
    def setUpClass(cls) -> None:
        if not HELPERS_PATH.is_file():
            raise unittest.SkipTest(f"amdsmi_helpers not installed at {HELPERS_PATH}")
        helpers_mod, _ = _import_cli_modules()
        cls.helpers_cls = helpers_mod.AMDSMIHelpers

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _bus(self, name: str) -> Path:
        bus = self.root / name
        bus.mkdir()
        return bus

    def _add_function(self, bus: Path, bdf: str, vendor: str, pci_class: str) -> None:
        device = self.root / "devices" / bus.name / bdf
        device.mkdir(parents=True)
        (device / "vendor").write_text(vendor + "\n", encoding="ascii")
        (device / "class").write_text(pci_class + "\n", encoding="ascii")
        (bus / bdf).symlink_to(device)

    def _present(self, bus: Path) -> bool:
        return self.helpers_cls.is_amd_gpu_present(pci_devices_path=bus)

    def test_non_gpu_functions_do_not_count(self) -> None:
        bus = self._bus("bus")
        self._add_function(bus, "0000:01:00.0", "0x1dd8", "0x020000")  # AI NIC
        self._add_function(bus, "0000:02:00.0", "0x1a03", "0x030000")  # BMC VGA
        self._add_function(bus, "0000:03:00.1", "0x1002", "0x040300")  # GPU HDMI audio
        self._add_function(bus, "0000:04:00.0", "0x1022", "0x120000")  # AMD CPU-vendor IP
        self._add_function(bus, "0000:05:00.0", "garbage", "0x030000")
        (bus / "0000:06:00.0").symlink_to(self.root / "unplugged")
        self.assertFalse(self._present(bus))

    def test_gpu_bound_to_another_driver_does_not_count(self) -> None:
        # modprobe amdgpu cannot claim a GPU that e.g. vfio-pci already owns
        for driver_name, expected in (("vfio-pci", False), ("amdgpu", True)):
            with self.subTest(driver=driver_name):
                bus = self._bus(driver_name)
                self._add_function(bus, "0000:03:00.0", "0x1002", "0x120000")
                driver = self.root / "drivers" / driver_name
                driver.mkdir(parents=True)
                (bus / "0000:03:00.0" / "driver").symlink_to(driver)
                self.assertEqual(self._present(bus), expected)

    def test_amd_gpu_base_classes_count(self) -> None:
        # VGA, other display (e.g. MI200), processing accelerator (e.g. MI300, MI450)
        for pci_class in ("0x030000", "0x038000", "0x120000"):
            with self.subTest(pci_class=pci_class):
                bus = self._bus(pci_class)
                self._add_function(bus, "0000:03:00.0", "0x1002", pci_class)
                self.assertTrue(self._present(bus))

    def test_missing_sysfs_reports_no_gpu(self) -> None:
        self.assertFalse(self._present(self.root / "absent"))


if __name__ == "__main__":
    unittest.main()
