#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A missing driver is named when its device is present, for every device type.

The CLI starts with whichever of amdgpu (GPU), the CPU HSMP driver, ionic (AI NIC) and
bnxt_en (Broadcom NIC) are loaded. Input that needs a missing driver reports that driver
(exit code 206) when the hardware is present, and keeps the generic "not supported" or
"invalid parameter" error when it is not.
"""

import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from common.common import (
    cli_search_order,
    fake_module,
    find_cli_dir,
    generated_version_stub,
    stub_modules_at_import,
)

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
if _CLI_DIR and _CLI_DIR not in sys.path:
    sys.path.append(_CLI_DIR)

cli_exc: Any = None
helpers_module: Any = None
parser_module: Any = None
_restore_stubs = None


def setUpModule() -> None:
    global cli_exc, helpers_module, parser_module, _restore_stubs
    if _CLI_DIR is None:
        raise unittest.SkipTest("amd-smi CLI not found")
    # amdsmi_helpers does ``from amdsmi_init import *``, which initializes the library, and
    # _version is generated at build time. Stub them unless another module already has:
    # generated_version_stub() cannot look past a stub that is already installed.
    stubs = {}
    if "amdsmi_init" not in sys.modules:
        from amdsmi import amdsmi_exception, amdsmi_interface

        stubs["amdsmi_init"] = fake_module(
            "amdsmi_init",
            AMDSMI_INIT_FLAG=0,
            AMDSMI_INITIALIZED=True,
            amdsmi_interface=amdsmi_interface,
            amdsmi_exception=amdsmi_exception,
        )
    if "_version" not in sys.modules:
        stubs.update(generated_version_stub())
    _restore_stubs = stub_modules_at_import(stubs)
    try:
        import amdsmi_cli_exceptions as cli_exc
        import amdsmi_helpers as helpers_module
        import amdsmi_parser as parser_module
    except ImportError as error:  # no CLI in this layout
        _restore_stubs()
        _restore_stubs = None
        raise unittest.SkipTest(f"amd-smi CLI not importable: {error}")


def tearDownModule() -> None:
    if _restore_stubs:
        _restore_stubs()


_DRIVERS = ("amdgpu", "hsmp_acpi", "ionic", "bnxt_en")
# GPU, CPU and AI NIC, as on an MI450 Helios node
_HELIOS = ("amdgpu", "hsmp_acpi", "ionic")


class _FakeHelpers:
    """AMDSMIHelpers stand-in: which drivers are loaded and which hardware is present."""

    def __init__(
        self,
        loaded=_HELIOS,
        present=_HELIOS,
        baremetal: bool = True,
        hypervisor: bool = False,
        output_format: str = "human_readable",
    ) -> None:
        self.loaded = set(loaded)
        self.present = set(present)
        self._baremetal = baremetal
        self._hypervisor = hypervisor
        self._output_format = output_format

    def is_amdgpu_initialized(self) -> bool:
        return "amdgpu" in self.loaded

    def is_amd_hsmp_initialized(self) -> bool:
        return "hsmp_acpi" in self.loaded

    def is_ainic_initialized(self) -> bool:
        return bool({"ionic", "bnxt_en"} & self.loaded)

    def is_brcm_nic_initialized(self) -> bool:
        return "bnxt_en" in self.loaded

    def is_brcm_switch_initialized(self) -> bool:
        return False

    def is_driver_loaded(self, driver: str) -> bool:
        return driver in self.loaded

    def get_devices_without_driver(self, driver: str) -> list:
        return ["0000:01:00.0"] if driver in self.present - self.loaded else []

    def get_missing_drivers(self) -> list:
        return [driver for driver in _DRIVERS if self.get_devices_without_driver(driver)]

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

    def get_gpu_choices(self):
        return {"0": "gpu0"}, "0"

    def get_cpu_choices(self):
        return {"0": "cpu0"}, "0"

    def get_core_choices(self):
        return {"0": "core0"}, "0"

    def get_nic_choices(self):
        return {"0": "nic0"}, "0"

    def get_switch_choices(self):
        return {}, ""

    def get_clock_types(self):
        return ["SYS", "MEM", "PCIE"], [0, 1, 2]

    def os_info(self) -> str:
        return "Linux Baremetal"

    def get_rocm_version(self) -> str:
        return "N/A"

    def get_output_format(self) -> str:
        return self._output_format


def _parse(argv, **state):
    """Parse ``amd-smi <argv>`` the way amdsmi_cli.py does, on fake driver state."""
    names = [
        name
        for name in inspect.signature(parser_module.AMDSMIParser.__init__).parameters
        if name not in ("self", "sys_argv", "helpers")
    ]
    funcs = {name: (lambda args: "ran") for name in names}
    sys_argv = ["amd-smi", *argv]
    with mock.patch.object(sys, "argv", sys_argv):
        parser = parser_module.AMDSMIParser(
            **funcs, sys_argv=sys_argv, helpers=_FakeHelpers(**state)
        )
        return parser.parse_args(list(argv))


class TestMissingDriverErrors(unittest.TestCase):
    def _error(self, *argv, **state):
        with self.assertRaises(cli_exc.AmdSmiException) as ctx:
            _parse(argv, **state)
        return ctx.exception

    def test_gpu_commands_name_amdgpu_when_a_gpu_is_present(self) -> None:
        for command in (
            "process",
            "event",
            "reset",
            "xgmi",
            "partition",
            "fabric",
            "bad-pages",
            "firmware",
            "ucode",
            "topology",
            "monitor",
            "dmon",
        ):
            with self.subTest(command=command):
                exc = self._error(command, loaded=("hsmp_acpi", "ionic"))
                self.assertIsInstance(exc, cli_exc.AmdSmiDriverNotLoadedException)
                self.assertEqual(exc.value, int(cli_exc.AmdSmiExitCode.DRIVERS_NOT_LOADED))
                self.assertEqual(exc.drivers, ["amdgpu"])
                self.assertIn(f"Command '{command}' requires the amdgpu driver", str(exc))
                self.assertIn("sudo modprobe amdgpu", str(exc))

    def test_gpu_commands_stay_not_supported_without_a_gpu(self) -> None:
        exc = self._error("process", loaded=("hsmp_acpi", "ionic"), present=("hsmp_acpi", "ionic"))
        self.assertIsInstance(exc, cli_exc.AmdSmiCommandNotSupportedException)

    def test_platform_restrictions_keep_not_supported(self) -> None:
        # Loading amdgpu would not enable these, so the driver is not the reason.
        for command, platform in (
            ("process", {"hypervisor": True}),
            ("bad-pages", {"baremetal": False}),
        ):
            with self.subTest(command=command):
                exc = self._error(command, loaded=("hsmp_acpi", "ionic"), **platform)
                self.assertIsInstance(exc, cli_exc.AmdSmiCommandNotSupportedException)

    def test_list_runs_with_only_a_nic_driver(self) -> None:
        args = _parse(["list"], loaded=("hsmp_acpi", "ionic"))
        self.assertIsNone(args.gpu)
        self.assertEqual(args.note_drivers, ("amdgpu", "ionic", "bnxt_en"))

    def test_list_names_every_missing_driver_with_hardware(self) -> None:
        exc = self._error("list", loaded=("hsmp_acpi",))
        self.assertIsInstance(exc, cli_exc.AmdSmiDriverNotLoadedException)
        self.assertEqual(exc.drivers, ["amdgpu", "ionic"])
        self.assertIn("requires the amdgpu or ionic driver but none is loaded", str(exc))

    def test_options_name_their_driver(self) -> None:
        for argv, loaded, driver in (
            (("static", "-g", "0"), ("hsmp_acpi", "ionic"), "amdgpu"),
            (("static", "--asic"), ("hsmp_acpi", "ionic"), "amdgpu"),
            (("static", "-U", "all"), ("amdgpu", "ionic"), "hsmp_acpi"),
            (("metric", "-O", "all"), ("amdgpu", "ionic"), "hsmp_acpi"),
            (("static", "-N", "all"), ("amdgpu", "hsmp_acpi"), "ionic"),
        ):
            with self.subTest(argv=argv):
                exc = self._error(*argv, loaded=loaded)
                self.assertIsInstance(exc, cli_exc.AmdSmiDriverNotLoadedException)
                self.assertEqual(exc.drivers, [driver])
                self.assertIn(f"Parameter '{argv[1]}' requires the {driver} driver", str(exc))

    def test_options_stay_invalid_without_their_hardware(self) -> None:
        exc = self._error(
            "static", "-U", "all", loaded=("amdgpu", "ionic"), present=("amdgpu", "ionic")
        )
        self.assertIsInstance(exc, cli_exc.AmdSmiInvalidParameterException)

    def test_unknown_options_stay_invalid(self) -> None:
        exc = self._error("static", "--no-such-option", loaded=("hsmp_acpi", "ionic"))
        self.assertIsInstance(exc, cli_exc.AmdSmiInvalidParameterException)

    def test_failed_rebuild_keeps_the_generic_error(self) -> None:
        with mock.patch.object(parser_module, "_AssumeDriverLoaded", side_effect=RuntimeError):
            exc = self._error("process", loaded=("hsmp_acpi", "ionic"))
        self.assertIsInstance(exc, cli_exc.AmdSmiCommandNotSupportedException)

    def test_broadcom_only_systems_run_the_broadcom_options(self) -> None:
        args = _parse(["firmware", "--brcm_nic"], loaded=("bnxt_en",))
        self.assertEqual(args.func(args), "ran")

    def test_broadcom_only_systems_explain_gpu_output(self) -> None:
        for present, expected in (
            (("amdgpu", "bnxt_en"), cli_exc.AmdSmiDriverNotLoadedException),
            (("bnxt_en",), cli_exc.AmdSmiCommandNotSupportedException),
        ):
            with self.subTest(present=present):
                args = _parse(["firmware"], loaded=("bnxt_en",), present=present)
                with self.assertRaises(expected):
                    args.func(args)

    def test_error_reports_the_driver_in_json(self) -> None:
        exc = cli_exc.AmdSmiDriverNotLoadedException("Command 'list'", ["amdgpu"], "json")
        payload = json.loads(str(exc))
        self.assertEqual(payload["code"], 206)
        self.assertEqual(
            payload["error"],
            "Command 'list' requires the amdgpu driver but it is not loaded."
            " Check amdgpu version and module status (sudo modprobe amdgpu).",
        )


class TestDriverNotes(unittest.TestCase):
    def _notes(self, note_drivers, **selected):
        missing = {
            "amdgpu": ["0001:01:00.0", "0002:01:00.0"],
            "hsmp_acpi": ["AMDI0097:00"],
            "ionic": ["0000:04:00.0"],
        }
        helpers = types.SimpleNamespace(
            get_devices_without_driver=lambda driver: missing.get(driver, [])
        )
        args = types.SimpleNamespace(note_drivers=note_drivers, **selected)
        return helpers_module.AMDSMIHelpers.get_driver_notes(helpers, args)

    def test_one_note_per_missing_driver_of_the_command(self) -> None:
        self.assertEqual(
            self._notes(_DRIVERS),
            [
                "Note: GPU 0001:01:00.0 0002:01:00.0: amdgpu driver not loaded"
                " (sudo modprobe amdgpu)",
                "Note: CPU AMDI0097:00: hsmp_acpi driver not loaded (sudo modprobe hsmp_acpi)",
                "Note: AI NIC 0000:04:00.0: ionic driver not loaded (sudo modprobe ionic)",
            ],
        )

    def test_selected_device_types_limit_the_notes(self) -> None:
        for selected, driver in (({"cpu": ["cpu0"]}, "hsmp_acpi"), ({"nic": ["nic0"]}, "ionic")):
            with self.subTest(selected=selected):
                notes = self._notes(_DRIVERS, **selected)
                self.assertEqual(len(notes), 1)
                self.assertIn(f"{driver} driver not loaded", notes[0])

    def test_commands_without_note_drivers_have_no_notes(self) -> None:
        self.assertEqual(self._notes(()), [])

    def test_cpu_hardware_is_the_acpi_hsmp_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "AMDI0097:00").mkdir()
            (Path(tmp) / "PNP0A08:00").mkdir()
            with mock.patch.object(helpers_module, "HSMP_ACPI_DEVICES", Path(tmp)):
                for loaded, expected in ((False, ["AMDI0097:00"]), (True, [])):
                    with self.subTest(loaded=loaded):
                        helpers = types.SimpleNamespace(is_driver_loaded=lambda _d: loaded)
                        self.assertEqual(
                            helpers_module.AMDSMIHelpers.get_devices_without_driver(
                                helpers, "hsmp_acpi"
                            ),
                            expected,
                        )


class TestPciDevicesWithoutDriver(unittest.TestCase):
    """Fixtures mirror sysfs: /sys/bus/pci/devices holds symlinks to device dirs."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bus = self.root / "bus"
        self.bus.mkdir()

    def _add(self, bdf: str, vendor: str, pci_class: str, driver: str = "") -> None:
        device = self.root / "devices" / bdf
        device.mkdir(parents=True)
        (device / "vendor").write_text(vendor + "\n")
        (device / "class").write_text(pci_class + "\n")
        if driver:
            target = self.root / "drivers" / driver
            target.mkdir(parents=True, exist_ok=True)
            (device / "driver").symlink_to(target)
        (self.bus / bdf).symlink_to(device)

    def _find(self, driver: str, bus=None) -> list:
        _, vendor_id, base_classes = helpers_module.DRIVER_DEVICES[driver]
        return helpers_module.AMDSMIHelpers.get_pci_devices_without_driver(
            vendor_id, base_classes, driver, bus or self.bus
        )

    def test_gpus_match_display_and_accelerator_classes(self) -> None:
        self._add("0000:03:00.0", "0x1002", "0x030000")  # display
        self._add("0001:01:00.0", "0x1002", "0x120000")  # accelerator, e.g. MI450
        self._add("0000:05:00.0", "0x1002", "0x120000", "amdgpu")
        self._add("0000:06:00.0", "0x1002", "0x120000", "vfio-pci")  # passed through
        self._add("0000:03:00.1", "0x1002", "0x040300")  # GPU audio function
        self._add("0000:00:18.0", "0x1022", "0x060000")  # CPU host bridge
        self.assertEqual(self._find("amdgpu"), ["0000:03:00.0", "0000:05:00.0", "0001:01:00.0"])

    def test_nics_match_their_vendor(self) -> None:
        self._add("0000:04:00.0", "0x1dd8", "0x020000")  # Pensando AI NIC
        self._add("0000:07:00.0", "0x14e4", "0x020000", "tg3")  # not a bnxt_en NIC
        self._add("0000:08:00.0", "0x14e4", "0x020000")
        self.assertEqual(self._find("ionic"), ["0000:04:00.0"])
        self.assertEqual(self._find("bnxt_en"), ["0000:08:00.0"])

    def test_unreadable_entries_and_missing_sysfs_are_skipped(self) -> None:
        self._add("0000:09:00.0", "garbage", "0x120000")
        (self.bus / "0000:0a:00.0").symlink_to(self.root / "gone")  # dangling link
        self.assertEqual(self._find("amdgpu"), [])
        self.assertEqual(self._find("amdgpu", bus=self.root / "absent"), [])
