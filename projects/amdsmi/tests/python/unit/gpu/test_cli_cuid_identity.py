#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consumer identity semantics with real CLI renderers and API fixtures."""

import argparse
import csv
import importlib.util
import io
import itertools
import json
import os
import sys
import types
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
if not _CLI_DIR:
    raise unittest.SkipTest("amd-smi CLI not found")
CLI = Path(_CLI_DIR)
CUID = "12345678-1234-8234-9234-123456789abc"
OTHER_CUID = "87654321-4321-8321-9321-cba987654321"
LEGACY_UUID = "11111111-2222-1333-8444-555555555555"
PRIMARY = "99999999-8888-8777-8666-555555555555"


class LibraryError(Exception):
    def __init__(self, code=10):
        self.code = code

    def get_error_code(self):
        return self.code

    def get_error_info(self):
        return f"amdsmi error {self.code}"


def set_module(stack, name, module):
    if name in sys.modules:
        stack.callback(sys.modules.__setitem__, name, sys.modules[name])
    else:
        stack.callback(sys.modules.pop, name, None)
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


def make_cli(stack):
    api = types.ModuleType("amdsmi.amdsmi_interface")
    api.amdsmi_wrapper = types.SimpleNamespace(
        AMDSMI_STATUS_NO_PERM=10,
        AMDSMI_STATUS_INVAL=1,
        AMDSMI_STATUS_NOT_SUPPORTED=2,
        AMDSMI_STATUS_UNEXPECTED_SIZE=42,
        AMDSMI_STATUS_UNEXPECTED_DATA=43,
    )
    api.amdsmi_get_gpu_cuid_info = Mock(
        return_value={
            "derived": CUID,
            "primary": PRIMARY,
            "component_type": "GPU",
            "source": "DRIVER",
            "auxiliary": False,
        }
    )
    api.amdsmi_get_gpu_device_cuid = Mock(return_value=CUID)
    api.amdsmi_get_gpu_device_uuid = Mock(return_value=LEGACY_UUID)
    api.amdsmi_get_gpu_device_bdf = Mock(return_value="0000:03:00.0")
    api.amdsmi_get_gpu_enumeration_info = Mock(return_value={"drm_render": 128})
    api.amdsmi_get_gpu_kfd_info = Mock(
        return_value={"kfd_id": 1, "node_id": 0, "current_partition_id": "N/A"}
    )
    api.amdsmi_get_cuid_seed_info = Mock(
        return_value={"provisioned": True, "fingerprint": "0102030405060708"}
    )
    api.amdsmi_get_gpu_asic_info = Mock(return_value={})
    errors = types.ModuleType("amdsmi.amdsmi_exception")
    errors.AmdSmiLibraryException = LibraryError
    package = types.ModuleType("amdsmi")
    package.amdsmi_interface = api
    package.amdsmi_exception = errors
    init = types.ModuleType("amdsmi_init")
    init.AMDSMI_INIT_FLAG = 0
    init.amdsmi_interface = api
    init.amdsmi_exception = errors
    for name, module in (
        ("amdsmi", package),
        ("amdsmi.amdsmi_interface", api),
        ("amdsmi.amdsmi_exception", errors),
        ("amdsmi_init", init),
    ):
        set_module(stack, name, module)
    stack.enter_context(mock.patch.object(sys, "path", [str(CLI), *sys.path]))
    for name in ("BDF", "amdsmi_cli_exceptions"):
        set_module(stack, name, None)

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, CLI / relative)
        module = importlib.util.module_from_spec(spec)
        set_module(stack, name, module)
        spec.loader.exec_module(module)
        return module

    helpers_module = load("amdsmi_helpers", "amdsmi_helpers.py")
    logger_module = load("amdsmi_logger", "amdsmi_logger.py")
    list_module = load("list_cuid_under_test", "subcommands/list_devices.py")
    static_module = load("static_identity_under_test", "subcommands/static.py")
    node_module = load("node_cuid_under_test", "subcommands/node.py")
    helpers = object.__new__(helpers_module.AMDSMIHelpers)
    helpers._is_linux = True
    helpers._is_baremetal = False
    helpers._is_virtual_os = True
    helpers._is_hypervisor = False
    helpers._is_passthrough = False
    helpers.get_gpu_id_from_device_handle = lambda handle: handle - 1
    return types.SimpleNamespace(
        api=api,
        helpers=helpers,
        helpers_module=helpers_module,
        logger=logger_module.AMDSMILogger,
        list=list_module.ListDevicesCommands,
        static=static_module.StaticCommands,
        node=node_module.NodeCommands,
    )


def capture_stdout(action):
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        action()
    return out.getvalue()


def run_list(cli, fmt, enumeration=False):
    def list_gpu():
        command = cli.list()
        command.helpers = cli.helpers
        command.logger = cli.logger(format=fmt, helpers=cli.helpers)
        command.group_check_printed = True
        command.list_gpu(argparse.Namespace(gpu=[1], enumeration=enumeration))

    output = capture_stdout(list_gpu)
    if fmt == "json":
        return json.loads(output)[0]
    return next(csv.DictReader(io.StringIO(output)))


NODE_COMPONENTS = [
    {
        "primary": "",
        "derived": OTHER_CUID,
        "component_type": "PLATFORM",
        "source": "LIBRARY",
        "auxiliary": True,
        "bdf": "",
        "device_path": "",
        "vendor_id": 0,
    },
    {
        "primary": PRIMARY,
        "derived": CUID,
        "component_type": "GPU",
        "source": "DRIVER",
        "auxiliary": False,
        "bdf": "0000:03:00.0",
        "device_path": "/sys/class/drm/renderD128",
        "vendor_id": 0x1002,
    },
    {
        "primary": PRIMARY,
        "derived": LEGACY_UUID,
        "component_type": "GPU",
        "source": "DRIVER",
        "auxiliary": False,
        "bdf": "0000:63:00.0",
        "device_path": "/sys/class/drm/renderD129",
        "vendor_id": 0x1002,
    },
]


def run_node(cli, fmt, components=NODE_COMPONENTS, **overrides):
    if isinstance(components, Exception):
        cli.api.amdsmi_get_cuid_components = Mock(side_effect=components)
    else:
        cli.api.amdsmi_get_cuid_components = Mock(return_value=components)
    cli.api.amdsmi_get_ttm_info = Mock(return_value={"current_pages": 0})
    cli.api.amdsmi_get_tray_info = Mock(return_value={})
    args = dict.fromkeys(("power_management", "base_board_temps", "gtt", "tray", "cuid"), False)
    args.update(overrides)

    def node():
        command = cli.node()
        command.helpers = cli.helpers
        command.helpers.read_pending_gtt_pages = lambda: None
        command.logger = cli.logger(format=fmt, helpers=cli.helpers)
        command.group_check_printed = True
        command.node_handle = None
        command.node(argparse.Namespace(**args))

    output = capture_stdout(node)
    if fmt == "json":
        return json.loads(output)[0]["node"]
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(output)))
    return output


def static_args(**overrides):
    args = dict.fromkeys(
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
            "numa",
            "partition",
            "cuid",
        ),
        False,
    )
    args.update(gpu=[1], cpu=None, nic=None)
    args.update(overrides)
    return argparse.Namespace(**args)


class TestCliCuidIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (CLI / "amdsmi_helpers.py").is_file():
            raise unittest.SkipTest(f"amd-smi CLI not found at {CLI}")

    @contextmanager
    def case(self, **params):
        with ExitStack() as outer:
            if params:
                outer.enter_context(self.subTest(**params))
            with ExitStack() as stack:
                yield make_cli(stack)

    def test_source_temporary_table(self):
        for source, auxiliary in itertools.product(
            ["DRIVER", "LIBRARY", "UNKNOWN"], [True, False, None]
        ):
            with self.case(source=source, auxiliary=auxiliary) as cli:
                cli.api.amdsmi_get_gpu_cuid_info.return_value.update(
                    source=source, auxiliary=auxiliary
                )
                result = cli.helpers.get_gpu_cuid_info(1)
                self.assertEqual(result["source"], source)
                self.assertEqual(result["temporary"], "unknown" if auxiliary is None else auxiliary)
                self.assertNotIn("effective_seed", result)
                cli.api.amdsmi_get_cuid_seed_info.assert_not_called()

    def test_enumeration_fields_precede_cuid_fields(self):
        for fmt in ["json", "csv"]:
            with self.case(fmt=fmt) as cli:
                cli.api.amdsmi_get_gpu_enumeration_info.return_value.update(
                    drm_card=0,
                    hsa_id=1,
                    hip_id=0,
                    hip_uuid=LEGACY_UUID,
                    oam_id=0,
                    physical_acc_id=2,
                )
                result = run_list(cli, fmt, enumeration=True)
                self.assertEqual(
                    list(result)[6:13],
                    ["render", "card", "hsa_id", "hip_id", "hip_uuid", "oam_id", "physical_acc_id"],
                )
                self.assertEqual(result["render"], "renderD128")
                self.assertEqual(result["hip_uuid"], LEGACY_UUID)
                self.assertEqual(result["cuid"], CUID)

    def test_human_list_reports_cuid_without_private_diagnostics(self):
        with self.case() as cli:

            def list_gpu():
                command = cli.list()
                command.helpers = cli.helpers
                command.logger = cli.logger(format="human_readable", helpers=cli.helpers)
                command.group_check_printed = True
                command.list_gpu(argparse.Namespace(gpu=[1], enumeration=False))

            output = capture_stdout(list_gpu)
            self.assertIn(f"UUID: {LEGACY_UUID}", output)
            self.assertIn(
                f"    CUID: {CUID}\n    SOURCE: DRIVER\n    TEMPORARY: False\n"
                "    CUID_METADATA_STATUS: available\n",
                output,
            )
            self.assertNotIn("SEED", output)
            self.assertNotIn("FINGERPRINT", output)
            self.assertNotIn("PRIMARY", output)

    def test_list_uuid_is_always_the_legacy_uuid(self):
        for fmt, cuid, legacy in itertools.product(["json", "csv"], [True, False], [True, False]):
            with self.case(fmt=fmt, cuid=cuid, legacy=legacy) as cli:
                if not cuid:
                    cli.api.amdsmi_get_gpu_cuid_info.side_effect = LibraryError(2)
                    cli.api.amdsmi_get_gpu_device_cuid.side_effect = LibraryError(2)
                if not legacy:
                    cli.api.amdsmi_get_gpu_device_uuid.side_effect = LibraryError(2)
                result = run_list(cli, fmt)
                alias = "gpu_uuid" if fmt == "csv" else "uuid"
                old_keys = (
                    ["gpu", "gpu_bdf", "gpu_uuid"] if fmt == "csv" else ["gpu", "bdf", "uuid"]
                )
                old_keys += ["kfd_id", "node_id", "partition_id"]
                self.assertEqual(
                    list(result), old_keys + ["cuid", "source", "temporary", "cuid_metadata_status"]
                )
                self.assertEqual(result[alias], LEGACY_UUID if legacy else "N/A")
                self.assertEqual(result["cuid"], CUID if cuid else "N/A")
                self.assertEqual(
                    result["cuid_metadata_status"], "available" if cuid else "not_supported"
                )
                self.assertNotIn(PRIMARY, str(result))
                self.assertNotIn("seed_fingerprint", result)
                cli.api.amdsmi_get_cuid_seed_info.assert_not_called()

    def test_metadata_failure_keeps_available_cuid(self):
        for fmt, code in itertools.product(["json", "csv"], [2, 10, 34]):
            with self.case(fmt=fmt, code=code) as cli:
                cli.api.amdsmi_get_gpu_cuid_info.side_effect = LibraryError(code)
                result = run_list(cli, fmt)
                self.assertEqual(result["cuid"], CUID)
                self.assertEqual(result["gpu_uuid" if fmt == "csv" else "uuid"], LEGACY_UUID)
                if code == 2:
                    self.assertEqual(result["source"], "N/A")
                    self.assertEqual(result["temporary"], "N/A")
                    self.assertEqual(result["cuid_metadata_status"], "not_supported")
                else:
                    self.assertEqual(result["source"], "UNKNOWN")
                    self.assertEqual(result["temporary"], "unknown")
                    self.assertEqual(result["cuid_metadata_status"], f"amdsmi_error_{code}")
                cli.api.amdsmi_get_gpu_enumeration_info.assert_not_called()

    def test_a_handle_without_a_cuid_reads_as_not_available(self):
        with self.case() as cli:
            cli.api.amdsmi_get_gpu_cuid_info.side_effect = LibraryError(2)
            cli.api.amdsmi_get_gpu_device_cuid.side_effect = LibraryError(2)
            result = cli.helpers.get_gpu_cuid_info(1)
            self.assertEqual(
                {k: result[k] for k in ("cuid", "component_type", "temporary", "source")},
                dict.fromkeys(("cuid", "component_type", "temporary", "source"), "N/A"),
            )
            self.assertEqual(result["cuid_metadata_status"], "not_supported")

    def test_old_binding_without_snapshot_keeps_cuid(self):
        with self.case() as cli:
            del cli.api.amdsmi_get_gpu_cuid_info
            result = cli.helpers.get_gpu_cuid_info(1)
            self.assertEqual(result["cuid"], CUID)
            self.assertEqual(result["cuid_metadata_status"], "unsupported")
            self.assertEqual(result["temporary"], "unknown")

    def test_temporary_serialization(self):
        for auxiliary, fmt in itertools.product([True, False, None], ["json", "csv"]):
            with self.case(auxiliary=auxiliary, fmt=fmt) as cli:
                cli.api.amdsmi_get_gpu_cuid_info.return_value["auxiliary"] = auxiliary
                result = run_list(cli, fmt)
                expected = "unknown" if auxiliary is None else auxiliary
                if fmt == "csv":
                    expected = str(expected)
                self.assertEqual(result["temporary"], expected)
                self.assertNotIn("auxiliary", result)

    def test_static_reports_identity_without_node_state(self):
        for auxiliary in [False, True]:
            with self.case(auxiliary=auxiliary) as cli:
                cli.api.amdsmi_get_gpu_cuid_info.return_value["auxiliary"] = auxiliary

                def static_gpu():
                    command = cli.static()
                    command.helpers = cli.helpers
                    command.logger = cli.logger(format="json", helpers=cli.helpers)
                    command.group_check_printed = True
                    command.static_gpu(static_args(cuid=True))
                    command.logger.combine_arrays_to_json()

                result = json.loads(capture_stdout(static_gpu))
                self.assertEqual(list(result), ["gpu_data"])
                cli.api.amdsmi_get_cuid_seed_info.assert_not_called()
                gpu = result["gpu_data"][0]["cuid"]
                self.assertEqual(
                    list(gpu),
                    ["cuid", "component_type", "source", "temporary", "cuid_metadata_status"],
                )
                self.assertIs(gpu["temporary"], auxiliary)

    def test_no_cli_field_carries_the_primary(self):
        with self.case() as cli:
            result = cli.helpers.get_gpu_cuid_info(1)
        self.assertNotIn(PRIMARY, str(result))
        self.assertEqual([key for key in result if "primary" in key], [])

    def test_static_without_cuid_does_not_query_identity(self):
        with self.case() as cli:

            def static_gpu():
                command = cli.static()
                command.helpers = cli.helpers
                command.logger = cli.logger(format="json", helpers=cli.helpers)
                command.group_check_printed = True
                command.static_gpu(static_args(asic=True))
                command.logger.combine_arrays_to_json()

            result = json.loads(capture_stdout(static_gpu))
            self.assertNotIn("cuid", result["gpu_data"][0])
            cli.api.amdsmi_get_gpu_cuid_info.assert_not_called()
            cli.api.amdsmi_get_cuid_seed_info.assert_not_called()


class TestCliNodeCuid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (CLI / "amdsmi_helpers.py").is_file():
            raise unittest.SkipTest(f"amd-smi CLI not found at {CLI}")

    @contextmanager
    def cli(self):
        with ExitStack() as stack:
            yield make_cli(stack)

    def test_every_component_is_named_by_type_and_position(self):
        with self.cli() as cli:
            cuid = run_node(cli, "json", cuid=True)["cuid"]
        self.assertEqual(list(cuid), ["seed_provisioned", "seed_fingerprint", "components"])
        self.assertIs(cuid["seed_provisioned"], True)
        self.assertEqual(cuid["seed_fingerprint"], "0102030405060708")
        self.assertEqual(list(cuid["components"]), ["PLATFORM", "GPU 0", "GPU 1"])
        platform = cuid["components"]["PLATFORM"]
        self.assertEqual(
            list(platform), ["cuid", "component_type", "source", "temporary", "bdf", "device_path"]
        )
        self.assertEqual(platform["cuid"], OTHER_CUID)
        self.assertEqual(platform["component_type"], "PLATFORM")
        self.assertIs(platform["temporary"], True)
        self.assertEqual(platform["bdf"], "N/A")
        self.assertEqual(platform["device_path"], "N/A")
        gpu = cuid["components"]["GPU 1"]
        self.assertEqual(gpu["cuid"], LEGACY_UUID)
        self.assertEqual(gpu["source"], "DRIVER")
        self.assertEqual(gpu["bdf"], "0000:63:00.0")
        self.assertNotIn(PRIMARY, str(cuid))

    def test_csv_has_one_row_per_component(self):
        with self.cli() as cli:
            rows = run_node(cli, "csv", cuid=True)
        self.assertEqual([row["component"] for row in rows], ["PLATFORM", "GPU 0", "GPU 1"])
        self.assertEqual({row["seed_fingerprint"] for row in rows}, {"0102030405060708"})
        self.assertEqual(rows[1]["cuid"], CUID)

    def test_human_readable_lists_each_component(self):
        with self.cli() as cli:
            output = run_node(cli, "human_readable", cuid=True)
        self.assertIn("    CUID:\n        SEED_PROVISIONED: True\n", output)
        self.assertIn(
            "        GPU 1:\n            CUID: " + LEGACY_UUID + "\n"
            "            COMPONENT_TYPE: GPU\n            SOURCE: DRIVER\n"
            "            TEMPORARY: False\n",
            output,
        )

    def test_a_library_failure_reports_no_components(self):
        with self.cli() as cli:
            cuid = run_node(cli, "json", components=LibraryError(2), cuid=True)["cuid"]
        self.assertEqual(cuid["components"], "N/A")

    def test_a_library_failure_reports_no_components_in_csv(self):
        with self.cli() as cli:
            rows = run_node(cli, "csv", components=LibraryError(2), cuid=True)
        self.assertEqual([row["components"] for row in rows], ["N/A"])

    def test_cuid_is_not_part_of_the_default_node_output(self):
        with self.cli() as cli:
            node = run_node(cli, "json")
            cli.api.amdsmi_get_cuid_components.assert_not_called()
            cli.api.amdsmi_get_cuid_seed_info.assert_not_called()
        self.assertNotIn("cuid", node)


class TestCliCuidOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub_modules(
            cls,
            {
                "_version": fake_module("_version", __version__="0.0.0+test"),
                "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=mock.Mock),
                "cuid_parser_under_test": None,
            },
        )
        cls.module = load_cli_module(
            "cuid_parser_under_test", str(CLI / "amdsmi_parser.py"), str(CLI)
        )

    def options(self, subcommand):
        helpers = mock.Mock()
        for name in (
            "is_amdgpu_initialized",
            "is_brcm_switch_initialized",
            "is_ainic_initialized",
            "is_brcm_nic_initialized",
            "is_amd_hsmp_initialized",
            "is_hypervisor",
            "is_windows",
            "is_virtual_os",
        ):
            getattr(helpers, name).return_value = False
        helpers.is_amdgpu_initialized.return_value = True
        helpers.get_gpu_choices.return_value = ({"0": 1}, "0")
        helpers.is_baremetal.return_value = True
        helpers.is_linux.return_value = True
        helpers.os_info.return_value = "Linux Baremetal"
        helpers.get_rocm_version.return_value = "unknown"
        helpers.get_clock_types.side_effect = lambda: (["SCLK", "MCLK", "PCIE"], "")
        with mock.patch.dict(os.environ):
            os.environ.pop("_ARGCOMPLETE", None)
            parser = self.module.AMDSMIParser(
                *([mock.Mock()] * 20), sys_argv=["amd-smi", subcommand], helpers=helpers
            )
        return set(parser.subparsers.choices[subcommand]._option_string_actions)

    def test_no_subcommand_offers_a_primary_cuid(self):
        for subcommand in ("static", "node"):
            with self.subTest(subcommand=subcommand):
                options = self.options(subcommand)
                self.assertIn("--cuid", options)
                self.assertIn("-Y", options)
                self.assertNotIn("--cuid-primary", options)
                self.assertNotIn("-y", options)
