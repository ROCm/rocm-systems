#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consumer identity semantics with real CLI renderers and API/sysfs fixtures."""

import argparse
import csv
import importlib.util
import io
import itertools
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

# The source tree's CLI, else the installed one beside the installed tests.
# Only amdsmi_path is needed from common.common, and a stale install makes it
# raise more than ImportError.
try:
    from common.common import amdsmi_path
except Exception:  # pragma: no cover - harness/install unavailable or stale
    amdsmi_path = None


def _resolve_cli():
    candidates = [Path(__file__).resolve().parents[4] / "amdsmi_cli"]
    if amdsmi_path:
        candidates.append(Path(amdsmi_path).parents[1] / "libexec" / "amdsmi_cli")
    for candidate in candidates:
        if (candidate / "amdsmi_helpers.py").is_file():
            return candidate
    return candidates[0]


CLI = _resolve_cli()
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
    api.amdsmi_wrapper = types.SimpleNamespace(AMDSMI_STATUS_NO_PERM=10, AMDSMI_STATUS_INVAL=1)
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


def make_sysfs(cli, stack):
    tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
    device = tmp_path / "devices/pci0000:00/0000:03:00.0"
    device.mkdir(parents=True)
    render = tmp_path / "class/drm/renderD128"
    render.mkdir(parents=True)
    (render / "device").symlink_to(device, target_is_directory=True)
    (device / "cuid_derived").write_text(CUID + "\n")
    (device / "cuid_seed_state").write_text("provisioned\n")

    def sysfs_path(path):
        assert path.startswith("/sys/")
        return tmp_path / path[len("/sys/") :]

    stack.enter_context(mock.patch.object(cli.helpers_module, "Path", sysfs_path))
    return device


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
    args = dict.fromkeys(
        ("power_management", "base_board_temps", "gtt", "tray", "cuid", "cuid_primary"), False
    )
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
            "partition",
            "cuid",
            "cuid_primary",
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
    def case(self, sysfs=True, **params):
        with ExitStack() as outer:
            if params:
                outer.enter_context(self.subTest(**params))
            with ExitStack() as stack:
                cli = make_cli(stack)
                yield cli, make_sysfs(cli, stack) if sysfs else None

    def test_source_auxiliary_table(self):
        for source, auxiliary in itertools.product(
            ["DRIVER", "LIBRARY", "UNKNOWN"], [True, False, None]
        ):
            with self.case(source=source, auxiliary=auxiliary) as (cli, sysfs):
                cli.api.amdsmi_get_gpu_cuid_info.return_value.update(
                    source=source, auxiliary=auxiliary
                )
                result = cli.helpers.get_gpu_cuid_info(1)
                self.assertEqual(result["source"], source)
                self.assertEqual(result["auxiliary"], "unknown" if auxiliary is None else auxiliary)
                expected = "temporary" if auxiliary is True else "unknown"
                if source == "DRIVER" and auxiliary is False:
                    expected = "provisioned"
                else:
                    cli.api.amdsmi_get_gpu_enumeration_info.assert_not_called()
                self.assertEqual(result["effective_seed"], expected)
                self.assertEqual(result["primary_cuid"], "N/A (not requested)")
                cli.api.amdsmi_get_cuid_seed_info.assert_not_called()

    def test_actual_reader_state(self):
        for state, partition in itertools.product(
            ["unprovisioned\n", "provisioned\n"], [None, 0, 1, 7]
        ):
            with self.case(state=state, partition=partition) as (cli, sysfs):
                node = sysfs
                if partition is not None:
                    node = sysfs / "xcp"
                    node.mkdir()
                    (node / "cuid_derived").write_text(CUID + "\n")
                    cli.api.amdsmi_get_gpu_kfd_info.return_value["current_partition_id"] = partition
                    (sysfs / "cuid_derived").write_text(OTHER_CUID + "\n")
                (node / "cuid_seed_state").write_text(state)
                self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], state.strip())

    def test_w6800_whole_device_with_kfd_zero(self):
        for state, fmt in itertools.product(["unprovisioned\n", "provisioned\n"], ["json", "csv"]):
            with self.case(state=state, fmt=fmt) as (cli, sysfs):
                cli.api.amdsmi_get_gpu_kfd_info.return_value["current_partition_id"] = 0
                (sysfs / "cuid_seed_state").write_text(state)
                result = run_list(cli, fmt)
                self.assertEqual(result["cuid"], CUID)
                self.assertEqual(result["source"], "DRIVER")
                self.assertEqual(result["auxiliary"], False if fmt == "json" else "False")
                self.assertEqual(result["effective_seed"], state.strip())

    def test_absent_or_malformed_state_is_unknown(self):
        for state in [
            None,
            "",
            "provisioned",
            "custom \n",
            "custom\r\n",
            "PROVISIONED\n",
            "unprovisioned\nprovisioned\n",
            "secure\n",
            "\xff\n",
        ]:
            with self.case(state=state) as (cli, sysfs):
                path = sysfs / "cuid_seed_state"
                if state is None:
                    path.unlink()
                else:
                    path.write_bytes(state.encode("latin-1"))
                result = cli.helpers.get_gpu_cuid_info(1)
                self.assertEqual(result["derived_cuid"], CUID)
                self.assertEqual(result["source"], "DRIVER")
                self.assertIs(result["auxiliary"], False)
                self.assertEqual(result["effective_seed"], "unknown")

    def test_permission_failure_is_unknown(self):
        for attribute in ["cuid_seed_state", "cuid_derived"]:
            with self.case(attribute=attribute) as (cli, sysfs):
                open_path = Path.open

                def denied(path, *args, **kwargs):
                    if path.name == attribute:
                        raise PermissionError("denied")
                    return open_path(path, *args, **kwargs)

                with mock.patch.object(Path, "open", denied):
                    self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_partitions_never_fall_back_to_parent(self):
        for partition, missing in itertools.product(
            [0, 1, 7], ["xcp", "cuid_seed_state", "cuid_derived"]
        ):
            with self.case(partition=partition, missing=missing) as (cli, sysfs):
                cli.api.amdsmi_get_gpu_kfd_info.return_value["current_partition_id"] = partition
                if partition == 0:
                    (sysfs / "current_compute_partition").write_text("CPX\n")
                if missing != "xcp":
                    node = sysfs / "xcp"
                    node.mkdir()
                    (node / "cuid_derived").write_text(CUID + "\n")
                    (node / "cuid_seed_state").write_text("unprovisioned\n")
                    (node / missing).unlink()
                self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_partition_interface_without_xcp_is_unknown(self):
        for partition in [0, "N/A", 0xFFFFFFFF]:
            with self.case(partition=partition) as (cli, sysfs):
                cli.api.amdsmi_get_gpu_kfd_info.return_value["current_partition_id"] = partition
                (sysfs / "current_compute_partition").write_text("CPX\n")
                self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_invalid_partition_id_is_unknown(self):
        for partition in [None, False, True, 0.0, "0", -1, "", 0x100000000]:
            with self.case(partition=partition) as (cli, sysfs):
                cli.api.amdsmi_get_gpu_kfd_info.return_value["current_partition_id"] = partition
                self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_partition_interface_appearing_during_whole_device_read(self):
        for attribute in ["xcp", "current_compute_partition"]:
            with self.case(attribute=attribute) as (cli, sysfs):
                cli.api.amdsmi_get_gpu_kfd_info.return_value["current_partition_id"] = 0
                read_bytes = Path.read_bytes

                def changed(path, *args, **kwargs):
                    value = read_bytes(path, *args, **kwargs)
                    if path.name == "cuid_seed_state":
                        (sysfs / attribute).mkdir()
                    return value

                with mock.patch.object(Path, "read_bytes", changed):
                    self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_broken_xcp_link_is_not_a_whole_device(self):
        with self.case() as (cli, sysfs):
            (sysfs / "xcp").symlink_to(sysfs / "removed-partition")
            self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_render_node_selects_partition_instead_of_bdf(self):
        with self.case() as (cli, sysfs):
            other = sysfs.parent / "amdgpu_xcp.1"
            (other / "xcp").mkdir(parents=True)
            (other / "xcp/cuid_derived").write_text(CUID + "\n")
            (other / "xcp/cuid_seed_state").write_text("unprovisioned\n")
            render_link = sysfs.parents[2] / "class/drm/renderD128/device"
            render_link.unlink()
            render_link.symlink_to(other, target_is_directory=True)
            self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unprovisioned")
            cli.api.amdsmi_get_gpu_device_bdf.assert_not_called()

    def test_invalid_render_is_unknown(self):
        for render in [None, "N/A", "128", -1, 0, 0xFFFFFFFF]:
            with self.case(render=render) as (cli, sysfs):
                cli.api.amdsmi_get_gpu_enumeration_info.return_value = {"drm_render": render}
                self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_topology_error_is_unknown(self):
        for api in ["amdsmi_get_gpu_enumeration_info", "amdsmi_get_gpu_kfd_info"]:
            with self.case(api=api) as (cli, sysfs):
                getattr(cli.api, api).side_effect = LibraryError()
                self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_missing_topology_is_unknown(self):
        with self.case() as (cli, sysfs):
            cli.api.amdsmi_get_gpu_kfd_info.return_value = {}
            self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_stale_or_changing_cuid_is_not_labeled(self):
        for values, partition in itertools.product(
            [(OTHER_CUID, OTHER_CUID), (CUID, OTHER_CUID), (OTHER_CUID, CUID)], [0, "N/A"]
        ):
            with self.case(values=values, partition=partition) as (cli, sysfs):
                cli.api.amdsmi_get_gpu_kfd_info.return_value["current_partition_id"] = partition
                read_text = Path.read_text
                observed = iter(values)

                def changed(path, *args, **kwargs):
                    if path.name == "cuid_derived":
                        return next(observed) + "\n"
                    return read_text(path, *args, **kwargs)

                with mock.patch.object(Path, "read_text", changed):
                    result = cli.helpers.get_gpu_cuid_info(1)
                self.assertEqual(result["derived_cuid"], CUID)
                self.assertEqual(result["effective_seed"], "unknown")

    def test_enumeration_fields_still_precede_new_fields(self):
        for fmt in ["json", "csv"]:
            with self.case(fmt=fmt) as (cli, sysfs):
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

    def test_human_list_reports_kind_without_private_diagnostics(self):
        with self.case() as (cli, sysfs):

            def list_gpu():
                command = cli.list()
                command.helpers = cli.helpers
                command.logger = cli.logger(format="human_readable", helpers=cli.helpers)
                command.group_check_printed = True
                command.list_gpu(argparse.Namespace(gpu=[1], enumeration=False))

            output = capture_stdout(list_gpu)
            self.assertIn(f"UUID: {CUID}", output)
            self.assertIn("IDENTIFIER_KIND: cuid", output)
            self.assertIn("EFFECTIVE_SEED: provisioned", output)
            self.assertNotIn("FINGERPRINT", output)
            self.assertNotIn("PRIMARY", output)

    def test_changing_render_device_is_unknown(self):
        with self.case() as (cli, sysfs):
            other = sysfs.parent / "replacement"
            other.mkdir()
            render_link = sysfs.parents[2] / "class/drm/renderD128/device"
            read_bytes = Path.read_bytes

            def changed(path, *args, **kwargs):
                value = read_bytes(path, *args, **kwargs)
                if path.name == "cuid_seed_state":
                    render_link.unlink()
                    render_link.symlink_to(other, target_is_directory=True)
                return value

            with mock.patch.object(Path, "read_bytes", changed):
                self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_inaccessible_partition_topology_has_no_parent_fallback(self):
        for attribute in ["xcp", "current_compute_partition"]:
            with self.case(attribute=attribute) as (cli, sysfs):
                lstat = Path.lstat

                def denied(path, *args, **kwargs):
                    if path.name == attribute:
                        raise PermissionError("denied")
                    return lstat(path, *args, **kwargs)

                with mock.patch.object(Path, "lstat", denied):
                    self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_missing_whole_device_cuid_is_unknown(self):
        with self.case() as (cli, sysfs):
            (sysfs / "cuid_derived").unlink()
            self.assertEqual(cli.helpers.get_gpu_cuid_info(1)["effective_seed"], "unknown")

    def test_list_alias_compatibility(self):
        for fmt, kind in itertools.product(["json", "csv"], ["cuid", "legacy_uuid", "unknown"]):
            with self.case(fmt=fmt, kind=kind) as (cli, sysfs):
                if kind != "cuid":
                    cli.api.amdsmi_get_gpu_cuid_info.side_effect = LibraryError(2)
                    cli.api.amdsmi_get_gpu_device_cuid.side_effect = LibraryError(2)
                if kind == "unknown":
                    cli.api.amdsmi_get_gpu_device_uuid.side_effect = LibraryError(2)
                result = run_list(cli, fmt)
                alias = "gpu_uuid" if fmt == "csv" else "uuid"
                old_keys = (
                    ["gpu", "gpu_bdf", "gpu_uuid"] if fmt == "csv" else ["gpu", "bdf", "uuid"]
                )
                old_keys += ["kfd_id", "node_id", "partition_id"]
                self.assertEqual(list(result)[:6], old_keys)
                self.assertEqual(
                    result[alias],
                    {"cuid": CUID, "legacy_uuid": LEGACY_UUID, "unknown": "N/A"}[kind],
                )
                self.assertEqual(result["cuid"], CUID if kind == "cuid" else "N/A")
                self.assertEqual(result["identifier_kind"], kind)
                self.assertEqual(
                    result["cuid_metadata_status"],
                    "available" if kind == "cuid" else "amdsmi_error_2",
                )
                self.assertEqual(
                    result["effective_seed"],
                    {"cuid": "provisioned", "legacy_uuid": "not_applicable", "unknown": "unknown"}[
                        kind
                    ],
                )
                self.assertNotIn("primary_cuid", result)
                self.assertNotIn("seed_fingerprint", result)
                self.assertNotIn("ready", result)
                self.assertNotIn(PRIMARY, str(result))
                cli.api.amdsmi_get_cuid_seed_info.assert_not_called()

    def test_metadata_failure_keeps_available_cuid(self):
        for fmt, code in itertools.product(["json", "csv"], [2, 10, 34]):
            with self.case(fmt=fmt, code=code) as (cli, sysfs):
                cli.api.amdsmi_get_gpu_cuid_info.side_effect = LibraryError(code)
                result = run_list(cli, fmt)
                self.assertEqual(result["identifier_kind"], "cuid")
                self.assertEqual(result["cuid"], CUID)
                self.assertEqual(result["source"], "UNKNOWN")
                self.assertEqual(result["auxiliary"], "unknown")
                self.assertEqual(result["effective_seed"], "unknown")
                self.assertEqual(result["cuid_metadata_status"], f"amdsmi_error_{code}")
                cli.api.amdsmi_get_gpu_device_uuid.assert_not_called()
                cli.api.amdsmi_get_gpu_enumeration_info.assert_not_called()

    def test_old_binding_without_snapshot_keeps_cuid(self):
        with self.case() as (cli, sysfs):
            del cli.api.amdsmi_get_gpu_cuid_info
            result = cli.helpers.get_gpu_cuid_info(1)
            self.assertEqual(result["identifier_kind"], "cuid")
            self.assertEqual(result["derived_cuid"], CUID)
            self.assertEqual(result["cuid_metadata_status"], "unsupported")
            self.assertEqual(result["auxiliary"], "unknown")

    def test_auxiliary_serialization(self):
        for auxiliary, fmt in itertools.product([True, False, None], ["json", "csv"]):
            with self.case(auxiliary=auxiliary, fmt=fmt) as (cli, sysfs):
                cli.api.amdsmi_get_gpu_cuid_info.return_value["auxiliary"] = auxiliary
                result = run_list(cli, fmt)
                expected = "unknown" if auxiliary is None else auxiliary
                if fmt == "csv":
                    expected = str(expected)
                self.assertEqual(result["auxiliary"], expected)

    def test_static_separates_local_seed_and_driver(self):
        for auxiliary in [False, True]:
            with self.case(auxiliary=auxiliary) as (cli, sysfs):
                (sysfs / "cuid_seed_state").write_text("unprovisioned\n")
                cli.api.amdsmi_get_gpu_cuid_info.return_value["auxiliary"] = auxiliary

                def static_gpu():
                    command = cli.static()
                    command.helpers = cli.helpers
                    command.logger = cli.logger(format="json", helpers=cli.helpers)
                    command.group_check_printed = True
                    command._report_cuid_seed()
                    command.static_gpu(static_args(cuid=True))
                    command.logger.combine_arrays_to_json()

                result = json.loads(capture_stdout(static_gpu))
                self.assertIs(result["seed_provisioned"], True)
                self.assertEqual(result["seed_fingerprint"], "0102030405060708")
                self.assertNotIn("seed_info_scope", result)
                gpu = result["gpu_data"][0]["cuid"]
                self.assertEqual(
                    gpu["effective_seed"], "temporary" if auxiliary else "unprovisioned"
                )
                self.assertEqual(gpu["primary_cuid"], "N/A (not requested)")
                self.assertNotIn("seed_fingerprint", gpu)

    def test_primary_requires_explicit_selection(self):
        with self.case() as (cli, sysfs):
            self.assertEqual(
                cli.helpers.get_gpu_cuid_info(1)["primary_cuid"], "N/A (not requested)"
            )
            self.assertEqual(
                cli.helpers.get_gpu_cuid_info(1, include_primary=True)["primary_cuid"], PRIMARY
            )

    def test_static_without_cuid_does_not_query_seed_or_identity(self):
        with self.case(sysfs=False) as (cli, _):

            def static_gpu():
                command = cli.static()
                command.helpers = cli.helpers
                command.logger = cli.logger(format="json", helpers=cli.helpers)
                command.group_check_printed = True
                command.static_gpu(static_args(asic=True))
                command.logger.combine_arrays_to_json()

            result = json.loads(capture_stdout(static_gpu))
            self.assertNotIn("cuid", result["gpu_data"][0])
            self.assertNotIn("seed_fingerprint", result)
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
        self.assertIs(cuid["seed_provisioned"], True)
        self.assertEqual(cuid["seed_fingerprint"], "0102030405060708")
        self.assertEqual(list(cuid["components"]), ["PLATFORM", "GPU 0", "GPU 1"])
        platform = cuid["components"]["PLATFORM"]
        self.assertEqual(platform["derived_cuid"], OTHER_CUID)
        self.assertIs(platform["auxiliary"], True)
        self.assertEqual(platform["bdf"], "N/A")
        self.assertEqual(platform["device_path"], "N/A")
        gpu = cuid["components"]["GPU 1"]
        self.assertEqual(gpu["derived_cuid"], LEGACY_UUID)
        self.assertEqual(gpu["source"], "DRIVER")
        self.assertEqual(gpu["bdf"], "0000:63:00.0")
        self.assertEqual(gpu["primary_cuid"], "N/A (not requested)")

    def test_primary_is_opt_in_and_says_when_it_needs_root(self):
        with self.cli() as cli:
            components = run_node(cli, "json", cuid_primary=True)["cuid"]["components"]
        self.assertEqual(components["GPU 0"]["primary_cuid"], PRIMARY)
        self.assertEqual(components["PLATFORM"]["primary_cuid"], "N/A (requires root)")

    def test_csv_has_one_row_per_component(self):
        with self.cli() as cli:
            rows = run_node(cli, "csv", cuid=True)
        self.assertEqual([row["component"] for row in rows], ["PLATFORM", "GPU 0", "GPU 1"])
        self.assertEqual({row["seed_fingerprint"] for row in rows}, {"0102030405060708"})
        self.assertEqual(rows[1]["derived_cuid"], CUID)

    def test_human_readable_lists_each_component(self):
        with self.cli() as cli:
            output = run_node(cli, "human_readable", cuid=True)
        self.assertIn("    CUID:\n        SEED_PROVISIONED: True", output)
        self.assertIn("        GPU 1:\n            DERIVED_CUID: " + LEGACY_UUID, output)

    def test_a_library_failure_reports_no_components(self):
        with self.cli() as cli:
            cuid = run_node(cli, "json", components=LibraryError(2), cuid=True)["cuid"]
        self.assertEqual(cuid["components"], "N/A")

    def test_cuid_is_not_part_of_the_default_node_output(self):
        with self.cli() as cli:
            node = run_node(cli, "json")
            cli.api.amdsmi_get_cuid_components.assert_not_called()
            cli.api.amdsmi_get_cuid_seed_info.assert_not_called()
        self.assertNotIn("cuid", node)


if __name__ == "__main__":
    unittest.main()
