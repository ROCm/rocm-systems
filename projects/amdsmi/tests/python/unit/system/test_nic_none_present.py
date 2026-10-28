#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""An explicit ``--nic`` on a host with no NICs reports that, and never prints GPU data.

NIC init is always requested, so ``--nic`` is registered on GPU-only hosts too. A bare ``--nic``
then selects an empty list, which is falsy: ``metric`` and ``firmware`` fell through to the GPU
path and ``static`` printed nothing. ``list`` already says "No AI NICs found"; these tests hold
the other three subcommands to the same behavior.
"""

import contextlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module

_CLI_DIR = find_cli_dir(*cli_search_order(str(Path(__file__).resolve().parent)))
_SUBCOMMANDS = Path(_CLI_DIR) / "subcommands" if _CLI_DIR else None
_NO_NICS_TEXT = "No AI NICs found"


class _LibraryError(Exception):
    def get_error_info(self) -> str:
        return "stub error"


def _amdsmi_stubs() -> dict:
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_LibraryError)
    interface = fake_module("amdsmi.amdsmi_interface", AMDSMI_MAX_RAIL_INDEX=0xFFFFFFFF)
    package = fake_module("amdsmi", amdsmi_exception=exception, amdsmi_interface=interface)
    return {
        "amdsmi": package,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_interface": interface,
    }


def _load(file_name: str, extra_stubs: dict = None) -> types.ModuleType:
    """Import one subcommand file in isolation, with ``amdsmi`` stubbed."""
    stubs = _amdsmi_stubs()
    stubs.update(extra_stubs or {})
    path = _SUBCOMMANDS / file_name if _SUBCOMMANDS else None
    with mock.patch.dict(sys.modules, stubs):
        return load_cli_module("nic_none_present_" + file_name, str(path) if path else None)


def _command(cls: type, **attributes) -> object:
    """Build a subcommand mixin without its __init__, wired to mocks."""

    class _Command(cls):
        def __init__(self) -> None:
            self.helpers = mock.Mock()
            self.helpers.is_ainic_initialized.return_value = True
            self.logger = mock.Mock()
            self.device_handles = [mock.Mock()]
            self.device_handles_ainics = []

    command = _Command()
    for name, value in attributes.items():
        setattr(command, name, value)
    return command


class TestNoNicsPrintsAMessage(unittest.TestCase):
    def test_metric_reports_no_nics_and_skips_the_gpu_path(self) -> None:
        command = _command(_load("metric.py").MetricCommands, metric_nic=mock.Mock())
        command.metric(types.SimpleNamespace(nic=[]))
        command.logger.print_no_ainics_found.assert_called_once_with()
        command.metric_nic.assert_not_called()
        command.helpers.handle_gpus.assert_not_called()

    def test_metric_still_dispatches_a_selected_nic(self) -> None:
        command = _command(_load("metric.py").MetricCommands, metric_nic=mock.Mock())
        args = types.SimpleNamespace(nic=[mock.Mock()])
        command.metric(args)
        command.metric_nic.assert_called_once()
        command.logger.print_no_ainics_found.assert_not_called()

    def test_firmware_reports_no_nics_and_skips_the_gpu_path(self) -> None:
        command = _command(_load("firmware.py").FirmwareCommands, firmware_nic=mock.Mock())
        command.firmware(types.SimpleNamespace(nic=[], gpu=None, fw_list=True))
        command.logger.print_no_ainics_found.assert_called_once_with()
        command.firmware_nic.assert_not_called()
        command.helpers.handle_gpus.assert_not_called()

    def test_firmware_still_dispatches_a_selected_nic(self) -> None:
        command = _command(_load("firmware.py").FirmwareCommands, firmware_nic=mock.Mock())
        command.firmware(types.SimpleNamespace(nic=[mock.Mock()], gpu=None, fw_list=True))
        command.firmware_nic.assert_called_once()
        command.logger.print_no_ainics_found.assert_not_called()

    def test_static_reports_no_nics_and_stops_before_cpu_and_gpu(self) -> None:
        static_stubs = {
            "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=object),
            "amdsmi_cli_exceptions": fake_module(
                "amdsmi_cli_exceptions", AmdSmiInvalidParameterException=Exception
            ),
        }
        command = _command(
            _load("static.py", static_stubs).StaticCommands, _static_ainic=mock.Mock()
        )
        is_handled = command._static_nics(types.SimpleNamespace(nic=[]), False, None)
        self.assertTrue(is_handled)
        command.logger.print_no_ainics_found.assert_called_once_with()
        command._static_ainic.assert_not_called()


class TestPrintNoAinicsFound(unittest.TestCase):
    """The shared message, rendered by the real logger."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI directory not found")
        helpers = fake_module("amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {}))
        with mock.patch.dict(sys.modules, {"amdsmi_helpers": helpers}):
            module = load_cli_module(
                "nic_none_present_logger", str(Path(_CLI_DIR) / "amdsmi_logger.py")
            )
        cls.logger_cls = module.AMDSMILogger

    def _render(self, format_name: str) -> str:
        logger = self.logger_cls.__new__(self.logger_cls)
        logger.format = format_name
        logger.destination = "stdout"
        logger.output = {"stale": "value"}
        logger.multiple_device_output = [{"stale": "value"}]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            logger.print_no_ainics_found()
        self.assertEqual(logger.output, {})
        self.assertEqual(logger.multiple_device_output, [])
        return out.getvalue()

    def test_human_readable_output_says_no_nics_were_found(self) -> None:
        self.assertEqual(self._render("human_readable"), _NO_NICS_TEXT + "\n")

    def test_json_output_is_an_empty_list(self) -> None:
        self.assertEqual(json.loads(self._render("json")), [])
