#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""``metric --rdma [idx]`` is registered like ``--port [idx]`` and does not need ``--port``.

The real ``AMDSMIParser._add_metric_parser`` runs against a stand-in ``self`` whose helpers report
only the AINIC driver as initialized, so no driver is needed.
"""

import argparse
import os
import sys
import unittest
from unittest import mock

from common.common import (
    cli_search_order,
    fake_module,
    find_cli_dir,
    generated_version_stub,
    stub_modules_at_import,
)

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
for _path in (_CLI_DIR, os.path.join(_CLI_DIR, "subcommands") if _CLI_DIR else None):
    if _path and _path not in sys.path:
        sys.path.append(_path)

# amdsmi_parser pulls in amdsmi_helpers, which does ``from amdsmi_init import *`` at load.
_stubs = {}
if "amdsmi_init" not in sys.modules:
    from amdsmi import amdsmi_exception as _amdsmi_exception
    from amdsmi import amdsmi_interface as _amdsmi_interface

    _stubs["amdsmi_init"] = fake_module(
        "amdsmi_init",
        AMDSMI_INIT_FLAG=0,
        AMDSMI_INITIALIZED=True,
        amdsmi_interface=_amdsmi_interface,
        amdsmi_exception=_amdsmi_exception,
    )
_stubs.update(generated_version_stub())
_restore_stubs = stub_modules_at_import(_stubs) if _stubs else None

_PARSER_IMPORT_ERROR = None
try:
    from amdsmi_parser import AMDSMIParser  # noqa: E402
except ImportError as _import_error:
    AMDSMIParser = None
    _PARSER_IMPORT_ERROR = _import_error


def setUpModule():
    if _PARSER_IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"amd-smi CLI parser unusable: {_PARSER_IMPORT_ERROR}")


def tearDownModule():
    if _restore_stubs is not None:
        _restore_stubs()


class _AinicOnlyHelpers:
    """Every ``is_*_initialized`` is False except the AINIC one; anything else is a mock."""

    def __getattr__(self, name):
        if name == "is_ainic_initialized":
            return lambda: True
        if name.startswith("is_"):
            return lambda *args, **kwargs: False
        return mock.MagicMock()


def _metric_parser():
    parser_self = mock.MagicMock()
    parser_self.helpers = _AinicOnlyHelpers()
    parser_self._guard_extended_requires_port = AMDSMIParser._guard_extended_requires_port
    parser_self._guard_nic_arguments_require_nic = AMDSMIParser._guard_nic_arguments_require_nic
    subparsers = argparse.ArgumentParser().add_subparsers()
    AMDSMIParser._add_metric_parser(parser_self, subparsers, lambda *args, **kwargs: None)
    parser = subparsers.choices["metric"]
    # The stand-in does not register the device arguments, so add the one the NIC checks read.
    parser.add_argument("--nic", nargs="*", default=None)
    return parser


class TestMetricNicRdmaParser(unittest.TestCase):
    def setUp(self):
        self.parser = _metric_parser()

    def _parse(self, *argv):
        namespace, _ = self.parser.parse_known_args(["--nic", *argv])
        return namespace

    def test_bare_rdma_means_every_port(self):
        self.assertEqual(self._parse("--rdma").rdma, -1)

    def test_rdma_takes_an_index(self):
        self.assertEqual(self._parse("--rdma", "1").rdma, 1)

    def test_rdma_defaults_to_off(self):
        self.assertIsNone(self._parse().rdma)

    def test_rdma_combines_with_port(self):
        namespace = self._parse("--rdma", "--port")
        self.assertEqual((namespace.rdma, namespace.port), (-1, -1))

    def test_help_points_at_static_and_explains_its_per_device_numbering(self):
        action = next(a for a in self.parser._actions if "--rdma" in a.option_strings)
        self.assertIn("static --nic", action.help)
        self.assertIn("per device", action.help)

    def test_extended_still_requires_port_when_rdma_is_given(self):
        with mock.patch.object(self.parser, "exit", side_effect=SystemExit(2)):
            with self.assertRaises(SystemExit):
                self._parse("--rdma", "--extended")


class TestMetricNicArgumentValidation(unittest.TestCase):
    """``--port``, ``--rdma`` and ``--sensors`` mean nothing without ``--nic``, and an index is not negative."""

    def setUp(self):
        self.parser = _metric_parser()

    def _parse(self, *argv):
        with mock.patch.object(self.parser, "exit", side_effect=SystemExit(2)):
            namespace, _ = self.parser.parse_known_args(list(argv))
        return namespace

    def _assert_rejected(self, *argv):
        with mock.patch.object(self.parser, "exit", side_effect=SystemExit(2)):
            with self.assertRaises(SystemExit):
                self.parser.parse_known_args(list(argv))

    def test_port_without_nic_is_rejected(self):
        self._assert_rejected("--port")

    def test_rdma_without_nic_is_rejected(self):
        self._assert_rejected("--rdma", "0")

    def test_sensors_without_nic_is_rejected(self):
        self._assert_rejected("--sensors")

    def test_the_nic_flags_are_accepted_with_nic(self):
        namespace = self._parse("--nic", "--port", "0", "--rdma", "--sensors")
        self.assertEqual((namespace.port, namespace.rdma, namespace.sensors), (0, -1, True))

    def test_a_negative_port_index_is_rejected(self):
        self._assert_rejected("--nic", "--port", "-5")

    def test_a_negative_rdma_index_is_rejected(self):
        self._assert_rejected("--nic", "--rdma", "-2")


class TestMetricNicSensorsParser(unittest.TestCase):
    def setUp(self):
        self.parser = _metric_parser()

    def _parse(self, *argv):
        namespace, _ = self.parser.parse_known_args(["--nic", *argv])
        return namespace

    def test_sensors_defaults_to_off(self):
        self.assertFalse(self._parse().sensors)

    def test_sensors_turns_on_without_a_value(self):
        self.assertTrue(self._parse("--sensors").sensors)

    def test_sensors_combines_with_port_and_rdma(self):
        namespace = self._parse("--sensors", "--port", "--rdma")
        self.assertEqual((namespace.sensors, namespace.port, namespace.rdma), (True, -1, -1))

    def test_help_says_it_shows_thresholds(self):
        action = next(a for a in self.parser._actions if "--sensors" in a.option_strings)
        self.assertIn("thresholds", action.help)
