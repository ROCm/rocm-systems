#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Direct unit tests for the REAL ``AMDSMIHelpers`` AMPP validators.

``test_cli_ampp_configure.py`` installs a hand-written ``_FakeAMDSMIHelpers``
into ``sys.modules`` so the CLI layers under test import cleanly, which means
those tests never touch the shipped implementation. This file imports
``amdsmi_cli/amdsmi_helpers.py`` for real (same fake-``amdsmi``-package
bootstrap ``test_cli_ras_cper_json.py`` uses) and pins down
``is_valid_ampp_field_name`` / ``parse_ampp_field_value`` -- the two checks
guarding every AMPP name and value the CLI hands to the library.
"""

import os
import sys
import unittest

from common.common import (
    add_class_cleanup,
    cli_search_order,
    fake_module,
    find_cli_dir,
    stub_modules,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = find_cli_dir(*cli_search_order(_THIS_DIR))
_HELPERS_SRC = os.path.join(_CLI_DIR, "amdsmi_helpers.py") if _CLI_DIR else ""

# Modules imported (directly or transitively) when amdsmi_helpers loads against
# the faked ``amdsmi`` package; a stub left in place here would corrupt a
# sibling test's real/installed import, so these are removed up front and
# restored via ``stub_modules_at_import``'s returned restore callable.
_CLI_MODULES = (
    "amdsmi",
    "amdsmi.amdsmi_interface",
    "amdsmi.amdsmi_exception",
    "amdsmi_init",
    "amdsmi_helpers",
    "amdsmi_cli_exceptions",
    "BDF",
)

# Mirrors py-interface/amdsmi_interface.py; the validators compare against it.
_MAX_STRING_LENGTH = 256


class _FakeInitFlags:
    INIT_ALL_PROCESSORS = 0xFFFFFFFF
    INIT_AMD_GPUS = 1
    INIT_AMD_CPUS = 2
    INIT_AMD_NICS = 4


class _FakeClkType:
    """Enough ``AmdSmiClkType`` members for ``amdsmi_helpers`` to import."""

    SYS = "SYS"
    MEM = "MEM"
    DF = "DF"
    SOC = "SOC"
    DCEF = "DCEF"
    VCLK0 = "VCLK0"
    VCLK1 = "VCLK1"
    DCLK0 = "DCLK0"
    DCLK1 = "DCLK1"


class _FakeException(Exception):
    def __init__(self, *args):
        super().__init__("mock error")


def _build_fake_amdsmi_modules():
    # AMDSMIHelpers reads these off amdsmi_wrapper at class-body-definition time
    # (CPER_DECODE_MESSAGES), i.e. during this module's import -- an empty stub
    # crashes the import before any test runs.
    wrapper = fake_module(
        "amdsmi.amdsmi_wrapper",
        AMDSMI_STATUS_INVAL=1,
        AMDSMI_STATUS_UNEXPECTED_SIZE=2,
        AMDSMI_STATUS_UNEXPECTED_DATA=3,
        AMDSMI_STATUS_NOT_SUPPORTED=4,
    )
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        amdsmi_wrapper=wrapper,
        AmdSmiInitFlags=_FakeInitFlags,
        AmdSmiClkType=_FakeClkType,
        AmdSmiLibraryException=_FakeException,
        AmdSmiParameterException=_FakeException,
        AMDSMI_MAX_STRING_LENGTH=_MAX_STRING_LENGTH,
        amdsmi_init=lambda _flag: None,
        amdsmi_shut_down=lambda: None,
        amdsmi_get_processor_handles=lambda: [],
    )
    exception = fake_module(
        "amdsmi.amdsmi_exception",
        AmdSmiLibraryException=_FakeException,
        AmdSmiParameterException=_FakeException,
    )
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)

    modules = {name: None for name in _CLI_MODULES}
    modules["amdsmi"] = amdsmi_pkg
    modules["amdsmi.amdsmi_interface"] = interface
    modules["amdsmi.amdsmi_exception"] = exception
    return modules


class TestAmppValidators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(_HELPERS_SRC):
            raise unittest.SkipTest(f"amd-smi CLI amdsmi_helpers.py not found at {_HELPERS_SRC}")

        stub_modules(cls, _build_fake_amdsmi_modules())
        path_added = _CLI_DIR not in sys.path
        if path_added:
            sys.path.insert(0, _CLI_DIR)
            add_class_cleanup(cls, sys.path.remove, _CLI_DIR)

        import amdsmi_helpers

        cls.helpers_cls = amdsmi_helpers.AMDSMIHelpers

    def test_valid_name_accepted(self):
        self.assertTrue(self.helpers_cls.is_valid_ampp_field_name("PPT0_Limit"))

    def test_empty_string_rejected(self):
        self.assertFalse(self.helpers_cls.is_valid_ampp_field_name(""))

    def test_non_str_rejected(self):
        for bad in (0, 1, True, False, None, 3.5, ["PPT0_Limit"], b"PPT0_Limit"):
            with self.subTest(bad=bad):
                self.assertFalse(self.helpers_cls.is_valid_ampp_field_name(bad))

    def test_length_boundary(self):
        # The library copies the name into an AMDSMI_MAX_STRING_LENGTH buffer
        # that must stay NUL-terminated, so the last usable length is
        # AMDSMI_MAX_STRING_LENGTH - 1 bytes.
        self.assertTrue(self.helpers_cls.is_valid_ampp_field_name("K" * (_MAX_STRING_LENGTH - 1)))
        self.assertFalse(self.helpers_cls.is_valid_ampp_field_name("K" * _MAX_STRING_LENGTH))

    def test_multibyte_length_measured_in_bytes_not_characters(self):
        # 2-byte UTF-8 characters: 127 chars fit, 128 do not.
        self.assertTrue(self.helpers_cls.is_valid_ampp_field_name("é" * 127))
        self.assertFalse(self.helpers_cls.is_valid_ampp_field_name("é" * 128))

    def test_lone_surrogate_rejected(self):
        # Not UTF-8-encodable; the library's name.encode("utf-8") would raise.
        self.assertFalse(self.helpers_cls.is_valid_ampp_field_name("\ud800"))
        self.assertFalse(self.helpers_cls.is_valid_ampp_field_name("PPT0_\udced\udca0\udc80"))

    def test_valid_values_parsed(self):
        self.assertEqual(self.helpers_cls.parse_ampp_field_value(300), 300)
        self.assertEqual(self.helpers_cls.parse_ampp_field_value("300"), 300)
        self.assertEqual(self.helpers_cls.parse_ampp_field_value(-1), -1)

    def test_int64_boundaries_accepted(self):
        self.assertEqual(self.helpers_cls.parse_ampp_field_value(2**63 - 1), 2**63 - 1)
        self.assertEqual(self.helpers_cls.parse_ampp_field_value(-(2**63)), -(2**63))

    def test_int64_overflow_rejected(self):
        self.assertIsNone(self.helpers_cls.parse_ampp_field_value(2**63))
        self.assertIsNone(self.helpers_cls.parse_ampp_field_value(-(2**63) - 1))

    def test_non_numeric_rejected(self):
        for bad in ("not_a_number", "", None, [1], {}, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                self.assertIsNone(self.helpers_cls.parse_ampp_field_value(bad))

    def test_bool_parsed_as_its_int_value(self):
        # bool subclasses int, so int(True)/int(False) succeed. JSON's true/false
        # therefore round-trip to 1/0 rather than being rejected -- pinned here
        # because it is behavior the driver sees, not an accident of the caller.
        self.assertEqual(self.helpers_cls.parse_ampp_field_value(True), 1)
        self.assertEqual(self.helpers_cls.parse_ampp_field_value(False), 0)

    def test_float_truncates_toward_zero(self):
        self.assertEqual(self.helpers_cls.parse_ampp_field_value(300.9), 300)
