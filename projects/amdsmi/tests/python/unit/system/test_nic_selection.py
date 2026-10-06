#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""``--nic`` selection by ID, UUID and BDF.

A selection matches ID or UUID across every choice before it is read as a BDF. Parsing it as a BDF
for each non-matching choice raised ``Invalid BDF format`` for any ID but the first.
"""

import os
import sys
import unittest

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

# amdsmi_helpers does ``from amdsmi_init import *`` at load; the real one needs a driver.
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

_HELPERS_IMPORT_ERROR = None
try:
    from amdsmi_helpers import AMDSMIHelpers  # noqa: E402
except ImportError as _import_error:
    AMDSMIHelpers = None
    _HELPERS_IMPORT_ERROR = _import_error


def setUpModule():
    if _HELPERS_IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"amd-smi CLI helpers unusable: {_HELPERS_IMPORT_ERROR}")


def tearDownModule():
    if _restore_stubs is not None:
        _restore_stubs()


def _choices():
    """Four NICs shaped like ``AMDSMIHelpers.nic_choices_from_nic_info`` builds them."""
    bdfs = ["0001:01:00.0", "0002:01:00.0", "0003:01:00.0", "0004:01:00.0"]
    return {
        str(nic_id): {"bdf": bdf, "UUID": f"uuid-{nic_id}", "Device Handle": f"handle-{nic_id}"}
        for nic_id, bdf in enumerate(bdfs)
    }


class TestNicSelection(unittest.TestCase):
    def setUp(self):
        self.helpers = AMDSMIHelpers()
        self.choices = _choices()

    def _select(self, selections):
        return self.helpers.get_device_handles_from_nic_selections(
            nic_selections=selections, nic_choices=self.choices
        )

    def test_first_id_selects_that_nic(self):
        self.assertEqual(self._select(["0"]), (True, ["handle-0"]))

    def test_non_first_id_selects_that_nic(self):
        self.assertEqual(self._select(["2"]), (True, ["handle-2"]))

    def test_non_first_uuid_selects_that_nic(self):
        self.assertEqual(self._select(["uuid-3"]), (True, ["handle-3"]))

    def test_uuid_match_ignores_case(self):
        self.assertEqual(self._select(["UUID-1"]), (True, ["handle-1"]))

    def test_non_first_bdf_selects_that_nic(self):
        self.assertEqual(self._select(["0003:01:00.0"]), (True, ["handle-2"]))

    def test_mixed_selections_keep_order(self):
        self.assertEqual(
            self._select(["3", "0001:01:00.0", "uuid-1"]),
            (True, ["handle-3", "handle-0", "handle-1"]),
        )

    def test_unknown_id_is_reported_as_the_failed_selection(self):
        self.assertEqual(self._select(["9"]), (False, "9"))

    def test_text_that_is_not_an_id_uuid_or_bdf_is_reported(self):
        self.assertEqual(self._select(["not-a-nic"]), (False, "not-a-nic"))

    def test_valid_bdf_of_no_nic_is_reported(self):
        self.assertEqual(self._select(["00ff:01:00.0"]), (False, "00ff:01:00.0"))

    def test_first_failure_stops_the_selection(self):
        self.assertEqual(self._select(["1", "9", "2"]), (False, "9"))


if __name__ == "__main__":
    unittest.main()
