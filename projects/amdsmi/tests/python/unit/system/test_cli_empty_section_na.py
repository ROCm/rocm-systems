#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for empty-section rendering in human-readable ``amd-smi`` output.

``custom_dump`` renders an empty nested dict as ``KEY: N/A`` rather than a bare
``KEY:`` header. The rule holds for every subcommand that renders through
``custom_dump`` (AI-NIC ``RDMA_DEVICES`` is only what prompted it), so the
populated and empty-list paths are pinned alongside it.

``_convert_json_to_human_readable`` stashes leftover keys under an internal
``AMDSMI_SPACING_REMOVAL`` marker and strips it afterwards by literal match, so
the marker must never itself become an empty dict.

``amdsmi_logger.py`` is loaded from the source tree so the test exercises the code
under development rather than a possibly-stale installed copy.
"""

import importlib.util
import os
import unittest

from common.common import amdsmi_path, fake_module, find_cli_dir, stub_modules

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# This directory first so a source checkout wins: the point is to exercise the
# logger under development, not a possibly-stale installed copy.
_CLI_DIR = find_cli_dir(_THIS_DIR, amdsmi_path)
LOGGER_PATH = os.path.join(_CLI_DIR, "amdsmi_logger.py") if _CLI_DIR else None


def _load_logger_module():
    spec = importlib.util.spec_from_file_location("amdsmi_logger_under_test", LOGGER_PATH)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"could not load amdsmi_logger from {LOGGER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCliEmptySectionNA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not LOGGER_PATH or not os.path.isfile(LOGGER_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI amdsmi_logger.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        # Registered before the stub is installed, so the restore still runs if
        # loading the logger raises: unittest skips tearDownClass on that path.
        helpers = fake_module("amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {}))
        stub_modules(cls, {"amdsmi_helpers": helpers})
        module = _load_logger_module()
        cls.logger = module.AMDSMILogger.__new__(module.AMDSMILogger)

    def _render(self, payload):
        return self.logger._convert_json_to_human_readable(payload)

    # Whole-output comparisons, not assertIn: the N/A line has to keep the
    # indentation of the section header it replaces, and assertIn passes just as
    # happily when the line lands in column 0.
    def test_empty_nested_dict_renders_na(self):
        # The AI-NIC case: a NIC whose RDMA driver is absent reports no devices.
        output = self._render({"ai_nic": 0, "rdma_devices": {}})
        self.assertEqual(output, "AI_NIC: 0\n    RDMA_DEVICES: N/A\n")

    def test_populated_nested_dict_still_nests(self):
        output = self._render({"ai_nic": 0, "rdma_devices": {"rdma_device_0": {"name": "ionic_0"}}})
        self.assertEqual(
            output,
            "AI_NIC: 0\n    RDMA_DEVICES:\n        RDMA_DEVICE_0:\n            NAME: ionic_0\n",
        )

    def test_empty_dict_nested_inside_a_populated_section(self):
        output = self._render({"ai_nic": 0, "asic": {"vendor_name": "AMD", "ports": {}}})
        self.assertEqual(
            output, "AI_NIC: 0\n    ASIC:\n        VENDOR_NAME: AMD\n        PORTS: N/A\n"
        )

    def test_empty_list_still_renders_na(self):
        output = self._render({"gpu": 0, "bad_pages": []})
        self.assertEqual(output, "GPU: 0\n    BAD_PAGES: N/A\n")

    def test_internal_spacing_marker_never_reaches_the_output(self):
        # The marker holds every non-device key, so it is empty when there are none.
        for payload in ({}, {"gpu": 0}, {"gpu": 0, "vram": {"type": "HBM"}}):
            with self.subTest(payload=payload):
                self.assertNotIn("AMDSMI_SPACING_REMOVAL", self._render(payload))

    def test_payload_of_only_a_device_key_renders_just_that_key(self):
        self.assertEqual(self._render({"gpu": 0}), "GPU: 0\n")

    def test_empty_payload_renders_nothing(self):
        self.assertEqual(self._render({}), "")
