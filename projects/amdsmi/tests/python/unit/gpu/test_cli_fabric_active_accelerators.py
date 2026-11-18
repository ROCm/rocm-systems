#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`amd-smi fabric` decodes vpod_active_accelerators as the documented bitmap.

amdsmi_get_gpu_fabric_info() returns 32 words in which bit N % 32 of word N / 32
marks accelerator N active. The CLI lists every active id, eight per row, so a
vPoD of more than 32 accelerators is shown whole.
"""

import os
import unittest

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = find_cli_dir(*cli_search_order(_THIS_DIR))
FABRIC_PATH = os.path.join(_CLI_DIR, "subcommands", "fabric.py") if _CLI_DIR else None

_WORDS = 32
_UNSET = (1 << 32) - 1


def _bitmap(ids):
    words = [0] * _WORDS
    for accel_id in ids:
        words[accel_id // 32] |= 1 << (accel_id % 32)
    return words


class TestFabricActiveAccelerators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        interface = fake_module("amdsmi.amdsmi_interface")
        exception = fake_module("amdsmi.amdsmi_exception")
        stub_modules(
            cls,
            {
                "amdsmi": fake_module(
                    "amdsmi", amdsmi_interface=interface, amdsmi_exception=exception
                ),
                "amdsmi.amdsmi_interface": interface,
                "amdsmi.amdsmi_exception": exception,
            },
        )
        cls.module = load_cli_module("fabric_active_accelerators_under_test", FABRIC_PATH)

    def test_vpod_larger_than_32_accelerators_is_listed_whole(self):
        rows = self.module._fabric_local_active_accelerators_rows(_bitmap(range(40)))

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0], "0, 1, 2, 3, 4, 5, 6, 7")
        self.assertEqual(rows[-1], "32, 33, 34, 35, 36, 37, 38, 39")

    def test_bits_map_to_ids_across_words(self):
        rows = self.module._fabric_local_active_accelerators_rows(_bitmap([0, 31, 32, 255]))

        self.assertEqual(rows, ["0, 31, 32, 255"])

    def test_full_word_is_32_accelerators_not_unset(self):
        rows = self.module._fabric_local_active_accelerators_rows([_UNSET] + [0] * (_WORDS - 1))

        self.assertEqual(", ".join(rows), ", ".join(str(i) for i in range(32)))

    def test_no_active_accelerator_is_not_available(self):
        self.assertIsNone(self.module._fabric_local_active_accelerators_rows([0] * _WORDS))
