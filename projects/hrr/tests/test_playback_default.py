#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the playback classification default."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import gen_hrr_api_args as gen  # noqa: E402


def entry(name: str, raw_type: str = "int", param: str = "value") -> gen.ApiEntry:
    return gen.ApiEntry(
        name=name,
        ret_type="hipError_t",
        params=[gen.Param(raw_type, param)],
        table="runtime",
    )


class PlaybackDefaultTest(unittest.TestCase):
    def test_translatable_runtime_api_calls_hip(self):
        text = gen.generate_playback_shim(entry("hipBrandNewApi", "int", "device"))
        self.assertIn("hipBrandNewApi((int)a->device)", text)
        self.assertNotIn("NOOP playback handler", text)

    def test_untranslated_handle_is_a_noop(self):
        text = gen.generate_playback_shim(
            entry("hipBrandNewApi", "hipKernel_t", "kernel")
        )
        self.assertIn(
            "NOOP playback handler called for hipBrandNewApi", text
        )
        self.assertNotIn("hipBrandNewApi((", text)

    def test_grandfathered_untranslated_handle_still_calls_hip(self):
        self.assertIn("hipLibraryGetGlobal", gen.DIRECT_PLAYBACK_APIS)
        text = gen.generate_playback_shim(
            entry("hipLibraryGetGlobal", "hipLibrary_t", "library")
        )
        self.assertIn("hipLibraryGetGlobal((hipLibrary_t)a->library)", text)
        self.assertNotIn("NOOP playback handler", text)

    def test_manual_classification_emits_only_an_extern(self):
        name = "hipMalloc"
        self.assertIn(name, gen.MANUAL_PLAYBACK_APIS)
        text = gen.generate_playback_shim(entry(name))
        self.assertEqual(
            text,
            "extern hipError_t playback_hipMalloc"
            "(PlaybackContext& ctx, const uint8_t* payload);\n",
        )

    def test_kernel_attribute_for_device_is_a_noop(self):
        name = "hipKernelSetAttributeForDevice"
        self.assertNotIn(name, gen.DIRECT_PLAYBACK_APIS)
        text = gen.generate_playback_shim(
            entry(name, "hipKernel_t", "kernel")
        )
        self.assertIn(
            "NOOP playback handler called for hipKernelSetAttributeForDevice",
            text,
        )
        self.assertNotIn("hipKernelSetAttributeForDevice((", text)


if __name__ == "__main__":
    unittest.main()
