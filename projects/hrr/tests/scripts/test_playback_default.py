#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the playback classification default."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

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
        # --multi-thread replay can run one handler on two threads at once.
        self.assertIn("static std::atomic<bool> warned{false};", text)
        self.assertIn("warned.exchange(true)", text)

    def test_grandfathered_untranslated_handle_is_refused(self):
        name = "hipLibraryGetGlobal"
        self.assertNotIn(name, gen.DIRECT_PLAYBACK_APIS)
        self.assertIn(name, gen.PLAYBACK_ONLY_UNREPLAYABLE_APIS)
        text = gen.generate_playback_shim(
            entry(name, "hipLibrary_t", "library")
        )
        self.assertIn(f'hrr_note_unreplayable(ctx, "{name}"', text)
        self.assertIn("return hipErrorNotSupported;", text)
        self.assertNotIn(f"{name}((", text)
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

    def test_elapsed_time_stays_hand_written(self):
        # hipExtModuleLaunchKernel marks its start and stop events inside the
        # runtime. The hand-written handler accepts a captured success when
        # those markers were not stored. Regenerating must not replace it.
        name = "hipEventElapsedTime"
        self.assertIn(name, gen.MANUAL_PLAYBACK_APIS)
        text = gen.generate_playback_shim(entry(name))
        self.assertEqual(
            text,
            "extern hipError_t playback_hipEventElapsedTime"
            "(PlaybackContext& ctx, const uint8_t* payload);\n",
        )

    def test_ext_launch_records_both_event_markers(self):
        capture = (
            Path(__file__).resolve().parents[3]
            / "clr/hipamd/src/hrr/hip_capture.cpp"
        )
        text = capture.read_text()
        start = text.index("hipError_t capture_hipExtModuleLaunchKernel(")
        end = text.index("\nhipError_t ", start + 1)
        body = text[start:end]
        self.assertEqual(body.count("record_ext_launch_event("), 2)
        self.assertLess(
            body.index("record_ext_launch_event(startEvent"),
            body.index("record_launch("),
        )
        self.assertLess(
            body.index("record_launch("),
            body.index("record_ext_launch_event(stopEvent"),
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
