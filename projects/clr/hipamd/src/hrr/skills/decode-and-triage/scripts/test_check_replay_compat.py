#!/usr/bin/env python3
"""Unit tests for check_replay_compat.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import check_replay_compat as crc  # noqa: E402

SAMPLE_METADATA = {
    "schema_version": 1,
    "runtime": {
        "hip_runtime_version": "7.13.0",
        "comgr_version": "2.8",
    },
    "device_count": 2,
    "captured_device_count": 2,
    "devices": [
        {
            "ordinal": 0,
            "properties": {
                "name": "Instinct MI300X",
                "gcn_arch_name": "gfx942:sramecc+:xnack-",
            },
        },
        {
            "ordinal": 1,
            "properties": {
                "name": "Instinct MI300X",
                "gcn_arch_name": "gfx942:sramecc+:xnack-",
            },
        },
    ],
}


class CheckReplayCompatTests(unittest.TestCase):
    def test_load_capture_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arch = Path(tmp)
            (arch / "manifest.json").write_text(
                json.dumps({"pid": 1, "metadata": SAMPLE_METADATA}),
                encoding="utf-8",
            )
            meta = crc.load_capture_metadata(arch)
            self.assertIsNotNone(meta)
            assert meta is not None
            self.assertEqual(meta.hip_runtime_version, "7.13.0")
            self.assertEqual(meta.device_count, 2)
            self.assertEqual(len(meta.devices), 2)

    def test_blocks_when_capture_needs_more_gpus(self) -> None:
        capture = crc.CaptureMetadata(
            device_count=2, devices=SAMPLE_METADATA["devices"]
        )
        replay = crc.ReplayEnvironment(visible_gpus=1, gpu_archs=["gfx942"])
        report = crc.evaluate_compat(capture, replay, gpu=0)
        self.assertFalse(report.ok)
        self.assertTrue(any("only exposes 1" in b for b in report.blocks))

    def test_blocks_when_requested_gpu_missing(self) -> None:
        capture = crc.CaptureMetadata(
            device_count=1, devices=[SAMPLE_METADATA["devices"][0]]
        )
        replay = crc.ReplayEnvironment(visible_gpus=1, gpu_archs=["gfx942"])
        report = crc.evaluate_compat(capture, replay, gpu=1)
        self.assertFalse(report.ok)
        self.assertTrue(any("requested replay GPU 1" in b for b in report.blocks))

    def test_warns_on_hip_version_mismatch(self) -> None:
        capture = crc.CaptureMetadata(
            device_count=1,
            hip_runtime_version="7.13.0",
            devices=[SAMPLE_METADATA["devices"][0]],
        )
        replay = crc.ReplayEnvironment(
            visible_gpus=1,
            gpu_archs=["gfx942"],
            hip_runtime_version="7.15.0",
        )
        report = crc.evaluate_compat(capture, replay, gpu=0)
        self.assertTrue(report.ok)
        self.assertTrue(
            any("HIP runtime version mismatch" in w for w in report.warnings)
        )

    def test_legacy_manifest_skips_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arch = Path(tmp)
            (arch / "manifest.json").write_text(
                json.dumps({"pid": 1, "complete": False}),
                encoding="utf-8",
            )
            self.assertIsNone(crc.load_capture_metadata(arch))

    @mock.patch("check_replay_compat._run")
    def test_parse_rocm_smi_gpu_count(self, run_mock: mock.Mock) -> None:
        run_mock.return_value = mock.Mock(
            stdout="GPU[0]\tGPU[1]\nCard series: Instinct MI300X\nCard series: Instinct MI300X\n",
            stderr="",
        )
        env = crc.probe_host_replay_env()
        self.assertEqual(env.visible_gpus, 2)


if __name__ == "__main__":
    unittest.main()
