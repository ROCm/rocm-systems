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

    def test_prompts_on_hip_version_mismatch(self) -> None:
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
            any("HIP runtime version mismatch" in p for p in report.prompts)
        )
        self.assertFalse(report.warnings)

    def test_prompts_on_comgr_version_mismatch(self) -> None:
        capture = crc.CaptureMetadata(
            device_count=1,
            comgr_version="3.0",
            devices=[SAMPLE_METADATA["devices"][0]],
        )
        replay = crc.ReplayEnvironment(
            visible_gpus=1,
            gpu_archs=["gfx942"],
            comgr_version="2.8",
        )
        report = crc.evaluate_compat(capture, replay, gpu=0)
        self.assertTrue(report.ok)
        self.assertTrue(any("comgr version mismatch" in p for p in report.prompts))

    def test_strict_version_blocks_without_prompt(self) -> None:
        capture = crc.CaptureMetadata(
            device_count=1,
            hip_runtime_version="7.13.0",
            comgr_version="3.0",
            devices=[SAMPLE_METADATA["devices"][0]],
        )
        replay = crc.ReplayEnvironment(
            visible_gpus=1,
            gpu_archs=["gfx942"],
            hip_runtime_version="7.15.0",
            comgr_version="2.8",
        )
        report = crc.evaluate_compat(
            capture, replay, gpu=0, strict_version=True
        )
        self.assertFalse(report.ok)
        self.assertTrue(report.blocks)
        self.assertFalse(report.prompts)

    @mock.patch("check_replay_compat.probe_comgr_version", return_value="3.0")
    @mock.patch("check_replay_compat._run")
    def test_probe_host_reads_comgr(
        self, run_mock: mock.Mock, _comgr_mock: mock.Mock
    ) -> None:
        run_mock.return_value = mock.Mock(
            stdout="GPU[0]\tGPU[1]\nCard series: Instinct MI300X\nCard series: Instinct MI300X\n",
            stderr="",
        )
        env = crc.probe_host_replay_env()
        self.assertEqual(env.visible_gpus, 2)
        self.assertEqual(env.comgr_version, "3.0")

    def test_parse_hip_version_formats(self) -> None:
        self.assertEqual(
            crc._parse_hip_version("HIP version : 7.13.0\n"),
            "7.13.0",
        )
        self.assertEqual(
            crc._parse_hip_version("7.13.99004-3309c6114a\n"),
            "7.13.99004-3309c6114a",
        )

    def test_hip_version_from_mounted_clr_lib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib_dir = Path(tmp)
            versioned = lib_dir / "libamdhip64.so.7.15.26291-2ed46569ce"
            versioned.write_bytes(b"")
            (lib_dir / "libamdhip64.so.7").symlink_to(versioned.name)
            (lib_dir / "libamdhip64.so").symlink_to("libamdhip64.so.7")
            self.assertEqual(
                crc.hip_version_from_clr_lib(lib_dir),
                "7.15.26291",
            )

    def test_resolve_clr_lib_from_playback_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib_dir = root / "hipamd" / "lib"
            lib_dir.mkdir(parents=True)
            versioned = lib_dir / "libamdhip64.so.7.15.26291-abc"
            versioned.write_bytes(b"")
            play = root / "hipamd" / "src" / "hrr" / "playback" / "hrr-playback"
            play.parent.mkdir(parents=True)
            play.write_bytes(b"")
            resolved = crc.resolve_clr_lib_dir(hrr_playback=str(play))
            self.assertEqual(resolved, lib_dir.resolve())

    def test_build_docker_ld_inside_image_native(self) -> None:
        self.assertEqual(
            crc.build_docker_ld_inside(
                clr_lib=None,
                rocr_lib=None,
                extra_ld="/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib",
            ),
            "/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib:/opt/rocm/lib",
        )

    def test_resolve_replay_mounts_skips_without_overlay(self) -> None:
        clr, rocr, extra = crc.resolve_replay_mounts(
            docker_image="rocm/vllm:test",
            clr_build="/build",
            hrr_playback="/build/hipamd/src/hrr/playback/hrr-playback",
            mount_clr=False,
        )
        self.assertIsNone(clr)
        self.assertIsNone(rocr)
        self.assertEqual(
            extra,
            "/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib",
        )

    def test_resolve_replay_mounts_with_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib_dir = root / "hipamd" / "lib"
            lib_dir.mkdir(parents=True)
            (lib_dir / "libamdhip64.so.7").write_bytes(b"")
            clr, rocr, extra = crc.resolve_replay_mounts(
                docker_image="rocm/vllm:test",
                clr_build=str(root),
                mount_clr=True,
            )
            self.assertEqual(clr, lib_dir.resolve())

    def test_docker_mount_clr_enabled(self) -> None:
        with mock.patch.dict("os.environ", {"HRR_DOCKER_MOUNT_CLR": "1"}):
            self.assertTrue(crc.docker_mount_clr_enabled())
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(crc.docker_mount_clr_enabled())

    def test_build_docker_ld_inside_matches_replay_docker(self) -> None:
        self.assertEqual(
            crc.build_docker_ld_inside(
                clr_lib=Path("/build/hipamd/lib"),
                rocr_lib=Path("/rocr/lib"),
                extra_ld="/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib",
            ),
            "/opt/hrr/lib:/opt/hrr/rocr:"
            "/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib:/opt/rocm/lib",
        )

    @mock.patch("check_replay_compat._docker_cmd", side_effect=lambda *args: list(args))
    @mock.patch("check_replay_compat._run")
    def test_probe_docker_mounts_clr_lib(
        self, run_mock: mock.Mock, _docker_mock: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib_dir = Path(tmp)
            versioned = lib_dir / "libamdhip64.so.7.15.26291-2ed46569ce"
            versioned.write_bytes(b"")
            (lib_dir / "libamdhip64.so.7").symlink_to(versioned.name)
            run_mock.side_effect = [
                mock.Mock(
                    stdout="GPU[0]\nCard series: AMD Radeon Graphics\n",
                    stderr="HIP version : 7.13.99004-3309c6114a\n",
                    returncode=0,
                ),
                mock.Mock(stdout="3.0\n", stderr="", returncode=0),
            ]
            env = crc.probe_docker_replay_env(
                "rocm/vllm:test",
                clr_lib=lib_dir,
                extra_ld=crc.default_docker_extra_ld("rocm/vllm:test"),
            )
            self.assertEqual(env.hip_runtime_version, "7.15.26291")
            docker_cmd = run_mock.call_args_list[0].args[0]
            self.assertIn("-v", docker_cmd)
            mount_idx = docker_cmd.index("-v")
            self.assertEqual(docker_cmd[mount_idx + 1], f"{lib_dir}:/opt/hrr/lib:ro")
            probe_shell = docker_cmd[-1]
            self.assertIn("/opt/hrr/lib", probe_shell)

    def test_no_prompt_when_mounted_hip_matches_capture(self) -> None:
        capture = crc.CaptureMetadata(
            device_count=1,
            hip_runtime_version="7.15.26291",
            comgr_version="3.0",
            devices=[SAMPLE_METADATA["devices"][0]],
        )
        replay = crc.ReplayEnvironment(
            visible_gpus=1,
            gpu_archs=["gfx950"],
            hip_runtime_version="7.15.26291",
            comgr_version="3.0",
            source="docker:rocm/vllm:test:overlay:/build/hipamd/lib",
        )
        report = crc.evaluate_compat(capture, replay, gpu=0)
        self.assertTrue(report.ok)
        self.assertFalse(report.prompts)

    def test_legacy_manifest_skips_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arch = Path(tmp)
            (arch / "manifest.json").write_text(
                json.dumps({"pid": 1, "complete": False}),
                encoding="utf-8",
            )
            self.assertIsNone(crc.load_capture_metadata(arch))

    def test_render_report_includes_capture_gcn_arch(self) -> None:
        capture = crc.CaptureMetadata(
            schema_version=1,
            hip_runtime_version="7.15.26291",
            comgr_version="3.0",
            device_count=1,
            devices=[SAMPLE_METADATA["devices"][0]],
        )
        report = crc.CompatReport(
            capture=capture,
            replay=crc.ReplayEnvironment(
                visible_gpus=1,
                gpu_archs=["gfx942"],
                hip_runtime_version="7.15.26291",
                comgr_version="3.0",
            ),
        )
        text = crc.render_report(report)
        self.assertIn("gcn_arch_name: gfx942:sramecc+:xnack-", text)


if __name__ == "__main__":
    unittest.main()
