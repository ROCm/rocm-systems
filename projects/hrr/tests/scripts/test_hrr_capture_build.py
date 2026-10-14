#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for hrr_capture_build.py."""

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrr_capture_build as cb  # noqa: E402

FINDER_FOUND = {
    "status": "found",
    "results": [
        {
            "github_repository_name": "ROCm/rocm-systems",
            "workflow_run_id": "123",
            "git_commit_sha": "abc",
            "workflow_run_html_url": "https://example/runs/123",
            "artifact_group": "gfx94X-dcgpu",
        }
    ],
}


class PickBaseRunTest(unittest.TestCase):
    def test_found(self):
        base = cb.pick_base_run(FINDER_FOUND)
        self.assertEqual(base["run_id"], "123")
        self.assertEqual(base["commit"], "abc")
        self.assertEqual(base["repo"], "ROCm/rocm-systems")

    def test_not_found(self):
        with self.assertRaises(SystemExit):
            cb.pick_base_run({"status": "not_found", "results": []})


class CommandTest(unittest.TestCase):
    def test_fetch_command_uses_recorded_run(self):
        base = cb.pick_base_run(FINDER_FOUND)
        cmd = [str(c) for c in cb.fetch_command(Path("/t"), base, Path("/d"), ("core-hip_lib",))]
        self.assertIn("--flatten", cmd)
        self.assertEqual(cmd[cmd.index("--run-id") + 1], "123")
        self.assertEqual(cmd[cmd.index("--run-github-repo") + 1], "ROCm/rocm-systems")
        self.assertEqual(cmd[-1], "core-hip_lib")

    def test_clr_compiles_capture_in(self):
        cmd = [str(c) for c in cb.clr_configure_command(Path("/s"), Path("/b"), Path("/d"))]
        self.assertIn("-DCMAKE_CXX_FLAGS=-DHIP_HRR_CAPTURE_ENABLED", cmd)
        self.assertIn("-DCMAKE_INSTALL_PREFIX=/d", cmd)
        self.assertIn("-DHIP_COMMON_DIR=/s/projects/hip", cmd)

    def test_hrr_builds_integration_suite_for_archs(self):
        cmd = [str(c) for c in cb.hrr_configure_command(Path("/s"), Path("/b"), Path("/d"), "gfx942;gfx950")]
        self.assertIn("-DGPU_TARGETS=gfx942;gfx950", cmd)
        self.assertIn("-DHRR_BUILD_INTEGRATION_TESTS=ON", cmd)
        self.assertIn("-DCMAKE_HIP_COMPILER=/d/lib/llvm/bin/clang++", cmd)

    def test_build_targets_cover_uploaded_executables(self):
        names = {Path(rel).name for rel in cb.EXECUTABLES}
        self.assertTrue(names <= set(cb.HRR_TARGETS))

    def test_runtime_artifacts_are_a_subset_of_build_artifacts(self):
        self.assertTrue(set(cb.RUNTIME_ARTIFACTS) <= set(cb.BUILD_ARTIFACTS))


class EmbeddedTargetsTest(unittest.TestCase):
    OUTPUT = (
        "OFFLOADING IMAGE [0]:\nkind  elf\narch  gfx942\ntriple amdgcn-amd-amdhsa\n"
        "OFFLOADING IMAGE [1]:\nkind  elf\narch  gfx1100\n"
    )

    def test_parses_arches(self):
        self.assertEqual(cb.embedded_targets(self.OUTPUT), {"gfx942", "gfx1100"})

    def test_missing(self):
        self.assertEqual(cb.missing_targets("gfx942;gfx950;gfx1100", self.OUTPUT), ["gfx950"])

    def test_none_missing(self):
        self.assertEqual(cb.missing_targets("gfx942;gfx1100", self.OUTPUT), [])


class LibraryOverlayTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make_lib(self, directory: Path, real: str, content: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / real).write_text(content)
        os.symlink(real, directory / "libamdhip64.so.7")
        os.symlink("libamdhip64.so.7", directory / "libamdhip64.so")

    def test_overlay_replaces_release_library_and_keeps_rocr(self):
        capture = self.tmp / "capture"
        dist = self.tmp / "dist" / "lib"
        self.make_lib(capture, "libamdhip64.so.7.2.1", "capture")
        self.make_lib(dist, "libamdhip64.so.7.2.0", "release")
        (dist / "libhiprtc.so.7").write_text("release")
        (capture / "libhiprtc.so.7").write_text("capture")
        (dist / "libhsa-runtime64.so.1").write_text("base")

        cb.remove_amdhip64(dist)
        copied = cb.copy_amdhip64(capture, dist)

        self.assertEqual(
            sorted(copied),
            ["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.7.2.1", "libhiprtc.so.7"],
        )
        self.assertFalse((dist / "libamdhip64.so.7.2.0").exists())
        self.assertEqual((dist / "libamdhip64.so").read_text(), "capture")
        self.assertTrue((dist / "libamdhip64.so.7").is_symlink())
        self.assertEqual((dist / "libhiprtc.so.7").read_text(), "capture")
        self.assertEqual((dist / "libhsa-runtime64.so.1").read_text(), "base")

    def test_copy_without_library_fails(self):
        (self.tmp / "empty").mkdir()
        with self.assertRaises(SystemExit):
            cb.copy_amdhip64(self.tmp / "empty", self.tmp / "out")


class LinkageTest(unittest.TestCase):
    LIB = Path("/w/dist/lib")

    def ldd(self, hsa: str, extra: str = "") -> str:
        return (
            "\tlinux-vdso.so.1 (0x00007ffd)\n"
            "\tlibamdhip64.so.7 => /w/dist/lib/libamdhip64.so.7 (0x00007f1)\n"
            f"\tlibhsa-runtime64.so.1 => {hsa} (0x00007f2)\n"
            f"{extra}"
            "\tlibc.so.6 => /lib64/libc.so.6 (0x00007f3)\n"
        )

    def test_both_from_dist(self):
        self.assertEqual(cb.linkage_errors(self.ldd("/w/dist/lib/libhsa-runtime64.so.1"), self.LIB), [])

    def test_rocr_from_elsewhere(self):
        errors = cb.linkage_errors(self.ldd("/opt/rocm/lib/libhsa-runtime64.so.1"), self.LIB)
        self.assertEqual(len(errors), 1)
        self.assertIn("libhsa-runtime64.so", errors[0])

    def test_two_rocr_copies(self):
        extra = "\tlibhsa-runtime64.so.2 => /opt/rocm/lib/libhsa-runtime64.so.2 (0x0)\n"
        errors = cb.linkage_errors(self.ldd("/w/dist/lib/libhsa-runtime64.so.1", extra), self.LIB)
        self.assertEqual(len(errors), 1)

    def test_missing_rocr(self):
        text = "\tlibamdhip64.so.7 => /w/dist/lib/libamdhip64.so.7 (0x1)\n"
        self.assertEqual(len(cb.linkage_errors(text, self.LIB)), 1)


class BuildTest(unittest.TestCase):
    def test_builds_clr_and_hrr_over_the_prebuilt_rocr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "dist"
            (dist / "lib").mkdir(parents=True)
            (dist / "lib" / "libhsa-runtime64.so.1").write_text("base")
            calls = []

            def runner(cmd):
                cmd = [str(c) for c in cmd]
                calls.append(cmd)
                if cmd[:2] == ["cmake", "--install"]:
                    (dist / "lib" / "libamdhip64.so.7").write_text("capture")

            args = argparse.Namespace(
                src=root / "src", build=root / "build", dist=dist,
                capture_lib=root / "build" / "capture-lib", archs="gfx942",
            )
            cb.cmd_build(args, runner=runner)
            sources = [c[c.index("-S") + 1] for c in calls if "-S" in c]
            self.assertEqual(sources, [f"{root}/src/projects/clr", f"{root}/src/projects/hrr"])
            self.assertEqual([p.name for p in args.capture_lib.iterdir()], ["libamdhip64.so.7"])
            self.assertEqual((dist / "lib" / "libhsa-runtime64.so.1").read_text(), "base")


class ArchsTest(unittest.TestCase):
    SRC = Path(__file__).resolve().parents[4]

    def test_default_is_the_integration_config(self):
        config = cb.hrr_test_config.load(self.SRC / cb.INTEGRATION_CONFIG)
        args = cb.parse_args(["build", "--src", str(self.SRC)])
        self.assertEqual(args.archs, ";".join(config["targets"]["arch"]))

    def test_override(self):
        args = cb.parse_args(["verify", "--src", str(self.SRC), "--archs", "gfx942;gfx1152"])
        self.assertEqual(args.archs, "gfx942;gfx1152")


class ExecutableBitsTest(unittest.TestCase):
    def test_restores_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            for rel in cb.EXECUTABLES:
                (build / rel).parent.mkdir(parents=True, exist_ok=True)
                (build / rel).write_text("")
                (build / rel).chmod(0o644)
            cb.restore_executable_bits(build)
            for rel in cb.EXECUTABLES:
                self.assertEqual((build / rel).stat().st_mode & 0o777, 0o755)


class FetchBaseTest(unittest.TestCase):
    def test_reuses_recorded_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_json = Path(tmp) / "base.json"
            base_json.write_text(json.dumps(cb.pick_base_run(FINDER_FOUND)))
            calls = []
            args = argparse.Namespace(
                base_json=base_json, therock=Path("/t"), dist=Path("/d")
            )
            cb.cmd_fetch_base(args, runner=calls.append)
            cmd = [str(c) for c in calls[0]]
            self.assertEqual(cmd[cmd.index("--run-id") + 1], "123")
            self.assertIn("core-runtime_dev", cmd)


if __name__ == "__main__":
    unittest.main()
