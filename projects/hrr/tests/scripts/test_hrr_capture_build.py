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

    def test_rocr_installs_like_therock(self):
        cmd = [str(c) for c in cb.rocr_configure_command(Path("/s"), Path("/b"), Path("/d"))]
        self.assertEqual(cmd[:2], ["env", "PKG_CONFIG_PATH=/d/lib/rocm_sysdeps/lib/pkgconfig"])
        self.assertEqual(cmd[cmd.index("-S") + 1], "/s/projects/rocr-runtime")
        self.assertIn("-DCMAKE_PREFIX_PATH=/d;/d/lib/rocm_sysdeps", cmd)
        self.assertIn("-DCMAKE_INSTALL_PREFIX=/d", cmd)
        self.assertIn("-DCMAKE_INSTALL_LIBDIR=lib", cmd)
        self.assertIn("-DCMAKE_INSTALL_RPATH=$ORIGIN;$ORIGIN/rocm_sysdeps/lib", cmd)
        self.assertIn("-DClang_DIR=/d/lib/llvm/lib/cmake/clang", cmd)

    def test_hrr_builds_integration_suite_for_archs(self):
        cmd = [str(c) for c in cb.hrr_configure_command(Path("/s"), Path("/b"), Path("/d"), "gfx942;gfx950")]
        self.assertIn("-DGPU_TARGETS=gfx942;gfx950", cmd)
        self.assertIn("-DHRR_BUILD_INTEGRATION_TESTS=ON", cmd)
        self.assertIn("-DCMAKE_HIP_COMPILER=/d/lib/llvm/bin/clang++", cmd)

    def test_build_targets_cover_uploaded_executables(self):
        names = {Path(rel).name for rel in cb.EXECUTABLES}
        self.assertTrue(names <= set(cb.HRR_TARGETS))

    def test_build_and_runtime_artifacts_are_generic_names(self):
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

    def make_lib(self, directory: Path, stem: str, soname: str, real: str, content: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / real).write_text(content)
        os.symlink(real, directory / soname)
        os.symlink(soname, directory / stem)

    def make_runtime(self, directory: Path, version: str, content: str) -> None:
        self.make_lib(directory, "libamdhip64.so", "libamdhip64.so.7",
                      f"libamdhip64.so.7.{version}", content)
        self.make_lib(directory, "libhsa-runtime64.so", "libhsa-runtime64.so.1",
                      f"libhsa-runtime64.so.1.{version}", content)

    def test_overlay_replaces_base_libraries(self):
        built = self.tmp / "built"
        dist = self.tmp / "dist" / "lib"
        self.make_runtime(built, "2.1", "pr")
        self.make_runtime(dist, "2.0", "base")
        (dist / "libhiprtc.so.7").write_text("keep")

        cb.remove_runtime_libs(dist)
        copied = cb.copy_runtime_libs(built, dist)

        self.assertEqual(len(copied), 6)
        self.assertFalse((dist / "libamdhip64.so.7.2.0").exists())
        self.assertFalse((dist / "libhsa-runtime64.so.1.2.0").exists())
        self.assertEqual((dist / "libamdhip64.so").read_text(), "pr")
        self.assertEqual((dist / "libhsa-runtime64.so").read_text(), "pr")
        self.assertTrue((dist / "libhsa-runtime64.so.1").is_symlink())
        self.assertEqual((dist / "libhiprtc.so.7").read_text(), "keep")

    def test_copy_without_rocr_fails(self):
        built = self.tmp / "built"
        self.make_lib(built, "libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.7.2.1", "pr")
        with self.assertRaises(SystemExit):
            cb.copy_runtime_libs(built, self.tmp / "out")

    def test_remove_rocr_install(self):
        dist = self.tmp / "dist"
        (dist / "include" / "hsa").mkdir(parents=True)
        (dist / "include" / "hsa" / "hsa.h").write_text("")
        (dist / "include" / "hip").mkdir(parents=True)
        self.make_runtime(dist / "lib", "2.0", "base")
        cb.remove_paths(dist, cb.ROCR_INSTALLED)
        self.assertFalse((dist / "include" / "hsa").exists())
        self.assertTrue((dist / "include" / "hip").exists())
        self.assertFalse(list((dist / "lib").glob("libhsa-runtime64.so*")))
        self.assertTrue((dist / "lib" / "libamdhip64.so.7").exists())


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
    def test_rocr_is_installed_before_clr_is_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "dist"
            (dist / "include" / "hsa").mkdir(parents=True)
            (dist / "lib").mkdir()
            calls = []

            def runner(cmd):
                cmd = [str(c) for c in cmd]
                calls.append(cmd)
                # The base's ROCr headers are gone before ROCr installs.
                if cmd[:2] == ["cmake", "--install"] and cmd[2].endswith("rocr"):
                    self.assertFalse((dist / "include" / "hsa").exists())
                if cmd[:2] == ["cmake", "--install"] and cmd[2].endswith("clr"):
                    for name in ("libamdhip64.so.7", "libhsa-runtime64.so.1"):
                        (dist / "lib" / name).write_text("")

            args = argparse.Namespace(
                src=root / "src", build=root / "build", dist=dist,
                runtime_lib=root / "build" / "runtime-lib", archs="gfx942",
            )
            cb.cmd_build(args, runner=runner)
            steps = [" ".join(c[c.index("cmake"):c.index("cmake") + 3]) for c in calls]
            rocr_install = steps.index(f"cmake --install {root}/build/rocr")
            clr_configure = next(i for i, c in enumerate(calls) if "projects/clr" in " ".join(c))
            self.assertLess(rocr_install, clr_configure)
            self.assertEqual(
                sorted(p.name for p in args.runtime_lib.iterdir()),
                ["libamdhip64.so.7", "libhsa-runtime64.so.1"],
            )


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
