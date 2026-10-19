#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for hrr_dev.py. Docker is not started."""

import sys
import os
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrr_dev  # noqa: E402


class ParseTest(unittest.TestCase):
    def test_emulator_and_gpu(self):
        self.assertEqual(hrr_dev.parse_args(["emulator"]).backend, "emulator")
        self.assertEqual(hrr_dev.parse_args(["gpu"]).backend, "gpu")


class DockerPlanTest(unittest.TestCase):
    def test_missing_image_is_a_docker_build(self):
        repo = Path("/repo")
        commands = hrr_dev.plan_docker(
            repo, "emulator", hrr_dev.DockerState(image=False, container=None)
        )
        self.assertEqual(commands[0][:3], ["docker", "build", "-t"])
        self.assertIn("hrr-dev", commands[0])
        self.assertIn(str(repo / "projects/hrr/tests/docker/Dockerfile"), commands[0])

    def test_stopped_container_is_started(self):
        commands = hrr_dev.plan_docker(
            Path("/repo"), "gpu", hrr_dev.DockerState(image=True, container="exited")
        )
        self.assertEqual(commands, [["docker", "start", "hrr-dev"]])

    def test_only_gpu_adds_devices(self):
        repo = Path("/repo")
        state = hrr_dev.DockerState(image=True, container=None)
        emulator = hrr_dev.plan_docker(repo, "emulator", state)[0]
        gpu = hrr_dev.plan_docker(repo, "gpu", state)[0]
        self.assertNotIn("/dev/kfd", emulator)
        self.assertNotIn("/dev/dri", emulator)
        self.assertEqual(
            gpu[gpu.index("--device"):gpu.index("--device") + 4],
            ["--device", "/dev/kfd", "--device", "/dev/dri"],
        )


class ScriptTest(unittest.TestCase):
    def test_emulator_uses_both_default_configs(self):
        script = hrr_dev.container_script(Path("/repo"), "emulator")
        self.assertIn("gfx950_mi355x_kmd_2gpu.json", script)
        self.assertIn("gfx1250_mi455x_kmd_4gpu.json", script)
        self.assertIn("--arch gfx950", script)
        self.assertIn("--arch gfx1250", script)
        self.assertIn("rocjitsu --daemon", script)
        self.assertIn("HIP_HRR_CAPTURE_ENABLED", script)
        self.assertIn("HRR_BUILD_INTEGRATION_TESTS=OFF", script)
        self.assertNotIn("/dev/kfd", script)

    def test_gpu_uses_arch_auto_and_no_launcher(self):
        script = hrr_dev.container_script(Path("/repo"), "gpu")
        self.assertIn("--arch auto", script)
        self.assertNotIn("rocjitsu", script)
        self.assertNotIn("gfx950_mi355x_kmd_2gpu.json", script)
        self.assertIn("detect_archs", script)
        self.assertNotIn("gfx950;gfx1250", script)

    def test_capture_library_on_path_in_both_modes(self):
        for backend in ("emulator", "gpu"):
            script = hrr_dev.container_script(Path("/repo"), backend)
            self.assertLess(
                script.index("/repo/build/hrr-dev/clr/hipamd/lib"),
                script.index("run_catch2.py --xml build/hrr-dev/integration"),
            )

    def test_handler_run_has_no_missing_config(self):
        script = hrr_dev.container_script(Path("/repo"), "emulator")
        self.assertNotIn("handler.yaml", script)

    def test_token_not_on_command_line(self):
        with mock.patch.dict(os.environ, {"GH_TOKEN": "secret"}):
            command = hrr_dev.exec_command(Path("/repo"), "gpu")
        self.assertNotIn("GH_TOKEN=secret", command)
        self.assertIn("GH_TOKEN", command)
        self.assertIn("HOME=/tmp", command)

    def test_user_install_scripts_on_path_before_sdk_use(self):
        script = hrr_dev.container_script(Path("/repo"), "emulator")
        self.assertLess(
            script.index('export PATH="$HOME/.local/bin:$PATH"'),
            script.index("rocm-sdk init"),
        )


if __name__ == "__main__":
    unittest.main()
