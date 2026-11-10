#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check topology-probe failures before the launcher starts its application."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TopologyProbeCliTest(unittest.TestCase):
    def test_permission_errors_prevent_application_launch(self):
        for mode in ("probe", "fallback-open"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                marker = Path(directory) / "application-started"
                env = dict(os.environ)
                env.update(
                    LD_PRELOAD=env["RJ_TOPOLOGY_PRELOAD"],
                    RJ_TOPOLOGY_FAILURE=mode,
                    ROCJITSU_RUNTIME_DIR=str(Path(directory) / "runtime"),
                    LC_ALL="C",
                )
                result = subprocess.run(
                    [
                        env["RJ_ROCJITSU"],
                        "--config",
                        env["RJ_DBT_CONFIG"],
                        "--",
                        sys.executable,
                        "-c",
                        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                        str(marker),
                    ],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("failed to load DBT execution topology", result.stderr)
                self.assertIn("Permission denied", result.stderr)
                self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
