#!/usr/bin/env python3
"""Unit tests for triage_archive.sh (stdlib unittest, no GPU required).

Device selection is exercised by lifting the `pick_gpu` function out of the
script and running it against recorded `rocm-smi` output, with a stub on PATH
standing in for the real tool. Picking the wrong device is not a cosmetic
mistake on a shared host: it sends the replay onto somebody else's card.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "triage_archive.sh"
FIXTURE = SCRIPT_DIR / "fixtures" / "rocm_smi_vram.txt"


def _pick_gpu_function() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^pick_gpu\(\)\s*\{.*?^\}", text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError("pick_gpu not found in triage_archive.sh")
    return match.group(0)


@unittest.skipUnless(shutil.which("bash"), "bash is required")
class PickGpuTests(unittest.TestCase):
    def _run_pick_gpu(self, smi_output: str, env_gpu: str | None = None) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            stub_dir = Path(tmp)
            data = stub_dir / "vram.txt"
            data.write_text(smi_output, encoding="utf-8")
            stub = stub_dir / "rocm-smi"
            stub.write_text(f'#!/bin/sh\ncat "{data}"\n', encoding="utf-8")
            stub.chmod(0o755)

            env = dict(os.environ)
            env["PATH"] = f"{stub_dir}{os.pathsep}{env.get('PATH', '')}"
            env.pop("GPU", None)
            if env_gpu is not None:
                env["GPU"] = env_gpu

            proc = subprocess.run(
                ["bash", "-c", f"{_pick_gpu_function()}\npick_gpu"],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            return proc.stdout.strip()

    def test_selects_the_device_with_the_most_free_memory(self) -> None:
        """The freest device is neither the first nor the largest here.

        Both devices report the same total, and the first is nearly full, so a
        parse that mistakes total for free selects the wrong one.
        """
        self.assertEqual(
            self._run_pick_gpu(FIXTURE.read_text(encoding="utf-8")), "1"
        )

    def test_used_line_is_not_read_as_the_total_line(self) -> None:
        """`VRAM Total Used Memory` also contains the word `Total`."""
        smi = (
            "GPU[0]\t\t: VRAM Total Memory (B): 100\n"
            "GPU[0]\t\t: VRAM Total Used Memory (B): 90\n"
            "GPU[1]\t\t: VRAM Total Memory (B): 100\n"
            "GPU[1]\t\t: VRAM Total Used Memory (B): 10\n"
        )
        self.assertEqual(self._run_pick_gpu(smi), "1")

    def test_explicit_gpu_env_wins(self) -> None:
        self.assertEqual(
            self._run_pick_gpu(FIXTURE.read_text(encoding="utf-8"), env_gpu="0"), "0"
        )


class ReplayMaskTests(unittest.TestCase):
    def test_native_replay_masks_with_hip_visible_devices(self) -> None:
        """hrr-playback is a HIP program, so the HIP mask is the one that applies.

        Setting ROCR_VISIBLE_DEVICES re-indexes devices underneath a HIP mask,
        which can place the replay on a device other than the one picked. The
        Windows and container paths already use the HIP mask.
        """
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('HIP_VISIBLE_DEVICES="$gpu"', text)
        assignments = [
            line.strip()
            for line in text.splitlines()
            if "ROCR_VISIBLE_DEVICES=" in line
        ]
        self.assertEqual(assignments, [], "ROCR_VISIBLE_DEVICES must not be set")


if __name__ == "__main__":
    unittest.main()
