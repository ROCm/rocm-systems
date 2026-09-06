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
FIXTURE = SCRIPT_DIR.parent / "evals" / "fixtures" / "rocm_smi_vram.txt"


def _shell_function(name: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}\(\)\s*\{{.*?^\}}", text, re.MULTILINE | re.DOTALL
    )
    if match is None:
        raise AssertionError(f"{name} not found in triage_archive.sh")
    return match.group(0)


def _pick_gpu_function() -> str:
    return _shell_function("pick_gpu")


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


def _bash_at_least(major: int, minor: int) -> bool:
    """The scripts target Linux bash. Expanding an empty array under `set -u`
    is an error before bash 4.4, which the macOS system bash still is."""
    bash = shutil.which("bash")
    if not bash:
        return False
    out = subprocess.run(
        [bash, "-c", "echo ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        got_major, got_minor = (int(part) for part in out.split(".")[:2])
    except ValueError:
        return False
    return (got_major, got_minor) >= (major, minor)


@unittest.skipUnless(shutil.which("bash"), "bash is required")
class RecordReplayStopTests(unittest.TestCase):
    """A stop the replay does not explain itself has to reach the log.

    Both cases end the log mid-stream. Without the marker the analyzer reads
    that truncation as insufficient signal and then looks for a kernel to
    blame, which reports a crash of the replay as a fault in the workload.
    """

    def _record(self, rc: int) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "replay.log"
            log.write_text("[HRR] Replaying 5 events\n", encoding="utf-8")
            script = (
                f'{_shell_function("record_replay_stop")}\n'
                f'record_replay_stop {rc} "{log}" >/dev/null\n'
            )
            subprocess.run(["bash", "-c", script], check=True)
            return log.read_text(encoding="utf-8")

    def test_a_timeout_is_recorded_as_a_timeout(self) -> None:
        self.assertIn("replay timed out after", self._record(124))

    def test_a_signal_death_is_recorded_with_its_signal(self) -> None:
        self.assertIn("replay killed by signal 11", self._record(139))

    def test_an_ordinary_failure_adds_nothing(self) -> None:
        """A replay that exits with an error has already said why in the log."""
        self.assertEqual(self._record(1), "[HRR] Replaying 5 events\n")


@unittest.skipUnless(shutil.which("bash"), "bash is required")
class LibraryPathTests(unittest.TestCase):
    def test_packaged_playback_puts_its_own_lib_on_the_path(self) -> None:
        """A playback shipped as bin/ and lib/ siblings, not a CLR build tree.

        Without its own lib dir the binary loads the system libamdhip64 and
        fails on the symbols it was built against, which takes out the
        metadata-only path as well as replay.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "lib").mkdir()
            play = root / "bin" / "hrr-playback"
            play.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            play.chmod(0o755)

            env = dict(os.environ)
            env["ROCM_PATH"] = str(root / "absent")
            env.pop("LD_LIBRARY_PATH", None)

            script = (
                f'SCRIPT_DIR={root}\nROCM_PATH="${{ROCM_PATH}}"\n'
                f'{_shell_function("setup_library_path")}\n'
                f'setup_library_path "{play}"\necho "$LD_LIBRARY_PATH"\n'
            )
            proc = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            self.assertIn(str(root / "lib"), proc.stdout.strip())


@unittest.skipUnless(_bash_at_least(4, 4), "needs bash >= 4.4 (Linux target)")
class EnsurePlaybackTests(unittest.TestCase):
    def test_existing_playback_is_returned_outside_a_clr_tree(self) -> None:
        """A provided binary must be usable where no CLR source tree exists.

        The script runs under `set -e`, so a helper whose last statement is a
        false conditional aborts it before the path is printed, and the caller
        reports a build failure even though the binary was there all along.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy(SCRIPT_DIR / "ensure_playback.sh", scripts)

            play = root / "bin" / "hrr-playback"
            play.parent.mkdir()
            play.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            play.chmod(0o755)

            env = dict(os.environ)
            env["HRR_PLAYBACK"] = str(play)
            env["ROCM_PATH"] = str(root / "absent")
            for stale in ("CLR_BUILD", "CLR_ROOT", "HRR_ROOT", "ROCR_LIB"):
                env.pop(stale, None)

            proc = subprocess.run(
                ["bash", str(scripts / "ensure_playback.sh"), "--build"],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(
                proc.returncode, 0, f"ensure_playback failed: {proc.stderr}"
            )
            self.assertEqual(proc.stdout.strip(), str(play))


if __name__ == "__main__":
    unittest.main()
