# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the rocprof-sys-attach binary (rocattach codepath).

Exercises a long-running HIP target (transpose in infinite mode) and runs
N attach/detach sessions against its PID. The target keeps a single output
directory across re-attaches, so each session adds a run-counter-suffixed
perfetto trace rather than a fresh per-session directory. Asserts that N
attach sessions yield at least N perfetto traces and that the target
survives all sessions.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.rocattach,
    pytest.mark.gpu,
    pytest.mark.attach,
    pytest.mark.rocm,
]


@pytest.mark.parametrize("n_sessions", [1, 2, 3])
class TestRocattach:
    """Attach the rocprof-sys-attach binary to a long-running transpose target."""

    SESSION_DURATION_S = 5
    TARGET_WARMUP_S = 2
    INTER_SESSION_PAUSE_S = 1
    ATTACH_TIMEOUT_S = 60
    TARGET_TERMINATE_TIMEOUT_S = 10

    def test_attach_loop(
        self,
        n_sessions: int,
        rocprof_config,
        test_output_dir: Path,
    ) -> None:
        """Run N attach sessions sequentially against a single transpose process."""
        if rocprof_config.rocprofsys_attach is None:
            pytest.skip("rocprof-sys-attach binary not found")

        try:
            transpose = rocprof_config.get_target_executable("transpose")
        except FileNotFoundError:
            pytest.skip("transpose target not found")

        target_env = {**os.environ, "ROCP_TOOL_ATTACH": "1"}
        target_proc = subprocess.Popen(
            [str(transpose), "2", "0"],  # 2 threads, nitr=0 (infinite)
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(test_output_dir),
            env=target_env,
        )

        try:
            time.sleep(self.TARGET_WARMUP_S)
            assert target_proc.poll() is None, (
                "transpose target exited before first attach "
                f"(returncode={target_proc.returncode})"
            )

            for session_idx in range(n_sessions):
                session_out = test_output_dir / f"session_{session_idx}"
                result = subprocess.run(
                    [
                        str(rocprof_config.rocprofsys_attach),
                        "-p",
                        str(target_proc.pid),
                        "-d",
                        str(self.SESSION_DURATION_S),
                        "-F",
                        "perfetto",
                        "-o",
                        str(session_out),
                    ],
                    timeout=self.ATTACH_TIMEOUT_S,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert result.returncode == 0, (
                    f"session {session_idx} failed (rc={result.returncode})\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
                assert target_proc.poll() is None, (
                    f"transpose target died after session {session_idx} "
                    f"(returncode={target_proc.returncode})"
                )
                time.sleep(self.INTER_SESSION_PAUSE_S)

            # Re-attaching to the same process reuses the output directory set
            # on the first attach; each session is distinguished by a run-counter
            # suffix (perfetto-trace-<pid>-<n>.pftrace), not a fresh session_N dir.
            traces = sorted(test_output_dir.rglob("perfetto-trace*.pftrace"))
            assert len(traces) >= n_sessions, (
                f"expected at least {n_sessions} perfetto-trace*.pftrace file(s) under "
                f"{test_output_dir}, found {len(traces)}: {traces}"
            )
        finally:
            if target_proc.poll() is None:
                target_proc.terminate()
                try:
                    target_proc.wait(timeout=self.TARGET_TERMINATE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    target_proc.kill()
                    target_proc.wait()


class TestRocattachMultiPid:
    """Attach one rocprof-sys-attach invocation to several transpose targets."""

    N_TARGETS = 3
    SESSION_DURATION_S = 5
    TARGET_WARMUP_S = 2
    ATTACH_TIMEOUT_S = 60
    TARGET_TERMINATE_TIMEOUT_S = 10

    def test_attach_multiple_pids(
        self,
        rocprof_config,
        test_output_dir: Path,
    ) -> None:
        """Attach to N targets with a single ``-p a,b,c`` and expect one trace per PID."""
        if rocprof_config.rocprofsys_attach is None:
            pytest.skip("rocprof-sys-attach binary not found")

        try:
            transpose = rocprof_config.get_target_executable("transpose")
        except FileNotFoundError:
            pytest.skip("transpose target not found")

        target_env = {**os.environ, "ROCP_TOOL_ATTACH": "1"}
        targets = [
            subprocess.Popen(
                [str(transpose), "2", "0"],  # 2 threads, nitr=0 (infinite)
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(test_output_dir),
                env=target_env,
            )
            for _ in range(self.N_TARGETS)
        ]

        try:
            time.sleep(self.TARGET_WARMUP_S)
            for target in targets:
                assert target.poll() is None, (
                    "transpose target exited before attach "
                    f"(returncode={target.returncode})"
                )

            pids = [target.pid for target in targets]
            result = subprocess.run(
                [
                    str(rocprof_config.rocprofsys_attach),
                    "-p",
                    ",".join(str(pid) for pid in pids),
                    "-d",
                    str(self.SESSION_DURATION_S),
                    "-F",
                    "perfetto",
                    "-o",
                    str(test_output_dir / "multi_pid"),
                ],
                timeout=self.ATTACH_TIMEOUT_S,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, (
                f"multi-pid attach failed (rc={result.returncode})\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

            for target in targets:
                assert target.poll() is None, (
                    f"transpose target {target.pid} died after detach "
                    f"(returncode={target.returncode})"
                )

            # Output files are named after the PID, so each target must own a trace.
            for pid in pids:
                traces = list(test_output_dir.rglob(f"perfetto-trace-{pid}*.pftrace"))
                assert traces, (
                    f"no perfetto trace for pid {pid} under {test_output_dir}; "
                    f"found {sorted(test_output_dir.rglob('*.pftrace'))}"
                )
        finally:
            for target in targets:
                if target.poll() is None:
                    target.terminate()
            for target in targets:
                try:
                    target.wait(timeout=self.TARGET_TERMINATE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    target.kill()
                    target.wait()
