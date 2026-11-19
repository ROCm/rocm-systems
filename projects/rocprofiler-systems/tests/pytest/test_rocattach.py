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
import select
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


@pytest.mark.class_name("rocattach")
class TestRocattachReattachFormat:
    """Re-attach to one target with a different output format on each session."""

    SESSION_DURATION_S = 5
    TARGET_WARMUP_S = 2
    INTER_SESSION_PAUSE_S = 1
    ATTACH_TIMEOUT_S = 60
    TARGET_TERMINATE_TIMEOUT_S = 10

    def test_reattach_uses_selected_format(
        self,
        rocprof_config,
        test_output_dir: Path,
    ) -> None:
        """Attach with ``-F rocpd`` then ``-F perfetto``; expect one output of each kind.

        Before the re-attach fix the second session reused the first session's
        format, so no perfetto trace was produced.
        """
        if rocprof_config.rocprofsys_attach is None:
            pytest.skip("rocprof-sys-attach binary not found")

        try:
            transpose = rocprof_config.get_target_executable("transpose")
        except FileNotFoundError:
            pytest.skip("transpose target not found")

        target_env = {
            **os.environ,
            "ROCP_TOOL_ATTACH": "1",
            "LD_LIBRARY_PATH": rocprof_config.get_library_path(),
        }
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

            for output_format in ("rocpd", "perfetto"):
                result = subprocess.run(
                    [
                        str(rocprof_config.rocprofsys_attach),
                        "-p",
                        str(target_proc.pid),
                        "-d",
                        str(self.SESSION_DURATION_S),
                        "-F",
                        output_format,
                        "-o",
                        str(test_output_dir / "attach_output"),
                    ],
                    timeout=self.ATTACH_TIMEOUT_S,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert result.returncode == 0, (
                    f"-F {output_format} session failed (rc={result.returncode})\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
                assert target_proc.poll() is None, (
                    f"transpose target died after the -F {output_format} session "
                    f"(returncode={target_proc.returncode})"
                )
                time.sleep(self.INTER_SESSION_PAUSE_S)

            databases = sorted(test_output_dir.rglob("rocpd-*.db"))
            traces = sorted(test_output_dir.rglob("perfetto-trace*.pftrace"))
            assert len(databases) == 1, (
                f"expected exactly one rocpd-*.db under {test_output_dir} "
                f"(from the -F rocpd session), found {len(databases)}: {databases}"
            )
            assert len(traces) == 1, (
                f"expected exactly one perfetto-trace*.pftrace under {test_output_dir} "
                f"(from the -F perfetto session), found {len(traces)}: {traces}"
            )
        finally:
            if target_proc.poll() is None:
                target_proc.terminate()
                try:
                    target_proc.wait(timeout=self.TARGET_TERMINATE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    target_proc.kill()
                    target_proc.wait()


@pytest.mark.class_name("rocattach")
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


@pytest.mark.class_name("rocattach")
class TestRocattachExitedTarget:
    """A target that exits while attached must not trap the interactive prompt."""

    TARGET_WARMUP_S = 2
    PROMPT_TIMEOUT_S = 60
    DETACH_TIMEOUT_S = 60
    TARGET_TERMINATE_TIMEOUT_S = 10
    PROMPT_MARKER = b"to detach all:"

    def _read_until(
        self, proc: subprocess.Popen, marker: bytes, timeout_s: float
    ) -> bytes:
        """Read ``proc`` stdout until ``marker`` appears (the prompt has no newline)."""
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        data = b""
        while marker not in data:
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"timed out waiting for {marker!r}; got:\n{data!r}"
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 4096)
            assert chunk, f"attach tool exited before printing {marker!r}:\n{data!r}"
            data += chunk
        return data

    def test_exited_target_does_not_block_detach_all(
        self,
        rocprof_config,
        test_output_dir: Path,
    ) -> None:
        """Kill one of two attached targets, then enter ``a`` and keep stdin open.

        The exited target cannot be detached from, so its session is dropped and
        reported as a failure. The live target is still detached (and finalized),
        the tool exits on its own without waiting for end of input, and its exit
        status is non-zero.
        """
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
            for _ in range(2)
        ]
        doomed, survivor = targets
        attach = None

        try:
            time.sleep(self.TARGET_WARMUP_S)
            for target in targets:
                assert target.poll() is None, (
                    "transpose target exited before attach "
                    f"(returncode={target.returncode})"
                )

            attach = subprocess.Popen(
                [
                    str(rocprof_config.rocprofsys_attach),
                    "-p",
                    f"{doomed.pid},{survivor.pid}",
                    "-F",
                    "perfetto",
                    "-o",
                    str(test_output_dir / "exited_target"),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            output = self._read_until(attach, self.PROMPT_MARKER, self.PROMPT_TIMEOUT_S)

            doomed.kill()
            doomed.wait()

            # Keep stdin open: if the exited session were kept, the prompt would
            # wait for more input here and this wait() would time out.
            assert attach.stdin is not None
            attach.stdin.write(b"a\n")
            attach.stdin.flush()
            try:
                returncode = attach.wait(timeout=self.DETACH_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pytest.fail(
                    "rocprof-sys-attach did not exit after 'a' although all remaining "
                    "sessions were handled: the exited target's session was kept\n"
                    f"output so far:\n{output.decode(errors='replace')}"
                )
            assert attach.stdout is not None
            output += attach.stdout.read()
            text = output.decode(errors="replace")

            assert returncode != 0, (
                "attach reported success although a target exited while attached\n"
                f"output:\n{text}"
            )
            assert survivor.poll() is None, (
                f"transpose target {survivor.pid} died "
                f"(returncode={survivor.returncode})"
            )
            traces = list(
                test_output_dir.rglob(f"perfetto-trace-{survivor.pid}[-.]*pftrace")
            )
            assert traces, (
                f"no perfetto trace for pid {survivor.pid} under {test_output_dir}: "
                "the live target was not detached after the exited one failed\n"
                f"output:\n{text}"
            )
        finally:
            if attach is not None:
                if attach.stdin is not None:
                    attach.stdin.close()
                if attach.poll() is None:
                    attach.kill()
                attach.wait()
                if attach.stdout is not None:
                    attach.stdout.close()
            for target in targets:
                if target.poll() is None:
                    target.terminate()
            for target in targets:
                try:
                    target.wait(timeout=self.TARGET_TERMINATE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    target.kill()
                    target.wait()


@pytest.mark.class_name("rocattach")
class TestRocattachPartialAttach:
    """One PID in ``-p`` cannot be attached; the others must still be profiled."""

    SESSION_DURATION_S = 3
    TARGET_WARMUP_S = 2
    ATTACH_TIMEOUT_S = 60
    TARGET_TERMINATE_TIMEOUT_S = 10

    def test_detaches_attached_targets_when_one_pid_fails(
        self,
        rocprof_config,
        test_output_dir: Path,
    ) -> None:
        """A failed attach must not leave the successfully attached target hanging.

        The tool exits non-zero because one attach failed, but the live target is
        still detached (and therefore finalized) after ``-d`` seconds, so its trace
        exists and the process keeps running.
        """
        if rocprof_config.rocprofsys_attach is None:
            pytest.skip("rocprof-sys-attach binary not found")

        try:
            transpose = rocprof_config.get_target_executable("transpose")
        except FileNotFoundError:
            pytest.skip("transpose target not found")

        # pid_max + 1 can never name a live process, so this attach always fails.
        dead_pid = int(Path("/proc/sys/kernel/pid_max").read_text().strip()) + 1

        target_env = {**os.environ, "ROCP_TOOL_ATTACH": "1"}
        target = subprocess.Popen(
            [str(transpose), "2", "0"],  # 2 threads, nitr=0 (infinite)
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(test_output_dir),
            env=target_env,
        )

        try:
            time.sleep(self.TARGET_WARMUP_S)
            assert target.poll() is None, (
                "transpose target exited before attach "
                f"(returncode={target.returncode})"
            )

            result = subprocess.run(
                [
                    str(rocprof_config.rocprofsys_attach),
                    "-p",
                    f"{target.pid},{dead_pid}",
                    "-d",
                    str(self.SESSION_DURATION_S),
                    "-F",
                    "perfetto",
                    "-o",
                    str(test_output_dir / "partial_attach"),
                ],
                timeout=self.ATTACH_TIMEOUT_S,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode != 0, (
                "attach reported success although one PID could not be attached\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

            assert target.poll() is None, (
                f"transpose target {target.pid} died after detach "
                f"(returncode={target.returncode})"
            )

            traces = list(test_output_dir.rglob(f"perfetto-trace-{target.pid}*.pftrace"))
            assert traces, (
                f"no perfetto trace for pid {target.pid} under {test_output_dir}: "
                "the attached target was not detached after the other PID failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        finally:
            if target.poll() is None:
                target.terminate()
                try:
                    target.wait(timeout=self.TARGET_TERMINATE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    target.kill()
                    target.wait()
