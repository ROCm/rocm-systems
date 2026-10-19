#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fake-process tests for scripts/run-rccl-scaleout.py."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SCRIPT = Path(__file__).parents[2] / "scripts" / "run-rccl-scaleout.py"
SPEC = importlib.util.spec_from_file_location("run_rccl_scaleout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


FAKE_ROCJITSU = '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

separator = sys.argv.index("--")
rank_binary = sys.argv[separator + 1]
rank_args = sys.argv[separator + 2:]
rank = int(next(arg.split("=", 1)[1] for arg in rank_args if arg.startswith("--rank=")))
shared = Path(next(arg.split("=", 1)[1] for arg in rank_args if arg.startswith("--shared-dir=")))
(shared / f"daemon{rank}.json").write_text(json.dumps({
    "argv": sys.argv[1:], "pid": os.getpid(), "pgid": os.getpgrp()
}))
os.execv(rank_binary, [rank_binary, *rank_args])
'''


FAKE_RANK = '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

rank = int(next(arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--rank=")))
shared = Path(next(arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--shared-dir=")))
environment_names = (
    "ROCJITSU_RUNTIME_DIR", "XDG_RUNTIME_DIR", "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "NCCL_NET",
    "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
    "GTEST_TOTAL_SHARDS", "GTEST_SHARD_INDEX", "GTEST_SHUFFLE", "GTEST_REPEAT",
)
(shared / f"rank{rank}.json").write_text(json.dumps({
    "argv": sys.argv[1:],
    "pid": os.getpid(),
    "pgid": os.getpgrp(),
    "env": {name: os.environ.get(name) for name in environment_names},
}))

behavior = "success"
selected_filter = next(arg.split("=", 1)[1] for arg in sys.argv[1:]
                       if arg.startswith("--gtest_filter="))
cases = (
    "RcclTest.AllReduce", "RcclTest.Broadcast", "RcclTest.AllGather",
    "RcclTest.ReduceScatter", "RcclTest.SendRecv",
)
selected_cases = list(cases) if selected_filter == "RcclTest.*" else [selected_filter]
if behavior == "partial_output":
    selected_cases = selected_cases[:1]
if behavior == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(3600)"])
    (shared / f"grandchild{rank}.pid").write_text(str(child.pid))
    while True:
        time.sleep(3600)
if behavior == "nonzero":
    print("fake rank failure", flush=True)
    raise SystemExit(7)
if behavior == "gtest_false_pass":
    print("[  PASSED  ] 1 test.", flush=True)
    print("NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0", flush=True)
    raise SystemExit(0)

if behavior == "gtest_fail":
    print(f"[ RUN      ] {selected_cases[0]}", flush=True)
    print(f"[  FAILED  ] {selected_cases[0]}", flush=True)
    print("[  PASSED  ] 0 tests.", flush=True)
else:
    for test_name in selected_cases:
        print(f"[ RUN      ] {test_name}", flush=True)
        if behavior != "missing_channel":
            print("NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0", flush=True)
        print(f"[       OK ] {test_name} (1 ms)", flush=True)
    count = len(selected_cases)
    noun = "test" if count == 1 else "tests"
    print(f"[  PASSED  ] {count} {noun}.", flush=True)
'''


def write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def process_is_running(pid: int) -> bool:
    """Treat an adopted zombie as stopped; it cannot execute after SIGKILL."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return stat.rsplit(") ", 1)[1][0] not in {"Z", "X"}


@unittest.skipUnless(sys.platform.startswith("linux"), "process groups need Linux")
class RunRcclScaleoutTest(unittest.TestCase):
    def prepare(self, root: Path) -> tuple[Path, Path, Path, Path]:
        rocjitsu = root / "rocjitsu"
        rank_binary = root / "hip_rccl_gfx1250_test"
        config = root / "system.json"
        output = root / "run"
        write_executable(rocjitsu, FAKE_ROCJITSU)
        write_executable(rank_binary, FAKE_RANK)
        config.write_text(
            json.dumps(
                {
                    "system": {
                        "nodes": [
                            {"id": "node0", "gpus": [{"id": "g0"}]},
                            {"id": "node1", "gpus": [{"id": "g1"}]},
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        return rocjitsu, rank_binary, config, output

    def arguments(self, root: Path, *extra: str) -> list[str]:
        rocjitsu, rank_binary, config, output = self.prepare(root)
        return [
            "--rocjitsu",
            str(rocjitsu),
            "--config",
            str(config),
            "--rank-binary",
            str(rank_binary),
            "--node",
            "node0",
            "--node",
            "node1",
            "--output",
            str(output),
            *extra,
        ]

    def set_rank_behavior(self, root: Path, rank: int, behavior: str) -> None:
        executable = root / "hip_rccl_gfx1250_test"
        source = executable.read_text(encoding="utf-8")
        marker = 'behavior = "success"'
        self.assertIn(marker, source)
        write_executable(
            executable,
            source.replace(
                marker, f'behavior = "{behavior}" if rank == {rank} else "success"', 1
            ),
        )

    def invoke(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = RUNNER.main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def read_rank(self, output: Path, rank: int) -> dict[str, object]:
        return json.loads((output / "shared" / f"rank{rank}.json").read_text())

    def assert_stopped(self, pid: int) -> None:
        deadline = time.monotonic() + 3
        while process_is_running(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(process_is_running(pid), f"process {pid} survived cleanup")

    def launcher_child_groups(self, launcher_pid: int) -> set[int]:
        """Capture rank groups before a stuck launcher is force-killed."""

        children = Path(f"/proc/{launcher_pid}/task/{launcher_pid}/children")
        try:
            child_pids = children.read_text().split()
        except OSError:
            return set()
        groups: set[int] = set()
        for child_pid in child_pids:
            try:
                groups.add(os.getpgid(int(child_pid)))
            except OSError:
                pass
        return groups

    def kill_recorded_rank_groups(
        self, shared: Path, extra_groups: set[int] | None = None
    ) -> None:
        """Emergency cleanup even if the launcher failed before handling SIGTERM."""

        groups = set(extra_groups or ())
        for rank in range(2):
            record = shared / f"daemon{rank}.json"
            try:
                group = int(json.loads(record.read_text())["pgid"])
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue
            if group > 0 and group != os.getpgrp():
                groups.add(group)
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_rejects_overlong_daemon_socket_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root)
            output = root / ("long-output-component-" * 4)
            args[args.index("--output") + 1] = str(output)
            status, _, stderr = self.invoke(args)
            self.assertNotEqual(status, 0)
            self.assertIn("daemon socket path would exceed", stderr)
            self.assertFalse(output.exists())

    def test_success_proves_node_launch_environment_and_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root, "--seed", "123")
            with mock.patch.dict(
                os.environ,
                {
                    "HIP_VISIBLE_DEVICES": "7",
                    "ROCR_VISIBLE_DEVICES": "8",
                    "CUDA_VISIBLE_DEVICES": "9",
                    "NCCL_NET": "IB",
                    "NCCL_SOCKET_IFNAME": "eth0",
                    "GTEST_TOTAL_SHARDS": "2",
                    "GTEST_SHARD_INDEX": "1",
                    "GTEST_SHUFFLE": "1",
                    "GTEST_REPEAT": "3",
                },
            ):
                status, stdout, stderr = self.invoke(args)
            self.assertEqual(status, 0, stderr)
            self.assertIn("NET/Socket smoke passed", stdout)
            self.assertIn(str(root / "run" / "rank0.log"), stdout)
            self.assertIn(str(root / "run" / "rank1.log"), stdout)

            records = [self.read_rank(root / "run", rank) for rank in range(2)]
            daemons = [
                json.loads((root / "run" / "shared" / f"daemon{rank}.json").read_text())
                for rank in range(2)
            ]
            for rank, (record, daemon) in enumerate(zip(records, daemons)):
                self.assertEqual(
                    daemon["argv"],
                    [
                        "--daemon",
                        "--config",
                        str(root / "system.json"),
                        "--node",
                        f"node{rank}",
                        "--",
                        str(root / "hip_rccl_gfx1250_test"),
                        f"--rank={rank}",
                        "--world-size=2",
                        f"--shared-dir={root / 'run' / 'shared'}",
                        "--expected-devices=1",
                        "--gtest_filter=RcclTest.AllReduce",
                        "--seed=123",
                    ],
                )
                self.assertEqual(record["argv"], daemon["argv"][7:])
                self.assertEqual(record["pgid"], daemon["pid"])
                self.assertEqual(
                    record["env"]["ROCJITSU_RUNTIME_DIR"],
                    str(root / "run" / f"runtime{rank}"),
                )
                self.assertEqual(
                    record["env"]["XDG_RUNTIME_DIR"], str(root / "run" / f"xdg{rank}")
                )
                for name in RUNNER.VISIBILITY_VARIABLES:
                    self.assertIsNone(record["env"][name])
                for name in (
                    "GTEST_TOTAL_SHARDS",
                    "GTEST_SHARD_INDEX",
                    "GTEST_SHUFFLE",
                    "GTEST_REPEAT",
                ):
                    self.assertIsNone(record["env"][name])
                for name in (
                    "NCCL_NET",
                    "NCCL_SOCKET_IFNAME",
                    "NCCL_IB_DISABLE",
                    "NCCL_P2P_DISABLE",
                    "NCCL_SHM_DISABLE",
                    "NCCL_DEBUG",
                    "NCCL_DEBUG_SUBSYS",
                ):
                    self.assertEqual(record["env"][name], RUNNER.RCCL_ENVIRONMENT[name])
                self.assertTrue((root / "run" / f"runtime{rank}").is_dir())
                self.assertTrue((root / "run" / f"xdg{rank}").is_dir())
                self.assert_stopped(int(record["pid"]))
            self.assertNotEqual(records[0]["pgid"], records[1]["pgid"])

    def test_all_cases_filter_runs_every_collective_on_both_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root, "--gtest-filter", "RcclTest.*")
            status, _stdout, stderr = self.invoke(args)
            self.assertEqual(status, 0, stderr)
            for rank in range(2):
                record = self.read_rank(root / "run", rank)
                self.assertIn("--gtest_filter=RcclTest.*", record["argv"])
                log = (root / "run" / f"rank{rank}.log").read_text()
                for case in RUNNER.RCCL_CASES:
                    self.assertIn(f"[ RUN      ] {case}", log)
                    self.assertIn(f"[       OK ] {case}", log)
                self.assertIn("[  PASSED  ] 5 tests.", log)

    def test_all_cases_filter_rejects_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root, "--gtest-filter", "RcclTest.*")
            self.set_rank_behavior(root, 1, "partial_output")
            status, _stdout, stderr = self.invoke(args)
            self.assertEqual(status, 1)
            self.assertIn("rank 1 log lacks a passing RcclTest result", stderr)
            log = (root / "run" / "rank1.log").read_text()
            self.assertIn("[  PASSED  ] 1 test.", log)

    def test_missing_channel_is_failure_even_with_passing_gtest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root)
            self.set_rank_behavior(root, 1, "missing_channel")
            status, _stdout, stderr = self.invoke(args)
            self.assertEqual(status, 1)
            self.assertIn("rank 1 log lacks a NET/Socket channel", stderr)
            self.assertIn(str(root / "run" / "rank1.log"), stderr)

    def test_nonzero_rank_stops_run_and_reports_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root)
            self.set_rank_behavior(root, 1, "nonzero")
            status, _stdout, stderr = self.invoke(args)
            self.assertEqual(status, 1)
            self.assertIn("rank 1 exited with status 7", stderr)
            self.assertTrue((root / "run" / "rank1.log").is_file())
            for rank in range(2):
                self.assert_stopped(int(self.read_rank(root / "run", rank)["pid"]))

    def test_gtest_summary_without_matching_result_is_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root)
            self.set_rank_behavior(root, 0, "gtest_false_pass")
            status, _stdout, stderr = self.invoke(args)
            self.assertEqual(status, 1)
            self.assertIn("rank 0 log lacks a passing RcclTest result", stderr)

    def test_cleanup_waits_for_group_after_launcher_exits(self) -> None:
        launcher = mock.Mock()
        launcher.pid = 12345
        launcher.poll.return_value = 0
        events: list[object] = []
        with (
            mock.patch.object(
                RUNNER,
                "signal_process_group",
                side_effect=lambda _process, signum: events.append(signum),
            ),
            mock.patch.object(
                RUNNER, "process_group_exists", side_effect=[True, False], create=True
            ) as group_exists,
            mock.patch.object(
                RUNNER.time, "sleep", side_effect=lambda _seconds: events.append("wait")
            ),
        ):
            RUNNER.cleanup_process_groups([launcher])
        self.assertEqual(events, [signal.SIGTERM, "wait", signal.SIGKILL])
        self.assertEqual(group_exists.call_count, 2)
        launcher.wait.assert_called_once()

    def test_timeout_kills_process_group_and_reaps_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root, "--timeout", "0.7")
            self.set_rank_behavior(root, 1, "hang")
            with mock.patch.object(RUNNER, "TERMINATION_GRACE_SECONDS", 0.1):
                status, _stdout, stderr = self.invoke(args)
            self.assertEqual(status, 1)
            self.assertIn("did not finish within 0.7 seconds", stderr)
            self.assert_stopped(int(self.read_rank(root / "run", 1)["pid"]))
            child_pid = int((root / "run" / "shared" / "grandchild1.pid").read_text())
            self.assert_stopped(child_pid)

    def test_sigterm_cleans_both_process_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root, "--timeout", "30")
            shared = root / "run" / "shared"
            self.set_rank_behavior(root, 1, "hang")
            launcher = subprocess.Popen(
                [sys.executable, str(SCRIPT), *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not all(
                    (shared / name).exists()
                    for name in ("rank0.json", "rank1.json", "grandchild1.pid")
                ):
                    self.assertIsNone(
                        launcher.poll(), "launcher exited before rank startup"
                    )
                    self.assertLess(time.monotonic(), deadline, "ranks did not start")
                    time.sleep(0.01)
                launcher.send_signal(signal.SIGTERM)
                _stdout, stderr = launcher.communicate(timeout=10)
                self.assertEqual(launcher.returncode, 1, stderr)
                self.assertIn("launcher received SIGTERM", stderr)
                for rank in range(2):
                    self.assert_stopped(int(self.read_rank(root / "run", rank)["pid"]))
                self.assert_stopped(int((shared / "grandchild1.pid").read_text()))
            finally:
                child_groups = (
                    self.launcher_child_groups(launcher.pid)
                    if launcher.poll() is None
                    else set()
                )
                try:
                    if launcher.poll() is None:
                        launcher.send_signal(signal.SIGTERM)
                    try:
                        launcher.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        if launcher.poll() is None:
                            launcher.kill()
                        launcher.communicate(timeout=5)
                finally:
                    self.kill_recorded_rank_groups(shared, child_groups)
                    if launcher.poll() is None:
                        launcher.kill()
                    launcher.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
