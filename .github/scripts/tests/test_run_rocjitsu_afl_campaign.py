# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the bounded rocjitsu AFL++ campaign runner."""

import os
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
RUNNER = SCRIPTS_DIR / "run_rocjitsu_afl_campaign.py"

RUNNER_SPEC = spec_from_file_location("run_rocjitsu_afl_campaign", RUNNER)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER_MODULE = module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_process_groups_with_only_zombies_are_not_live(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    for process_id, name, state, process_group in (
        (100, "zombie leader", "Z", 100),
        (200, "live group leader", "Z", 200),
        (201, "live member", "S", 200),
        (300, "unrelated", "R", 300),
    ):
        process_dir = proc_root / str(process_id)
        process_dir.mkdir(parents=True)
        (process_dir / "stat").write_text(
            f"{process_id} ({name}) {state} 1 {process_group} 0 0\n"
        )

    assert RUNNER_MODULE.live_process_groups({100, 200}, proc_root) == {200}


def _run_early_exit_worker_with_session_child(
    tmp_path: Path, monkeypatch, exit_code: int
) -> tuple[int, Path]:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_executable(
        tools / "afl-fuzz",
        f"""#!/usr/bin/env python3
import pathlib
import subprocess
import sys
import time

arguments = sys.argv[1:]
output_dir = pathlib.Path(arguments[arguments.index("-o") + 1])
instance = arguments[arguments.index("-S") + 1]
instance_dir = output_dir / instance
instance_dir.mkdir(parents=True)
heartbeat = instance_dir / "child-heartbeat"
subprocess.Popen(
    [
        sys.executable,
        "-c",
        '''import pathlib, sys, time
heartbeat = pathlib.Path(sys.argv[1])
while True:
    with heartbeat.open("a") as stream:
        stream.write("x")
    time.sleep(0.02)
''',
        heartbeat,
    ],
    start_new_session=True,
)
while not heartbeat.exists():
    time.sleep(0.01)
(instance_dir / "fuzzer_stats").write_text(
    "execs_done : 4\\nexecs_per_sec : 2.0\\n"
)
sys.exit({exit_code})
""",
    )
    binary = tmp_path / "target"
    _write_executable(binary, "#!/usr/bin/env python3\n")
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "seed").write_bytes(b"seed")
    output = tmp_path / "findings"
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    monkeypatch.setattr(RUNNER_MODULE, "available_cpu_count", lambda: 1)

    status = RUNNER_MODULE.run_campaign(
        binary,
        seeds,
        output,
        10,
        1,
        worker_stop_timeout_seconds=0.1,
    )
    return status, output / "worker-00" / "child-heartbeat"


def test_cleans_up_session_child_after_successful_worker_exit(
    tmp_path: Path, monkeypatch
) -> None:
    status, heartbeat = _run_early_exit_worker_with_session_child(
        tmp_path, monkeypatch, 0
    )

    assert status == 0
    heartbeat_size = heartbeat.stat().st_size
    time.sleep(0.1)
    assert heartbeat.stat().st_size == heartbeat_size


def test_cleans_up_session_child_after_failed_worker_exit(
    tmp_path: Path, monkeypatch
) -> None:
    status, heartbeat = _run_early_exit_worker_with_session_child(
        tmp_path, monkeypatch, 1
    )

    assert status == 1
    heartbeat_size = heartbeat.stat().st_size
    time.sleep(0.1)
    assert heartbeat.stat().st_size == heartbeat_size


def test_preserves_a_saved_crash_when_a_worker_fails(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_executable(
        tools / "afl-fuzz",
        """#!/usr/bin/env python3
import pathlib
import sys

arguments = sys.argv[1:]
output_dir = pathlib.Path(arguments[arguments.index("-o") + 1])
instance = arguments[arguments.index("-S") + 1]
instance_dir = output_dir / instance
(instance_dir / "crashes").mkdir(parents=True)
(instance_dir / "hangs").mkdir()
(instance_dir / "fuzzer_stats").write_text("execs_done : 10\\nexecs_per_sec : 2.5\\n")
if instance == "worker-01":
    (instance_dir / "crashes/id:000000").write_bytes(b"crash")
    sys.exit(1)
""",
    )
    binary = tmp_path / "target"
    _write_executable(binary, "#!/usr/bin/env python3\n")
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "seed").write_bytes(b"seed")
    output = tmp_path / "findings"
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    monkeypatch.setattr(RUNNER_MODULE, "available_cpu_count", lambda: 3)

    status = RUNNER_MODULE.run_campaign(binary, seeds, output, 1, 3)

    assert status == 1
    captured = capsys.readouterr()
    assert "AFL++ campaign started: 3 workers for 1s" in captured.out
    assert "AFL++ campaign complete:" in captured.out
    assert "30 inputs executed (7.50 inputs/s)" in captured.out
    assert (output / "worker-01" / "crashes" / "id:000000").read_bytes() == b"crash"
    assert (Path(f"{output}-logs") / "worker-01.log").is_file()


def test_stops_a_worker_process_group_that_overruns_the_shutdown_grace(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_executable(
        tools / "afl-fuzz",
        """#!/usr/bin/env python3
import pathlib
import signal
import subprocess
import sys
import time

arguments = sys.argv[1:]
output_dir = pathlib.Path(arguments[arguments.index("-o") + 1])
instance = arguments[arguments.index("-S") + 1]
instance_dir = output_dir / instance
instance_dir.mkdir(parents=True)
signal.signal(signal.SIGINT, signal.SIG_IGN)
heartbeat = instance_dir / "child-heartbeat"
subprocess.Popen(
    [
        sys.executable,
        "-c",
        '''import pathlib, signal, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
heartbeat = pathlib.Path(sys.argv[1])
while True:
    with heartbeat.open("a") as stream:
        stream.write("x")
    time.sleep(0.02)
''',
        heartbeat,
    ],
    start_new_session=True,
)
time.sleep(30)
""",
    )
    binary = tmp_path / "target"
    _write_executable(binary, "#!/usr/bin/env python3\n")
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "seed").write_bytes(b"seed")
    output = tmp_path / "findings"
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    monkeypatch.setattr(RUNNER_MODULE, "available_cpu_count", lambda: 1)

    started_at = time.monotonic()
    status = RUNNER_MODULE.run_campaign(
        binary,
        seeds,
        output,
        1,
        1,
        shutdown_grace_seconds=0,
        worker_stop_timeout_seconds=0.1,
    )
    elapsed = time.monotonic() - started_at

    assert status == 1
    assert elapsed < 3
    heartbeat = output / "worker-00" / "child-heartbeat"
    heartbeat_size = heartbeat.stat().st_size
    time.sleep(0.1)
    assert heartbeat.stat().st_size == heartbeat_size
    captured = capsys.readouterr()
    assert "duration reached; requesting graceful shutdown" in captured.out
    assert "reached its 0s shutdown grace with 1 unresponsive worker(s)" in captured.out
    assert "AFL++ campaign complete:" in captured.out
    assert "worker-00 did not stop within the shutdown grace" in captured.err
    assert "worker-00 did not publish fuzzer_stats" in captured.err


def test_requests_graceful_shutdown_at_the_campaign_deadline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_executable(
        tools / "afl-fuzz",
        """#!/usr/bin/env python3
import pathlib
import signal
import sys
import time

arguments = sys.argv[1:]
output_dir = pathlib.Path(arguments[arguments.index("-o") + 1])
instance = arguments[arguments.index("-S") + 1]
instance_dir = output_dir / instance
instance_dir.mkdir(parents=True)
(instance_dir / "fuzzer_stats").write_text("execs_done : 10\\nexecs_per_sec : 2.5\\n")
signal.signal(signal.SIGINT, lambda _signum, _frame: sys.exit(0))
time.sleep(30)
""",
    )
    binary = tmp_path / "target"
    _write_executable(binary, "#!/usr/bin/env python3\n")
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "seed").write_bytes(b"seed")
    output = tmp_path / "findings"
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    monkeypatch.setattr(RUNNER_MODULE, "available_cpu_count", lambda: 1)

    started_at = time.monotonic()
    status = RUNNER_MODULE.run_campaign(
        binary,
        seeds,
        output,
        1,
        1,
        shutdown_grace_seconds=2,
        worker_stop_timeout_seconds=0.1,
    )
    elapsed = time.monotonic() - started_at

    assert status == 0
    assert elapsed < 3
    output_text = capsys.readouterr().out
    assert (
        "duration reached; requesting graceful shutdown from 1 worker(s)" in output_text
    )
    assert "unresponsive worker" not in output_text
    assert "AFL++ campaign complete:" in output_text


def test_fails_when_a_worker_exits_without_publishing_stats(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_executable(tools / "afl-fuzz", "#!/usr/bin/env python3\n")
    binary = tmp_path / "target"
    _write_executable(binary, "#!/usr/bin/env python3\n")
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "seed").write_bytes(b"seed")
    output = tmp_path / "findings"
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    monkeypatch.setattr(RUNNER_MODULE, "available_cpu_count", lambda: 1)

    status = RUNNER_MODULE.run_campaign(binary, seeds, output, 1, 1)

    assert status == 1
    captured = capsys.readouterr()
    assert "worker-00 did not publish fuzzer_stats" in captured.err


def test_fails_when_a_worker_reports_zero_executions(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_executable(
        tools / "afl-fuzz",
        """#!/usr/bin/env python3
import pathlib
import sys

arguments = sys.argv[1:]
output_dir = pathlib.Path(arguments[arguments.index("-o") + 1])
instance = arguments[arguments.index("-S") + 1]
instance_dir = output_dir / instance
instance_dir.mkdir(parents=True)
(instance_dir / "fuzzer_stats").write_text("execs_done : 0\\nexecs_per_sec : 0\\n")
""",
    )
    binary = tmp_path / "target"
    _write_executable(binary, "#!/usr/bin/env python3\n")
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "seed").write_bytes(b"seed")
    output = tmp_path / "findings"
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    monkeypatch.setattr(RUNNER_MODULE, "available_cpu_count", lambda: 1)

    status = RUNNER_MODULE.run_campaign(binary, seeds, output, 1, 1)

    assert status == 1
    assert "worker-00 did not execute any inputs" in capsys.readouterr().err
