#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import signal
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

RUNNER = Path(__file__).parents[3] / "fuzz" / "execution" / "run_campaign.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("execution_campaign_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load execution campaign runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def campaign_args(binary: Path, output: Path, duration: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        binary=binary,
        output=output,
        target="auto",
        duration=duration,
        start_seed=0,
        batch_size=1,
        workers=1,
        threads_per_worker=1,
        async_mode="auto",
        async_helpers=1,
        issuers="auto",
    )


class ExecutionCampaignTest(unittest.TestCase):
    def test_interrupted_crash_publication_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "campaign"
            output.mkdir()
            current = root / "current.bin"
            current.write_bytes(b"failing input")
            runner = load_runner()
            real_copyfile = runner.shutil.copyfile

            def interrupt_after_copy(source, destination):
                real_copyfile(source, destination)
                raise KeyboardInterrupt

            arguments = (
                output,
                current,
                root / "fuzzer",
                7,
                1,
                "failure",
                "gfx1250",
                True,
                "auto",
                1,
                "auto",
            )
            with (
                mock.patch.object(
                    runner.shutil, "copyfile", side_effect=interrupt_after_copy
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                runner.save_crash(*arguments)

            self.assertEqual(list((output / "crashes").glob("*.bin")), [])
            digest, duplicate = runner.save_crash(*arguments)
            crash = output / "crashes" / f"{digest}.bin"
            self.assertFalse(duplicate)
            self.assertEqual(crash.read_bytes(), current.read_bytes())
            self.assertTrue(crash.with_suffix(".json").is_file())

    def test_missing_crash_metadata_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "campaign"
            current = root / "current.bin"
            current.write_bytes(b"failing input")
            digest = hashlib.sha256(current.read_bytes()).hexdigest()
            crash = output / "crashes" / f"{digest}.bin"
            crash.parent.mkdir(parents=True)
            crash.write_bytes(current.read_bytes())
            runner = load_runner()

            saved_digest, duplicate = runner.save_crash(
                output,
                current,
                root / "fuzzer",
                7,
                1,
                "failure",
                "gfx1250",
                True,
                "auto",
                1,
                "auto",
            )

            self.assertEqual(saved_digest, digest)
            self.assertFalse(duplicate)
            metadata = json.loads(crash.with_suffix(".json").read_text())
            self.assertEqual(metadata["attributed_seed"], 7)

    def test_campaign_finishes_at_uint64_seed_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "fake-worker.py"
            worker.write_text(textwrap.dedent("""\
                    #!/usr/bin/env python3
                    import json
                    from pathlib import Path
                    import sys

                    args = sys.argv[1:]
                    output = Path(args[args.index('--output') + 1])
                    seed = int(args[args.index('--seed') + 1])
                    count = int(args[args.index('--random') + 1])
                    (output / 'launch.txt').write_text(f'{seed} {count}\\n')
                    keys = (
                        'executed', 'async_executed', 'async_capable_inputs',
                        'async_enabled_inputs', 'concurrent_inputs', 'issuer_executions',
                        'waves_halted', 'rounds', 'budget_exhausted', 'inputs',
                        'inputs_executed', 'inputs_halted', 'inputs_stopped',
                    )
                    stats = {key: 0 for key in keys}
                    stats['inputs'] = count
                    print(json.dumps(stats))
                    """))
            worker.chmod(0o755)
            output = root / "campaign"
            max_seed = (1 << 64) - 1
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--binary",
                    str(worker),
                    "--output",
                    str(output),
                    "--duration",
                    "1",
                    "--start-seed",
                    str(max_seed),
                    "--batch-size",
                    "2",
                    "--workers",
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (output / "work" / "worker-0" / "launch.txt").read_text(),
                f"{max_seed} 1\n",
            )
            self.assertFalse((output / "work" / "worker-1").exists())
            summary = json.loads(result.stdout)
            self.assertEqual(summary["totals"]["inputs"], 1)
            self.assertEqual(summary["next_seed"], 1 << 64)

    def test_threaded_failure_replays_serially_without_stale_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "fake-worker.py"
            worker.write_text(textwrap.dedent("""\
                    #!/usr/bin/env python3
                    from pathlib import Path
                    import sys
                    import time

                    args = sys.argv[1:]
                    output = Path(args[args.index('--output') + 1])
                    seed = int(args[args.index('--seed') + 1])
                    count = int(args[args.index('--random') + 1])
                    threads = int(args[args.index('--threads') + 1])
                    assert args[args.index('--async') + 1] == 'on'
                    assert args[args.index('--async-helpers') + 1] == '4'
                    assert args[args.index('--issuers') + 1] == '2'
                    launches = output / 'launches.txt'
                    previous = launches.read_text() if launches.exists() else ''
                    launch = len(previous.splitlines()) + 1
                    launches.write_text(previous + f'{threads} {seed} {count}\\n')

                    if launch == 1:
                        for thread in range(threads):
                            directory = output / f'thread-{thread}'
                            directory.mkdir(parents=True, exist_ok=True)
                            (directory / 'current.seed').write_text(f'{seed + thread}\\n')
                            (directory / 'current.bin').write_bytes(bytes([seed + thread, 0x42]))
                        sys.stderr.write('x' * 100_000)
                        sys.stderr.flush()
                        time.sleep(0.2)
                        raise SystemExit(1)

                    if launch == 2:
                        (output / 'current.seed').write_text(f'{seed + 1}\\n')
                        (output / 'current.bin').write_bytes(bytes([seed + 1, 0x42]))
                        raise SystemExit(1)

                    keys = (
                        'executed', 'async_executed', 'async_capable_inputs',
                        'async_enabled_inputs', 'concurrent_inputs', 'issuer_executions',
                        'waves_halted', 'rounds', 'budget_exhausted', 'inputs',
                        'inputs_executed', 'inputs_halted', 'inputs_stopped',
                    )
                    stats = {key: 0 for key in keys}
                    stats['inputs'] = count
                    print(__import__('json').dumps(stats))
                    """))
            worker.chmod(0o755)
            output = root / "campaign"
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--binary",
                    str(worker),
                    "--output",
                    str(output),
                    "--duration",
                    "1",
                    "--batch-size",
                    "8",
                    "--workers",
                    "1",
                    "--threads-per-worker",
                    "2",
                    "--async",
                    "on",
                    "--async-helpers",
                    "4",
                    "--issuers",
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            retained = list((output / "crashes").glob("*.bin"))
            self.assertEqual(len(retained), 2)
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            failures = [event for event in events if event["type"] == "crash-candidate"]
            self.assertEqual([event["seed"] for event in failures[:3]], [0, 1, 1])
            self.assertEqual(
                [event["exact_attribution"] for event in failures[:3]],
                [False, False, True],
            )
            launches = (output / "work" / "worker-0" / "launches.txt").read_text()
            self.assertIn("2 0 8\n", launches)
            self.assertIn("1 0 8\n", launches)
            self.assertIn("1 2 6\n", launches)
            exact = json.loads((retained[1].with_suffix(".json")).read_text())
            if exact["first_seen_seed"] != 1:
                exact = json.loads((retained[0].with_suffix(".json")).read_text())
            self.assertTrue(exact["exact_attribution"])
            self.assertEqual(exact["attributed_seed"], 1)
            self.assertEqual(
                exact["replay"][-6:],
                ["--async", "on", "--async-helpers", "4", "--issuers", "2"],
            )

    def test_threaded_failure_retains_exact_and_sibling_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "fake-worker.py"
            worker.write_text(textwrap.dedent("""\
                    #!/usr/bin/env python3
                    from pathlib import Path
                    import shutil
                    import sys
                    import time

                    args = sys.argv[1:]
                    output = Path(args[args.index('--output') + 1])
                    seed = int(args[args.index('--seed') + 1])
                    count = int(args[args.index('--random') + 1])
                    threads = int(args[args.index('--threads') + 1])
                    launches = output / 'launches.txt'
                    launch = int(launches.read_text()) + 1 if launches.exists() else 1
                    launches.write_text(str(launch))

                    if launch == 1:
                        for thread in range(threads):
                            directory = output / f'thread-{thread}'
                            directory.mkdir(parents=True, exist_ok=True)
                            (directory / 'current.bin').write_bytes(bytes([seed + thread, 0x42]))
                            (directory / 'current.seed').write_text(f'{seed + thread}\\n')
                        shutil.copyfile(output / 'thread-0' / 'current.bin', output / 'failure.bin')
                        (output / 'failure.seed').write_text(f'{seed}\\n')
                        raise SystemExit(1)

                    keys = (
                        'executed', 'async_executed', 'async_capable_inputs',
                        'async_enabled_inputs', 'concurrent_inputs', 'issuer_executions',
                        'waves_halted', 'rounds', 'budget_exhausted', 'inputs',
                        'inputs_executed', 'inputs_halted', 'inputs_stopped',
                    )
                    stats = {key: 0 for key in keys}
                    stats['inputs'] = count
                    print(__import__('json').dumps(stats))
                    time.sleep(0.05)
                    """))
            worker.chmod(0o755)
            output = root / "campaign"
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--binary",
                    str(worker),
                    "--output",
                    str(output),
                    "--duration",
                    "1",
                    "--batch-size",
                    "8",
                    "--workers",
                    "1",
                    "--threads-per-worker",
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = [
                json.loads(path.read_text())
                for path in (output / "crashes").glob("*.json")
            ]
            self.assertEqual(len(metadata), 2)
            by_seed = {record["first_seen_seed"]: record for record in metadata}
            self.assertEqual(set(by_seed), {0, 1})
            self.assertTrue(by_seed[0]["exact_attribution"])
            self.assertEqual(by_seed[0]["attributed_seed"], 0)
            self.assertFalse(by_seed[1]["exact_attribution"])
            self.assertNotIn("attributed_seed", by_seed[1])
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            failures = [event for event in events if event["type"] == "crash-candidate"]
            self.assertEqual([event["seed"] for event in failures[:2]], [0, 1])
            self.assertEqual(
                [event["exact_attribution"] for event in failures[:2]],
                [True, False],
            )

    def test_failure_between_deadline_polls_is_recorded_as_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "fake-worker"
            worker.write_bytes(b"fake worker")
            output = root / "campaign"
            runner = load_runner()

            class DeadlineRaceProcess:
                def __init__(self, command, **_kwargs) -> None:
                    self.returncode = None
                    self.poll_count = 0
                    self.terminated = False
                    work = Path(command[command.index("--output") + 1])
                    seed = int(command[command.index("--seed") + 1])
                    (work / "failure.bin").write_bytes(b"failure")
                    (work / "failure.seed").write_text(f"{seed}\n")

                def poll(self):
                    self.poll_count += 1
                    if self.poll_count == 1:
                        return None
                    self.returncode = 1
                    return self.returncode

                def terminate(self) -> None:
                    self.terminated = True
                    self.returncode = -signal.SIGTERM

                def wait(self, timeout=None):
                    if self.returncode is None:
                        self.returncode = 1
                    return self.returncode

                def kill(self) -> None:
                    self.returncode = -signal.SIGKILL

            process = None

            def launch_process(*args, **kwargs):
                nonlocal process
                process = DeadlineRaceProcess(*args, **kwargs)
                return process

            moments = iter((0.0, 0.0, 2.0))

            def monotonic() -> float:
                return next(moments, 2.0)

            with (
                mock.patch.object(
                    runner, "parse_args", return_value=campaign_args(worker, output)
                ),
                mock.patch.object(
                    runner.subprocess, "Popen", side_effect=launch_process
                ),
                mock.patch.object(runner.time, "monotonic", side_effect=monotonic),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = runner.main()

            self.assertEqual(result, 0)
            self.assertIsNotNone(process)
            self.assertFalse(process.terminated)
            crashes = list((output / "crashes").glob("*.bin"))
            self.assertEqual(len(crashes), 1)
            self.assertEqual(list((output / "stopped").glob("*.bin")), [])
            metadata = json.loads(crashes[0].with_suffix(".json").read_text())
            self.assertEqual(metadata["returncode"], 1)
            self.assertFalse(metadata["timed_out"])
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["type"], "crash-candidate")

    def test_interrupt_during_launch_stops_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "hung-worker.py"
            worker.write_text(textwrap.dedent("""\
                    #!/usr/bin/env python3
                    import time

                    time.sleep(60)
                    """))
            worker.chmod(0o755)
            output = root / "campaign"
            runner = load_runner()
            real_popen = subprocess.Popen
            children = []

            def interrupting_popen(*args, **kwargs):
                child = real_popen(*args, **kwargs)
                children.append(child)
                os.kill(os.getpid(), signal.SIGINT)
                return child

            try:
                with (
                    mock.patch.object(
                        runner,
                        "parse_args",
                        return_value=campaign_args(worker, output, duration=10),
                    ),
                    mock.patch.object(
                        runner.subprocess,
                        "Popen",
                        side_effect=interrupting_popen,
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    result = runner.main()

                self.assertEqual(result, 130)
                self.assertEqual(len(children), 1)
                self.assertEqual(children[0].wait(timeout=2), -signal.SIGTERM)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait()

    def test_deadline_stops_hung_worker_and_retains_active_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "hung-worker.py"
            worker.write_text(textwrap.dedent("""\
                    #!/usr/bin/env python3
                    from pathlib import Path
                    import sys
                    import time

                    args = sys.argv[1:]
                    output = Path(args[args.index('--output') + 1])
                    seed = int(args[args.index('--seed') + 1])
                    (output / 'current.seed').write_text(f'{seed}\\n')
                    (output / 'current.bin').write_bytes(b'\\x05\\x00\\x00\\x00\\x80\\xbf')
                    time.sleep(60)
                    """))
            worker.chmod(0o755)
            output = root / "campaign"
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--binary",
                    str(worker),
                    "--output",
                    str(output),
                    "--duration",
                    "1",
                    "--batch-size",
                    "1",
                    "--workers",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            retained = list((output / "stopped").glob("*.bin"))
            self.assertEqual(len(retained), 1)
            metadata = json.loads(retained[0].with_suffix(".json").read_text())
            self.assertFalse(metadata["exact_attribution"])
            self.assertTrue(metadata["timed_out"])
            self.assertEqual(metadata["stop_reason"], "deadline")
            self.assertEqual(list((output / "crashes").glob("*.bin")), [])
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["type"], "deadline-candidate")
            self.assertTrue((output / "campaign-state.json").is_file())

    def test_interrupt_preserves_active_input_across_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "hung-worker.py"
            worker.write_text(textwrap.dedent("""\
                    #!/usr/bin/env python3
                    from pathlib import Path
                    import sys
                    import time

                    args = sys.argv[1:]
                    output = Path(args[args.index('--output') + 1])
                    seed = int(args[args.index('--seed') + 1])
                    launch_file = output / 'launches.txt'
                    launch = int(launch_file.read_text()) + 1 if launch_file.exists() else 1
                    launch_file.write_text(str(launch))
                    (output / 'current.bin').write_bytes(bytes([launch]))
                    (output / 'current.seed').write_text(f'{seed}\\n')
                    time.sleep(60)
                    """))
            worker.chmod(0o755)
            output = root / "campaign"
            command = [
                sys.executable,
                str(RUNNER),
                "--binary",
                str(worker),
                "--output",
                str(output),
                "--duration",
                "1",
                "--batch-size",
                "1",
                "--workers",
                "1",
            ]
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ) as process:
                try:
                    marker = output / "work" / "worker-0" / "current.seed"
                    deadline = time.monotonic() + 5
                    while not marker.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(
                        marker.exists(), "worker did not publish its active input"
                    )
                    process.send_signal(signal.SIGINT)
                    _, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 130, stderr.decode())
                finally:
                    if process.poll() is None:
                        process.kill()
            retained = list((output / "stopped").glob("*.bin"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_bytes(), b"\x01")
            metadata = json.loads(retained[0].with_suffix(".json").read_text())
            self.assertFalse(metadata["exact_attribution"])
            self.assertFalse(metadata["timed_out"])
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["type"], "interrupt-candidate")
            resumed = subprocess.run(command, capture_output=True, timeout=10)
            self.assertEqual(resumed.returncode, 0, resumed.stderr.decode())
            summary = json.loads(resumed.stdout)
            self.assertEqual(summary["failures_this_run"], 0)
            self.assertEqual(summary["retained_inputs"], 0)
            self.assertEqual(summary["stopped_inputs"], 2)
            self.assertEqual(retained[0].read_bytes(), b"\x01")
            self.assertEqual(len(list((output / "stopped").glob("*.bin"))), 2)

    def test_deadline_ignores_partially_written_active_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "partial-worker.py"
            worker.write_text(textwrap.dedent("""\
                    #!/usr/bin/env python3
                    from pathlib import Path
                    import sys
                    import time

                    args = sys.argv[1:]
                    output = Path(args[args.index('--output') + 1])
                    seed = int(args[args.index('--seed') + 1])
                    (output / 'current.seed').write_text('')
                    (output / 'current.bin').write_bytes(b'partial')
                    thread = output / 'thread-0'
                    thread.mkdir(parents=True, exist_ok=True)
                    (thread / 'current.seed').write_text(f'{seed}\\n')
                    (thread / 'current.bin').write_bytes(b'complete')
                    time.sleep(60)
                    """))
            worker.chmod(0o755)
            output = root / "campaign"
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--binary",
                    str(worker),
                    "--output",
                    str(output),
                    "--duration",
                    "1",
                    "--batch-size",
                    "2",
                    "--workers",
                    "1",
                    "--threads-per-worker",
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            retained = list((output / "stopped").glob("*.bin"))
            self.assertEqual(len(retained), 1)
            metadata = json.loads(retained[0].with_suffix(".json").read_text())
            self.assertTrue(metadata["timed_out"])
            self.assertFalse(metadata["exact_attribution"])


if __name__ == "__main__":
    unittest.main()
