#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

RUNNER = Path(__file__).parents[3] / "fuzz" / "execution" / "run_campaign.py"


class ExecutionCampaignTest(unittest.TestCase):
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
            retained = list((output / "crashes").glob("*.bin"))
            self.assertEqual(len(retained), 1)
            metadata = json.loads(retained[0].with_suffix(".json").read_text())
            self.assertFalse(metadata["exact_attribution"])
            self.assertTrue(metadata["timed_out"])
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["type"], "timeout-candidate")
            self.assertTrue((output / "campaign-state.json").is_file())

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
            retained = list((output / "crashes").glob("*.bin"))
            self.assertEqual(len(retained), 1)
            metadata = json.loads(retained[0].with_suffix(".json").read_text())
            self.assertTrue(metadata["timed_out"])
            self.assertFalse(metadata["exact_attribution"])


if __name__ == "__main__":
    unittest.main()
