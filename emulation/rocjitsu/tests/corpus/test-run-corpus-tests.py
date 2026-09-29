#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise corpus orchestration without compiling or launching GPU programs."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

SOURCE_DIR = Path(__file__).resolve().parents[2]


class RunCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='corpus runner ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.calls = self.root / 'calls.jsonl'
        self.env = {
            **os.environ,
            'PATH': f'{self.bin}{os.pathsep}{os.environ["PATH"]}',
            'ROCM_PATH': str(self.root / 'rocm'),
            'ROCJITSU_SOURCE_DIR': str(SOURCE_DIR),
            'CORPUS_TEST_CALLS': str(self.calls),
        }
        self.write_tool('rocjitsu', '#!/bin/sh\nexit 0\n')
        self.write_tool(
            'pytest',
            f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "--suite" not in args:
    sys.exit(0)
suite = args[args.index("--suite") + 1]
target = args[args.index("--target") + 1]
with open(os.environ["CORPUS_TEST_CALLS"], "a") as log:
    log.write(json.dumps({"args": args, "config": os.getenv("ROCJITSU_RACE_CONFIG")}) + "\\n")
if "--junitxml" in args:
    report = Path(args[args.index("--junitxml") + 1])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("<testsuites/>")
sys.exit(1 if os.getenv("CORPUS_TEST_FAIL") and target == "gfx950" and "--last-failed" not in args else 0)
''',
        )

    def write_tool(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def run_script(self, *args):
        result = subprocess.run(
            ['bash', str(SOURCE_DIR / 'tests/corpus/run-corpus-tests.sh'), *args],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        return result, calls

    @staticmethod
    def option(call, name):
        args = call['args']
        return args[args.index(name) + 1]

    def test_race_suite_is_opt_in(self):
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 5)
        self.assertTrue(
            all(self.option(c, '--suite') == 'iree,kernels,cts' for c in calls)
        )

    def test_race_uses_private_config_and_shared_launch_wrapper(self):
        result, calls = self.run_script(
            '--race-tests', '--workers', '3', '--soft-timeout', '17'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 7)
        for call in calls:
            wrapper = shlex.split(self.option(call, '--run-wrapper'))
            self.assertIn(
                str(SOURCE_DIR / 'tests/corpus/corpus-process-supervisor.sh'), wrapper
            )
            self.assertIn('17s', wrapper)
            self.assertEqual(self.option(call, '-n'), '3')
            self.assertEqual(self.option(call, '--timeout'), '32')
            if self.option(call, '--suite') == 'race':
                self.assertEqual(wrapper[wrapper.index('--config') + 1], '{config}')
                self.assertTrue(Path(call['config']).is_file())
                self.assertNotIn('--skip-tests-config', call['args'])
            else:
                self.assertIn('--skip-tests-config', call['args'])
        self.assertEqual(
            [self.option(c, '--target') for c in calls[-2:]], ['gfx950', 'gfx1151']
        )
        reports = self.root / '.pytest-artifacts/junit'
        self.assertTrue((reports / 'race-gfx950.xml').is_file())
        self.assertTrue((reports / 'race-gfx1151.xml').is_file())

    def test_successful_retry_preserves_failure_for_both_suites(self):
        self.env['CORPUS_TEST_FAIL'] = '1'
        result, calls = self.run_script(
            '--race-tests', '--rerun-failed', '--hard-timeout', '43'
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        retries = [c for c in calls if '--last-failed' in c['args']]
        self.assertEqual(
            [self.option(c, '--suite') for c in retries], ['iree,kernels,cts', 'race']
        )
        for call in retries:
            self.assertIn('--last-failed-no-failures=none', call['args'])
            self.assertNotIn('--junitxml', call['args'])
            self.assertIn('43s', shlex.split(self.option(call, '--run-wrapper')))
        self.assertEqual(self.option(calls[-1], '--target'), 'gfx1151')

    def test_race_preserves_clang_asan_child_preload(self):
        rocm = Path(self.env['ROCM_PATH'])
        (rocm / 'lib/llvm/bin').mkdir(parents=True)
        symbolizer = rocm / 'lib/llvm/bin/llvm-symbolizer'
        symbolizer.touch()
        symbolizer.chmod(0o755)
        (rocm / 'lib/libamdhip64.so').touch()
        result, calls = self.run_script('--race-tests', '--sanitizer', 'clang-asan')
        self.assertEqual(result.returncode, 0, result.stderr)
        for call in calls:
            wrapper = shlex.split(self.option(call, '--run-wrapper'))
            self.assertEqual(
                wrapper[-3:],
                [
                    'bash',
                    str(SOURCE_DIR / 'tests/corpus/corpus-hip-preload.sh'),
                    str(rocm / 'lib/libamdhip64.so'),
                ],
            )


if __name__ == '__main__':
    unittest.main()
