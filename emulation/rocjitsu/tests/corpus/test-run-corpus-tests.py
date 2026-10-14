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
            'ROCJITSU_CPU_THREAD_BUDGET': '4',
        }
        self.env.pop('ROCJITSU_RACE_CASE_TIMEOUT', None)
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
    log.write(json.dumps({"args": args, "config": os.getenv("ROCJITSU_RACE_CONFIG"),
                         "case_timeout": os.getenv("ROCJITSU_RACE_CASE_TIMEOUT")}) + "\\n")
if "--junitxml" in args:
    report = Path(args[args.index("--junitxml") + 1])
    report.parent.mkdir(parents=True, exist_ok=True)
    duration = os.getenv("CORPUS_TEST_RACE_DURATION", "0") if suite == "race" else "0"
    report.write_text(f'<testsuites><testsuite><testcase classname="{suite}" name="passing" time="{duration}"/></testsuite></testsuites>')
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
        self.assertEqual(len(calls), 8)
        for call in calls:
            wrapper = shlex.split(self.option(call, '--run-wrapper'))
            self.assertIn(
                str(SOURCE_DIR / 'tests/corpus/corpus-process-supervisor.sh'), wrapper
            )
            self.assertIn('17s', wrapper)
            self.assertEqual(wrapper[wrapper.index('--cpu-thread-budget') + 1], '4')
            self.assertEqual(self.option(call, '-n'), '3')
            self.assertEqual(self.option(call, '--timeout'), '32')
            if self.option(call, '--suite') == 'race':
                self.assertEqual(wrapper[wrapper.index('--config') + 1], '{config}')
                self.assertTrue(Path(call['config']).is_file())
                self.assertNotIn('--skip-tests-config', call['args'])
                self.assertEqual(call['case_timeout'], '17')
            else:
                self.assertIn('--skip-tests-config', call['args'])
                self.assertIsNone(call['case_timeout'])
        self.assertEqual(
            [self.option(c, '--target') for c in calls[-3:]],
            ['gfx942', 'gfx950', 'gfx1151'],
        )
        reports = self.root / '.pytest-artifacts/junit'
        self.assertTrue((reports / 'race-gfx942.xml').is_file())
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
            if self.option(call, '--suite') == 'race':
                self.assertNotIn('--skip-tests-config', call['args'])
                self.assertEqual(call['case_timeout'], '43')
            else:
                self.assertIsNone(call['case_timeout'])
                self.assertEqual(
                    Path(self.option(call, '--skip-tests-config')),
                    SOURCE_DIR / 'tests/corpus/gfx950_skip_tests.json',
                )
        self.assertEqual(self.option(calls[-1], '--target'), 'gfx1151')
        self.assertTrue(
            all(
                c['case_timeout'] == '30'
                for c in calls
                if '--last-failed' not in c['args']
                and self.option(c, '--suite') == 'race'
            )
        )

    def test_race_retry_deadline_does_not_leak_into_first_pass(self):
        self.env['CORPUS_TEST_FAIL'] = '1'
        self.env['ROCJITSU_RACE_CASE_TIMEOUT'] = '7'
        result, calls = self.run_script(
            '--race-tests',
            '--rerun-failed',
            '--soft-timeout',
            '17',
            '--hard-timeout',
            '43',
            '--rerun-timeout',
            '120',
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        race_calls = [c for c in calls if self.option(c, '--suite') == 'race']
        self.assertEqual([c['case_timeout'] for c in race_calls], ['17', '43', '17'])

    def test_race_near_timeout_reporting_uses_the_execution_deadline(self):
        for timeout, duration in [('30', '29'), ('360', '349')]:
            with self.subTest(timeout=timeout):
                self.calls.unlink(missing_ok=True)
                self.env['CORPUS_TEST_RACE_DURATION'] = duration
                result, calls = self.run_script(
                    '--race-tests', '--soft-timeout', timeout, '--warn-perf'
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                race_calls = [c for c in calls if self.option(c, '--suite') == 'race']
                for call in race_calls:
                    self.assertEqual(call['case_timeout'], timeout)
                    self.assertIn(
                        f'{timeout}s', shlex.split(self.option(call, '--run-wrapper'))
                    )
                self.assertIn(f'close to the {timeout}s timeout', result.stdout)
                self.assertIn(f'{duration}s, race-gfx950, race::passing', result.stdout)
                self.assertIn(
                    f'{duration}s, race-gfx1151, race::passing', result.stdout
                )
                self.assertIn('::warning::Near-timeout tests found.', result.stdout)

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
