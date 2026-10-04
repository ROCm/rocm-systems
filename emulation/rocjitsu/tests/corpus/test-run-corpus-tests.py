#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Test corpus budget forwarding without launching simulator workloads.
# Usage: python tests/corpus/test-run-corpus-tests.py

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]
SCRIPT = SOURCE / 'tests/corpus/run-corpus-tests.sh'


class ThreadBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        binary = self.root / 'bin'
        binary.mkdir()
        stubs = {
            'rocjitsu': '#!/bin/bash\nprintf "%s\\0" "$@" >> "$PREFLIGHT_LOG"\n',
            'pytest': (
                '#!/bin/bash\n'
                'printf "%s\\0" "$@" >> "$ARGV_LOG"\n'
                'printf "\\0" >> "$ARGV_LOG"\n'
                'if [[ "$*" == *--junitxml* && "${FORCE_RETRY:-0}" == 1 ]]; then exit 1; fi\n'
            ),
        }
        for name, content in stubs.items():
            path = binary / name
            path.write_text(content)
            path.chmod(0o755)
        symbolizer = self.root / 'lib/llvm/bin/llvm-symbolizer'
        symbolizer.parent.mkdir(parents=True)
        symbolizer.write_text('#!/bin/bash\nexit 0\n')
        symbolizer.chmod(0o755)
        (self.root / 'lib/libamdhip64.so').touch()
        self.env = dict(
            os.environ,
            PATH=f'{binary}:{os.environ["PATH"]}',
            ROCM_PATH=str(self.root),
            ROCJITSU_SOURCE_DIR=str(SOURCE),
            ARGV_LOG=str(self.root / 'argv'),
            PREFLIGHT_LOG=str(self.root / 'preflight'),
        )

    def run_script(self, *args):
        return subprocess.run(
            ['bash', str(SCRIPT), *args],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def assert_budget(self, argv, budget):
        self.assertNotIn('--rocjitsu-thread-budget', argv)
        if budget is None:
            self.assertNotIn('--cpu-thread-budget', argv)
        else:
            self.assertEqual(argv.count('--cpu-thread-budget'), 1)
            self.assertEqual(argv[argv.index('--cpu-thread-budget') + 1], budget)

    def test_budget_across_initial_runs_reruns_and_preflight(self):
        for budget in (None, '0', '3'):
            for sanitizer in ('none', 'clang-asan', 'gcc-asan'):
                with self.subTest(budget=budget, sanitizer=sanitizer):
                    (self.root / 'argv').write_bytes(b'')
                    (self.root / 'preflight').write_bytes(b'')
                    self.env['FORCE_RETRY'] = '1'
                    args = ['--rerun-failed', '--sanitizer', sanitizer]
                    if budget is not None:
                        args += ['--rocjitsu-thread-budget', budget]
                    result = self.run_script(*args)
                    # A successful retry must not hide the original failure.
                    self.assertEqual(result.returncode, 1, result.stderr)
                    calls = (self.root / 'argv').read_bytes().split(b'\0\0')
                    wrappers = []
                    for call in calls:
                        argv = [item.decode() for item in call.split(b'\0')]
                        if '--run-wrapper' in argv:
                            wrapper = shlex.split(argv[argv.index('--run-wrapper') + 1])
                            self.assert_budget(wrapper, budget)
                            wrappers.append(wrapper)
                    self.assertEqual(len(wrappers), 10)
                    preflight = (self.root / 'preflight').read_bytes()
                    if sanitizer == 'clang-asan':
                        self.assertTrue(preflight)
                        self.assert_budget(preflight.decode().split('\0'), budget)
                    else:
                        self.assertFalse(preflight)

    def test_invalid_budget(self):
        for value in ('', '-1', 'abc'):
            with self.subTest(value=value):
                result = self.run_script('--rocjitsu-thread-budget', value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('requires a non-negative integer', result.stderr)
        result = self.run_script('--rocjitsu-thread-budget')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--rocjitsu-thread-budget requires a value', result.stderr)


if __name__ == '__main__':
    unittest.main()
