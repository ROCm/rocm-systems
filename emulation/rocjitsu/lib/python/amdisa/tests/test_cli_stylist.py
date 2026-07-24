# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Focused tests for the code generator's stylist integration."""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from amdisa.__main__ import _run_stylist


class CliStylistTest(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, '-m', 'amdisa', *arguments],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_run_stylist_passes_all_selected_output_paths(self) -> None:
        with mock.patch('amdisa.__main__.subprocess.run') as run:
            _run_stylist('isa-output', None, 'dbt-output')

        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]).name, 'stylist.py')
        self.assertEqual(command[1:], ['--format-only', 'isa-output', 'dbt-output'])
        self.assertEqual(run.call_args.kwargs, {'check': True})

    def test_run_stylist_reports_missing_command(self) -> None:
        with (
            mock.patch('amdisa.__main__.subprocess.run', side_effect=FileNotFoundError),
            self.assertRaisesRegex(RuntimeError, 'stylist not found'),
        ):
            _run_stylist('isa-output')

    def test_run_stylist_reports_exit_status(self) -> None:
        failure = subprocess.CalledProcessError(2, ['stylist'])
        with (
            mock.patch('amdisa.__main__.subprocess.run', side_effect=failure),
            self.assertRaisesRegex(RuntimeError, 'exit code 2'),
        ):
            _run_stylist('isa-output')

    def test_isa_generation_requires_output_path(self) -> None:
        result = self.run_cli('missing.xml')

        self.assertEqual(result.returncode, 2)
        self.assertIn('--isa-output is required', result.stderr)

    def test_multi_isa_dbt_generation_requires_output_path(self) -> None:
        result = self.run_cli(
            '--multi',
            'test:missing.xml',
            '--isa-output',
            'isa-output',
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn('--dbt-output is required', result.stderr)


if __name__ == '__main__':
    unittest.main()
