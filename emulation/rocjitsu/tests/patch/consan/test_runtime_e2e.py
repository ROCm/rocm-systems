# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check integration-driver preconditions without requiring an HSA device."""

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from consan_runtime_e2e import Driver, HardwareUnavailable


class RuntimeIntegration(unittest.TestCase):
    def test_preparation_failures_cannot_continue_the_pipeline(self):
        for status in (0, 2):
            with self.subTest(
                status=status
            ), tempfile.TemporaryDirectory() as directory:
                driver = Driver(
                    argparse.Namespace(root=Path(directory), case='disjoint')
                )
                command = ['prepare', 'original', 'target', 'entry', 'artifact']
                result = argparse.Namespace(
                    returncode=status, stdout='prepared', stderr='rejected'
                )
                with mock.patch(
                    'consan_runtime_e2e.subprocess.run', return_value=result
                ) as run:
                    if status:
                        with self.assertRaisesRegex(
                            AssertionError, 'expected exit 0, got 2'
                        ):
                            driver.run('prepare', command)
                    else:
                        self.assertEqual(driver.run('prepare', command), 'prepared')
                self.assertEqual(run.call_args.args[0], command)
                self.assertEqual(driver.commands[-1]['exit_status'], status)
                self.assertEqual(
                    (driver.directory / 'prepare.stderr').read_text(), 'rejected'
                )

    def test_missing_statistics_checks_the_requested_agent_first(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            source = build / 'source'
            source.mkdir()
            (build / 'CMakeCache.txt').write_text(
                f'CMAKE_HOME_DIRECTORY:INTERNAL={source}\n'
            )
            binaries = {}
            for name in ('prepare', 'runner', 'inspect', 'plugin'):
                binaries[name] = build / name
                binaries[name].write_bytes(b'test driver input')
            fixtures = build / 'fixtures'
            fixture = fixtures / 'targets' / 'gfx950' / 'wave64' / 'native-2.hsaco'
            fixture.parent.mkdir(parents=True)
            fixture.write_bytes(b'preflight fixture; never executed by this unit test')
            args = argparse.Namespace(
                **binaries,
                source=source,
                fixtures=fixtures,
                root=build / 'evidence',
                case='missing-statistics',
                fixture_kind='assembly',
                target='gfx950',
                wave_size=64,
                threads=128,
                launcher=None,
                test_statistics=False,
                amd_signals=False,
            )
            driver = Driver(args)
            with mock.patch.object(
                driver, 'run', side_effect=HardwareUnavailable
            ) as run:
                with self.assertRaises(HardwareUnavailable):
                    driver.execute()
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], 'original-baseline')
            self.assertIn('--baseline', run.call_args.args[1])
            self.assertFalse((driver.directory / 'artifact').exists())


if __name__ == '__main__':
    unittest.main()
