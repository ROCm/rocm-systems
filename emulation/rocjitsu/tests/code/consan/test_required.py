# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pin false-green prevention independently of a working simulator or GPU."""

import unittest
import xml.etree.ElementTree as ET

from run_required import (
    physical_names,
    required_names,
    validate_physical_registrations,
    validate_results,
    validate_selected,
)


def physical_registrations():
    tests = []
    for name in sorted(physical_names()):
        _, gpu, wave, kind, scenario = name.split('.')
        wave_size = int(wave.removeprefix('wave'))
        tests.append(
            {
                'name': name,
                'properties': [
                    {'name': 'LABELS', 'value': ['consan', 'physical-gpu']},
                    {'name': 'SKIP_RETURN_CODE', 'value': 77},
                    {'name': 'RESOURCE_LOCK', 'value': [f'consan-{gpu}']},
                ],
                'command': [
                    'python3',
                    'consan_e2e.py',
                    '--target',
                    gpu,
                    '--wave-size',
                    str(wave_size),
                    '--fixture-kind',
                    kind,
                    '--threads',
                    str(3 * wave_size + 1 if scenario == 'partial' else 2 * wave_size),
                    '--case',
                    'disjoint' if scenario == 'partial' else scenario,
                ],
            }
        )
    return tests


class RequiredCoverage(unittest.TestCase):
    def test_missing_physical_registration_fails(self):
        tests = physical_registrations()
        validate_physical_registrations(tests)
        for selected in ([], tests[1:]):
            with self.assertRaisesRegex(
                ValueError, 'Physical ConSan registrations missing'
            ):
                validate_physical_registrations(selected)

    def test_physical_registration_execution_contract_is_checked(self):
        name = 'ConSanPhysicalE2E.gfx1250.wave32.assembly.partial'
        for property_name, invalid in (
            ('LABELS', ['consan']),
            ('LABELS', ['physical-gpu', 'simulator']),
            ('DISABLED', True),
            ('SKIP_RETURN_CODE', 0),
            ('RESOURCE_LOCK', ['consan-gfx1100']),
        ):
            with self.subTest(property_name=property_name, invalid=invalid):
                tests = physical_registrations()
                test = next(test for test in tests if test['name'] == name)
                test['properties'] = [
                    item for item in test['properties'] if item['name'] != property_name
                ]
                test['properties'].append({'name': property_name, 'value': invalid})
                with self.assertRaisesRegex(
                    ValueError, 'invalid physical execution properties'
                ):
                    validate_physical_registrations(tests)
        for option, invalid in (
            ('--target', 'gfx1100'),
            ('--wave-size', '64'),
            ('--fixture-kind', 'hip'),
            ('--threads', '64'),
            ('--case', 'partial'),
        ):
            with self.subTest(option=option):
                tests = physical_registrations()
                test = next(test for test in tests if test['name'] == name)
                test['command'][test['command'].index(option) + 1] = invalid
                with self.assertRaisesRegex(ValueError, f'expected {option}'):
                    validate_physical_registrations(tests)
        for option in ('--launcher', '--require-agent'):
            with self.subTest(option=option):
                tests = physical_registrations()
                test = next(test for test in tests if test['name'] == name)
                test['command'].append(option)
                with self.assertRaisesRegex(ValueError, 'requests simulator execution'):
                    validate_physical_registrations(tests)

    def test_missing_or_empty_selection_fails(self):
        for names in (set(), required_names() - {next(iter(required_names()))}):
            with self.assertRaises(ValueError):
                validate_selected([{'name': name} for name in names])
        validate_selected([{'name': name} for name in required_names()])

    def test_missing_or_skipped_execution_fails(self):
        root = ET.Element('testsuite')
        with self.assertRaises(ValueError):
            validate_results(root)
        for name in required_names():
            ET.SubElement(root, 'testcase', name=name)
        validate_results(root)
        ET.SubElement(root[0], 'skipped')
        with self.assertRaises(ValueError):
            validate_results(root)


if __name__ == '__main__':
    unittest.main()
