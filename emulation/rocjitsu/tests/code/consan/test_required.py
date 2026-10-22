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


def paired_registrations():
    tests = []
    for name in sorted(physical_names()):
        # The validator owns agreement, not the CMake scenario's geometry or
        # fixture mapping. Use arbitrary shared values instead of repeating it.
        _, gpu, _, _, _ = name.split('.')
        command = [
            'python3',
            'driver.py',
            '--target',
            gpu,
            '--wave-size',
            '64',
            '--fixture-kind',
            'assembly',
            '--threads',
            '137',
            '--case',
            'shared-fixture',
        ]
        tests.append(
            {
                'name': name,
                'properties': [
                    {'name': 'LABELS', 'value': ['consan', 'physical-gpu']},
                    {'name': 'SKIP_RETURN_CODE', 'value': 77},
                    {'name': 'RESOURCE_LOCK', 'value': [f'consan-{gpu}']},
                ],
                'command': list(command),
            }
        )
        tests.append(
            {
                'name': name.replace('ConSanPhysical', 'ConSanSim', 1),
                'properties': [
                    {
                        'name': 'LABELS',
                        'value': ['consan', 'simulator', 'required-consan'],
                    }
                ],
                'command': command
                + [
                    '--launcher',
                    'rocjitsu',
                    '--config',
                    'config.json',
                    '--require-agent',
                ],
            }
        )
    return tests


def set_property(test, name, value):
    test['properties'] = [item for item in test['properties'] if item['name'] != name]
    test['properties'].append({'name': name, 'value': value})


class RequiredCoverage(unittest.TestCase):
    def test_missing_physical_registration_fails(self):
        tests = paired_registrations()
        validate_physical_registrations(tests)
        for selected in ([], tests[1:], tests[2:]):
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
                tests = paired_registrations()
                test = next(test for test in tests if test['name'] == name)
                set_property(test, property_name, invalid)
                with self.assertRaisesRegex(
                    ValueError, 'invalid physical execution properties'
                ):
                    validate_physical_registrations(tests)
        for option in ('--launcher', '--config', '--require-agent'):
            with self.subTest(option=option):
                tests = paired_registrations()
                test = next(test for test in tests if test['name'] == name)
                test['command'].append(option)
                with self.assertRaisesRegex(ValueError, 'requests simulator execution'):
                    validate_physical_registrations(tests)

    def test_simulator_twin_is_required_and_cannot_skip(self):
        name = 'ConSanSimE2E.gfx1250.wave32.assembly.partial'
        tests = paired_registrations()
        with self.assertRaisesRegex(ValueError, 'required simulator twin missing'):
            validate_physical_registrations(
                [test for test in tests if test['name'] != name]
            )
        for property_name, invalid in (
            ('LABELS', ['simulator']),
            ('LABELS', ['simulator', 'required-consan', 'physical-gpu']),
            ('DISABLED', True),
            ('SKIP_RETURN_CODE', 77),
        ):
            with self.subTest(property_name=property_name):
                tests = paired_registrations()
                twin = next(test for test in tests if test['name'] == name)
                set_property(twin, property_name, invalid)
                with self.assertRaisesRegex(
                    ValueError, 'invalid required simulator properties'
                ):
                    validate_physical_registrations(tests)
        for option, width in (
            ('--launcher', 2),
            ('--config', 2),
            ('--require-agent', 1),
        ):
            with self.subTest(option=option):
                tests = paired_registrations()
                twin = next(test for test in tests if test['name'] == name)
                offset = twin['command'].index(option)
                del twin['command'][offset : offset + width]
                with self.assertRaisesRegex(ValueError, f'expected one {option}'):
                    validate_physical_registrations(tests)

    def test_twins_must_agree_on_application_arguments(self):
        for option in (
            '--target',
            '--wave-size',
            '--fixture-kind',
            '--threads',
            '--case',
        ):
            with self.subTest(option=option):
                tests = paired_registrations()
                tests[0]['command'][tests[0]['command'].index(option) + 1] = 'different'
                with self.assertRaisesRegex(
                    ValueError, 'command differs from required simulator twin'
                ):
                    validate_physical_registrations(tests)

    def test_twins_must_agree_on_nonbackend_flags(self):
        for flag in ('--utf8-entry', '--test-statistics'):
            with self.subTest(flag=flag):
                tests = paired_registrations()
                tests[0]['command'].append(flag)
                with self.assertRaisesRegex(
                    ValueError, 'command differs from required simulator twin'
                ):
                    validate_physical_registrations(tests)

    def test_additional_physical_registration_also_needs_a_twin(self):
        tests = paired_registrations()
        extra = dict(tests[0])
        extra['name'] = extra['name'].rsplit('.', 1)[0] + '.additional-case'
        tests.append(extra)
        with self.assertRaisesRegex(ValueError, 'required simulator twin missing'):
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
