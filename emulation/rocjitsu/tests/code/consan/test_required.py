# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pin false-green prevention independently of a working simulator or GPU."""

import unittest
import xml.etree.ElementTree as ET

from run_required import required_names, validate_results, validate_selected


class RequiredCoverage(unittest.TestCase):
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
