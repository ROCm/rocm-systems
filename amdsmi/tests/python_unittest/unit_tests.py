#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import unittest

import sys
sys.path.append("/opt/rocm/libexec/amdsmi_cli/")

try:
    import amdsmi
except ImportError:
    raise ImportError("Could not import /opt/rocm/libexec/amdsmi_cli/amdsmi_cli.py")

class TestAmdSmiPythonBDF(unittest.TestCase):
    valid_bdfs = {
        "00:00.0": [0, 0, 0, 0],
        "01:01.1": [0, 1, 1, 1],
        "FF:1F.7": [0, 255, 31, 7],
        "FF:00.7": [0, 255, 0, 7],
        "11:01.2": [0, 17, 1, 2],
        "11:0a.2": [0, 17, 10, 2],
        "0000:FF:1F.7": [0, 255, 31, 7],
        "0001:ff:1F.7": [1, 255, 31, 7],
        "ffff:FF:1f.7": [65535, 255, 31, 7],
    }
    invalid_bdfs = {
        # invalid bdf strings, expect None
        None: None,
        "": None,
        "00:00:0": None,
        "00.00:0": None,
        "00:00.Z": None,
        "00:0Z.0": None,
        "0Z:00.0": None,
        "Z00:00.0": None,
        "A00:00.0": None,
        "0A00:00.0": None,
        "00:00.07": None,
        "00:00.8": None,
        "00:00.10": None,
        "00:00.11": None,
        "00:00.-1": None,
        "00:00.*-1": None,
        "00:00.123": None,
        "00:20.0": None,
        "00:45.0": None,
        "00:200.0": None,
        "00:002.0": None,
        "100:00.0": None,
        "0100:00.0": None,
        "00100:00.0": None,
        "0101:00.0": None,
        "00001:00.0": None,
        "10001:00.0": None,
        "45:0.0": None,
        ".00:00.0": None,
        "00.00.0": None,
        "00.0.0": None,
        "0.00.0": None,
        "000.00.0": None,
        "00 00 0": None,
        " 00:00.0": None,
        "00:00.0 ": None,
        "0000:00.00.0": None,
        "000:00:00.0": None,
        "00:00:00.1": None,
        "0:00:00.1": None,
        "0000 00 00 0": None,
        "-1-1:00:00.0": None,
        "AAAA:00:AA.0": None,
        "*1*1:00:00.0": None,
        "0000:00:00.07": None,
        "0000:00:00.8": None,
        "0000:00:00.10": None,
        "0000:00:00.11": None,
        "0000:00:00.-1": None,
        "0000:00:00.*-1": None,
        "0000:00:00.123": None,
        "0000:00:20.0": None,
        "0000:00:45.0": None,
        "0000:00:200.0": None,
        "0000:00:002.0": None,
        "0000:100:00.0": None,
        "0000:0100:00.0": None,
        "0000:00100:00.0": None,
        "0000:0101:00.0": None,
        "0000:00001:00.0": None,
        "0000:10001:00.0": None,
        "0000:45:0.0": None,
        ".0000.00:00.0": None,
        "0000.00.0.0": None,
        " 0000:00:00.0": None,
        "0000:00:00.0 ": None,
    }
    def test_parse_bdf(self):
        # go through all bdfs
        expectations = self.valid_bdfs.copy()
        expectations.update(self.invalid_bdfs)
        for bdf in expectations:
            expected = expectations[bdf]
            result = amdsmi.amdsmi_interface._parse_bdf(bdf)
            self.assertEqual(result, expected,
                             "Expected {} for bdf {}, but got {}".format(
                                 expected, bdf, result))
    @classmethod
    def _convert_bdf_to_long(clz, bdf):
        if len(bdf) == 12:
            return bdf
        if len(bdf) == 7:
            return "0000:" + bdf
        return None
    def test_format_bdf(self):
        # go through valid bdfs
        expectations = self.valid_bdfs.copy()
        for bdf_string in expectations:
            # use key as result and value as input
            bdf_list = expectations[bdf_string]
            smi_bdf = amdsmi.amdsmi_interface._make_amdsmi_bdf_from_list(bdf_list)
            expected = TestAmdSmiPythonBDF._convert_bdf_to_long(bdf_string)
            expected = expected.lower()
            result = amdsmi.amdsmi_interface._format_bdf(smi_bdf)
            self.assertEqual(result, expected,
                             "Expected {} for bdf {}, but got {}".format(
                                 expected, bdf_string, result))
    def test_check_res(self):
        # expect retry error to raise SmiRetryException
        with self.assertRaises(amdsmi.AmdSmiRetryException) as retry_test:
            amdsmi.amdsmi_interface._check_res(
                (lambda: amdsmi.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_RETRY)())
        # except retry error to have AMDSMI_STATUS_RETRY error code
        self.assertEqual(retry_test.exception.get_error_code(),
                         amdsmi.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_RETRY)
        # expect invalid args error to raise AmdSmiLibraryException
        with self.assertRaises(amdsmi.AmdSmiLibraryException) as inval_test:
            amdsmi.amdsmi_interface._check_res(
                (lambda: amdsmi.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_INVAL)())
        # expect invalid args error to have AMDSMI_STATUS_INVAL error code
        self.assertEqual(inval_test.exception.get_error_code(),
                         amdsmi.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_INVAL)
        # for successfull call, expect no error is given
        result = amdsmi.amdsmi_interface._check_res(
            (lambda: amdsmi.amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS)())
        self.assertEqual(None, result)
if __name__ == '__main__':
    unittest.main()
