# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
import subprocess
import unittest
import os
import sys
from unittest.mock import patch, mock_open

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import abi_check

# Fragments of real abi-compliance-checker 2.3 reports.
SUMMARY = (
    "<tr><th>Removed Symbols</th><td>High</td><td>0</td></tr>\n"
    "<tr><th>Problems with<br/>Data Types</th><td>High</td><td>0</td></tr>\n"
)
ADDED = (
    "<a name='Binary_Added'></a>"
    "<h2>Added Symbols <span class='new'>&nbsp;13&nbsp;</span></h2><hr/>\n"
)
REMOVED = (
    "<a name='Binary_Removed'></a><a name='Binary_Withdrawn'></a>"
    "<h2>Removed Symbols <span class='failed'>&nbsp;1&nbsp;</span></h2><hr/>\n"
)
MEDIUM = (
    "<a name='Type_Binary_Problems_Medium'></a>\n"
    "<h2>Problems with Data Types, Medium Severity "
    "<span class='failed'>&nbsp;2&nbsp;</span></h2><hr/>\n"
)
LOW_CONSTANT = (
    "<a name='Constant_Source_Problems_Low'></a>\n"
    "<h2>Problems with Constants, Low Severity "
    "<span class='warning'>&nbsp;1&nbsp;</span></h2><hr/>\n"
)
LOW_SIZE_CHANGE = (
    "<h2>Problems with Data Types, Low Severity "
    "<span class='warning'>&nbsp;5&nbsp;</span></h2><hr/>\n"
    "<span class='ext'>[+]</span> <span class='ttype'>struct</span> amdsmi_driver_info_t "
    "<span class='warning'>&nbsp;5&nbsp;</span></span>\n"
    "<td>Size of this type has been changed from <b>768</b> bytes to <b>1792</b> bytes.</td>\n"
)
NESTED_SIZE_CHANGE = (
    "<span class='ext'>[+]</span> <span class='ttype'>union</span> "
    "amdsmi_fabric_info_t::fabric_info_ <span class='warning'>&nbsp;1&nbsp;</span></span>\n"
    "<td>Size of this type has been changed from <b>244</b> bytes to <b>480</b> bytes.</td>\n"
)


class AbiCheckTest(unittest.TestCase):
    @patch("subprocess.check_call")
    @patch("os.path.exists")
    def test_prepare_headers_success(self, mock_exists, mock_subprocess):
        mock_exists.return_value = True

        with patch("builtins.open", mock_open()) as mock_file:
            result = abi_check.prepare_headers("origin/develop", "projects/amdsmi")

            self.assertTrue(result)
            mock_subprocess.assert_any_call(
                ["cp", "projects/amdsmi/include/amd_smi/amdsmi.h", "amdsmi_new.h"],
                shell=False,
                stdout=None,
                stderr=None,
            )

            git_call_found = False
            for args, kwargs in mock_subprocess.call_args_list:
                cmd_list = args[0]
                if cmd_list[0] == "git" and cmd_list[1] == "show":
                    git_call_found = True
                    self.assertIn(
                        "origin/develop:projects/amdsmi/include/amd_smi/amdsmi.h", cmd_list[2]
                    )

            self.assertTrue(git_call_found)

    @patch("os.path.exists")
    def test_prepare_headers_missing_new(self, mock_exists):
        mock_exists.return_value = False
        result = abi_check.prepare_headers("origin/develop", "projects/amdsmi")
        self.assertFalse(result)

    @patch("subprocess.check_call")
    def test_run_abi_checker_major(self, mock_subprocess):
        abi_check.run_abi_checker(strict=False)

        args, _ = mock_subprocess.call_args
        cmd = args[0]
        self.assertIn("abi-compliance-checker", cmd)
        self.assertNotIn("-strict", cmd)
        self.assertIn("-lib", cmd)
        self.assertIn("amdsmi", cmd)

    @patch("subprocess.check_call")
    def test_run_abi_checker_minor(self, mock_subprocess):
        abi_check.run_abi_checker(strict=True)

        args, _ = mock_subprocess.call_args
        cmd = args[0]
        self.assertIn("-strict", cmd)

    def _check(self, html_content):
        out = io.StringIO()
        with patch("builtins.open", mock_open(read_data=html_content)):
            with patch("os.path.exists", return_value=True), contextlib.redirect_stdout(out):
                return abi_check.check_report_content("report.html"), out.getvalue()

    def test_check_report_added_symbols_are_clean(self):
        self.assertTrue(self._check(SUMMARY + ADDED)[0])

    def test_check_report_low_constant_change_is_clean(self):
        self.assertTrue(self._check(SUMMARY + LOW_CONSTANT)[0])

    def test_check_report_removed_symbols(self):
        self.assertFalse(self._check(SUMMARY + REMOVED)[0])

    def test_check_report_medium_problems(self):
        self.assertFalse(self._check(SUMMARY + MEDIUM)[0])

    def test_check_report_struct_size_change(self):
        clean, out = self._check(SUMMARY + LOW_SIZE_CHANGE)
        self.assertFalse(clean)
        self.assertIn("amdsmi_driver_info_t changed from 768 to 1792 bytes", out)

    def test_check_report_names_nested_type(self):
        clean, out = self._check(LOW_SIZE_CHANGE + NESTED_SIZE_CHANGE)
        self.assertFalse(clean)
        self.assertIn("amdsmi_fabric_info_t::fabric_info_ changed from 244 to 480 bytes", out)

    def test_check_report_missing_file(self):
        with patch("os.path.exists", return_value=False):
            self.assertFalse(abi_check.check_report_content("missing.html"))

    @patch("subprocess.check_output", return_value="0123abcd\n")
    def test_merge_base_is_the_fork_point(self, mock_output):
        self.assertEqual(abi_check.merge_base("develop"), "0123abcd")
        mock_output.assert_called_once_with(["git", "merge-base", "develop", "HEAD"], text=True)

    @patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git"))
    def test_merge_base_falls_back_to_base_ref(self, _):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(abi_check.merge_base("develop"), "develop")

    def _main(self, tool_success, report_clean):
        argv = ["abi_check.py", "--base-ref", "develop", "--mode", "major"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(
            io.StringIO()
        ), patch.object(abi_check, "merge_base", return_value="0123abcd"), patch.object(
            abi_check, "prepare_headers", return_value=True
        ) as prepare, patch.object(
            abi_check, "run_abi_checker", return_value=tool_success
        ), patch.object(abi_check, "check_report_content", return_value=report_clean):
            abi_check.main()
        return prepare

    def test_main_compares_against_merge_base(self):
        prepare = self._main(tool_success=True, report_clean=True)
        prepare.assert_called_once_with("0123abcd", "projects/amdsmi")

    def test_main_fails_when_checker_reports_incompatible(self):
        with self.assertRaises(SystemExit) as ctx:
            self._main(tool_success=False, report_clean=True)
        self.assertEqual(ctx.exception.code, 1)

    def test_main_fails_when_report_has_problems(self):
        with self.assertRaises(SystemExit) as ctx:
            self._main(tool_success=True, report_clean=False)
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
