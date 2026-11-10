# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
import subprocess
import unittest
import os
import sys
import tempfile
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
BINARY_CONSTANT = (
    "<div id='BinaryTab' class='tab'>\n"
    "<a name='Constant_Binary_Problems_Low'></a>\n"
    "<h2>Problems with Constants, Low Severity "
    "<span class='warning'>&nbsp;1&nbsp;</span></h2><hr/>\n"
    "<td>The value of constant <b>AMDSMI_MAX_MM_IP_COUNT</b> has been changed from <b>8</b> "
    "to <b>16</b>.</td>\n"
)
BINARY_STRING_CONSTANT = (
    "<div id='BinaryTab' class='tab'>\n"
    "<td>The value of constant <b>AMDSMI_TIME_FORMAT</b> has been changed from "
    "<b>&quot;%02d:%02d:%02d.%03d&quot;</b> to <b>&quot;%02d:%02d:%02d.%06d&quot;</b>.</td>\n"
)
SOURCE_VERSION_CONSTANT = (
    "<div id='BinaryTab' class='tab'>\n"
    "<div id='SourceTab' class='tab'>\n"
    + LOW_CONSTANT
    + "<td>The value of constant <b>AMDSMI_LIB_VERSION_MINOR</b> has been changed from <b>2</b> "
    "to <b>3</b>.</td>\n"
)
LOW_SIZE_CHANGE = (
    "<h2>Problems with Data Types, Low Severity "
    "<span class='warning'>&nbsp;5&nbsp;</span></h2><hr/>\n"
    "<span class='ext'>[+]</span> <span class='ttype'>struct</span> amdsmi_driver_info_t "
    "<span class='warning'>&nbsp;5&nbsp;</span></span>\n"
    "<td>Size of this type has been changed from <b>768</b> bytes to <b>1792</b> bytes.</td>\n"
)
HIGH = (
    "<a name='Type_Binary_Problems_High'></a>\n"
    "<h2>Problems with Data Types, High Severity "
    "<span class='failed'>&nbsp;1&nbsp;</span></h2><hr/>\n"
)
SYMBOLS_MEDIUM = (
    "<a name='Symbol_Binary_Problems_Medium'></a>\n"
    "<h2>Problems with Symbols, Medium Severity "
    "<span class='failed'>&nbsp;1&nbsp;</span></h2><hr/>\n"
)
LOW_TYPE_NOT_SIZE = (
    "<h2>Problems with Data Types, Low Severity "
    "<span class='warning'>&nbsp;1&nbsp;</span></h2><hr/>\n"
    "<span class='ext'>[+]</span> <span class='ttype'>enum</span> amdsmi_link_type_t "
    "<span class='warning'>&nbsp;1&nbsp;</span></span>\n"
    "<td>Value of member <b>AMDSMI_LINK_TYPE_UNKNOWN</b> has been changed from <b>4</b> "
    "to <b>5</b>.</td>\n"
)
SIZE_ROW_WITHOUT_TYPE = (
    "<h2>Problems with Data Types, Low Severity "
    "<span class='warning'>&nbsp;1&nbsp;</span></h2><hr/>\n"
    "<td>Size of this type has been changed from <b>8</b> bytes to <b>16</b> bytes.</td>\n"
)
NESTED_SIZE_CHANGE = (
    "<span class='ext'>[+]</span> <span class='ttype'>union</span> "
    "amdsmi_fabric_info_t::fabric_info_ <span class='warning'>&nbsp;1&nbsp;</span></span>\n"
    "<td>Size of this type has been changed from <b>244</b> bytes to <b>480</b> bytes.</td>\n"
)


class AbiCheckTest(unittest.TestCase):
    def test_prepare_headers_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = os.path.join(tmp, "projects/amdsmi/include/amd_smi/amdsmi.h")
            os.makedirs(os.path.dirname(header))
            with open(header, "wb") as f:
                f.write(b"new \xb5\n")

            def git_show(cmd, stdout):
                self.assertEqual(
                    cmd, ["git", "show", "origin/develop:projects/amdsmi/include/amd_smi/amdsmi.h"]
                )
                stdout.write(b"old \xb5\n")

            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with patch("subprocess.check_call", side_effect=git_show):
                    self.assertTrue(abi_check.prepare_headers("origin/develop", "projects/amdsmi"))
                with open(abi_check.OLD_HEADER, "rb") as f:
                    old = f.read()
                with open(abi_check.NEW_HEADER, "rb") as f:
                    new = f.read()
            finally:
                os.chdir(cwd)

        self.assertEqual(old, b"#define ENABLE_ESMI_LIB 1\nold \xb5\n")
        self.assertEqual(new, b"#define ENABLE_ESMI_LIB 1\nnew \xb5\n")

    @patch("os.path.exists")
    def test_prepare_headers_missing_new(self, mock_exists):
        mock_exists.return_value = False
        result = abi_check.prepare_headers("origin/develop", "projects/amdsmi")
        self.assertFalse(result)

    def test_prepare_headers_fails_without_old_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = os.path.join(tmp, "projects/amdsmi/include/amd_smi/amdsmi.h")
            os.makedirs(os.path.dirname(header))
            with open(header, "w") as f:
                f.write("new\n")
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with patch(
                    "subprocess.check_call", side_effect=subprocess.CalledProcessError(128, "git")
                ), contextlib.redirect_stdout(io.StringIO()):
                    self.assertFalse(abi_check.prepare_headers("deadbeef", "projects/amdsmi"))
            finally:
                os.chdir(cwd)

    @patch("subprocess.call", return_value=0)
    def test_run_abi_checker_major(self, mock_subprocess):
        abi_check.run_abi_checker(strict=False)

        args, _ = mock_subprocess.call_args
        cmd = args[0]
        self.assertIn("abi-compliance-checker", cmd)
        self.assertNotIn("-strict", cmd)
        self.assertIn("-lib", cmd)
        self.assertIn("amdsmi", cmd)

    @patch("subprocess.call", return_value=0)
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

    def test_check_report_constant_change(self):
        clean, out = self._check(SUMMARY + BINARY_CONSTANT)
        self.assertFalse(clean)
        self.assertIn("Value of AMDSMI_MAX_MM_IP_COUNT changed from 8 to 16", out)

    def test_check_report_version_constant_is_clean(self):
        self.assertTrue(self._check(SUMMARY + SOURCE_VERSION_CONSTANT)[0])

    def test_check_report_format_string_is_clean(self):
        self.assertTrue(self._check(SUMMARY + BINARY_STRING_CONSTANT)[0])

    def test_check_report_removed_symbols(self):
        self.assertFalse(self._check(SUMMARY + REMOVED)[0])

    def test_check_report_medium_problems(self):
        self.assertFalse(self._check(SUMMARY + MEDIUM)[0])

    def test_check_report_high_problems(self):
        self.assertFalse(self._check(SUMMARY + HIGH)[0])

    def test_check_report_symbol_problems(self):
        self.assertFalse(self._check(SUMMARY + SYMBOLS_MEDIUM)[0])

    def test_check_report_low_type_problem_other_than_size_is_clean(self):
        self.assertTrue(self._check(SUMMARY + LOW_TYPE_NOT_SIZE)[0])

    def test_check_report_size_row_does_not_inherit_earlier_type(self):
        clean, out = self._check(SUMMARY + LOW_SIZE_CHANGE + SIZE_ROW_WITHOUT_TYPE)
        self.assertFalse(clean)
        self.assertIn("Size of unknown type changed from 8 to 16 bytes", out)

    def test_check_report_without_marker_is_unusable(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self._check(ADDED)[0])

    def test_check_report_struct_size_change(self):
        clean, out = self._check(SUMMARY + LOW_SIZE_CHANGE)
        self.assertFalse(clean)
        self.assertIn("amdsmi_driver_info_t changed from 768 to 1792 bytes", out)

    def test_check_report_names_nested_type(self):
        clean, out = self._check(SUMMARY + LOW_SIZE_CHANGE + NESTED_SIZE_CHANGE)
        self.assertFalse(clean)
        self.assertIn("amdsmi_fabric_info_t::fabric_info_ changed from 244 to 480 bytes", out)

    def test_check_report_missing_file(self):
        with patch("os.path.exists", return_value=False), contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(abi_check.check_report_content("missing.html"))

    @patch("subprocess.check_output", return_value="0123abcd\n")
    def test_merge_base_is_the_fork_point(self, mock_output):
        self.assertEqual(abi_check.merge_base("develop"), "0123abcd")
        mock_output.assert_called_once_with(
            ["git", "merge-base", "develop", "HEAD"], universal_newlines=True
        )

    def test_merge_base_with_real_git(self):
        # Unmocked: a mocked check_output accepts keywords this interpreter rejects.
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True)
        self.assertEqual(abi_check.merge_base("HEAD"), head.strip())

    @patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git"))
    def test_merge_base_missing_is_an_error(self, _):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(abi_check.merge_base("develop"))

    def _main(self, returncode=0, report_clean=True, base="0123abcd"):
        argv = ["abi_check.py", "--base-ref", "develop", "--mode", "major"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(
            io.StringIO()
        ), patch.object(abi_check, "merge_base", return_value=base), patch.object(
            abi_check, "prepare_headers", return_value=True
        ) as prepare, patch.object(
            abi_check, "run_abi_checker", return_value=returncode
        ), patch.object(abi_check, "check_report_content", return_value=report_clean):
            abi_check.main()
        return prepare

    def _exit_code(self, **kwargs):
        with self.assertRaises(SystemExit) as ctx:
            self._main(**kwargs)
        return ctx.exception.code

    def test_main_compares_against_merge_base(self):
        prepare = self._main()
        prepare.assert_called_once_with("0123abcd", "projects/amdsmi")

    def test_main_fails_when_checker_reports_incompatible(self):
        self.assertEqual(self._exit_code(returncode=1), abi_check.EXIT_BREAK)

    def test_main_fails_when_report_has_problems(self):
        self.assertEqual(self._exit_code(report_clean=False), abi_check.EXIT_BREAK)

    def test_main_reports_checker_error_separately(self):
        self.assertEqual(self._exit_code(returncode=6), abi_check.EXIT_ERROR)

    def test_main_reports_unusable_report_separately(self):
        self.assertEqual(self._exit_code(report_clean=None), abi_check.EXIT_ERROR)

    def test_main_errors_without_merge_base(self):
        self.assertEqual(self._exit_code(base=None), abi_check.EXIT_ERROR)


if __name__ == "__main__":
    unittest.main()
