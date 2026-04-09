import unittest
import os
import sys
from unittest.mock import patch, mock_open

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import abi_check


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

    def test_check_report_clean(self):
        html_content = "<html><body>No changes found.</body></html>"
        with patch("builtins.open", mock_open(read_data=html_content)):
            with patch("os.path.exists", return_value=True):
                self.assertTrue(abi_check.check_report_content("report.html"))

    def test_check_report_removed_symbols(self):
        html_content = "<html><body><h2>Removed Symbols (1)</h2></body></html>"
        with patch("builtins.open", mock_open(read_data=html_content)):
            with patch("os.path.exists", return_value=True):
                self.assertFalse(abi_check.check_report_content("report.html"))

    def test_check_report_missing_file(self):
        with patch("os.path.exists", return_value=False):
            self.assertFalse(abi_check.check_report_content("missing.html"))


if __name__ == "__main__":
    unittest.main()
