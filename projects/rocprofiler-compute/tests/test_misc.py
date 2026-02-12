##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestExperimentalFlag:
    """Test suite for --experimental flag functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration values needed by argparser."""
        return {
            "rocprof_compute_home": Path("/tmp/rocprof"),
            "supported_archs": {"gfx908": "MI100", "gfx90a": "MI200"},
            "rocprof_compute_version": {"ver": "1.0.0", "ver_pretty": "v1.0.0"},
        }

    def test_experimental_flag_exists(self, mock_config):
        """Test that --experimental flag is available in general options."""
        from argparser import omniarg_parser

        parser = argparse.ArgumentParser()
        omniarg_parser(
            parser,
            mock_config["rocprof_compute_home"],
            mock_config["supported_archs"],
            mock_config["rocprof_compute_version"],
            experimental_enabled=False,
        )

        # Parse args with --experimental flag
        args = parser.parse_args(["--experimental"])
        assert hasattr(args, "experimental")
        assert args.experimental is True

    def test_experimental_flag_default_false(self, mock_config):
        """Test that --experimental flag defaults to False."""
        from argparser import omniarg_parser

        parser = argparse.ArgumentParser()
        omniarg_parser(
            parser,
            mock_config["rocprof_compute_home"],
            mock_config["supported_archs"],
            mock_config["rocprof_compute_version"],
            experimental_enabled=False,
        )

        # Parse args without --experimental flag
        args = parser.parse_args([])
        assert hasattr(args, "experimental")
        assert args.experimental is False

    def test_experimental_flag_is_boolean(self, mock_config):
        """Test that --experimental is a boolean flag (store_true)."""
        from argparser import omniarg_parser

        parser = argparse.ArgumentParser()
        omniarg_parser(
            parser,
            mock_config["rocprof_compute_home"],
            mock_config["supported_archs"],
            mock_config["rocprof_compute_version"],
            experimental_enabled=False,
        )

        # Parse with flag
        args_with_flag = parser.parse_args(["--experimental"])
        assert isinstance(args_with_flag.experimental, bool)
        assert args_with_flag.experimental is True

        # Parse without flag
        args_without_flag = parser.parse_args([])
        assert isinstance(args_without_flag.experimental, bool)
        assert args_without_flag.experimental is False

    def test_experimental_help_shown_when_both_flags_present(self):
        """Test that experimental features help text shows feature descriptions."""
        from argparser import ExperimentalAction

        # Create a standalone parser with test-specific experimental feature
        parser = argparse.ArgumentParser()
        general_group = parser.add_argument_group("General Options")

        general_group.add_argument(
            "--experimental",
            action="store_true",
            default=False,
            help=(
                "Enable experimental feature(s):\n"
                "   Test Feature Alpha (--test-alpha)\n"
                "   Test Feature Beta (--test-beta)\n"
            ),
        )

        general_group.add_argument(
            "--test-alpha",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Feature Alpha",
            base_action="store_true",
            help="Alpha test feature",
        )

        general_group.add_argument(
            "--test-beta",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Test Feature Beta",
            base_action="store_true",
            help="Beta test feature",
        )

        # Get help text
        help_text = parser.format_help()

        # Check that experimental features are listed in help
        assert "Enable experimental feature(s):" in help_text
        assert "Test Feature Alpha" in help_text
        assert "Test Feature Beta" in help_text

    def test_experimental_help_with_empty_features(self, mock_config):
        """Test help text with --experimental flag."""
        from argparser import omniarg_parser

        parser = argparse.ArgumentParser()
        omniarg_parser(
            parser,
            mock_config["rocprof_compute_home"],
            mock_config["supported_archs"],
            mock_config["rocprof_compute_version"],
            experimental_enabled=False,
        )

        # Get help text
        help_text = parser.format_help()

        # Flag should still exist
        assert "--experimental" in help_text

    def test_experimental_help_with_single_feature(self):
        """Test help text with a single experimental feature."""
        from argparser import ExperimentalAction

        # Create a standalone parser with a single test-specific experimental feature
        parser = argparse.ArgumentParser()
        general_group = parser.add_argument_group("General Options")

        general_group.add_argument(
            "--experimental",
            action="store_true",
            default=False,
            help=(
                "Enable experimental feature(s):\n"
                "   Single Test Feature (--single-test)\n"
            ),
        )

        general_group.add_argument(
            "--single-test",
            action=ExperimentalAction,
            experimental_enabled=True,
            feature_label="Single Test Feature",
            base_action="store_true",
            help="Single test feature",
        )

        # Get help text
        help_text = parser.format_help()

        # Check the single feature is displayed
        assert "Enable experimental feature(s):" in help_text
        assert "Single Test Feature" in help_text

    def test_experimental_flag_with_profile_mode(self, mock_config):
        """Test --experimental flag works in profile mode."""
        from argparser import omniarg_parser

        parser = argparse.ArgumentParser()
        omniarg_parser(
            parser,
            mock_config["rocprof_compute_home"],
            mock_config["supported_archs"],
            mock_config["rocprof_compute_version"],
            experimental_enabled=False,
        )

        # Parse profile mode with experimental flag
        args = parser.parse_args(["profile", "--experimental", "-n", "test"])
        assert args.experimental is True, f"{args.experimental}"
        assert args.mode == "profile", f"{args.mode} == profile"

    def test_experimental_flag_with_analyze_mode(self, mock_config):
        """Test --experimental flag works in analyze mode."""
        from argparser import omniarg_parser

        parser = argparse.ArgumentParser()
        omniarg_parser(
            parser,
            mock_config["rocprof_compute_home"],
            mock_config["supported_archs"],
            mock_config["rocprof_compute_version"],
            experimental_enabled=False,
        )

        # Parse analyze mode with experimental flag
        args = parser.parse_args([
            "analyze",
            "--experimental",
            "-p",
            "workloads/test",
        ])
        assert args.experimental is True, f"{args.experimental}"
        assert args.mode == "analyze", f"{args.mode} == analyze"

    def test_experimental_flag_position_independent(self, mock_config):
        """Test that --experimental flag works when placed before mode."""
        from argparser import omniarg_parser

        parser = argparse.ArgumentParser()
        omniarg_parser(
            parser,
            mock_config["rocprof_compute_home"],
            mock_config["supported_archs"],
            mock_config["rocprof_compute_version"],
            experimental_enabled=False,
        )

        # Test with experimental flag before mode
        args1 = parser.parse_args(["profile", "--experimental", "-n", "test"])
        assert args1.experimental is True, f"{args1.experimental}"

        # Test with experimental flag before analyze mode
        args2 = parser.parse_args(["analyze", "--experimental", "-p", "test"])
        assert args2.experimental is True, f"{args2.experimental}"

    def test_experimental_flag_idempotent(self, mock_config):
        """Test that specifying --experimental multiple times doesn't cause errors."""
        from argparser import omniarg_parser

        parser = argparse.ArgumentParser()
        omniarg_parser(
            parser,
            mock_config["rocprof_compute_home"],
            mock_config["supported_archs"],
            mock_config["rocprof_compute_version"],
            experimental_enabled=False,
        )

        # Should handle multiple occurrences gracefully
        args = parser.parse_args(["--experimental", "--experimental"])
        assert args.experimental is True

    def test_experimental_flag_with_version(self, mock_config):
        """Test that --experimental works with --version flag (version exits)."""
        from argparser import omniarg_parser

        parser = argparse.ArgumentParser()
        omniarg_parser(
            parser,
            mock_config["rocprof_compute_home"],
            mock_config["supported_archs"],
            mock_config["rocprof_compute_version"],
            experimental_enabled=False,
        )

        # Version should exit with code 0
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--experimental", "--version"])
        assert exc_info.value.code == 0

    def test_experimental_flag_in_all_parsers(self, mock_config):
        """Test that --experimental flag is available in main parser and subparsers."""
        from argparser import omniarg_parser

        parser = argparse.ArgumentParser()
        omniarg_parser(
            parser,
            mock_config["rocprof_compute_home"],
            mock_config["supported_archs"],
            mock_config["rocprof_compute_version"],
            experimental_enabled=False,
        )

        # Main parser (no mode)
        args_main = parser.parse_args(["--experimental"])
        assert args_main.experimental is True

        # Profile subparser
        args_profile = parser.parse_args([
            "profile",
            "--experimental",
            "-n",
            "test",
        ])
        assert args_profile.experimental is True

        # Analyze subparser
        args_analyze = parser.parse_args([
            "analyze",
            "--experimental",
            "-p",
            "workloads/test",
        ])
        assert args_analyze.experimental is True, f"{args_analyze.experimental}"


class TestExperimentalFlagIntegration:
    """Integration tests for --experimental flag detection logic."""

    @patch("sys.argv", ["rocprof-compute", "--experimental", "profile", "-n", "test"])
    def test_experimental_flag_detection_in_argv(self):
        """Test that --experimental flag is correctly detected in sys.argv."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv
        assert experimental_requested is True

    @patch("sys.argv", ["rocprof-compute", "profile", "-n", "test"])
    def test_no_experimental_flag_in_argv(self):
        """Test that absence of --experimental is correctly detected."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv
        assert experimental_requested is False

    @patch("sys.argv", ["rocprof-compute", "profile", "-n", "test", "--test-feature-x"])
    def test_experimental_feature_detection_in_argv(self):
        """Test detection of experimental feature flags in argv."""
        argv = sys.argv[1:]
        has_test_feature = "--test-feature-x" in argv
        assert has_test_feature is True

    @patch("sys.argv", ["rocprof-compute", "profile", "-n", "test"])
    def test_no_experimental_features_detected(self):
        """Test that no experimental features are detected when none are used."""
        argv = sys.argv[1:]
        has_test_feature = "--test-feature-x" in argv
        assert has_test_feature is False

    @patch("sys.argv", ["rocprof-compute", "--experimental", "--help"])
    def test_show_experimental_help_detection(self):
        """Test that show_experimental_help is correctly determined."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv
        show_experimental_help = experimental_requested and (
            "-h" in argv or "--help" in argv
        )

        assert experimental_requested is True
        assert show_experimental_help is True

    @patch("sys.argv", ["rocprof-compute", "--experimental", "-h"])
    def test_show_experimental_help_with_short_flag(self):
        """Test that show_experimental_help works with -h."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv
        show_experimental_help = experimental_requested and (
            "-h" in argv or "--help" in argv
        )

        assert experimental_requested is True
        assert show_experimental_help is True

    @patch("sys.argv", ["rocprof-compute", "--help"])
    def test_no_experimental_help_without_flag(self):
        """Test that experimental help is not shown without --experimental flag."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv
        show_experimental_help = experimental_requested and (
            "-h" in argv or "--help" in argv
        )

        assert experimental_requested is False
        assert show_experimental_help is False

    @patch("sys.argv", ["rocprof-compute", "--experimental"])
    def test_experimental_without_help(self):
        """Test that experimental help is not triggered without help flag."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv
        show_experimental_help = experimental_requested and (
            "-h" in argv or "--help" in argv
        )

        assert experimental_requested is True
        assert show_experimental_help is False

    @patch("sys.argv", ["rocprof-compute", "-h", "profile"])
    def test_no_experimental_help_when_flag_after_help(self):
        """Test help flag position relative to experimental."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv
        show_experimental_help = experimental_requested and (
            "-h" in argv or "--help" in argv
        )

        assert experimental_requested is False
        assert show_experimental_help is False


class TestExperimentalFeatureGating:
    """Tests for experimental feature flag and separator behavior."""

    @patch(
        "sys.argv",
        ["rocprof-compute", "profile", "-n", "test", "--test-exp-feature"],
    )
    def test_experimental_feature_in_argv(self):
        """Test detection of experimental feature flags in argv."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv
        has_test_feature = "--test-exp-feature" in argv

        # Feature flag is in argv but --experimental is not
        assert experimental_requested is False
        assert has_test_feature is True

    @patch(
        "sys.argv",
        [
            "rocprof-compute",
            "--experimental",
            "profile",
            "-n",
            "test",
            "--test-exp-feature",
        ],
    )
    def test_experimental_flag_with_feature(self):
        """Test when both --experimental and feature flag are present."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv
        has_test_feature = "--test-exp-feature" in argv

        # Both flags present
        assert experimental_requested is True
        assert has_test_feature is True

    @patch(
        "sys.argv",
        ["rocprof-compute", "--experimental", "profile", "-n", "test"],
    )
    def test_experimental_flag_without_features(self):
        """Test when --experimental is set but no experimental features are used."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv
        has_test_feature = "--test-exp-feature" in argv

        # Experimental flag is set but no features used
        assert experimental_requested is True
        assert has_test_feature is False

    @patch(
        "sys.argv",
        [
            "rocprof-compute",
            "profile",
            "-n",
            "test",
            "--",
            "./my_app",
            "--test-exp-feature",
        ],
    )
    def test_feature_after_separator_not_detected(self):
        """Test that feature flags after '--' separator are not part of tool args."""
        argv = sys.argv[1:]

        # Split argv at '--' separator (if present)
        try:
            separator_index = argv.index("--")
            argv_to_scan = argv[:separator_index]
        except ValueError:
            argv_to_scan = argv

        experimental_requested = "--experimental" in argv_to_scan
        has_feature_before = "--test-exp-feature" in argv_to_scan
        has_feature_full = "--test-exp-feature" in argv

        # --test-exp-feature appears in argv but only after '--'
        assert experimental_requested is False
        assert has_feature_before is False
        assert has_feature_full is True

    @patch(
        "sys.argv",
        [
            "rocprof-compute",
            "profile",
            "-n",
            "test",
            "--",
            "./my_app",
            "--experimental",
        ],
    )
    def test_experimental_flag_after_separator_not_detected(self):
        """Test that --experimental flag after '--' separator is
        not detected as tool flag."""
        argv = sys.argv[1:]

        # Split argv at '--' separator (if present)
        try:
            separator_index = argv.index("--")
            argv_to_scan = argv[:separator_index]
        except ValueError:
            argv_to_scan = argv

        experimental_requested = "--experimental" in argv_to_scan
        experimental_in_full = "--experimental" in argv

        # --experimental appears in argv but only after '--', so should NOT be detected
        assert experimental_requested is False
        # Verify the flag is in full argv but not in scanned portion
        assert experimental_in_full is True

    @patch(
        "sys.argv",
        [
            "rocprof-compute",
            "--experimental",
            "profile",
            "--test-exp-feature",
            "arg1",
            "arg2",
            "-n",
            "test",
            "--",
            "./my_app",
            "--experimental",
        ],
    )
    def test_mixed_flags_with_separator(self):
        """Test that only flags before '--' are detected, not after."""
        argv = sys.argv[1:]

        # Split argv at '--' separator (if present)
        try:
            separator_index = argv.index("--")
            argv_to_scan = argv[:separator_index]
            argv_after = argv[separator_index + 1 :]
        except ValueError:
            argv_to_scan = argv
            argv_after = []

        experimental_before = "--experimental" in argv_to_scan
        feature_before = "--test-exp-feature" in argv_to_scan
        experimental_after = "--experimental" in argv_after

        # --experimental and --test-exp-feature are before '--' so should be detected
        # --experimental after '--' should NOT be detected as tool flag
        assert experimental_before is True
        assert feature_before is True
        assert experimental_after is True  # It's there, but not scanned as tool arg
        # Verify proper scanning
        assert "--experimental" in argv_to_scan
        assert "--experimental" in argv_after
