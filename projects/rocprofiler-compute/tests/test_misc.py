##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

    @pytest.fixture
    def mock_experimental_features(self):
        """Mock experimental features for testing."""
        return [
            {"label": "Feature Alpha", "flags": ["--feature-alpha"]},
            {"label": "Feature Beta", "flags": ["--feature-beta", "--beta"]},
            {"label": "Feature Gamma", "flags": ["--feature-gamma"]},
        ]

    @pytest.fixture
    def empty_experimental_features(self):
        """Mock empty experimental features list."""
        return []

    @pytest.fixture
    def single_experimental_feature(self):
        """Mock single experimental feature."""
        return [
            {"label": "Test Feature", "flags": ["--test-feature"]},
        ]

    def test_experimental_flag_exists(self, mock_config):
        """Test that --experimental flag is available in general options."""
        with patch("argparser.EXPERIMENTAL_FEATURES", []):
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
        with patch("argparser.EXPERIMENTAL_FEATURES", []):
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
        with patch("argparser.EXPERIMENTAL_FEATURES", []):
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

    def test_experimental_help_shown_when_both_flags_present(
        self, mock_config, mock_experimental_features
    ):
        """Test that experimental features are shown in help when
        show_experimental_help=True."""
        with patch("argparser.EXPERIMENTAL_FEATURES", mock_experimental_features):
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

            # Check that experimental features are listed in help
            assert "Enable experimental feature(s):" in help_text

            # Verify all mocked experimental features appear in help
            for feature in mock_experimental_features:
                assert feature["label"] in help_text, (
                    f"{feature['label']} in {help_text}"
                )

    def test_experimental_help_with_empty_features(
        self, mock_config, empty_experimental_features
    ):
        """Test help text when no experimental features are registered."""
        with patch("argparser.EXPERIMENTAL_FEATURES", empty_experimental_features):
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

    def test_experimental_help_with_single_feature(
        self, mock_config, single_experimental_feature
    ):
        """Test help text with a single experimental feature."""
        with patch("argparser.EXPERIMENTAL_FEATURES", single_experimental_feature):
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

            # Check single feature is displayed
            assert "Test Feature" in help_text, f'"Test Feature" in {help_text}'

    def test_experimental_flag_with_profile_mode(self, mock_config):
        """Test --experimental flag works in profile mode."""
        with patch("argparser.EXPERIMENTAL_FEATURES", []):
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
        with patch("argparser.EXPERIMENTAL_FEATURES", []):
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
        with patch("argparser.EXPERIMENTAL_FEATURES", []):
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
        with patch("argparser.EXPERIMENTAL_FEATURES", []):
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
        with patch("argparser.EXPERIMENTAL_FEATURES", []):
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
        with patch("argparser.EXPERIMENTAL_FEATURES", []):
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

    @pytest.fixture
    def mock_experimental_features(self):
        """Mock experimental features for testing."""
        return [
            {"label": "Feature Alpha", "flags": ["--feature-alpha"]},
            {"label": "Feature Beta", "flags": ["--feature-beta", "--beta"]},
            {"label": "Feature Gamma", "flags": ["--feature-gamma"]},
        ]

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

    @patch("sys.argv", ["rocprof-compute", "profile", "-n", "test", "--feature-alpha"])
    def test_experimental_feature_detection_single_flag(
        self, mock_experimental_features
    ):
        """Test detection of a single experimental feature flag."""
        argv = sys.argv[1:]

        experimental_used_labels = [
            feat["label"]
            for feat in mock_experimental_features
            if any(flag in argv for flag in feat["flags"])
        ]

        assert len(experimental_used_labels) == 1
        assert "Feature Alpha" in experimental_used_labels

    @patch("sys.argv", ["rocprof-compute", "profile", "-n", "test", "--beta"])
    def test_experimental_feature_detection_alias_flag(
        self, mock_experimental_features
    ):
        """Test detection using an alias flag."""
        argv = sys.argv[1:]

        experimental_used_labels = [
            feat["label"]
            for feat in mock_experimental_features
            if any(flag in argv for flag in feat["flags"])
        ]

        assert len(experimental_used_labels) == 1
        assert "Feature Beta" in experimental_used_labels

    @patch(
        "sys.argv",
        [
            "rocprof-compute",
            "profile",
            "--feature-alpha",
            "--feature-gamma",
            "-n",
            "test",
        ],
    )
    def test_experimental_feature_detection_multiple_flags(
        self, mock_experimental_features
    ):
        """Test detection of multiple experimental features."""
        argv = sys.argv[1:]

        experimental_used_labels = [
            feat["label"]
            for feat in mock_experimental_features
            if any(flag in argv for flag in feat["flags"])
        ]

        assert len(experimental_used_labels) == 2
        assert "Feature Alpha" in experimental_used_labels
        assert "Feature Gamma" in experimental_used_labels

    @patch("sys.argv", ["rocprof-compute", "profile", "-n", "test"])
    def test_no_experimental_features_detected(self, mock_experimental_features):
        """Test that no features are detected when none are used."""
        argv = sys.argv[1:]

        experimental_used_labels = [
            feat["label"]
            for feat in mock_experimental_features
            if any(flag in argv for flag in feat["flags"])
        ]

        assert len(experimental_used_labels) == 0

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


class TestExperimentalFeatureRegistry:
    """Tests for EXPERIMENTAL_FEATURES registry validation."""

    @pytest.fixture
    def valid_experimental_features(self):
        """Valid experimental features structure."""
        return [
            {"label": "Feature One", "flags": ["--feature-one"]},
            {"label": "Feature Two", "flags": ["--feature-two", "--f2"]},
        ]

    @pytest.fixture
    def invalid_features_no_label(self):
        """Invalid: missing label."""
        return [
            {"flags": ["--feature-one"]},
        ]

    @pytest.fixture
    def invalid_features_no_flags(self):
        """Invalid: missing flags."""
        return [
            {"label": "Feature One"},
        ]

    @pytest.fixture
    def invalid_features_empty_flags(self):
        """Invalid: empty flags list."""
        return [
            {"label": "Feature One", "flags": []},
        ]

    def test_valid_registry_structure(self, valid_experimental_features):
        """Test that valid registry structure passes validation."""
        for feature in valid_experimental_features:
            assert isinstance(feature, dict)
            assert "label" in feature
            assert "flags" in feature
            assert isinstance(feature["label"], str)
            assert isinstance(feature["flags"], list)
            assert len(feature["flags"]) > 0
            for flag in feature["flags"]:
                assert isinstance(flag, str)
                assert flag.startswith("--")

    def test_registry_with_unique_labels(self, valid_experimental_features):
        """Test that all feature labels are unique."""
        labels = [feat["label"] for feat in valid_experimental_features]
        assert len(labels) == len(set(labels))

    def test_registry_with_unique_flags(self, valid_experimental_features):
        """Test that all feature flags are unique."""
        all_flags = []
        for feat in valid_experimental_features:
            all_flags.extend(feat["flags"])
        assert len(all_flags) == len(set(all_flags))

    def test_flags_follow_naming_convention(self, valid_experimental_features):
        """Test that flags follow naming conventions."""
        for feature in valid_experimental_features:
            for flag in feature["flags"]:
                # Must start with --
                assert flag.startswith("--")
                # No spaces
                assert " " not in flag
                # Lowercase and hyphens
                assert flag.islower() or "-" in flag

    def test_empty_registry_is_valid(self):
        """Test that an empty registry is valid."""
        empty_features = []
        assert isinstance(empty_features, list)
        assert len(empty_features) == 0


class TestExperimentalFeatureGating:
    """Tests for experimental feature gating logic."""

    @pytest.fixture
    def mock_experimental_features(self):
        """Mock experimental features for testing."""
        return [
            {"label": "Advanced Mode", "flags": ["--advanced-mode"]},
            {"label": "Debug Feature", "flags": ["--debug-feature", "--dbg"]},
        ]

    @patch(
        "sys.argv",
        ["rocprof-compute", "profile", "-n", "test", "--advanced-mode"],
    )
    def test_feature_used_without_experimental_flag(self, mock_experimental_features):
        """Test detection when feature is used without --experimental flag."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv

        experimental_used_labels = [
            feat["label"]
            for feat in mock_experimental_features
            if any(flag in argv for flag in feat["flags"])
        ]

        # Feature is used but experimental flag is not set
        assert experimental_requested is False
        assert len(experimental_used_labels) == 1
        assert "Advanced Mode" in experimental_used_labels

    @patch(
        "sys.argv",
        [
            "rocprof-compute",
            "--experimental",
            "profile",
            "-n",
            "test",
            "--advanced-mode",
        ],
    )
    def test_feature_used_with_experimental_flag(self, mock_experimental_features):
        """Test detection when feature is used with --experimental flag."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv

        experimental_used_labels = [
            feat["label"]
            for feat in mock_experimental_features
            if any(flag in argv for flag in feat["flags"])
        ]

        # Feature is used and experimental flag is set
        assert experimental_requested is True
        assert len(experimental_used_labels) == 1
        assert "Advanced Mode" in experimental_used_labels

    @patch(
        "sys.argv",
        ["rocprof-compute", "--experimental", "profile", "-n", "test"],
    )
    def test_experimental_flag_without_features(self, mock_experimental_features):
        """Test when --experimental is set but no features are used."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv

        experimental_used_labels = [
            feat["label"]
            for feat in mock_experimental_features
            if any(flag in argv for flag in feat["flags"])
        ]

        # Experimental flag is set but no features used
        assert experimental_requested is True
        assert len(experimental_used_labels) == 0

    @patch(
        "sys.argv",
        [
            "rocprof-compute",
            "--experimental",
            "profile",
            "--advanced-mode",
            "--dbg",
            "-n",
            "test",
        ],
    )
    def test_multiple_features_with_experimental_flag(self, mock_experimental_features):
        """Test multiple features with --experimental flag."""
        argv = sys.argv[1:]
        experimental_requested = "--experimental" in argv

        experimental_used_labels = [
            feat["label"]
            for feat in mock_experimental_features
            if any(flag in argv for flag in feat["flags"])
        ]

        assert experimental_requested is True
        assert len(experimental_used_labels) == 2
        assert "Advanced Mode" in experimental_used_labels
        assert "Debug Feature" in experimental_used_labels

    @patch(
        "sys.argv",
        [
            "rocprof-compute",
            "profile",
            "-n",
            "test",
            "--",
            "./my_app",
            "--advanced-mode",
        ],
    )
    def test_workload_args_after_separator_not_detected(
        self, mock_experimental_features
    ):
        """Test that arguments after '--' separator are not
        scanned for experimental features."""
        argv = sys.argv[1:]

        # Split argv at '--' separator (if present)
        try:
            separator_index = argv.index("--")
            argv_to_scan = argv[:separator_index]
        except ValueError:
            argv_to_scan = argv

        experimental_requested = "--experimental" in argv_to_scan

        experimental_used_labels = [
            feat["label"]
            for feat in mock_experimental_features
            if any(flag in argv_to_scan for flag in feat["flags"])
        ]

        # --advanced-mode appears in argv but only after '--', so should NOT be detected
        assert experimental_requested is False
        assert len(experimental_used_labels) == 0
        # Verify the flag is in full argv but not in scanned portion
        assert "--advanced-mode" in argv
        assert "--advanced-mode" not in argv_to_scan

    @patch(
        "sys.argv",
        [
            "rocprof-compute",
            "--experimental",
            "profile",
            "--advanced-mode",
            "-n",
            "test",
            "--",
            "./my_app",
            "--dbg",
        ],
    )
    def test_mixed_experimental_features_with_separator(
        self, mock_experimental_features
    ):
        """Test that only features before '--' are detected, not after."""
        argv = sys.argv[1:]

        # Split argv at '--' separator (if present)
        try:
            separator_index = argv.index("--")
            argv_to_scan = argv[:separator_index]
        except ValueError:
            argv_to_scan = argv

        experimental_requested = "--experimental" in argv_to_scan

        experimental_used_labels = [
            feat["label"]
            for feat in mock_experimental_features
            if any(flag in argv_to_scan for flag in feat["flags"])
        ]

        # --advanced-mode is before '--' so should be detected
        # --dbg is after '--' so should NOT be detected
        assert experimental_requested is True
        assert len(experimental_used_labels) == 1
        assert "Advanced Mode" in experimental_used_labels
        assert "Debug Feature" not in experimental_used_labels
        # Verify both flags exist in full argv
        assert "--advanced-mode" in argv
        assert "--dbg" in argv
        # But only one is in the scanned portion
        assert "--advanced-mode" in argv_to_scan
        assert "--dbg" not in argv_to_scan
