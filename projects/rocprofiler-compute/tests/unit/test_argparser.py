# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for rocprof-compute general CLI options.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
from common import SUPPORTED_ARCHS

from argparser import omniarg_parser

HOME = Path.cwd()
VERSION = {"ver_pretty": "rocprof-compute (unit test)"}


def build_args(argv, experimental=False):
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tool", usage="rocprof-compute [mode] [options]"
    )
    omniarg_parser(parser, HOME, SUPPORTED_ARCHS, VERSION, experimental)
    return parser.parse_args(argv)


# =============================================================================
# -v / --version
# =============================================================================


@pytest.mark.parametrize("flag", ["-v", "--version"])
def test_version_success_exits_zero(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        build_args([flag])
    assert exc.value.code == 0
    assert "unit test" in capsys.readouterr().out


# =============================================================================
# -h / --help
# =============================================================================


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_success_exits_zero(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        build_args([flag])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_rejects_explicit_value(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        build_args([f"{flag}=now"])
    assert exc.value.code == 2
    assert "--help" in capsys.readouterr().err


# =============================================================================
# -V / --verbose
# =============================================================================


@pytest.mark.parametrize("flag", ["-V", "--verbose"])
def test_verbose_success_counts(flag):
    assert build_args([]).verbose == 0
    assert build_args([flag]).verbose == 1
    assert build_args([flag, flag, flag]).verbose == 3


@pytest.mark.parametrize("flag", ["-V", "--verbose"])
def test_verbose_rejects_explicit_value(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        build_args([f"{flag}=2"])
    assert exc.value.code == 2
    assert "--verbose" in capsys.readouterr().err


# =============================================================================
# -q / --quiet
# =============================================================================


@pytest.mark.parametrize("flag", ["-q", "--quiet"])
def test_quiet_success_sets_flag(flag):
    assert build_args([]).quiet is False
    assert build_args([flag]).quiet is True


@pytest.mark.parametrize("flag", ["-q", "--quiet"])
def test_quiet_rejects_explicit_value(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        build_args([f"{flag}=loud"])
    assert exc.value.code == 2
    assert "--quiet" in capsys.readouterr().err


# =============================================================================
# --config-dir
# =============================================================================


def test_config_dir_success_stores_value():
    assert build_args(["--config-dir", "/tmp/cfg"]).config_dir == "/tmp/cfg"


def test_config_dir_requires_value(capsys):
    with pytest.raises(SystemExit) as exc:
        build_args(["--config-dir"])
    assert exc.value.code == 2
    assert "--config-dir" in capsys.readouterr().err


# =============================================================================
# profile -d / --output-directory
# =============================================================================


@pytest.mark.parametrize("flag", ["-d", "--output-directory"])
def test_profile_output_directory(flag):
    args = build_args(["profile", flag, "/tmp/out", "--", "./vcopy"])
    assert args.output_directory == "/tmp/out"


def test_profile_rejects_dispatch():
    """--dispatch belongs to analyze mode only."""
    with pytest.raises(SystemExit) as exc:
        build_args(["profile", "--dispatch", "1", "--", "./vcopy"])
    assert exc.value.code == 2


def test_analyze_dispatch_unaffected():
    args = build_args(["analyze", "--path", "/tmp/wl", "-d", "3"])
    assert args.gpu_dispatch_id == [["3"]]


def test_pc_sampling_analyze_options():
    """Defaults, overrides, and validation for the analyze PC sampling options."""
    defaults = build_args(["analyze"])
    assert defaults.pc_sampling_sorting_type == "count"
    assert defaults.pc_sampling_rows == 10

    overrides = build_args([
        "analyze",
        "--pc-sampling-sorting-type",
        "offset",
        "--pc-sampling-rows",
        "25",
    ])
    assert overrides.pc_sampling_sorting_type == "offset"
    assert overrides.pc_sampling_rows == 25

    # 0 is allowed and means "show all rows".
    assert build_args(["analyze", "--pc-sampling-rows", "0"]).pc_sampling_rows == 0

    # Negative row counts trigger an argparse error.
    with patch.object(
        argparse.ArgumentParser, "error", side_effect=SystemExit(2)
    ) as mock_error:
        with pytest.raises(SystemExit):
            build_args(["analyze", "--pc-sampling-rows", "-1"])
    mock_error.assert_called_once()


# =============================================================================
# Experimental Feature Tests
# =============================================================================


@pytest.mark.experimental_feature
def test_experimental_feature_without_flag_errors(monkeypatch, capsys):
    """Test that using experimental feature without --experimental flag raises error."""
    import argparse

    from argparser import ExperimentalAction

    # Monkeypatch sys.argv to simulate command-line usage
    monkeypatch.setattr("sys.argv", ["rocprof-compute", "--test-exp-feature"])

    # Create a self-contained parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-exp-feature",
        action=ExperimentalAction,
        experimental_enabled=False,
        feature_label="Test experimental feature",
        base_action="store_const",
        nargs=0,
        const=True,
        default=False,
        help="Custom Help",
    )

    # Test that using experimental feature without --experimental causes error
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args()

    assert exc_info.value.code == 2  # argparse error exit code
    captured = capsys.readouterr()
    assert "experimental feature" in captured.err.lower()
    assert "--experimental" in captured.err.lower()


@pytest.mark.experimental_feature
def test_experimental_feature_with_flag_succeeds(monkeypatch, caplog):
    """Test that using experimental feature with --experimental flag succeeds."""
    import argparse

    from argparser import ExperimentalAction

    # Monkeypatch sys.argv to simulate command-line usage with --experimental
    monkeypatch.setattr("sys.argv", ["rocprof-compute", "--test-exp-feature"])

    # Create a self-contained parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-exp-feature",
        action=ExperimentalAction,
        experimental_enabled=True,
        feature_label="Test experimental feature",
        base_action="store_const",
        nargs=0,
        const=True,
        default=False,
        help="Custom Help",
    )

    # Parse args - should succeed and print warning
    parser.parse_args()

    # Verify warning was logged
    assert "Test experimental feature" in caplog.text
    assert "experimental" in caplog.text.lower()
    assert "may change in future releases" in caplog.text.lower()


@pytest.mark.experimental_feature
def test_experimental_flag_parsing_before_separator(monkeypatch, caplog):
    """Test that prelim parser correctly detects --experimental
    before '--' separator."""
    import argparse

    from argparser import ExperimentalAction

    # Monkeypatch sys.argv with --experimental before separator
    monkeypatch.setattr(
        "sys.argv",
        ["rocprof-compute", "--experimental", "profile", "-n", "test", "--", "./app"],
    )

    # Create a self-contained prelim parser
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument("--experimental", action="store_true", default=False)
    prelim_parser.parse_known_args()

    # Create full parser with experimental feature
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimental", action="store_true", default=False)
    parser.add_argument(
        "--test-exp-feature",
        action=ExperimentalAction,
        experimental_enabled=True,
        feature_label="Test experimental feature",
        base_action="store_const",
        nargs=0,
        const=True,
        default=False,
        help="Custom Help",
    )

    # Parse with just the experimental feature flag
    monkeypatch.setattr("sys.argv", ["rocprof-compute", "--test-exp-feature"])
    parser.parse_args()

    assert "experimental" in caplog.text.lower()


@pytest.mark.experimental_feature
def test_experimental_flag_parsing_after_separator(monkeypatch, capsys):
    """Test that prelim parser ignores --experimental after '--' separator."""
    import argparse

    from argparser import ExperimentalAction

    # Monkeypatch sys.argv with --experimental after separator
    monkeypatch.setattr(
        "sys.argv",
        ["rocprof-compute", "profile", "-n", "test", "--", "./app", "--experimental"],
    )

    # Create a self-contained prelim parser
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument("--experimental", action="store_true", default=False)
    prelim_parser.parse_known_args()

    # Create full parser with experimental feature
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-exp-feature",
        action=ExperimentalAction,
        experimental_enabled=False,
        feature_label="Test experimental feature",
        base_action="store_const",
        nargs=0,
        const=True,
        default=False,
        help="Custom Help",
    )

    with pytest.raises(SystemExit):
        parser.parse_args()

    captured = capsys.readouterr()
    assert "use --experimental" not in captured.err.lower()


@pytest.mark.experimental_feature
def test_experimental_flag_without_features(monkeypatch, capsys):
    """Test that --experimental flag is parsed correctly even without
    experimental features."""
    import argparse

    # Monkeypatch sys.argv with --experimental but no experimental features
    monkeypatch.setattr(
        "sys.argv", ["rocprof-compute", "--experimental", "profile", "-n", "test"]
    )

    # Create a self-contained parser with just --experimental flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimental", action="store_true", default=False)
    parser.add_argument("profile", nargs="?")
    parser.add_argument("-n", "--name", type=str)

    # Parse args - should succeed without errors since no experimental features used
    parser.parse_args()

    # Verify no errors or warnings
    captured = capsys.readouterr()
    assert captured.err == "", f"{captured.err}"


@pytest.mark.experimental_feature
def test_experimental_action_help_suppression():
    """Test that ExperimentalAction suppresses help when experimental_enabled=False."""
    import argparse

    from argparser import ExperimentalAction

    # Create parser without experimental enabled
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-exp-feature",
        action=ExperimentalAction,
        experimental_enabled=False,
        feature_label="Test experimental feature",
        base_action="store_const",
        nargs=0,
        const=True,
        default=False,
        help="Test help text",
    )

    # Get help text
    help_text = parser.format_help()

    # Help should be suppressed
    assert "--test-exp-feature" not in help_text, f"{help_text}"
