# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for rocprof-compute general CLI options.
"""

import argparse
from pathlib import Path
from unittest.mock import call, patch

import pytest
from common import SUPPORTED_ARCHS, patch_console

from argparser import CliHelpFormatter, DeprecatedAction, build_parser

HOME = Path.cwd()
VERSION = {"ver_pretty": "rocprof-compute (unit test)"}
REMOVAL = "is deprecated and will be removed in a future release."


def build_args(argv, experimental=False):
    """Parse argv with the rocprof-compute parser."""
    if experimental:
        argv = ["--experimental", *argv]
    return build_parser(HOME, SUPPORTED_ARCHS, VERSION, argv).parse_args(argv)


def help_text(home, argv, capsys):
    """Return the help output for argv with the given rocprof-compute home."""
    parser = build_parser(home, SUPPORTED_ARCHS, VERSION, argv)
    with pytest.raises(SystemExit):
        parser.parse_args([*argv, "--help"])
    return capsys.readouterr().out


def write_skills_readme(skills_dir):
    """Create a skills README and return its path."""
    skills_dir.mkdir(parents=True)
    readme = skills_dir / "README.md"
    readme.write_text("skills")
    return readme


def deprecated_option_parser():
    """Return a parser with a renamed option and two wholly deprecated options."""
    parser = argparse.ArgumentParser(formatter_class=CliHelpFormatter)
    parser.add_argument(
        "--new",
        "--old",
        dest="value",
        metavar="<value>",
        action=DeprecatedAction,
        base_action="store",
        deprecated_names=["--old"],
        help="Set the value.",
    )
    parser.add_argument(
        "--gone",
        action=DeprecatedAction,
        base_action="store_true",
        advice="Use --new instead.",
        help="Turn it on.",
    )
    parser.add_argument(
        "--retired",
        action=DeprecatedAction,
        base_action="store_true",
        help="Turn it off.",
    )
    return parser


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


# =============================================================================
# profile --kernel-iteration-range
# =============================================================================


def test_profile_kernel_iteration_range():
    args = build_args([
        "profile",
        "--kernel-iteration-range",
        "1",
        "3:5",
        "--",
        "./vcopy",
    ])
    assert args.kernel_iteration_range == ["1", "3:5"]


# =============================================================================
# analyze debug options
# =============================================================================


def test_analyze_debug_options():
    """Debug options work without --debug, and without -p."""
    defaults = build_args(["analyze", "-p", "/tmp/workload"])
    assert defaults.debug_metric_expression is False
    assert defaults.debug_table_view is False
    assert defaults.debug_verify_deps is False

    args = build_args([
        "analyze",
        "--debug-metric-expression",
        "--debug-table-view",
        "--debug-verify-deps",
    ])
    assert args.debug_metric_expression is True
    assert args.debug_table_view is True
    assert args.debug_verify_deps is True


def test_pc_sampling_analyze_options():
    """Defaults, overrides, and validation for the analyze PC sampling options."""
    defaults = build_args(["analyze"])
    assert defaults.pc_sampling_rows == 10

    overrides = build_args(["analyze", "--pc-sampling-rows", "25"])
    assert overrides.pc_sampling_rows == 25

    # 0 is allowed and means "show all kernels".
    assert build_args(["analyze", "--pc-sampling-rows", "0"]).pc_sampling_rows == 0

    # Negative row counts trigger an argparse error.
    with patch.object(
        argparse.ArgumentParser, "error", side_effect=SystemExit(2)
    ) as mock_error:
        with pytest.raises(SystemExit):
            build_args(["analyze", "--pc-sampling-rows", "-1"])
    mock_error.assert_called_once()


# =============================================================================
# ML API trace options
# =============================================================================


@pytest.mark.parametrize(
    "values, expected",
    [([], []), (["torch", "triton"], ["torch", "triton"])],
)
def test_profile_trace_ml_api_keeps_workload_command(values, expected):
    args = build_args(
        ["profile", "--trace-ml-api", *values, "--", "./vcopy"], experimental=True
    )
    assert args.trace_ml_api == expected
    assert args.remaining == ["--", "./vcopy"]


def test_analyze_operator_options_take_space_separated_values():
    args = build_args(
        [
            "analyze",
            "--list-operators",
            "torch",
            "user",
            "--operator",
            "torch:*conv*",
            "*relu*",
        ],
        experimental=True,
    )
    assert args.list_operators == ["torch", "user"]
    assert args.operator == ["torch:*conv*", "*relu*"]


@pytest.mark.parametrize(
    "argv",
    [
        ["profile", "--trace-ml-api", "user", "--", "./vcopy"],
        ["profile", "--trace-ml-api", "torch,triton", "--", "./vcopy"],
        ["analyze", "--list-operators", "pytorch"],
    ],
)
def test_ml_api_options_reject_unknown_values(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        build_args(argv, experimental=True)
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


# =============================================================================
# Deprecated option names
# =============================================================================


@pytest.mark.parametrize(
    ("argv", "warnings"),
    [
        (["--old", "1"], [call(f"--old {REMOVAL} Use --new instead.")]),
        (["--new", "1"], []),
        (["--gone"], [call(f"--gone {REMOVAL} Use --new instead.")]),
        (["--retired"], [call(f"--retired {REMOVAL}")]),
    ],
    ids=["old_name", "new_name", "whole_option", "whole_option_without_advice"],
)
def test_deprecated_action_warns_only_for_deprecated_names(monkeypatch, argv, warnings):
    mocks = patch_console(monkeypatch, "argparser", "warning")
    deprecated_option_parser().parse_args(argv)
    assert mocks["warning"].call_args_list == warnings


def test_deprecated_alias_fills_the_same_dest(monkeypatch):
    patch_console(monkeypatch, "argparser", "warning")
    assert deprecated_option_parser().parse_args(["--old", "1"]).value == "1"


def test_deprecated_action_adds_the_deprecation_to_the_help():
    help_text = deprecated_option_parser().format_help()
    assert "Set the value.\n" in help_text
    assert "Deprecated alias: --old." in help_text
    assert "DEPRECATED: Use --new instead." in help_text
    assert "DEPRECATED: Will be removed in a future release." in help_text


# =============================================================================
# Help notation
# =============================================================================


@pytest.mark.parametrize(
    ("option_strings", "nargs", "metavar", "expected"),
    [
        (["-b", "--block"], "+", "<ids>", "-b, --block <ids>..."),
        (["--operator"], "*", "patterns", "--operator [patterns]..."),
        (["-t", "--time-unit"], None, "<unit>", "-t, --time-unit <unit>"),
        (["--view"], "?", "view", "--view [view]"),
    ],
    ids=["one_or_more", "zero_or_more", "one_value", "optional_value"],
)
def test_cli_help_formatter_shows_option_names_and_argument_once(
    option_strings, nargs, metavar, expected
):
    parser = argparse.ArgumentParser(formatter_class=CliHelpFormatter)
    parser.add_argument(*option_strings, nargs=nargs, metavar=metavar)
    assert expected in parser.format_help()


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
        base_action="store_true",
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
        base_action="store_true",
        help="Custom Help",
    )

    # Parse args - should succeed and print warning
    parser.parse_args()

    # Verify warning was logged
    assert "Test experimental feature" in caplog.text
    assert "experimental" in caplog.text.lower()
    assert "may change in future releases" in caplog.text.lower()


@pytest.mark.experimental_feature
def test_experimental_option_works_with_experimental_before_workload():
    argv = ["--experimental", "profile", "--pc-sampling", "-n", "test", "--", "./app"]
    args = build_parser(HOME, SUPPORTED_ARCHS, VERSION, argv).parse_args(argv)
    assert args.pc_sampling


@pytest.mark.experimental_feature
def test_experimental_option_fails_with_experimental_in_workload(capsys):
    argv = ["profile", "--pc-sampling", "-n", "test", "--", "./app", "--experimental"]
    with pytest.raises(SystemExit) as exc:
        build_parser(HOME, SUPPORTED_ARCHS, VERSION, argv).parse_args(argv)
    assert exc.value.code == 2
    assert "experimental feature" in capsys.readouterr().err.lower()


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
        base_action="store_true",
        help="Test help text",
    )

    # Get help text
    help_text = parser.format_help()

    # Help should be suppressed
    assert "--test-exp-feature" not in help_text, f"{help_text}"


# =============================================================================
# Agent Skills note in --help
# =============================================================================


@pytest.mark.parametrize("mode", [[], ["profile"], ["analyze"]])
def test_help_shows_skills_in_source_checkout(tmp_path, capsys, mode):
    readme = write_skills_readme(tmp_path / "skills")
    output = help_text(tmp_path / "src", mode, capsys)
    assert f"Agent Skills: see {readme}" in output


@pytest.mark.parametrize("mode", [[], ["profile"], ["analyze"]])
def test_help_shows_skills_in_install(tmp_path, capsys, mode):
    readme = write_skills_readme(tmp_path / "share" / "rocprofiler-compute" / "skills")
    home = tmp_path / "libexec" / "rocprofiler-compute"
    output = help_text(home, mode, capsys)
    assert f"Agent Skills: see {readme}" in output


def test_help_omits_skills_when_not_shipped(tmp_path, capsys):
    assert "Agent Skills" not in help_text(tmp_path / "src", [], capsys)
