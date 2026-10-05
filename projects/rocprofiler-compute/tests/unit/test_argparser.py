# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for rocprof-compute general CLI options.
"""

import argparse
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from common import SUPPORTED_ARCHS

from argparser import (
    DEPRECATED_OPTIONS,
    CommaListAction,
    apply_panel_shortcuts,
    non_negative_int,
    omniarg_parser,
    warn_deprecated_options,
)

HOME = Path.cwd()
VERSION = {"ver_pretty": "rocprof-compute (unit test)"}

DEPRECATED_ALIASES = [
    # (mode argv, old argv, new argv, dest, expected value)
    (["profile"], ["--roof-only"], ["--roofline"], "roofline", True),
    (["profile"], ["--bench-only"], ["--roofline-bench-only"], "bench_only", True),
    (["profile"], ["--device", "2"], ["--roofline-device", "2"], "device", 2),
    (["analyze"], ["--sort", "dispatches"], ["--roofline-sort", "dispatches"],
     "sort", "dispatches"),
    (["analyze"], ["--mem-level", "HBM"], ["--roofline-mem-level", "HBM"],
     "mem_level", ["HBM"]),
    (["analyze"], ["--roofline-data-type", "FP16"], ["--roofline-data-types", "FP16"],
     "roofline_data_type", ["FP16"]),
]  # fmt: skip

# Options whose (Default: ...) describes the default in words
DESCRIBED_DEFAULTS = {
    "--output-directory",
    "--config-dir",
    "--rocprofiler-sdk-tool-path",
    "--pc-sampling-interval",
    "--cols",
    "--view",
}
# "(Default: X)" is the default; "(Default when given without a value: X)" the const
HELP_DEFAULT_RE = re.compile(r"\(Default( when given without a value)?: ([^)]*)\)")


def build_args(argv, experimental=False):
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tool", usage="rocprof-compute [mode] [options]"
    )
    omniarg_parser(parser, HOME, SUPPORTED_ARCHS, VERSION, experimental)
    return parser.parse_args(argv)


def help_text(home, argv, capsys):
    """Return the help output for argv with the given rocprof-compute home."""
    parser = argparse.ArgumentParser(
        prog="tool",
        description="CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    omniarg_parser(parser, home, SUPPORTED_ARCHS, VERSION)
    with pytest.raises(SystemExit):
        parser.parse_args([*argv, "--help"])
    return capsys.readouterr().out


def write_skills_readme(skills_dir):
    """Create a skills README and return its path."""
    skills_dir.mkdir(parents=True)
    readme = skills_dir / "README.md"
    readme.write_text("skills")
    return readme


def all_parsers():
    """Return the top-level parser and the profile and analyze parsers."""
    parser = argparse.ArgumentParser()
    omniarg_parser(parser, HOME, SUPPORTED_ARCHS, VERSION, True)
    return [parser, *parser._subparsers._group_actions[0].choices.values()]


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
# analyze --verify-deps
# =============================================================================


def test_analyze_verify_deps_defaults_off():
    assert build_args(["analyze", "-p", "/tmp/workload"]).verify_deps is False


def test_analyze_verify_deps_needs_no_workload():
    """The flag is a standalone environment check, so -p stays optional."""
    assert build_args(["analyze", "--verify-deps"]).verify_deps is True


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
# Deprecated option names
# =============================================================================


@pytest.mark.parametrize(
    ("mode", "old", "new", "dest", "expected"),
    DEPRECATED_ALIASES,
    ids=[alias[1][0] for alias in DEPRECATED_ALIASES],
)
def test_deprecated_alias_matches_new_name(mode, old, new, dest, expected):
    assert getattr(build_args(mode), dest) != expected
    assert getattr(build_args(mode + new), dest) == expected
    assert getattr(build_args(mode + old), dest) == expected


@pytest.mark.parametrize("mode", ["profile", "analyze"])
def test_deprecated_aliases_listed_in_help(mode, capsys):
    with pytest.raises(SystemExit):
        build_args([mode, "--help"])
    out = capsys.readouterr().out
    for alias_mode, old, new, _, _ in DEPRECATED_ALIASES:
        if alias_mode == [mode]:
            assert f"{new[0]}, {old[0]}" in out


def test_deprecated_options_are_real_options():
    option_strings = {
        option for parser in all_parsers() for option in parser._option_string_actions
    }
    assert set(DEPRECATED_OPTIONS) <= option_strings


@pytest.mark.parametrize(
    ("argv", "workload", "warned"),
    [
        (["profile", "--roof-only"], [], ["--roof-only"]),
        (["analyze", "--mem-level=HBM", "--retain-rocpd-output"], [],
         ["--mem-level", "--retain-rocpd-output"]),
        # Options of the workload belong to the workload
        (["profile", "-n", "x", "--", "./app", "--device", "1"],
         ["--", "./app", "--device", "1"], []),
        (["profile", "--roofline-device", "1"], [], []),
    ],
    ids=["old_name", "equals_form", "workload_options", "new_name"],
)  # fmt: skip
def test_warn_deprecated_options(argv, workload, warned, caplog):
    warn_deprecated_options(argv, workload)
    for option in warned:
        assert f"{option} is deprecated" in caplog.text
    assert caplog.text.count("is deprecated") == len(warned)


# =============================================================================
# Help notation
# =============================================================================


def test_help_shows_option_metavars(capsys):
    with pytest.raises(SystemExit):
        build_args(["analyze", "--help"], experimental=True)
    out = capsys.readouterr().out
    assert "-b, --block <ids>..." in out
    assert "-t, --time-unit <unit>" in out
    assert "--torch-operator [patterns]..." in out
    assert "--list-metrics <arch>" in out
    assert "Values: ALL, HBM, L2, vL1D, L0, LDS" in out


def test_help_defaults_match_parser_defaults():
    """Each "(Default: X)" in the help is the option's real default."""
    for action in (a for parser in all_parsers() for a in parser._actions):
        name = action.option_strings[-1] if action.option_strings else action.dest
        if name in DESCRIBED_DEFAULTS:
            continue
        help_text = " ".join((action.help or "").split())
        for match in HELP_DEFAULT_RE.finditer(help_text):
            value = action.const if match.group(1) else action.default
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            assert match.group(2) == str(value), name


def test_profile_help_lists_conflicting_options(capsys):
    with pytest.raises(SystemExit):
        build_args(["profile", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    # Options in the same group as --roofline are not listed
    assert "Cannot be used with --set or --roofline-bench-only." in out
    assert (
        "Cannot be used with --block, --speed-of-light, --memory-chart, "
        "--roofline, --set or --no-roof." in out
    )


# =============================================================================
# Comma separated lists
# =============================================================================


@pytest.mark.parametrize(
    ("argv", "dest", "expected"),
    [
        (["-R", "FP16,FP32"], "roofline_data_type", ["FP16", "FP32"]),
        (["-R", "FP16", "FP32"], "roofline_data_type", ["FP16", "FP32"]),
        (["-R", "FP16", "-R", "FP32"], "roofline_data_type", ["FP16", "FP32"]),
        (["--cols", "0,2"], "cols", [0, 2]),
    ],
    ids=["comma", "space", "repeated", "item_type"],
)
def test_comma_list_forms_are_equivalent(argv, dest, expected):
    assert getattr(build_args(["analyze"] + argv), dest) == expected


@pytest.mark.parametrize(
    ("options", "value", "error"),
    [
        ({"item_choices": ["A", "B"]}, "A,C", "invalid choice: 'C'"),
        ({"item_type": int}, "1,x", "invalid int value: 'x'"),
        ({"item_type": non_negative_int}, "1,-2", "must be a non-negative integer"),
    ],
    ids=["choice", "value_error", "type_error_message"],
)
def test_comma_list_rejects_bad_items(options, value, error, capsys):
    parser = argparse.ArgumentParser(prog="rocprof-compute")
    parser.add_argument("--items", action=CommaListAction, **options)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--items", value])
    assert exc.value.code == 2
    assert f"argument --items: {error}" in capsys.readouterr().err


def test_comma_list_append_keeps_occurrences():
    args = build_args(["analyze", "-k", "1,2", "-k", "3", "-d", "4,5"])
    assert args.gpu_kernel == [[1, 2], [3]]
    assert args.gpu_dispatch_id == [["4", "5"]]


# =============================================================================
# --speed-of-light / --memory-chart / --roofline
# =============================================================================


@pytest.mark.parametrize(
    ("argv", "dest", "expected"),
    [
        (["profile", "-b", "5,sol", "--roofline"], "filter_blocks", ["5", "sol", "4"]),
        (["profile", "-b", "4", "--roofline"], "filter_blocks", ["4"]),
        (
            ["analyze", "--roofline", "--speed-of-light", "--memory-chart"],
            "filter_metrics",
            ["2", "3", "4"],
        ),
        # analyze treats filter_metrics=None as "show everything"
        (["analyze"], "filter_metrics", None),
    ],
    ids=["append_to_block", "no_duplicate", "all_three", "none_selected"],
)
def test_apply_panel_shortcuts(argv, dest, expected):
    args = build_args(argv)
    apply_panel_shortcuts(args, dest)
    assert getattr(args, dest) == expected


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
