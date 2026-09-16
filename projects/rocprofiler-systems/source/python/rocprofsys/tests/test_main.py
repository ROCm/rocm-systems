# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Lightweight unit tests for rocprofsys/__main__.py's pure-Python argument
parsing and environment handling. These do not require libpyrocprofsys (the
native bindings) to be built, and don't spawn a subprocess.
"""

import importlib.util
import os
from pathlib import Path

import pytest

_MAIN_PY = Path(__file__).resolve().parent.parent / "__main__.py"
_spec = importlib.util.spec_from_file_location("rocprofsys_main_under_test", _MAIN_PY)
rocprofsys_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rocprofsys_main)


# The functions under test mutate os.environ directly, so snapshot and restore
# it around every test to keep them order-independent.
@pytest.fixture(autouse=True)
def _restore_environ():
    saved = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


# =============================================================================
# find_script
# =============================================================================


def test_find_script_direct_path(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("pass\n")
    assert rocprofsys_main.find_script(str(script)) == str(script)


def test_find_script_via_path_env(tmp_path, monkeypatch):
    script = tmp_path / "on_path.py"
    script.write_text("pass\n")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert rocprofsys_main.find_script("on_path.py") == str(script)


def test_find_script_missing_raises(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(SystemExit):
        rocprofsys_main.find_script("does_not_exist.py")


# =============================================================================
# get_value
# =============================================================================


def test_get_value_uses_arg_when_given(monkeypatch):
    monkeypatch.delenv("SOME_VAR", raising=False)
    assert rocprofsys_main.get_value("SOME_VAR", 1, int, arg="5") == 5


def test_get_value_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("SOME_VAR", "7")
    assert rocprofsys_main.get_value("SOME_VAR", 1, int) == 7


def test_get_value_falls_back_to_default_and_sets_env(monkeypatch):
    monkeypatch.delenv("SOME_VAR", raising=False)
    assert rocprofsys_main.get_value("SOME_VAR", 3, int) == 3
    assert os.environ["SOME_VAR"] == "3"


# =============================================================================
# apply_config_option
# =============================================================================


def test_apply_config_option_noop_when_no_config(monkeypatch):
    monkeypatch.delenv("ROCPROFSYS_CONFIG_FILE", raising=False)
    rocprofsys_main.apply_config_option(None)
    assert "ROCPROFSYS_CONFIG_FILE" not in os.environ


def test_apply_config_option_sets_env_var(monkeypatch):
    monkeypatch.delenv("ROCPROFSYS_CONFIG_FILE", raising=False)
    rocprofsys_main.apply_config_option("cli.cfg")
    assert os.environ["ROCPROFSYS_CONFIG_FILE"] == "cli.cfg"


def test_apply_config_option_merges_with_existing_env_var(monkeypatch):
    monkeypatch.setenv("ROCPROFSYS_CONFIG_FILE", "env.cfg")
    rocprofsys_main.apply_config_option("cli.cfg")
    assert os.environ["ROCPROFSYS_CONFIG_FILE"] == "env.cfg" + os.pathsep + "cli.cfg"


@pytest.mark.parametrize("existing", ["env.cfg:", "env.cfg;", "env.cfg::"])
def test_apply_config_option_strips_trailing_separators(existing, monkeypatch):
    """A trailing separator in the env var must not produce an empty entry."""
    monkeypatch.setenv("ROCPROFSYS_CONFIG_FILE", existing)
    rocprofsys_main.apply_config_option("cli.cfg")
    assert os.environ["ROCPROFSYS_CONFIG_FILE"] == "env.cfg" + os.pathsep + "cli.cfg"


def test_apply_config_option_ignores_separator_only_env_var(monkeypatch):
    monkeypatch.setenv("ROCPROFSYS_CONFIG_FILE", ":")
    rocprofsys_main.apply_config_option("cli.cfg")
    assert os.environ["ROCPROFSYS_CONFIG_FILE"] == "cli.cfg"


# =============================================================================
# parse_args
# =============================================================================


def test_parse_args_defaults_are_none():
    opts = rocprofsys_main.parse_args([])
    assert opts.config is None
    assert opts.verbosity is None
    assert opts.full_filepath is None
    assert opts.label is None
    assert opts.trace_c is None
    assert opts.annotate_trace is None


@pytest.mark.parametrize("flag", ["-c", "--config"])
def test_parse_args_config_flag(flag):
    opts = rocprofsys_main.parse_args([flag, "my.cfg"])
    assert opts.config == "my.cfg"


def test_parse_args_rejects_abbreviations():
    """Abbreviated flags are rejected, since the parser sets allow_abbrev=False."""
    with pytest.raises(SystemExit):
        rocprofsys_main.parse_args(["--conf", "my.cfg"])
    with pytest.raises(SystemExit):
        rocprofsys_main.parse_args(["--full", "true"])


def test_parse_args_label_accepts_multiple_values():
    opts = rocprofsys_main.parse_args(["--label", "args", "file"])
    assert opts.label == ["args", "file"]


def test_parse_args_label_rejects_invalid_choice():
    with pytest.raises(SystemExit):
        rocprofsys_main.parse_args(["--label", "bogus"])


def test_parse_args_verbosity_int_conversion():
    opts = rocprofsys_main.parse_args(["-v", "10"])
    assert opts.verbosity == 10


# =============================================================================
# parse_args: str2bool options (-F/--full-filepath, --trace-c, -a/--annotate-trace)
# =============================================================================


@pytest.mark.parametrize(
    "flag", ["-F", "--full-filepath", "--trace-c", "-a", "--annotate-trace"]
)
def test_parse_args_bool_flag_alone_defaults_to_true(flag):
    dest = {
        "-F": "full_filepath",
        "--full-filepath": "full_filepath",
        "--trace-c": "trace_c",
        "-a": "annotate_trace",
        "--annotate-trace": "annotate_trace",
    }[flag]
    opts = rocprofsys_main.parse_args([flag])
    assert getattr(opts, dest) is True


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("0", False),
    ],
)
def test_parse_args_bool_flag_explicit_value(value, expected):
    opts = rocprofsys_main.parse_args(["--trace-c", value])
    assert opts.trace_c is expected


def test_parse_args_bool_flag_rejects_invalid_value():
    with pytest.raises(SystemExit):
        rocprofsys_main.parse_args(["--trace-c", "not-a-bool"])


# =============================================================================
# parse_args: nargs="+" list options
# =============================================================================


@pytest.mark.parametrize(
    "flag,dest",
    [
        ("-I", "function_include"),
        ("--function-include", "function_include"),
        ("-E", "function_exclude"),
        ("--function-exclude", "function_exclude"),
        ("-R", "function_restrict"),
        ("--function-restrict", "function_restrict"),
        ("-MI", "module_include"),
        ("--module-include", "module_include"),
        ("-ME", "module_exclude"),
        ("--module-exclude", "module_exclude"),
        ("-MR", "module_restrict"),
        ("--module-restrict", "module_restrict"),
    ],
)
def test_parse_args_list_options_accept_multiple_values(flag, dest):
    opts = rocprofsys_main.parse_args([flag, "foo", "bar"])
    assert getattr(opts, dest) == ["foo", "bar"]


# =============================================================================
# parse_args: simple flags (-b/--builtin, -s/--setup)
# =============================================================================


def test_parse_args_builtin_flag():
    opts = rocprofsys_main.parse_args([])
    assert opts.builtin is False
    opts = rocprofsys_main.parse_args(["-b"])
    assert opts.builtin is True
    opts = rocprofsys_main.parse_args(["--builtin"])
    assert opts.builtin is True


def test_parse_args_setup_option():
    opts = rocprofsys_main.parse_args([])
    assert opts.setup is None
    opts = rocprofsys_main.parse_args(["-s", "setup.py"])
    assert opts.setup == "setup.py"
    opts = rocprofsys_main.parse_args(["--setup", "setup.py"])
    assert opts.setup == "setup.py"


# =============================================================================
# parse_args: combined invocation
# =============================================================================


def test_parse_args_combined_flags():
    opts = rocprofsys_main.parse_args(
        [
            "-c",
            "my.cfg",
            "-v",
            "5",
            "--builtin",
            "--label",
            "args",
            "line",
            "-I",
            "foo",
            "bar",
            "--trace-c",
            "false",
        ]
    )
    assert opts.config == "my.cfg"
    assert opts.verbosity == 5
    assert opts.builtin is True
    assert opts.label == ["args", "line"]
    assert opts.function_include == ["foo", "bar"]
    assert opts.trace_c is False
