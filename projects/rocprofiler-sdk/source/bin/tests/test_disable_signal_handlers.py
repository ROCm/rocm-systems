# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""GPU-free tests for --disable-signal-handlers parsing and environment forwarding.

The option has two spellings that mean different things: with no value it leaves the
handlers installed and stops the tool keeping them in front of the application's, and
with a list of signals it installs no handler for those. Both halves are exported, so
these assert the environment the launcher hands to the application rather than the
parsed namespace.
"""

import json

import pytest

ENABLED = "ROCPROF_SIGNAL_HANDLERS"
DISABLED = "ROCPROF_SIGNAL_HANDLERS_DISABLED"


@pytest.fixture
def launch(rocprofv3, monkeypatch):
    environments = []
    monkeypatch.delenv(ENABLED, raising=False)
    monkeypatch.delenv(DISABLED, raising=False)
    monkeypatch.setattr(rocprofv3, "resolve_library_path", lambda path, args: path)
    monkeypatch.setattr(
        rocprofv3.os,
        "execvpe",
        lambda executable, argv, env: environments.append(env),
    )
    monkeypatch.setattr(
        rocprofv3.subprocess,
        "check_call",
        lambda argv, env: environments.append(env) or 0,
    )

    def run(*args):
        rocprofv3.main(["--kernel-trace", *args, "--", "/bin/true"])
        assert len(environments) == 1
        return environments[0]

    return run


def test_omitted_touches_neither_variable(launch):
    env = launch()
    assert ENABLED not in env
    assert DISABLED not in env


def test_bare_disables_interposition_only(launch):
    env = launch("--disable-signal-handlers")
    assert env[ENABLED] == "0"
    assert env[DISABLED] == ""


@pytest.mark.parametrize(
    "value,expected", [("true", "0"), ("false", "1"), ("1", "0"), ("0", "1")]
)
def test_boolean_spellings_are_preserved(launch, value, expected):
    env = launch("--disable-signal-handlers", value)
    assert env[ENABLED] == expected
    assert env[DISABLED] == ""


def test_named_signals_keep_interposition_for_the_rest(launch):
    env = launch("--disable-signal-handlers", "SIGINT")
    assert env[ENABLED] == "1"
    assert env[DISABLED] == "SIGINT"


def test_named_signals_are_sorted_and_deduplicated(launch):
    env = launch("--disable-signal-handlers", "SIGQUIT", "SIGINT", "SIGINT")
    assert env[DISABLED] == "SIGINT,SIGQUIT"


@pytest.mark.parametrize("spelling", ["SIGINT", "sigint", "sig_int", "int", "2"])
def test_signal_spellings_resolve(launch, spelling):
    assert launch("--disable-signal-handlers", spelling)[DISABLED] == "SIGINT"


def test_inherited_value_does_not_survive_a_bare_invocation(launch, monkeypatch):
    monkeypatch.setenv(DISABLED, "SIGTERM")
    assert launch("--disable-signal-handlers")[DISABLED] == ""


def test_inherited_value_is_left_alone_when_the_option_is_absent(launch, monkeypatch):
    # The export sits behind `disable_signal_handlers is not None`, so an invocation
    # that does not use the option leaves the caller's environment as it found it.
    monkeypatch.setenv(DISABLED, "SIGTERM")
    assert launch()[DISABLED] == "SIGTERM"


@pytest.mark.parametrize("name", ["SIGABRT", "SIGTERM"])
def test_logging_claimed_signals_are_reported(launch, capsys, name):
    env = launch("--disable-signal-handlers", name)
    assert env[DISABLED] == name
    assert "logging failure handler still claims it" in capsys.readouterr().err


@pytest.mark.parametrize("alias,canonical", [("SIGIOT", "SIGABRT"), ("iot", "SIGABRT")])
def test_aliases_resolve_to_the_signal_they_share_a_number_with(launch, alias, canonical):
    assert launch("--disable-signal-handlers", alias)[DISABLED] == canonical


def test_unhandled_signal_is_dropped_but_handled_ones_remain(launch):
    env = launch("--disable-signal-handlers", "SIGUSR1", "SIGINT")
    assert env[DISABLED] == "SIGINT"


def test_only_unhandled_signals_is_an_error(rocprofv3, capsys):
    # Nothing would be disabled, and the export would carry more signal handling than
    # the bare option.
    with pytest.raises(SystemExit) as exc:
        rocprofv3.parse_arguments(["--disable-signal-handlers", "SIGUSR1", "SIGUSR2"])
    assert exc.value.code == 2
    assert "installs a handler for none of" in capsys.readouterr().err


def test_comma_separated_is_an_error(rocprofv3, capsys):
    with pytest.raises(SystemExit) as exc:
        rocprofv3.parse_arguments(["--disable-signal-handlers", "SIGINT,SIGQUIT"])
    assert exc.value.code == 2
    # An unrecognized token also exits 2, so the message is what separates this branch
    # from the resolver rejecting the token whole.
    assert "space-separated" in capsys.readouterr().err


def test_input_file_list_is_rejected(rocprofv3, tmp_path, monkeypatch, capsys):
    # An input file bypasses the argparse action, so a list arrives as a list. Both
    # launch paths are stubbed, so `launched` records any attempt to start the
    # application.
    launched = []
    monkeypatch.setattr(rocprofv3, "resolve_library_path", lambda path, args: path)
    monkeypatch.setattr(
        rocprofv3.os, "execvpe", lambda executable, argv, env: launched.append(env)
    )
    monkeypatch.setattr(
        rocprofv3.subprocess,
        "check_call",
        lambda argv, env: launched.append(env) or 0,
    )
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"jobs": [{"disable_signal_handlers": ["SIGINT"]}]}))
    with pytest.raises(SystemExit) as exc:
        rocprofv3.main(["-i", str(path), "--", "/bin/true"])
    assert exc.value.code == 1
    assert "command line only" in capsys.readouterr().err
    assert not launched


@pytest.mark.parametrize("value", ["true", "false"])
def test_input_file_quoted_boolean_is_rejected(
    rocprofv3, tmp_path, monkeypatch, capsys, value
):
    # `not "false"` is False, so a quoted boolean used to export the true form.
    launched = []
    monkeypatch.setattr(rocprofv3, "resolve_library_path", lambda path, args: path)
    monkeypatch.setattr(
        rocprofv3.os, "execvpe", lambda executable, argv, env: launched.append(env)
    )
    monkeypatch.setattr(
        rocprofv3.subprocess,
        "check_call",
        lambda argv, env: launched.append(env) or 0,
    )
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"jobs": [{"disable_signal_handlers": value}]}))
    with pytest.raises(SystemExit) as exc:
        rocprofv3.main(["-i", str(path), "--", "/bin/true"])
    assert exc.value.code == 1
    assert "unquoted true or false" in capsys.readouterr().err
    assert not launched


@pytest.mark.parametrize("token", ["SIGTERN", "banana", "999"])
def test_non_signal_token_is_an_error(rocprofv3, token):
    with pytest.raises(SystemExit) as exc:
        rocprofv3.parse_arguments(["--disable-signal-handlers", token])
    assert exc.value.code == 2
