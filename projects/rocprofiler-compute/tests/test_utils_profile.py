# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for helpers in utils.utils_profile."""

from pathlib import Path
from typing import Any, NoReturn

import pytest

from utils import utils_profile
from utils.utils_profile import is_shell_target


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("bash", id="bare_bash"),
        pytest.param("/usr/bin/bash -c true", id="bash_dash_c"),
        pytest.param("sh script.sh", id="sh_with_script"),
        pytest.param("./run.sh", id="relative_shell_script"),
        pytest.param("job.bash", id="bash_extension"),
        pytest.param("RUN.SH", id="uppercase_extension"),
        pytest.param("/opt/tools/zsh", id="absolute_zsh_path"),
        pytest.param("fish -i", id="fish_shell"),
        pytest.param("tcsh", id="tcsh_basename"),
        pytest.param('"/path with spaces/run.sh"', id="quoted_path_with_spaces"),
        pytest.param("'/opt/my tools/zsh' -c true", id="quoted_shell_binary_path"),
        pytest.param('bash -c "echo hi"', id="bash_with_quoted_arg"),
    ],
)
def test_is_shell_target_true(target: str, mock_etc_shells: None) -> None:
    assert is_shell_target(target) is True


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("/bin/true", id="bin_true"),
        pytest.param("python3 app.py", id="python3_script"),
        pytest.param("./vcopy", id="custom_binary"),
        pytest.param("-- /bin/true", id="leading_dash_dash"),
        pytest.param("/usr/local/bin/my_app --flag", id="binary_with_flag"),
        pytest.param("python3 -m json.tool", id="python_module"),
        pytest.param("", id="empty_string"),
        pytest.param('"unterminated', id="malformed_quoting"),
    ],
)
def test_is_shell_target_false(target: str, mock_etc_shells: None) -> None:
    assert is_shell_target(target) is False


def test_is_shell_target_falls_back_to_extensions_when_etc_shells_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When /etc/shells is unavailable, well-known shells still work."""

    def raise_missing(self: Path, *_args: Any, **_kwargs: Any) -> NoReturn:
        raise FileNotFoundError(self)

    monkeypatch.setattr(Path, "read_text", raise_missing)
    utils_profile.shell_basenames.cache_clear()

    assert is_shell_target("./run.sh") is True
    assert is_shell_target("bash") is True


def test_well_known_shells_recognised_when_etc_shells_is_minimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """zsh/fish/tcsh must work even if /etc/shells lists only sh + bash."""

    def fake_read_text(self: Path, *_args: Any, **_kwargs: Any) -> str:
        return "/bin/sh\n/bin/bash\n"

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    utils_profile.shell_basenames.cache_clear()

    for cmd in ("zsh -c true", "fish", "tcsh", "csh script.csh", "ksh"):
        assert is_shell_target(cmd) is True, cmd


def test_etc_shells_parser_skips_comments_and_blank_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_read_text(self: Path, *_args: Any, **_kwargs: Any) -> str:
        return "# header comment\n\n/bin/sh\n   \n/bin/bash\n"

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    utils_profile.shell_basenames.cache_clear()

    assert utils_profile.read_etc_shells_basenames() == frozenset({"sh", "bash"})


def test_etc_shells_includes_arbitrary_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever /etc/shells advertises is treated as a shell."""

    def fake_read_text(self: Path, *_args: Any, **_kwargs: Any) -> str:
        return "/usr/bin/xonsh\n/usr/local/bin/elvish\n"

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    utils_profile.shell_basenames.cache_clear()

    assert is_shell_target("xonsh -c 'true'") is True
    assert is_shell_target("/usr/local/bin/elvish") is True
