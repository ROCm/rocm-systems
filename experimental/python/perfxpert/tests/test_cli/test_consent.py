"""Tests for `perfxpert.cli._consent` (Task 3, I10).

Covers:

* Round-trip grant + check for a (backend, cwd, file-set) triple.
* Invalidation when file-set changes.
* Invalidation when cwd changes.
* `PERFXPERT_ASSUME_CONSENT=1` env override skips the prompt.
* Non-TTY stdin without env override refuses with guidance.
* `XDG_CONFIG_HOME` is honored.
* HOME isolation — tests never touch the developer's real
  `~/.config/perfxpert/consent.yaml`.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from unittest import mock

import pytest

from perfxpert.cli import _consent
from perfxpert.cli._consent import (
    CONSENT_ASSUME_ENV,
    consent_path,
    cwd_hash,
    file_set_hash,
    grant_consent,
    has_consent,
    prompt_consent_interactive,
    revoke_consent,
)


# ---------------------------------------------------------------------------
# Fixtures — isolate HOME + XDG_CONFIG_HOME so no tests touch the real home.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HOME + XDG_CONFIG_HOME to a tmp dir.

    The consent cache lives under `$XDG_CONFIG_HOME/perfxpert/`; by
    redirecting both we guarantee the real developer home never sees
    test writes.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    return tmp_path


# ---------------------------------------------------------------------------
# Hash helpers.
# ---------------------------------------------------------------------------


def test_cwd_hash_stable_for_same_path(tmp_path: Path) -> None:
    h1 = cwd_hash(tmp_path)
    h2 = cwd_hash(tmp_path)
    assert h1 == h2
    assert len(h1) == 8


def test_cwd_hash_differs_across_paths(tmp_path: Path) -> None:
    h1 = cwd_hash(tmp_path)
    h2 = cwd_hash(tmp_path / "sub")
    assert h1 != h2


def test_file_set_hash_is_order_independent(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    h1 = file_set_hash([(a, True, False), (b, False, True)])
    h2 = file_set_hash([(b, False, True), (a, True, False)])
    assert h1 == h2


def test_file_set_hash_changes_on_tracked_flag(tmp_path: Path) -> None:
    """Critical I10 invariant: (path, tracked=False) -> (path, tracked=True)
    must invalidate consent because the user newly tracks the file in git.
    """
    a = tmp_path / "a"
    untracked = file_set_hash([(a, True, False)])
    tracked = file_set_hash([(a, True, True)])
    assert untracked != tracked


# ---------------------------------------------------------------------------
# Round-trip + invalidation.
# ---------------------------------------------------------------------------


def test_consent_round_trip(isolated_home: Path) -> None:
    fset = file_set_hash([(isolated_home / "CLAUDE.md", False, False)])
    assert has_consent("claude", isolated_home, fset) is False

    grant_consent("claude", isolated_home, fset)
    assert has_consent("claude", isolated_home, fset) is True


def test_consent_invalidated_when_file_set_changes(isolated_home: Path) -> None:
    fset1 = file_set_hash([(isolated_home / "CLAUDE.md", False, False)])
    fset2 = file_set_hash([(isolated_home / "CLAUDE.md", True, True)])

    grant_consent("claude", isolated_home, fset1)
    assert has_consent("claude", isolated_home, fset1) is True
    # New file-set = different consent key → re-prompt required.
    assert has_consent("claude", isolated_home, fset2) is False


def test_consent_invalidated_when_cwd_changes(
    isolated_home: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    fset = file_set_hash([(other / "CLAUDE.md", False, False)])

    grant_consent("claude", isolated_home, fset)
    # Different cwd → consent does NOT transfer.
    assert has_consent("claude", other, fset) is False


def test_consent_invalidated_across_backends(isolated_home: Path) -> None:
    fset = file_set_hash([(isolated_home / "AGENTS.md", False, False)])
    grant_consent("claude", isolated_home, fset)
    # Gemini consent stored separately.
    assert has_consent("gemini", isolated_home, fset) is False


def test_revoke_consent_removes_all_fset_variants(isolated_home: Path) -> None:
    f1 = file_set_hash([(isolated_home / "a", False, False)])
    f2 = file_set_hash([(isolated_home / "b", True, True)])
    grant_consent("claude", isolated_home, f1)
    grant_consent("claude", isolated_home, f2)
    assert has_consent("claude", isolated_home, f1)
    assert has_consent("claude", isolated_home, f2)

    revoke_consent("claude", isolated_home)
    assert has_consent("claude", isolated_home, f1) is False
    assert has_consent("claude", isolated_home, f2) is False


# ---------------------------------------------------------------------------
# Interactive prompt + env overrides.
# ---------------------------------------------------------------------------


def test_env_override_auto_grants(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONSENT_ASSUME_ENV, "1")
    buf = io.StringIO()
    assert (
        prompt_consent_interactive(
            "claude", isolated_home, ["install stuff"], stream=buf
        )
        is True
    )
    assert "auto-grant" in buf.getvalue()


@pytest.mark.parametrize("env_val", ["true", "YES", "1"])
def test_env_override_accepts_truthy_values(
    env_val: str, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONSENT_ASSUME_ENV, env_val)
    buf = io.StringIO()
    assert (
        prompt_consent_interactive("claude", isolated_home, [], stream=buf) is True
    )


def test_non_tty_refuses_without_env(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CONSENT_ASSUME_ENV, raising=False)
    # Force stdin.isatty to return False.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    buf = io.StringIO()
    assert (
        prompt_consent_interactive(
            "claude", isolated_home, ["action 1"], stream=buf
        )
        is False
    )
    assert CONSENT_ASSUME_ENV in buf.getvalue()
    assert "non-interactive" in buf.getvalue()


def test_interactive_yes_grants(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CONSENT_ASSUME_ENV, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "y")
    buf = io.StringIO()
    assert (
        prompt_consent_interactive(
            "claude", isolated_home, ["action 1"], stream=buf
        )
        is True
    )


def test_interactive_no_declines(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CONSENT_ASSUME_ENV, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "n")
    buf = io.StringIO()
    assert (
        prompt_consent_interactive(
            "claude", isolated_home, ["action 1"], stream=buf
        )
        is False
    )


def test_interactive_enter_declines(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default (bare Enter) should be the safe "no" path."""
    monkeypatch.delenv(CONSENT_ASSUME_ENV, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "")
    buf = io.StringIO()
    assert (
        prompt_consent_interactive("claude", isolated_home, ["x"], stream=buf)
        is False
    )


# ---------------------------------------------------------------------------
# XDG + HOME isolation.
# ---------------------------------------------------------------------------


def test_xdg_config_home_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xdg = tmp_path / "custom_xdg"
    xdg.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    # HOME should be irrelevant once XDG is set.
    monkeypatch.setenv("HOME", str(tmp_path / "some_other_home"))

    p = consent_path()
    assert str(xdg) in str(p)
    assert p.name == "consent.yaml"


def test_home_isolation_defaults_to_dot_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With HOME set and XDG_CONFIG_HOME unset, consent lives under HOME/.config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    p = consent_path()
    assert str(tmp_path) in str(p)
    assert ".config/perfxpert/consent.yaml" in str(p)


def test_grant_does_not_touch_other_home(
    isolated_home: Path,
) -> None:
    """Sanity: grant writes only under the configured cache path."""
    fset = file_set_hash([(isolated_home / "x", False, False)])
    grant_consent("claude", isolated_home, fset)
    path = consent_path()
    assert path.is_file()
    # Must live under the isolated tmp tree.
    assert str(path).startswith(str(isolated_home)), path


def test_consent_file_is_yaml(isolated_home: Path) -> None:
    import yaml

    fset = file_set_hash([(isolated_home / "x", False, False)])
    grant_consent("claude", isolated_home, fset)
    data = yaml.safe_load(consent_path().read_text())
    assert isinstance(data, dict)
    assert len(data) == 1
    record = next(iter(data.values()))
    assert record["backend"] == "claude"
    assert record["cwd_hash"] == cwd_hash(isolated_home)
    assert record["file_set_hash"] == fset


def test_corrupt_cache_is_treated_as_empty(
    isolated_home: Path,
) -> None:
    """If the YAML file gets corrupted, we should not crash — just
    start fresh (the user will re-prompt)."""
    p = consent_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(": not yaml\n" * 50)
    fset = file_set_hash([(isolated_home / "x", False, False)])
    assert has_consent("claude", isolated_home, fset) is False
