"""Tests for perfxpert-code launcher subcommand dispatch.

Covers two phases of review findings:

* Review-finding I4 (phase 7): `perfxpert-code --help` must surface the
  perfxpert-owned subcommands (`doctor`, `analyze`, `config`, `providers`)
  so users can discover them, not silently forward to opencode.
* Issue 2 (phase 8): `perfxpert-code doctor` errored with
  ``Failed to change directory to .../doctor`` because opencode
  interpreted ``doctor`` as a positional CWD. The launcher now routes
  known subcommands explicitly. See
  ``docs/superpowers/plans/2026-04-18-perfxpert-phase8-pr2-user-issues.md``.
"""

from __future__ import annotations

import pytest

from perfxpert.cli import opencode_launcher
from perfxpert.cli.opencode_launcher import (
    _OPENCODE_SUBCOMMANDS,
    _PERFXPERT_DISPATCH_SUBCOMMANDS,
    _PERFXPERT_SUBCOMMANDS,
    main,
    route_subcommand,
)


# ---------------------------------------------------------------------------
# --help / -h handling (review-finding I4).
# ---------------------------------------------------------------------------


class TestHelpFlag:
    """Bare `perfxpert-code --help` must print the perfxpert-owned banner.

    Per review I4: help flag discovery must list doctor / analyze / config
    / providers BEFORE falling through to opencode's generic help.
    """

    @pytest.mark.parametrize("flag", ["--help", "-h"])
    def test_help_flag_prints_perfxpert_banner_and_lists_subcommands(
        self, flag, capsys, monkeypatch
    ):
        """--help / -h before any subcommand prints the perfxpert help banner."""
        # Force resolve_opencode_binary to fail so we exit early after the banner
        # instead of spawning the real opencode process.
        def _no_binary():
            raise FileNotFoundError("opencode binary not bundled in this test")

        monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", _no_binary)

        rc = opencode_launcher.main([flag])
        out = capsys.readouterr().out

        assert rc == 0, "help should exit 0 even when opencode binary absent"
        # The perfxpert subcommand list must be visible.
        for sub in ("analyze", "config", "doctor", "providers"):
            assert sub in out, f"help output must mention {sub!r}"
        # And the branding line must be there.
        assert "perfxpert" in out.lower() or "PerfXpert" in out

    def test_help_flag_after_subcommand_is_passthrough(self, monkeypatch):
        """`perfxpert-code run --help` is NOT a perfxpert-owned help request;
        the positional 'run' comes first, so the flag should fall through to
        opencode without the perfxpert banner short-circuiting discovery.
        """
        # The helper boolean is False → perfxpert banner not printed.
        assert opencode_launcher._help_flag_precedes_subcommand(["run", "--help"]) is False
        assert opencode_launcher._help_flag_precedes_subcommand(["stats", "-h"]) is False

    def test_help_flag_before_subcommand_is_perfxpert_owned(self):
        """`perfxpert-code --help run` treats --help as perfxpert's own."""
        assert opencode_launcher._help_flag_precedes_subcommand(["--help", "run"]) is True
        assert opencode_launcher._help_flag_precedes_subcommand(["-h"]) is True

    def test_help_flag_with_only_flags_preceding_is_still_help(self):
        """`perfxpert-code --verbose --help` — verbose is a flag, not a positional."""
        assert (
            opencode_launcher._help_flag_precedes_subcommand(
                ["--verbose", "--help"]
            )
            is True
        )

    def test_help_flag_missing_returns_false(self):
        assert opencode_launcher._help_flag_precedes_subcommand([]) is False
        assert opencode_launcher._help_flag_precedes_subcommand(["run"]) is False

    def test_perfxpert_subcommands_registry_is_non_empty(self):
        """The perfxpert subcommand catalog must list at least doctor + analyze."""
        subs = opencode_launcher._PERFXPERT_SUBCOMMANDS
        assert "doctor" in subs, "doctor must be listed for review I4"
        assert "analyze" in subs
        # Each description must be a non-empty string so `--help` is useful.
        for name, desc in subs.items():
            assert isinstance(desc, str) and desc.strip(), name


# ---------------------------------------------------------------------------
# route_subcommand() — pure function, no side effects.
# ---------------------------------------------------------------------------


def test_route_empty_argv_is_default() -> None:
    kind, out = route_subcommand([])
    assert kind == "opencode_default"
    assert out == []


def test_route_doctor_is_perfxpert_owned() -> None:
    kind, out = route_subcommand(["doctor"])
    assert kind == "perfxpert"
    assert out == ["doctor"]


@pytest.mark.parametrize("sub", sorted(_OPENCODE_SUBCOMMANDS))
def test_route_known_opencode_subcommand(sub: str) -> None:
    kind, out = route_subcommand([sub])
    assert kind == "opencode_subcommand", f"{sub!r} must be recognized"
    assert out == [sub]


def test_route_unknown_positional_is_default() -> None:
    # Backward compat: passing a project path should still go to opencode
    # as the default (interactive) mode, which opencode treats as CWD.
    kind, out = route_subcommand(["/tmp/my-project"])
    assert kind == "opencode_default"
    assert out == ["/tmp/my-project"]


def test_route_flags_are_skipped_when_finding_first_positional() -> None:
    # `--print-logs stats` => stats is the first positional and must route.
    kind, out = route_subcommand(["--print-logs", "stats"])
    assert kind == "opencode_subcommand"
    assert out == ["--print-logs", "stats"]


def test_route_run_with_prompt_routes_as_subcommand() -> None:
    kind, out = route_subcommand(["run", "explain this kernel"])
    assert kind == "opencode_subcommand"
    assert out == ["run", "explain this kernel"]


def test_dispatch_set_contains_doctor() -> None:
    """Guardrail: doctor must short-circuit before opencode resolution."""
    assert "doctor" in _PERFXPERT_DISPATCH_SUBCOMMANDS


# ---------------------------------------------------------------------------
# main() — end-to-end dispatch, mocked at subprocess.run.
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


def test_main_doctor_invokes_python_m_perfxpert(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProc(0)

    monkeypatch.setattr("perfxpert.cli.opencode_launcher.subprocess.run", _fake_run)
    rc = main(["doctor"])
    assert rc == 0
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[1:] == ["-m", "perfxpert", "doctor"], (
        "expected python -m perfxpert doctor; got " + repr(cmd)
    )


def test_main_stats_is_passthrough_without_cwd_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Issue 2: `perfxpert-code stats` must NOT be interpreted as CWD."""
    # Fake binary + config dir so resolve_* succeeds.
    fake_bin = tmp_path / "opencode"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    fake_cfg = tmp_path / "cfg"
    fake_cfg.mkdir()

    monkeypatch.setattr(
        "perfxpert.cli.opencode_launcher.resolve_opencode_binary",
        lambda: fake_bin,
    )
    monkeypatch.setattr(
        "perfxpert.cli.opencode_launcher.resolve_config_dir",
        lambda: fake_cfg,
    )
    # Banner suppression.
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProc(0)

    monkeypatch.setattr("perfxpert.cli.opencode_launcher.subprocess.run", _fake_run)
    rc = main(["stats"])
    assert rc == 0
    cmd = captured["cmd"]
    assert cmd[0] == str(fake_bin)
    assert cmd[1:] == ["stats"]
    # Critical: opencode subcommands must run from the user's CWD, NOT the
    # bundled runtime_cfg_dir. Our wrapper passes cwd=None in that branch.
    assert captured["kwargs"].get("cwd") is None  # type: ignore[union-attr]


def test_main_run_passes_prompt_through(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_bin = tmp_path / "opencode"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    fake_cfg = tmp_path / "cfg"
    fake_cfg.mkdir()

    monkeypatch.setattr(
        "perfxpert.cli.opencode_launcher.resolve_opencode_binary", lambda: fake_bin,
    )
    monkeypatch.setattr(
        "perfxpert.cli.opencode_launcher.resolve_config_dir", lambda: fake_cfg,
    )
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProc(0)

    monkeypatch.setattr("perfxpert.cli.opencode_launcher.subprocess.run", _fake_run)
    rc = main(["run", "explain this kernel"])
    assert rc == 0
    cmd = captured["cmd"]
    assert cmd[0] == str(fake_bin)
    assert cmd[1:] == ["run", "explain this kernel"]
    assert captured["kwargs"].get("cwd") is None  # type: ignore[union-attr]


def test_main_default_invocation_stages_runtime_cfg_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Default (interactive) mode still cd's into the bundled runtime dir."""
    fake_bin = tmp_path / "opencode"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    fake_cfg = tmp_path / "cfg"
    fake_cfg.mkdir()
    (fake_cfg / "opencode.json").write_text("{}")

    monkeypatch.setattr(
        "perfxpert.cli.opencode_launcher.resolve_opencode_binary", lambda: fake_bin,
    )
    monkeypatch.setattr(
        "perfxpert.cli.opencode_launcher.resolve_config_dir", lambda: fake_cfg,
    )
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["kwargs"] = kwargs
        return _FakeProc(0)

    monkeypatch.setattr("perfxpert.cli.opencode_launcher.subprocess.run", _fake_run)
    rc = main([])
    assert rc == 0
    # Default interactive: should stage into a runtime cache dir, NOT None.
    assert captured["kwargs"].get("cwd") is not None  # type: ignore[union-attr]
