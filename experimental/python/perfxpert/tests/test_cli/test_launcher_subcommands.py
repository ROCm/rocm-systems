"""Tests for perfxpert-code launcher subcommand dispatch.

Covers two distinct review findings:

* Review-finding I4: `perfxpert-code --help` must surface the
  perfxpert-owned subcommands (`init`, `diff`, `ci`, `doctor`, `analyze`,
  `config`, `providers`) so users can discover them, not silently forward
  to opencode.
* Issue 2: `perfxpert-code doctor` errored with
  ``Failed to change directory to .../doctor`` because opencode
  interpreted ``doctor`` as a positional CWD. The launcher now routes
  known subcommands explicitly.
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


@pytest.fixture(autouse=True)
def _disable_repo_local_patched_binary(monkeypatch):
    monkeypatch.setattr(
        opencode_launcher,
        "_repo_local_patched_opencode_paths",
        lambda: [],
    )


# ---------------------------------------------------------------------------
# --help / -h handling (review-finding I4).
# ---------------------------------------------------------------------------


class TestHelpFlag:
    """Bare `perfxpert-code --help` must print the perfxpert-owned banner.

    Per review I4: help flag discovery must list the documented
    perfxpert subcommands before falling through to opencode's generic help.
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
        for sub in ("analyze", "init", "diff", "ci", "config", "doctor", "providers"):
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
        """The perfxpert subcommand catalog must list the documented CLI surface."""
        subs = opencode_launcher._PERFXPERT_SUBCOMMANDS
        assert "doctor" in subs, "doctor must be listed for review I4"
        assert "analyze" in subs
        assert "init" in subs
        assert "diff" in subs
        assert "ci" in subs
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


@pytest.mark.parametrize("subcommand", ["init", "diff", "ci"])
def test_route_restored_cli_subcommands_are_perfxpert_owned(subcommand: str) -> None:
    kind, out = route_subcommand([subcommand])
    assert kind == "perfxpert"
    assert out == [subcommand]


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


def test_dispatch_set_contains_install_patches() -> None:
    """install-patches must short-circuit before opencode resolution
    (it's the command that CREATES the opencode binary; it'd deadlock if it
    required opencode first)."""
    assert "install-patches" in _PERFXPERT_DISPATCH_SUBCOMMANDS


def test_install_patches_listed_in_perfxpert_subcommands() -> None:
    """Help banner must discover install-patches."""
    assert "install-patches" in _PERFXPERT_SUBCOMMANDS
    assert _PERFXPERT_SUBCOMMANDS["install-patches"].strip()


def test_route_install_patches_is_perfxpert_owned() -> None:
    kind, out = route_subcommand(["install-patches"])
    assert kind == "perfxpert"
    assert out == ["install-patches"]


# ---------------------------------------------------------------------------
# Bundled-binary priority: the launcher must prefer
# perfxpert/_bundled/opencode over upstream installs on disk.
# ---------------------------------------------------------------------------


class TestBundledPriority:
    """resolve_opencode_binary() priority order:
       env override → _bundled → well-known → PATH.

    Requirement: the bundled binary wins over
    ~/.opencode/bin/opencode (upstream, unpatched).
    """

    def test_bundled_wins_over_wellknown(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """If both bundled and upstream exist, we pick bundled."""
        # Stub importlib.resources so _bundled/opencode is "present".
        bundled = tmp_path / "bundled" / "opencode"
        bundled.parent.mkdir(parents=True)
        bundled.write_text("#!/bin/sh\nexit 0\n")
        bundled.chmod(0o755)

        # Also create an upstream well-known entry — it must NOT win.
        upstream = tmp_path / "home" / ".opencode" / "bin" / "opencode"
        upstream.parent.mkdir(parents=True)
        upstream.write_text("#!/bin/sh\nexit 1\n")
        upstream.chmod(0o755)

        # Make well-known list point at the upstream binary.
        monkeypatch.setattr(
            opencode_launcher,
            "_wellknown_opencode_paths",
            lambda: [upstream],
        )

        # Patch importlib.resources.files() for _bundled lookup.
        from contextlib import contextmanager

        @contextmanager
        def _fake_as_file(p):
            yield bundled

        class _FakeTraversable:
            def __truediv__(self, other):
                return self

            def is_file(self):
                return True

        monkeypatch.setattr(
            opencode_launcher.resources, "as_file", _fake_as_file
        )
        monkeypatch.setattr(
            opencode_launcher.resources,
            "files",
            lambda _pkg: _FakeTraversable(),
        )
        monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)

        resolved = opencode_launcher.resolve_opencode_binary()
        assert resolved == bundled, (
            f"bundled must win over upstream; got {resolved}"
        )

    def test_env_override_wins_over_bundled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """PERFXPERT_OPENCODE_PATH trumps bundled — power-user escape hatch."""
        explicit = tmp_path / "mycustom" / "opencode"
        explicit.parent.mkdir(parents=True)
        explicit.write_text("#!/bin/sh\nexit 0\n")
        explicit.chmod(0o755)

        monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", str(explicit))
        resolved = opencode_launcher.resolve_opencode_binary()
        assert resolved == explicit

    def test_wellknown_emits_warning_when_bundled_absent(
        self,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        """Falling through to upstream must warn the user that patches aren't active."""
        upstream = tmp_path / "home" / ".opencode" / "bin" / "opencode"
        upstream.parent.mkdir(parents=True)
        upstream.write_text("#!/bin/sh\nexit 0\n")
        upstream.chmod(0o755)

        # Bundled absent.
        class _FakeTraversable:
            def __truediv__(self, other):
                return self

            def is_file(self):
                return False

        from contextlib import contextmanager

        @contextmanager
        def _fake_as_file(p):
            yield tmp_path / "bundled" / "missing"

        monkeypatch.setattr(
            opencode_launcher.resources, "as_file", _fake_as_file
        )
        monkeypatch.setattr(
            opencode_launcher.resources,
            "files",
            lambda _pkg: _FakeTraversable(),
        )
        monkeypatch.setattr(
            opencode_launcher,
            "_wellknown_opencode_paths",
            lambda: [upstream],
        )
        monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
        monkeypatch.delenv("PERFXPERT_SILENCE_UNPATCHED_WARNING", raising=False)

        resolved = opencode_launcher.resolve_opencode_binary()
        assert resolved == upstream

        captured = capsys.readouterr()
        assert "unpatched" in captured.err.lower()
        assert "install-patches" in captured.err


class TestRunAutoAgentInject:
    """`perfxpert-code run ...` must load the perfxpert agent
    so AGENTS.md (with the tool-priority gate) applies.

    opencode's `run` otherwise defaults to agent=build and ignores our
    bundled opencode.json. The launcher injects `--agent perfxpert` when
    the user did not already specify `--agent`.
    """

    def test_inject_adds_agent_perfxpert_for_run(self) -> None:
        out = opencode_launcher._inject_perfxpert_agent_for_run(
            ["run", "optimize kernel X"]
        )
        assert out == ["run", "--agent", "perfxpert", "optimize kernel X"]

    def test_inject_preserves_explicit_agent(self) -> None:
        out = opencode_launcher._inject_perfxpert_agent_for_run(
            ["run", "--agent", "build", "explain this"]
        )
        # User override wins — inject is a no-op.
        assert out == ["run", "--agent", "build", "explain this"]

    def test_inject_handles_agent_equals_form(self) -> None:
        out = opencode_launcher._inject_perfxpert_agent_for_run(
            ["run", "--agent=plan", "my message"]
        )
        assert out == ["run", "--agent=plan", "my message"]

    def test_inject_noop_for_non_run_subcommand(self) -> None:
        # stats / auth / models don't take --agent; we must NOT inject.
        assert opencode_launcher._inject_perfxpert_agent_for_run(["stats"]) == [
            "stats"
        ]
        assert opencode_launcher._inject_perfxpert_agent_for_run(
            ["models", "anthropic"]
        ) == ["models", "anthropic"]

    def test_inject_honors_opt_out_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERFXPERT_NO_AUTO_AGENT", "1")
        out = opencode_launcher._inject_perfxpert_agent_for_run(
            ["run", "explain"]
        )
        # Opt-out → no inject.
        assert out == ["run", "explain"]

    def test_main_run_sets_opencode_config_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """The launcher must set OPENCODE_CONFIG so opencode loads our
        bundled opencode.json even when `run` keeps the user's CWD."""
        fake_bin = tmp_path / "opencode"
        fake_bin.write_text("#!/bin/sh\nexit 0\n")
        fake_bin.chmod(0o755)
        fake_cfg = tmp_path / "cfg"
        fake_cfg.mkdir()
        (fake_cfg / "opencode.json").write_text("{}")

        monkeypatch.setattr(
            "perfxpert.cli.opencode_launcher.resolve_opencode_binary",
            lambda: fake_bin,
        )
        monkeypatch.setattr(
            "perfxpert.cli.opencode_launcher.resolve_config_dir",
            lambda: fake_cfg,
        )
        monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")
        monkeypatch.delenv("OPENCODE_CONFIG", raising=False)

        captured: dict[str, object] = {}

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["env"] = kwargs.get("env", {})
            return _FakeProc(0)

        monkeypatch.setattr(
            "perfxpert.cli.opencode_launcher.subprocess.run", _fake_run
        )
        rc = main(["run", "hello"])
        assert rc == 0
        env = captured["env"]
        assert isinstance(env, dict)
        cfg = env.get("OPENCODE_CONFIG")
        assert cfg is not None
        assert cfg.endswith("opencode.json"), cfg

    def test_main_run_preserves_user_opencode_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """If user already exported OPENCODE_CONFIG, don't clobber it."""
        fake_bin = tmp_path / "opencode"
        fake_bin.write_text("#!/bin/sh\nexit 0\n")
        fake_bin.chmod(0o755)
        fake_cfg = tmp_path / "cfg"
        fake_cfg.mkdir()
        (fake_cfg / "opencode.json").write_text("{}")

        monkeypatch.setattr(
            "perfxpert.cli.opencode_launcher.resolve_opencode_binary",
            lambda: fake_bin,
        )
        monkeypatch.setattr(
            "perfxpert.cli.opencode_launcher.resolve_config_dir",
            lambda: fake_cfg,
        )
        monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")
        monkeypatch.setenv("OPENCODE_CONFIG", "/user/custom.json")

        captured: dict[str, object] = {}

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["env"] = kwargs.get("env", {})
            return _FakeProc(0)

        monkeypatch.setattr(
            "perfxpert.cli.opencode_launcher.subprocess.run", _fake_run
        )
        rc = main(["run", "hello"])
        assert rc == 0
        env = captured["env"]
        assert env.get("OPENCODE_CONFIG") == "/user/custom.json"  # type: ignore[union-attr]


def test_main_install_patches_dispatches_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`perfxpert-code install-patches` must invoke the build script,
    NOT `python -m perfxpert install-patches` (which doesn't exist)."""
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        return _FakeProc(0)

    monkeypatch.setattr(
        "perfxpert.cli.opencode_launcher.subprocess.run", _fake_run
    )
    rc = main(["install-patches"])
    assert rc == 0
    cmd = captured["cmd"]
    # The inline handler invokes `bash <path>/build-bundled-opencode.sh`.
    assert isinstance(cmd, list)
    assert cmd[0] == "bash"
    assert cmd[1].endswith("build-bundled-opencode.sh")


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
    # Auto-inject: `run` without explicit --agent gets
    # `--agent perfxpert` so AGENTS.md (tool-priority gate) loads.
    assert cmd[1:] == ["run", "--agent", "perfxpert", "explain this kernel"]
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


# ---------------------------------------------------------------------------
# --help / -h routing (cycle-2 N2 coverage).
# ---------------------------------------------------------------------------


class TestHelpFlag:
    """Help-flag handling composed from `_handle_help_flag`.

    Bare ``--help`` / ``-h`` before any positional must print the AMD
    PerfXpert banner (via _print_perfxpert_help) and then fall through
    to ``opencode --help``. A help flag AFTER a positional (e.g.
    ``run --help``) is subcommand-specific and passes through verbatim.
    """

    @pytest.mark.parametrize("flag", ["--help", "-h"])
    def test_bare_help_flag_prints_perfxpert_banner(
        self, flag: str, capsys, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure opencode is unresolved so main() short-circuits after
        # printing the banner (per the composed fallback behavior).
        def _raise(*_args, **_kwargs):
            raise FileNotFoundError("opencode not installed")

        monkeypatch.setattr(
            "perfxpert.cli.opencode_launcher.resolve_opencode_binary", _raise,
        )
        rc = main([flag])
        out = capsys.readouterr().out
        # Banner printed — bubble exit 0 so discovery path is clean.
        assert rc == 0
        assert "AMD ROCm PerfXpert" in out
        assert "perfxpert-owned subcommands" in out

    @pytest.mark.parametrize("flag", ["--help", "-h"])
    def test_help_after_opencode_subcommand_passes_through(
        self, flag: str, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """``perfxpert-code run --help`` must forward to opencode verbatim
        rather than printing our banner — that's opencode's own subcommand
        help request."""
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
        monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")
        captured: dict[str, object] = {}

        def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["cmd"] = list(cmd)
            captured["kwargs"] = kwargs
            return _FakeProc(0)

        monkeypatch.setattr(
            "perfxpert.cli.opencode_launcher.subprocess.run", _fake_run,
        )
        rc = main(["run", flag])
        assert rc == 0
        # The help flag was forwarded to opencode verbatim (not stripped).
        assert flag in captured["cmd"]  # type: ignore[operator]
        # And because `run` is a known opencode subcommand, cwd is None
        # (user's project CWD preserved).
        assert captured["kwargs"].get("cwd") is None  # type: ignore[union-attr]

    def test_bare_help_short_circuits_before_opencode_missing_error(
        self, capsys, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If opencode isn't installed, bare --help MUST still print our
        banner and return 0 (per the composed fallback behavior).
        """
        monkeypatch.setattr(
            "perfxpert.cli.opencode_launcher.resolve_opencode_binary",
            lambda: (_ for _ in ()).throw(
                FileNotFoundError("opencode binary not found.")
            ),
        )
        rc = main(["--help"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "AMD ROCm PerfXpert" in out
