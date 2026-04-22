"""Tests for perfxpert-code launcher subcommand dispatch.

Covers review-finding I4: `perfxpert-code --help` must surface the
perfxpert-owned subcommands (`doctor`, `analyze`, `config`, `providers`)
so users can discover them, not silently forward to opencode.
"""

from __future__ import annotations

import io
import sys

import pytest

from perfxpert.cli import opencode_launcher


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
