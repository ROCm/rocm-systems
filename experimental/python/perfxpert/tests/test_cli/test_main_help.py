"""Top-level CLI registration tests for ``python -m perfxpert``."""

from __future__ import annotations

import importlib

import pytest

from perfxpert import __main__ as main_mod


def test_top_level_help_lists_documented_subcommands(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main_mod.main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    for subcommand in ("analyze", "config", "providers", "doctor", "init", "diff", "ci"):
        assert subcommand in out


@pytest.mark.parametrize(
    ("argv", "module_name", "func_name"),
    [
        (["init"], "perfxpert.cli.init_cmd", "run_init"),
        (["diff", "baseline.db", "candidate.db"], "perfxpert.cli.diff_cmd", "run_diff"),
        (["ci", "baseline.db", "candidate.db"], "perfxpert.cli.ci_cmd", "run_ci"),
    ],
)
def test_main_dispatches_restored_subcommands(monkeypatch, argv, module_name, func_name) -> None:
    module = importlib.import_module(module_name)
    called = {}

    def _fake(args):
        called["subcommand"] = args.subcommand
        return 0

    monkeypatch.setattr(module, func_name, _fake)

    with pytest.raises(SystemExit) as exc:
        main_mod.main(argv)

    assert exc.value.code == 0
    assert called["subcommand"] == argv[0]
