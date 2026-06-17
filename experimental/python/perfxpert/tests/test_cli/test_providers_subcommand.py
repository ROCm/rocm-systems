"""Tests for `perfxpert providers list` subcommand."""

import subprocess
import sys

import pytest

from perfxpert.__main__ import main as perfxpert_main


def test_providers_list_prints_all_five(capsys):
    rc = perfxpert_main(["providers", "list"])
    out = capsys.readouterr().out.lower()
    assert rc == 0
    for p in ("anthropic", "openai", "ollama", "private", "opencode"):
        assert p in out


def test_providers_list_shows_descriptions(capsys):
    perfxpert_main(["providers", "list"])
    out = capsys.readouterr().out
    assert "Anthropic Claude" in out
    assert "OpenAI GPT" in out or "gpt" in out.lower()


def test_providers_list_shows_status(capsys, monkeypatch):
    monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-test")
    perfxpert_main(["providers", "list"])
    out = capsys.readouterr().out.lower()
    assert "configured" in out or "status" in out
