"""Tests for the `perfxpert config` subcommand."""

import textwrap
from io import StringIO
from unittest.mock import patch

from perfxpert.config._cli import run_config_show, run_config_set


def test_config_show_prints_yaml(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    run_config_show()
    captured = capsys.readouterr().out
    assert "provider:" in captured
    assert "anthropic" in captured


def test_config_set_writes_yaml(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    run_config_set("provider", "openai")
    path = tmp_path / ".config" / "perfxpert" / "config.yaml"
    assert path.exists()
    assert "provider" in path.read_text()
    assert "openai" in path.read_text()


def test_config_set_preserves_other_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".config" / "perfxpert"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.yaml").write_text("provider: openai\nmodel: gpt-4o\n")
    run_config_set("max_tokens", "4096")
    content = (cfg_dir / "config.yaml").read_text()
    assert "openai" in content
    assert "gpt-4o" in content
    assert "4096" in content


def test_config_set_rejects_invalid_field(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SystemExit):
        run_config_set("bogus_field", "x")


def test_config_set_coerces_bool(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    run_config_set("airgap", "true")
    content = (tmp_path / ".config" / "perfxpert" / "config.yaml").read_text()
    assert "airgap" in content
    assert "true" in content.lower()
