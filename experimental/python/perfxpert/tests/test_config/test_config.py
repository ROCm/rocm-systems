"""Tests for perfxpert.config."""

import textwrap

import pytest
from pydantic import ValidationError

from perfxpert.config import PerfXpertConfig, load_config
from perfxpert.config import _config


def test_defaults():
    c = PerfXpertConfig()
    assert c.provider == "anthropic"
    assert c.model is None
    assert c.max_tokens == 2048
    assert c.fence_profile == "standard"
    assert c.airgap is False
    assert c.regression_threshold_pct == 3.0
    assert c.hot_kernel_coverage_pct == 80.0


def test_frozen_immutable():
    c = PerfXpertConfig()
    with pytest.raises(ValidationError):
        c.provider = "openai"  # type: ignore[misc]


def test_invalid_provider_rejected():
    with pytest.raises(ValidationError):
        PerfXpertConfig(provider="bogus")


def test_invalid_fence_profile_rejected():
    with pytest.raises(ValidationError):
        PerfXpertConfig(fence_profile="maximal")


def test_load_defaults_when_no_env_no_file(monkeypatch, tmp_path):
    for k in (
        "PERFXPERT_PROVIDER",
        "PERFXPERT_MODEL",
        "PERFXPERT_MAX_TOKENS",
        "PERFXPERT_FENCE_PROFILE",
        "PERFXPERT_AIRGAP",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    c = load_config()
    assert c.provider == "anthropic"
    assert c.airgap is False


def test_load_defaults_when_home_unresolvable(monkeypatch):
    def _raise_home_error():
        raise RuntimeError("home is unavailable")

    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setattr(_config.Path, "home", _raise_home_error)
    c = load_config()
    assert c.provider == "anthropic"
    assert c.model is None
    assert c.max_tokens == 2048
    assert c.fence_profile == "standard"
    assert c.airgap is False


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PERFXPERT_PROVIDER", "ollama")
    monkeypatch.setenv("PERFXPERT_MODEL", "llama3:70b")
    monkeypatch.setenv("PERFXPERT_MAX_TOKENS", "4096")
    monkeypatch.setenv("PERFXPERT_FENCE_PROFILE", "full")
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    c = load_config()
    assert c.provider == "ollama"
    assert c.model == "llama3:70b"
    assert c.max_tokens == 4096
    assert c.fence_profile == "full"
    assert c.airgap is True


def test_yaml_file_loaded(monkeypatch, tmp_path):
    for k in (
        "PERFXPERT_PROVIDER",
        "PERFXPERT_MODEL",
        "PERFXPERT_MAX_TOKENS",
        "PERFXPERT_FENCE_PROFILE",
        "PERFXPERT_AIRGAP",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".config" / "perfxpert"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.yaml").write_text(
        textwrap.dedent(
            """
            provider: openai
            model: gpt-4o
            fence_profile: minimal
            regression_threshold_pct: 5.0
            """
        )
    )
    c = load_config()
    assert c.provider == "openai"
    assert c.model == "gpt-4o"
    assert c.fence_profile == "minimal"
    assert c.regression_threshold_pct == 5.0


def test_env_beats_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".config" / "perfxpert"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.yaml").write_text("provider: openai\n")
    monkeypatch.setenv("PERFXPERT_PROVIDER", "ollama")
    c = load_config()
    assert c.provider == "ollama"


def test_regression_threshold_numeric(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PERFXPERT_REGRESSION_THRESHOLD_PCT", "7.5")
    c = load_config()
    assert c.regression_threshold_pct == 7.5
