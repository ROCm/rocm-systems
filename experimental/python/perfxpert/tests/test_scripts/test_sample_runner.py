"""Tests for scripts/test-samples.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test-samples.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("test_samples_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_python_samples_use_current_interpreter(monkeypatch):
    module = _load_script()
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_python_sample("print('ok')")

    assert result["passed"] is True
    assert captured["argv"][:2] == [sys.executable, "-c"]


def test_bash_sample_skips_when_bash_missing(monkeypatch):
    module = _load_script()
    monkeypatch.setattr(module.shutil, "which", lambda _: None)

    result = module.run_sample({"type": "bash", "code": "echo ok"})

    assert result["status"] == "SKIPPED"
    assert result["reason"] == "bash not found on PATH"
