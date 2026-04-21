"""Tests for `perfxpert init` — first-run wizard (Confluence row #29).

Covers the five cases enumerated in the rollout plan:

1. GPU detection prints gfx id + peaks when the detector is mocked.
2. Framework detection prefers source-scan programming_model over Python
   import presence.
3. ``--config-path`` override causes the YAML to be written to a custom
   path with the expected keys.
4. PyTorch detection produces a ``python`` target in the suggested command.
5. Clean run returns rc=0; unwritable config-path returns rc=1.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from perfxpert.cli import init_cmd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mk_args(**kw):
    """Build a argparse.Namespace-like object with init_cmd defaults."""
    defaults = dict(
        source_dir=None,
        provider=None,
        arch=None,
        non_interactive=True,
        config_path=None,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# 1. GPU detection
# ---------------------------------------------------------------------------


def test_init_non_interactive_detects_gpu_mocked(
    tmp_path, monkeypatch, capsys
):
    """Step 1 prints the mocked gfx id and peak specs."""
    canned = {
        "gfx_id": "gfx942",
        "peaks": {
            "name": "MI300X",
            "cu_count": 304,
            "peak_fp32_tflops": 163.4,
            "memory_bandwidth_tbs": 5.3,
        },
    }
    monkeypatch.setattr(init_cmd, "_detect_gpu", lambda override=None: canned)

    cfg = tmp_path / "pxcfg.yaml"
    rc = init_cmd.run_init(_mk_args(
        source_dir=str(tmp_path),
        config_path=str(cfg),
    ))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Step 1/4" in out
    assert "gfx942" in out
    assert "MI300X" in out
    assert "163.4" in out
    assert "5.3" in out


# ---------------------------------------------------------------------------
# 2. Framework detection prefers source-scan programming_model
# ---------------------------------------------------------------------------


def test_init_framework_detection_prefers_source_scan_over_python_import(
    tmp_path, monkeypatch
):
    """A tmp dir with .hip content reports programming_model = HIP even when
    torch is importable."""
    src = tmp_path / "proj"
    src.mkdir()
    (src / "kernel.hip").write_text(
        "__global__ void add(float* a, float* b, float* c) { }\n"
        "void launch() { hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0); }\n"
    )

    # Pretend torch is importable in this process even though it may not be.
    class _FakeSpec:
        pass

    real_find = init_cmd.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "torch":
            return _FakeSpec()
        return real_find(name)

    monkeypatch.setattr(init_cmd.importlib.util, "find_spec", fake_find_spec)

    info = init_cmd._detect_framework(str(src))
    assert info["programming_model"] == "HIP"
    assert info["kernel_count"] >= 1
    assert info["python_framework"] == "PyTorch"


# ---------------------------------------------------------------------------
# 3. --config-path override
# ---------------------------------------------------------------------------


def test_init_writes_config_to_custom_path(tmp_path, monkeypatch, capsys):
    """`--config-path foo/bar.yaml` must land the file at that exact path
    and contain the expected keys."""
    cfg = tmp_path / "nested" / "pxcfg.yaml"
    monkeypatch.setattr(
        init_cmd, "_detect_gpu", lambda override=None: None
    )
    rc = init_cmd.run_init(_mk_args(
        source_dir=str(tmp_path),
        provider="opencode",
        config_path=str(cfg),
    ))
    assert rc == 0
    assert cfg.exists()
    content = cfg.read_text()
    assert "provider: opencode" in content
    assert "airgap" in content
    assert "max_tokens" in content


# ---------------------------------------------------------------------------
# 4. PyTorch shim in suggested command
# ---------------------------------------------------------------------------


def test_init_suggests_command_with_framework_shim():
    """PyTorch → primary uses `python train.py`; HIP kernels → emits --pmc line."""
    # Case A: PyTorch-only (no kernels in source)
    info_py = {
        "source_dir": ".",
        "python_framework": "PyTorch",
        "programming_model": "Unknown",
        "kernel_count": 0,
        "file_count": 0,
        "suggested_first_command": "",
        "suggested_counters": [],
    }
    cmds = init_cmd._suggest_first_command(info_py)
    assert len(cmds) == 1
    assert "rocprofv3" in cmds[0]
    assert "--sys-trace" in cmds[0]
    assert "-- python train.py" in cmds[0]

    # Case B: HIP kernels present → aux --pmc line appended.
    info_hip = dict(info_py)
    info_hip.update(kernel_count=3, programming_model="HIP",
                    suggested_counters=["SQ_WAVES", "GRBM_COUNT", "GRBM_GUI_ACTIVE"])
    cmds = init_cmd._suggest_first_command(info_hip)
    assert len(cmds) == 2
    assert "--pmc" in cmds[1]
    assert "SQ_WAVES" in cmds[1]


# ---------------------------------------------------------------------------
# 5. Return codes
# ---------------------------------------------------------------------------


def test_init_returns_rc0_on_clean_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(init_cmd, "_detect_gpu", lambda override=None: None)
    cfg = tmp_path / "pxcfg.yaml"
    rc = init_cmd.run_init(_mk_args(
        source_dir=str(tmp_path),
        provider="opencode",
        config_path=str(cfg),
    ))
    assert rc == 0


def test_init_returns_rc1_on_unwritable_config_path(
    tmp_path, monkeypatch, capsys
):
    """An OSError when writing the config must bubble up as rc=1, NOT traceback."""
    monkeypatch.setattr(init_cmd, "_detect_gpu", lambda override=None: None)

    def _boom(cfg, target, *, non_interactive, stream=None):
        raise OSError("read-only filesystem (simulated)")

    monkeypatch.setattr(init_cmd, "_write_config", _boom)

    cfg = tmp_path / "ro" / "pxcfg.yaml"
    rc = init_cmd.run_init(_mk_args(
        source_dir=str(tmp_path),
        provider="opencode",
        config_path=str(cfg),
    ))
    assert rc == 1
