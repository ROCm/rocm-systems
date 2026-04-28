"""Tests for ``perfxpert init``."""

from __future__ import annotations

from types import SimpleNamespace

from perfxpert.cli import init_cmd


def _mk_args(**kwargs):
    defaults = dict(
        source_dir=None,
        provider=None,
        arch=None,
        non_interactive=True,
        config_path=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_init_non_interactive_detects_gpu_mocked(tmp_path, monkeypatch, capsys) -> None:
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
    rc = init_cmd.run_init(_mk_args(source_dir=str(tmp_path), config_path=str(cfg)))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Step 1/4" in out
    assert "gfx942" in out
    assert "MI300X" in out
    assert "163.4" in out
    assert "5.3" in out


def test_init_framework_detection_prefers_source_scan_over_python_import(tmp_path, monkeypatch) -> None:
    src = tmp_path / "proj"
    src.mkdir()
    (src / "kernel.hip").write_text(
        "__global__ void add(float* a, float* b, float* c) { }\n"
        "void launch() { hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0); }\n"
    )

    class _FakeSpec:
        pass

    real_find = init_cmd.importlib.util.find_spec

    def _fake_find_spec(name):
        if name == "torch":
            return _FakeSpec()
        return real_find(name)

    monkeypatch.setattr(init_cmd.importlib.util, "find_spec", _fake_find_spec)

    info = init_cmd._detect_framework(str(src))
    assert info["programming_model"] == "HIP"
    assert info["kernel_count"] >= 1
    assert info["python_framework"] == "PyTorch"


def test_init_writes_config_to_custom_path(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "nested" / "pxcfg.yaml"
    monkeypatch.setattr(init_cmd, "_detect_gpu", lambda override=None: None)

    rc = init_cmd.run_init(
        _mk_args(source_dir=str(tmp_path), provider="opencode", config_path=str(cfg))
    )

    assert rc == 0
    assert cfg.exists()
    content = cfg.read_text()
    assert "provider: opencode" in content
    assert "airgap" in content
    assert "max_tokens" in content


def test_init_suggests_command_with_framework_shim() -> None:
    info_python = {
        "source_dir": ".",
        "python_framework": "PyTorch",
        "programming_model": "Unknown",
        "kernel_count": 0,
        "file_count": 0,
        "suggested_first_command": "",
        "suggested_counters": [],
    }
    commands = init_cmd._suggest_first_command(info_python)
    assert len(commands) == 1
    assert "rocprofv3" in commands[0]
    assert "--sys-trace" in commands[0]
    assert "-- python train.py" in commands[0]

    info_hip = dict(info_python)
    info_hip.update(
        kernel_count=3,
        programming_model="HIP",
        suggested_counters=["SQ_WAVES", "GRBM_COUNT", "GRBM_GUI_ACTIVE"],
    )
    commands = init_cmd._suggest_first_command(info_hip)
    assert len(commands) == 2
    assert "--pmc" in commands[1]
    assert "SQ_WAVES" in commands[1]


def test_init_returns_rc0_on_clean_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(init_cmd, "_detect_gpu", lambda override=None: None)
    cfg = tmp_path / "pxcfg.yaml"
    rc = init_cmd.run_init(
        _mk_args(source_dir=str(tmp_path), provider="opencode", config_path=str(cfg))
    )
    assert rc == 0


def test_init_detects_ollama_via_new_local_url_env(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PERFXPERT_LLM_PRIVATE_URL", raising=False)
    monkeypatch.delenv("PRIVATE_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("PERFXPERT_LLM_LOCAL_URL", "http://localhost:11434")

    assert init_cmd._detect_configured_provider() == "ollama"


def test_init_returns_rc1_on_unwritable_config_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(init_cmd, "_detect_gpu", lambda override=None: None)

    def _boom(config, target, *, non_interactive, stream=None):
        raise OSError("read-only filesystem (simulated)")

    monkeypatch.setattr(init_cmd, "_write_config", _boom)

    cfg = tmp_path / "ro" / "pxcfg.yaml"
    rc = init_cmd.run_init(
        _mk_args(source_dir=str(tmp_path), provider="opencode", config_path=str(cfg))
    )
    assert rc == 1
