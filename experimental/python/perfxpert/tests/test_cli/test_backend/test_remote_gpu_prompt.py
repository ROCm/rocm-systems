from __future__ import annotations

from perfxpert.cli._backend.codex import CodexAdapter


def test_codex_prompt_scopes_gpu_discovery_to_ssh_target_host() -> None:
    rendered = CodexAdapter()._render_prompt_for_codex()

    assert "GPU discovery must" in rendered
    assert "remote execution host" in rendered
    assert "Do not let local runtime GPU specs override remote-target" in rendered
    assert "PERFXPERT_DISABLE_RUNTIME_GPU_SPECS=1" in rendered
