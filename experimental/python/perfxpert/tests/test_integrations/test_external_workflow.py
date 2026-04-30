"""Unit tests for external workflow adapter inspection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from perfxpert.integrations.external_workflow import (
    ExternalWorkflowError,
    inspect_external_workflow,
)


def _write_adapter_fixture(root):
    root.mkdir()
    (root / "README.md").write_text(
        "\n".join(
            [
                "The profiler runner uses rocprof-compute PMC counters.",
                "Source mapping provides source line PC sampling attribution.",
                "Trace inspection reads HSA packet traces and AQL assembly.",
                "Kernel replay can extract kernel launches.",
                "Validation checks correctness and regression behavior.",
                "Install with: pip install external-tool",
            ]
        ),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project.scripts]\nadapter-mcp = "adapter.server:main"\n',
        encoding="utf-8",
    )
    (root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"adapter": {"command": "adapter-mcp", "args": ["serve"]}}}),
        encoding="utf-8",
    )
    docs = root / "docs" / "knowledge"
    docs.mkdir(parents=True)
    (docs / "optimization.md").write_text("Prefer measured counters.\n", encoding="utf-8")
    skill = root / "skills" / "optimizer"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("Use this skill for optimization workflow.\n", encoding="utf-8")


def test_inspect_external_workflow_discovers_advisory_hooks(tmp_path) -> None:
    source = tmp_path / "adapter"
    _write_adapter_fixture(source)

    plan = inspect_external_workflow(
        str(source),
        interactive=True,
        cache_root=tmp_path / "cache",
    )

    kinds = {cap["kind"] for cap in plan["capabilities"]}
    assert {
        "profiling",
        "source_correlation",
        "trace_inspection",
        "kernel_isolation",
        "correctness",
        "mcp_tools",
        "agent_skill",
        "entry_point",
    } <= kinds
    assert plan["interactive_only"] is True
    assert plan["execution_allowed"] is False
    assert plan["mcp_servers"][0]["activation"].startswith("requires explicit active-session TUI consent")
    assert any(link["path"] == "docs/knowledge/optimization.md" for link in plan["knowledge_links"])
    assert any("do not run them" in warning for warning in plan["warnings"])
    assert plan["manifest_path"]
    manifest = json.loads(Path(plan["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["adapter_id"] == plan["adapter_id"]
    assert Path(plan["manifest_path"]).parent == tmp_path / "cache" / "adapters"


def test_url_source_requires_interactive_network_consent(tmp_path) -> None:
    with pytest.raises(ExternalWorkflowError, match="TUI-interactive"):
        inspect_external_workflow(
            "https://github.com/example/perf-workflow",
            interactive=False,
            cache_root=tmp_path / "cache",
        )

    with pytest.raises(ExternalWorkflowError, match="--allow-network"):
        inspect_external_workflow(
            "https://github.com/example/perf-workflow",
            interactive=True,
            allow_network=False,
            cache_root=tmp_path / "cache",
        )

    with pytest.raises(ExternalWorkflowError, match="https://"):
        inspect_external_workflow(
            "http://example.com/perf-workflow",
            interactive=True,
            allow_network=True,
            cache_root=tmp_path / "cache",
        )

    with pytest.raises(ExternalWorkflowError, match="embedded credentials"):
        inspect_external_workflow(
            "https://user:token@example.com/perf-workflow",
            interactive=True,
            allow_network=True,
            cache_root=tmp_path / "cache",
        )

    with pytest.raises(ExternalWorkflowError, match="query strings or fragments"):
        inspect_external_workflow(
            "https://github.com/example/perf-workflow?token=secret",
            interactive=True,
            allow_network=True,
            cache_root=tmp_path / "cache",
        )

    with pytest.raises(ExternalWorkflowError, match="query strings or fragments"):
        inspect_external_workflow(
            "https://github.com/example/perf-workflow#secret",
            interactive=True,
            allow_network=True,
            cache_root=tmp_path / "cache",
        )


def test_https_source_clone_and_cache_reuse(tmp_path, monkeypatch) -> None:
    source_url = "https://github.com/example/perf-workflow"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == ["git", "clone", "--depth"]:
            checkout = Path(cmd[-1])
            checkout.mkdir(parents=True)
            (checkout / ".git").mkdir()
            (checkout / "README.md").write_text("Profiler and validation workflow.\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[0] == "git" and cmd[3:6] == ["remote", "get-url", "origin"]:
            return SimpleNamespace(returncode=0, stdout=source_url + "\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr("perfxpert.integrations.external_workflow.subprocess.run", fake_run)

    first = inspect_external_workflow(
        source_url,
        interactive=True,
        allow_network=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )
    second = inspect_external_workflow(
        source_url,
        interactive=True,
        allow_network=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )

    clone_calls = [cmd for cmd in calls if cmd[:3] == ["git", "clone", "--depth"]]
    assert len(clone_calls) == 1
    assert clone_calls[0][3] == "1"
    assert clone_calls[0][4] == source_url
    assert first["materialized_path"] == second["materialized_path"]
    assert first["source"] == source_url


def test_https_clone_failure_is_runtime_error(tmp_path, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="fatal: could not read token\n")

    monkeypatch.setattr("perfxpert.integrations.external_workflow.subprocess.run", fake_run)

    with pytest.raises(ExternalWorkflowError, match="git clone failed"):
        inspect_external_workflow(
            "https://github.com/example/perf-workflow",
            interactive=True,
            allow_network=True,
            cache_root=tmp_path / "cache",
        )


def test_corrupt_file_cache_is_replaced_for_url_source(tmp_path, monkeypatch) -> None:
    source_url = "https://github.com/example/perf-workflow"
    cache_root = tmp_path / "cache"
    corrupt = cache_root / "sources" / "perf-workflow-dd6c86ae6ab1"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("not a checkout\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["git", "clone", "--depth"]:
            checkout = Path(cmd[-1])
            checkout.mkdir(parents=True)
            (checkout / ".git").mkdir()
            (checkout / "README.md").write_text("Profiler workflow.\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr("perfxpert.integrations.external_workflow.subprocess.run", fake_run)

    plan = inspect_external_workflow(
        source_url,
        interactive=True,
        allow_network=True,
        cache_root=cache_root,
        persist=False,
    )

    assert Path(plan["materialized_path"]).is_dir()
    assert (Path(plan["materialized_path"]) / ".git").is_dir()


def test_clone_finalize_failure_is_runtime_error(tmp_path, monkeypatch) -> None:
    source_url = "https://github.com/example/perf-workflow"

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["git", "clone", "--depth"]:
            checkout = Path(cmd[-1])
            checkout.mkdir(parents=True)
            (checkout / ".git").mkdir()
            (checkout / "README.md").write_text("Profiler workflow.\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd!r}")

    original_write_text = Path.write_text

    def fail_sentinel_write(self, *args, **kwargs):
        if self.name == ".perfxpert-adapter-clone-complete":
            raise PermissionError("read-only cache")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr("perfxpert.integrations.external_workflow.subprocess.run", fake_run)
    monkeypatch.setattr(Path, "write_text", fail_sentinel_write)

    with pytest.raises(ExternalWorkflowError, match="failed to finalize cached external workflow source"):
        inspect_external_workflow(
            source_url,
            interactive=True,
            allow_network=True,
            cache_root=tmp_path / "cache",
            persist=False,
        )


def test_inspection_prioritizes_docs_over_large_test_tree(tmp_path) -> None:
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "README.md").write_text("External workflow with advisory knowledge.\n", encoding="utf-8")
    tests = source / "tests"
    tests.mkdir()
    for idx in range(180):
        (tests / f"test_noise_{idx}.py").write_text("def test_noise(): pass\n", encoding="utf-8")
    knowledge = source / "docs" / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "critical.md").write_text("Important workflow hint.\n", encoding="utf-8")

    plan = inspect_external_workflow(
        str(source),
        interactive=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )

    assert "docs/knowledge/critical.md" in plan["inspected_files"]
    assert not any(path.startswith("tests/") for path in plan["inspected_files"])


def test_scan_caps_noisy_root_but_keeps_docs(tmp_path) -> None:
    source = tmp_path / "adapter"
    source.mkdir()
    for idx in range(180):
        (source / f"noise_{idx:03d}.txt").write_text("noise\n", encoding="utf-8")
    docs = source / "docs"
    docs.mkdir()
    (docs / "workflow.md").write_text("Profiler workflow knowledge.\n", encoding="utf-8")
    oversized = source / "large.md"
    oversized.write_bytes(b"x" * (256 * 1024 + 1))

    plan = inspect_external_workflow(
        str(source),
        interactive=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )

    assert "docs/workflow.md" in plan["inspected_files"]
    assert "large.md" not in plan["inspected_files"]
    assert len(plan["inspected_files"]) == 120
    assert any("candidate files" in warning for warning in plan["warnings"])


def test_child_cap_prioritizes_special_files_and_docs(tmp_path) -> None:
    source = tmp_path / "adapter"
    source.mkdir()
    for idx in range(1200):
        (source / f"noise_{idx:04d}.txt").write_text("noise\n", encoding="utf-8")
    (source / "README.md").write_text("Profiler workflow.\n", encoding="utf-8")
    (source / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"adapter": {"command": "adapter-mcp", "args": []}}}),
        encoding="utf-8",
    )
    docs = source / "docs"
    docs.mkdir()
    (docs / "workflow.md").write_text("Workflow knowledge.\n", encoding="utf-8")

    plan = inspect_external_workflow(
        str(source),
        interactive=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )

    assert "README.md" in plan["inspected_files"]
    assert ".mcp.json" in plan["inspected_files"]
    assert "docs/workflow.md" in plan["inspected_files"]
    assert plan["mcp_servers"][0]["name"] == "adapter"
    assert any("1024 children" in warning for warning in plan["warnings"])


def test_scan_directory_count_cap_warns(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("perfxpert.integrations.external_workflow.MAX_SCANNED_DIRS", 4)
    source = tmp_path / "adapter"
    source.mkdir()
    for idx in range(10):
        child = source / f"dir_{idx}"
        child.mkdir()
        (child / "README.md").write_text("Profiler workflow.\n", encoding="utf-8")

    plan = inspect_external_workflow(
        str(source),
        interactive=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )

    assert any("4 directories" in warning for warning in plan["warnings"])


def test_scan_depth_cap_warns(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("perfxpert.integrations.external_workflow.MAX_SCAN_DEPTH", 1)
    source = tmp_path / "adapter"
    deep = source / "level1" / "level2"
    deep.mkdir(parents=True)
    (deep / "README.md").write_text("Profiler workflow.\n", encoding="utf-8")

    plan = inspect_external_workflow(
        str(source),
        interactive=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )

    assert any("deeper than 1 levels" in warning for warning in plan["warnings"])


def test_file_source_uses_parent_directory_and_package_bins(tmp_path) -> None:
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "README.md").write_text("A profiler helper.\n", encoding="utf-8")
    (source / "package.json").write_text(
        json.dumps({"bin": {"adapter-prof": "bin/prof.js", "adapter-mcp": "bin/mcp.js"}}),
        encoding="utf-8",
    )

    plan = inspect_external_workflow(
        str(source / "README.md"),
        interactive=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )

    names = {cap["name"] for cap in plan["capabilities"] if cap["kind"] == "entry_point"}
    assert names == {"adapter-mcp", "adapter-prof"}
    assert plan["materialized_path"] == str(source)


def test_relative_source_and_default_cache_use_workload_cwd(tmp_path, monkeypatch) -> None:
    workload = tmp_path / "workload"
    source = workload / "adapter"
    source.mkdir(parents=True)
    (source / "README.md").write_text("Profiler workflow.\n", encoding="utf-8")
    staged_tui = tmp_path / "staged-tui"
    staged_tui.mkdir()
    monkeypatch.chdir(staged_tui)
    monkeypatch.setenv("PERFXPERT_WORKLOAD_CWD", str(workload))

    plan = inspect_external_workflow("adapter", interactive=True)

    assert plan["materialized_path"] == str(source)
    assert Path(plan["manifest_path"]).parent == workload / ".perfxpert" / "external-workflows" / "adapters"


def test_no_persist_omits_manifest_and_stable_adapter_id(tmp_path) -> None:
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "README.md").write_text("Profiler workflow.\n", encoding="utf-8")

    first = inspect_external_workflow(
        str(source),
        interactive=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )
    second = inspect_external_workflow(
        str(source),
        interactive=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )

    assert first["manifest_path"] is None
    assert second["manifest_path"] is None
    assert first["adapter_id"] == second["adapter_id"]
    assert not (tmp_path / "cache" / "adapters").exists()


def test_cache_symlink_is_rejected(tmp_path) -> None:
    cache = tmp_path / "cache"
    target = tmp_path / "target"
    target.mkdir()
    cache.symlink_to(target, target_is_directory=True)

    source = tmp_path / "adapter"
    source.mkdir()
    (source / "README.md").write_text("Profiler workflow.\n", encoding="utf-8")

    with pytest.raises(ExternalWorkflowError, match="symlink"):
        inspect_external_workflow(
            str(source),
            interactive=True,
            cache_root=cache,
        )


def test_malformed_mcp_descriptors_are_sanitized(tmp_path) -> None:
    source = tmp_path / "adapter"
    source.mkdir()
    (source / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "bad": "not-an-object",
                    "dangerous": {"command": "adapter-mcp; rm -rf /", "args": ["serve"]},
                    "invalid": {"command": 42, "args": "serve"},
                    "valid": {"command": "valid-mcp", "args": ["serve"]},
                }
            }
        ),
        encoding="utf-8",
    )

    plan = inspect_external_workflow(
        str(source),
        interactive=True,
        cache_root=tmp_path / "cache",
        persist=False,
    )

    servers = {server["name"]: server for server in plan["mcp_servers"]}
    assert set(servers) == {"dangerous", "invalid", "valid"}
    assert servers["dangerous"]["command"] is None
    assert servers["invalid"]["command"] is None
    assert servers["invalid"]["args"] == []
    assert servers["valid"]["command"] == "valid-mcp"
    assert any("unsafe MCP command" in warning for warning in plan["warnings"])


def test_invalid_mcp_json_and_non_dict_servers_are_ignored(tmp_path) -> None:
    invalid_json = tmp_path / "invalid-json"
    invalid_json.mkdir()
    (invalid_json / ".mcp.json").write_text("{not-json", encoding="utf-8")
    invalid_plan = inspect_external_workflow(
        str(invalid_json),
        interactive=True,
        cache_root=tmp_path / "cache-invalid",
        persist=False,
    )
    assert invalid_plan["mcp_servers"] == []

    non_dict = tmp_path / "non-dict"
    non_dict.mkdir()
    (non_dict / ".mcp.json").write_text(json.dumps({"mcpServers": []}), encoding="utf-8")
    non_dict_plan = inspect_external_workflow(
        str(non_dict),
        interactive=True,
        cache_root=tmp_path / "cache-non-dict",
        persist=False,
    )
    assert non_dict_plan["mcp_servers"] == []


def test_missing_source_is_rejected(tmp_path) -> None:
    with pytest.raises(ExternalWorkflowError, match="not found"):
        inspect_external_workflow(
            str(tmp_path / "missing"),
            interactive=True,
            cache_root=tmp_path / "cache",
        )
