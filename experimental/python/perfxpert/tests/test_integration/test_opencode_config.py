"""Tests for _bundled/opencode_config/* — well-formed JSON, AMD branding present."""

import json
from pathlib import Path


CONFIG_DIR = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "_bundled" / "opencode_config"
)


def test_config_dir_exists():
    assert CONFIG_DIR.is_dir(), f"missing config dir {CONFIG_DIR}"


def test_opencode_json_is_valid_and_has_amd_branding():
    data = json.loads((CONFIG_DIR / "opencode.json").read_text())
    assert data["$schema"] == "https://opencode.ai/config.json"
    assert data["mcp"]["perfxpert"]["command"] == ["perfxpert-mcp"]


def test_theme_json_is_valid():
    theme = json.loads((CONFIG_DIR / "amd-theme.json").read_text())
    assert "palette" in theme
    assert theme["palette"]["primary"] == "#ED1C24"  # AMD red


def test_mcp_json_configures_perfxpert_mcp():
    mcp = json.loads((CONFIG_DIR / "mcp.json").read_text())
    assert "mcpServers" in mcp
    assert "perfxpert" in mcp["mcpServers"]
    assert mcp["mcpServers"]["perfxpert"]["command"] == "perfxpert-mcp"


def test_agents_md_is_nontrivial():
    agents = (CONFIG_DIR / "AGENTS.md").read_text()
    assert len(agents) > 500
    assert "perfxpert" in agents.lower()
    assert "mcp" in agents.lower()
    # Must warn against the forbidden behaviors listed in the master.md
    assert "never" in agents.lower() or "NEVER" in agents


def test_theme_has_prompt_prefix_branding():
    theme = json.loads((CONFIG_DIR / "amd-theme.json").read_text())
    assert "ROCm PerfXpert" in theme.get("prompt_prefix", "")
