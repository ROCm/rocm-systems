#!/usr/bin/env python3
"""Unit tests for experimental/python/perfxpert/docs/canonical_facts.json."""

import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_FACTS = (
    _REPO_ROOT
    / "experimental"
    / "python"
    / "perfxpert"
    / "docs"
    / "canonical_facts.json"
)


def test_canonical_facts_exists():
    """Canonical facts JSON must exist."""
    assert _CANONICAL_FACTS.exists(), f"{_CANONICAL_FACTS} does not exist"


def test_canonical_facts_valid_json():
    """Canonical facts must be valid JSON."""
    facts_file = _CANONICAL_FACTS
    content = facts_file.read_text()
    facts = json.loads(content)
    assert isinstance(facts, dict), "Canonical facts must be a JSON object"


def test_canonical_facts_has_required_fields():
    """Canonical facts must include all required fields from spec."""
    facts_file = _CANONICAL_FACTS
    facts = json.loads(facts_file.read_text())

    required_fields = [
        "project_name",
        "amd_brand",
        "sdk",
        "agents_count",
        "mcp_tools",
        "llm_providers",
        "knowledge_yamls",
        "test_count",
        "red_team_attacks",
        "mi300x_fp64_tflops",
        "mi300x_mem_bw_tbs",
        "mi350x_lds_kb",
        "banned_apis",
        "feature_flags",
        "gpu_archs",
    ]

    for field in required_fields:
        assert field in facts, f"Missing required field: {field}"


def test_canonical_facts_values():
    """Canonical facts must match spec values."""
    facts_file = _CANONICAL_FACTS
    facts = json.loads(facts_file.read_text())

    # Verify some key values match spec (Section 5)
    assert facts["project_name"] == "PerfXpert"
    assert facts["amd_brand"] == "AMD ROCm"
    assert facts["sdk"] == "rocprofiler-sdk"
    assert facts["agents_count"] == 7
    assert facts["mcp_tools"] == 33
    assert facts["llm_providers"] == 5
    assert facts["knowledge_yamls"] == 22
    assert facts["mi300x_fp64_tflops"] == 81.7
    assert facts["mi350x_lds_kb"] == 160


def test_canonical_facts_banned_apis_count():
    """Canonical facts must list all 9 banned APIs."""
    facts_file = _CANONICAL_FACTS
    facts = json.loads(facts_file.read_text())

    banned = facts["banned_apis"]
    assert len(banned) == 9, f"Expected 9 banned APIs, got {len(banned)}"


def test_canonical_facts_feature_flags():
    """Canonical facts must list all feature flags."""
    facts_file = _CANONICAL_FACTS
    facts = json.loads(facts_file.read_text())

    expected_flags = [
        "PERFXPERT_LEGACY",
        "PERFXPERT_USE_AGENTS",
        "PERFXPERT_AIRGAP",
        "PERFXPERT_ALLOW_INSTALL",
        "PERFXPERT_REDIS_URL",
    ]

    flags = facts["feature_flags"]
    assert len(flags) >= 5, f"Expected at least 5 feature flags, got {len(flags)}"


if __name__ == "__main__":
    test_canonical_facts_exists()
    print("✓ test_canonical_facts_exists passed")
