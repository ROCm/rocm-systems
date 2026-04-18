"""Validate bundled opencode.json against the real opencode JSON schema.

This test ensures our opencode.json conforms to the official opencode
schema (https://opencode.ai/config.json). Schema is cached on disk to
avoid network requests during CI.
"""

import json
import urllib.request
from importlib import resources
from pathlib import Path

import pytest


_OPENCODE_SCHEMA_URL = "https://opencode.ai/config.json"
_SCHEMA_CACHE = Path("/tmp/opencode-config-schema.json")


def _get_schema():
    """Fetch opencode schema, using disk cache if available."""
    if _SCHEMA_CACHE.exists():
        return json.loads(_SCHEMA_CACHE.read_text())
    try:
        req = urllib.request.Request(
            _OPENCODE_SCHEMA_URL,
            headers={"User-Agent": "perfxpert-test/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("utf-8")
        _SCHEMA_CACHE.write_text(text)
        return json.loads(text)
    except Exception as e:
        pytest.skip(f"opencode schema fetch failed (offline?): {e}")


def test_bundled_opencode_json_validates_against_opencode_schema():
    """Validate our opencode.json against the real opencode schema."""
    import jsonschema

    schema = _get_schema()
    with resources.as_file(
        resources.files("perfxpert") / "_bundled" / "opencode_config" / "opencode.json"
    ) as p:
        config = json.loads(p.read_text())
    jsonschema.validate(config, schema)


def test_bundled_config_files_exist():
    """Ensure all 4 required config files are present."""
    config_dir = (
        resources.files("perfxpert") / "_bundled" / "opencode_config"
    )
    required_files = ["opencode.json", "mcp.json", "amd-theme.json", "AGENTS.md"]
    for filename in required_files:
        with resources.as_file(config_dir / filename) as p:
            assert p.exists(), f"missing {filename} in bundled config"
