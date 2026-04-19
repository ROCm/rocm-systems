#!/usr/bin/env python3
"""Unit tests for perfxpert.tools._safety env sanitization (task #52)."""

import os
import sys
from pathlib import Path

# Add perfxpert to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experimental/python/perfxpert"))

from perfxpert.tools._safety import build_safe_env


def test_build_safe_env_strips_api_key_suffix():
    """build_safe_env must strip env vars ending in _API_KEY."""
    # Create test env
    test_env = {
        'ANTHROPIC_API_KEY': 'sk-ant-...',
        'OPENAI_API_KEY': 'sk-org-...',
        'PATH': '/usr/bin',
        'HOME': '/home/user',
    }

    with patch_environ(test_env):
        safe = build_safe_env()
        assert 'ANTHROPIC_API_KEY' not in safe
        assert 'OPENAI_API_KEY' not in safe
        assert 'PATH' in safe
        assert 'HOME' in safe


def test_build_safe_env_strips_token_suffix():
    """build_safe_env must strip env vars ending in _TOKEN."""
    test_env = {
        'GITHUB_TOKEN': 'ghp_...',
        'NPM_TOKEN': 'npm_...',
        'PATH': '/usr/bin',
    }

    with patch_environ(test_env):
        safe = build_safe_env()
        assert 'GITHUB_TOKEN' not in safe
        assert 'NPM_TOKEN' not in safe
        assert 'PATH' in safe


def test_build_safe_env_strips_secret_suffix():
    """build_safe_env must strip env vars ending in _SECRET."""
    test_env = {
        'MY_SECRET': 'secret_value',
        'ANOTHER_SECRET': 'another_value',
        'PATH': '/usr/bin',
    }

    with patch_environ(test_env):
        safe = build_safe_env()
        assert 'MY_SECRET' not in safe
        assert 'ANOTHER_SECRET' not in safe
        assert 'PATH' in safe


def test_build_safe_env_allows_extra():
    """build_safe_env must allow caller to add extra safe vars."""
    test_env = {
        'API_KEY': 'secret',  # Should be stripped
        'PATH': '/usr/bin',
    }

    with patch_environ(test_env):
        safe = build_safe_env(extra={'SAFE_CUSTOM_VAR': 'value'})
        assert 'API_KEY' not in safe
        assert 'SAFE_CUSTOM_VAR' in safe
        assert safe['SAFE_CUSTOM_VAR'] == 'value'


def test_build_safe_env_rocm_vars():
    """build_safe_env must allow ROCm-related env vars."""
    test_env = {
        'ROCM_PATH': '/opt/rocm',
        'HIP_PATH': '/opt/hip',
        'ROCPROFV3_METRICS': 'all',
        'ROCPROFILER_OPTION': 'value',
        'PATH': '/usr/bin',
    }

    with patch_environ(test_env):
        safe = build_safe_env()
        assert 'ROCM_PATH' in safe
        assert 'HIP_PATH' in safe
        assert 'ROCPROFV3_METRICS' in safe
        assert 'ROCPROFILER_OPTION' in safe


class patch_environ:
    """Context manager to temporarily replace os.environ."""

    def __init__(self, new_env):
        self.new_env = new_env
        self.old_env = None

    def __enter__(self):
        self.old_env = os.environ.copy()
        os.environ.clear()
        os.environ.update(self.new_env)
        return self

    def __exit__(self, *args):
        os.environ.clear()
        os.environ.update(self.old_env)


if __name__ == "__main__":
    test_build_safe_env_strips_api_key_suffix()
    print("✓ test_build_safe_env_strips_api_key_suffix passed")
