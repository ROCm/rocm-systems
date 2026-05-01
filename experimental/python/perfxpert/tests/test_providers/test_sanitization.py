"""Tests for perfxpert.providers._sanitization."""

import pytest

from perfxpert.providers._sanitization import (
    redact_paths,
    sanitize_messages,
)


def test_redact_paths_absolute():
    """Test redaction of absolute file paths."""
    text = "/home/user/repo/src/matmul.hip failed"
    out = redact_paths(text)
    assert "/home/user/repo/src/matmul.hip" not in out
    assert "[REDACTED]" in out


def test_redact_paths_relative():
    """Test redaction of relative paths with traversal."""
    text = "File ../../secret/file.hip was accessed"
    out = redact_paths(text)
    assert "../../secret/file.hip" not in out
    assert "[REDACTED_PATH]" in out


def test_redact_paths_windows():
    """Test redaction of Windows paths."""
    text = r"C:\Users\dev\project\kernel.cu compiled"
    out = redact_paths(text)
    assert r"C:\Users\dev\project\kernel.cu" not in out
    assert "[REDACTED]" in out


def test_redact_paths_multiple():
    """Test redaction of multiple paths in one string."""
    text = "/opt/rocm/lib and /tmp/output.db"
    out = redact_paths(text)
    assert "/opt/rocm/lib" not in out
    assert "/tmp/output.db" not in out
    assert out.count("[REDACTED]") >= 2


def test_sanitize_messages_recurses_nested_content():
    messages = [
        {
            "role": "user",
            "content": {
                "db": "/tmp/private/trace.db",
                "notes": ["../../secret/file.hip"],
            },
        }
    ]
    out = sanitize_messages(messages)
    assert "/tmp/private/trace.db" not in str(out)
    assert "../../secret/file.hip" not in str(out)
    assert "[REDACTED]" in str(out)
    assert "[REDACTED_PATH]" in str(out)


def test_sanitize_messages_redacts_path_field_values_by_key():
    messages = [
        {
            "role": "user",
            "content": {
                "database_path": "relative/run.db",
                "source_dir": "demo-app",
                "include_path": ["private/include"],
                "analysis_options": {"att_dir": "att-output"},
            },
        }
    ]
    out = sanitize_messages(messages)
    rendered = str(out)
    assert "relative/run.db" not in rendered
    assert "demo-app" not in rendered
    assert "private/include" not in rendered
    assert "att-output" not in rendered
    assert rendered.count("[REDACTED_PATH]") == 4
