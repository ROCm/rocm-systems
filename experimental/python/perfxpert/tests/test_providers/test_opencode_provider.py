"""Tests for perfxpert.providers.opencode_provider — subprocess-based."""

from unittest.mock import MagicMock, patch

import pytest

from perfxpert.providers._exceptions import (
    DryRunResponse,
    ProviderError,
    TimeoutError as PTO,
)


def _fake_completed(stdout="opencode-out", returncode=0):
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = returncode
    return m


def test_dry_run_no_subprocess(monkeypatch):
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", "/usr/local/bin/opencode")
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    from perfxpert.providers.opencode_provider import OpencodeProvider
    with patch("perfxpert.providers.opencode_provider.subprocess.run") as mr:
        assert OpencodeProvider().complete([], dry_run=True) is DryRunResponse
        mr.assert_not_called()


def test_recursion_guard_raises(monkeypatch):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "1")
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", "/usr/local/bin/opencode")
    from perfxpert.providers.opencode_provider import OpencodeProvider
    prov = OpencodeProvider()
    with pytest.raises(ProviderError, match="recursion guard"):
        prov.complete([{"role": "user", "content": "hi"}])


def test_binary_path_from_env(monkeypatch):
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", "/custom/path/opencode")
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    from perfxpert.providers.opencode_provider import OpencodeProvider
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        return_value=_fake_completed(),
    ) as mr:
        OpencodeProvider().complete([{"role": "user", "content": "hi"}])
        cmd = mr.call_args.args[0]
        assert cmd[0] == "/custom/path/opencode"


def test_binary_path_from_shutil_which(monkeypatch):
    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    from perfxpert.providers.opencode_provider import OpencodeProvider
    with patch(
        "perfxpert.providers.opencode_provider.shutil.which",
        return_value="/opt/bin/opencode",
    ):
        with patch(
            "perfxpert.providers.opencode_provider.subprocess.run",
            return_value=_fake_completed(stdout="ok"),
        ) as mr:
            OpencodeProvider().complete([{"role": "user", "content": "hi"}])
            cmd = mr.call_args.args[0]
            assert cmd[0] == "/opt/bin/opencode"


def test_binary_path_from_bundled_launcher(monkeypatch):
    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    import perfxpert.cli.opencode_launcher as opencode_launcher
    from perfxpert.providers.opencode_provider import OpencodeProvider

    monkeypatch.setattr(
        opencode_launcher,
        "resolve_opencode_binary",
        lambda: "/pkg/perfxpert/_bundled/opencode",
    )
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        return_value=_fake_completed(stdout="ok"),
    ) as mr:
        OpencodeProvider().complete([{"role": "user", "content": "hi"}])
        cmd = mr.call_args.args[0]
        assert cmd[0] == "/pkg/perfxpert/_bundled/opencode"


def test_no_binary_found_raises(monkeypatch):
    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    import perfxpert.cli.opencode_launcher as opencode_launcher
    from perfxpert.providers.opencode_provider import OpencodeProvider
    monkeypatch.setattr(
        opencode_launcher,
        "resolve_opencode_binary",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    with patch("perfxpert.providers.opencode_provider.shutil.which", return_value=None):
        with pytest.raises(ProviderError, match="opencode"):
            OpencodeProvider()


def test_subprocess_output_parsed(monkeypatch):
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", "/bin/opencode")
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    from perfxpert.providers.opencode_provider import OpencodeProvider
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        return_value=_fake_completed(stdout="the-answer"),
    ):
        r = OpencodeProvider().complete([{"role": "user", "content": "hi"}])
        assert r.content == "the-answer"
        assert r.provider == "opencode"


def test_timeout_mapped(monkeypatch):
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", "/bin/opencode")
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    import subprocess as sp

    from perfxpert.providers.opencode_provider import OpencodeProvider
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        side_effect=sp.TimeoutExpired(cmd="opencode", timeout=1.0),
    ):
        with pytest.raises(PTO):
            OpencodeProvider(timeout=1.0).complete([{"role": "user", "content": "x"}])


def test_nonzero_exit_raises(monkeypatch):
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", "/bin/opencode")
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    from perfxpert.providers.opencode_provider import OpencodeProvider
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        return_value=_fake_completed(stdout="", returncode=2),
    ):
        with pytest.raises(ProviderError):
            OpencodeProvider().complete([{"role": "user", "content": "x"}])


def test_registered():
    from perfxpert.providers import registry
    import perfxpert.providers.opencode_provider  # noqa: F401
    assert "opencode" in registry.list_providers()
