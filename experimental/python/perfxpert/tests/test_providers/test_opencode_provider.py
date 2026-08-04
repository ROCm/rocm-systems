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
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    import perfxpert.cli.opencode_launcher as opencode_launcher
    from perfxpert.providers.opencode_provider import OpencodeProvider

    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: "/pkg/opencode")
    with patch("perfxpert.providers.opencode_provider.subprocess.run") as mr:
        assert OpencodeProvider().complete([], dry_run=True) is DryRunResponse
        mr.assert_not_called()


def test_recursion_guard_raises(monkeypatch):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "1")
    import perfxpert.cli.opencode_launcher as opencode_launcher
    from perfxpert.providers.opencode_provider import OpencodeProvider

    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: "/pkg/opencode")
    prov = OpencodeProvider()
    with pytest.raises(ProviderError, match="recursion guard"):
        prov.complete([{"role": "user", "content": "hi"}])


def test_explicit_constructor_binary_path(monkeypatch):
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", "/custom/path/opencode")
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    from perfxpert.providers.opencode_provider import OpencodeProvider
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        return_value=_fake_completed(),
    ) as mr:
        OpencodeProvider(opencode_path="/custom/path/opencode").complete(
            [{"role": "user", "content": "hi"}]
        )
        cmd = mr.call_args.args[0]
        assert cmd[0] == "/custom/path/opencode"


def test_env_override_is_ignored_for_bundled_provider(monkeypatch):
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", "/custom/path/opencode")
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


def test_token_ceiling_is_declared_unenforceable_rather_than_dropped(
    monkeypatch, caplog
):
    """A limit that is silently ignored is worse than one that is refused.

    ``complete()`` takes ``max_tokens`` for interface parity, but the opencode
    CLI has no output-token flag to forward it to, so callers were getting a
    ceiling that did nothing while the framework recorded one as applied.
    Trimming the reply afterwards would bound neither spend nor runtime, so
    the backend says plainly that it cannot honour it.
    """
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    import perfxpert.providers.opencode_provider as ocp
    from perfxpert.providers.opencode_provider import OpencodeProvider

    monkeypatch.setattr(ocp, "_TOKEN_CEILING_WARNED", False)
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        return_value=_fake_completed(stdout="ok"),
    ):
        with caplog.at_level("WARNING"):
            OpencodeProvider(opencode_path="/custom/opencode").complete(
                [{"role": "user", "content": "hi"}], max_tokens=256
            )

    assert "cannot enforce" in caplog.text


def test_no_token_ceiling_warning_when_none_was_asked_for(monkeypatch, caplog):
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    import perfxpert.providers.opencode_provider as ocp
    from perfxpert.providers.opencode_provider import OpencodeProvider

    monkeypatch.setattr(ocp, "_TOKEN_CEILING_WARNED", False)
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        return_value=_fake_completed(stdout="ok"),
    ):
        with caplog.at_level("WARNING"):
            OpencodeProvider(opencode_path="/custom/opencode").complete(
                [{"role": "user", "content": "hi"}]
            )

    assert "cannot enforce" not in caplog.text


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
    with pytest.raises(ProviderError, match="bundled patched binary"):
        OpencodeProvider()


def test_subprocess_output_parsed(monkeypatch):
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    from perfxpert.providers.opencode_provider import OpencodeProvider
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        return_value=_fake_completed(stdout="the-answer"),
    ):
        r = OpencodeProvider(opencode_path="/bin/opencode").complete(
            [{"role": "user", "content": "hi"}]
        )
        assert r.content == "the-answer"
        assert r.provider == "opencode"


def test_timeout_mapped(monkeypatch):
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    import subprocess as sp

    from perfxpert.providers.opencode_provider import OpencodeProvider
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        side_effect=sp.TimeoutExpired(cmd="opencode", timeout=1.0),
    ):
        with pytest.raises(PTO):
            OpencodeProvider(opencode_path="/bin/opencode", timeout=1.0).complete(
                [{"role": "user", "content": "x"}]
            )


def test_nonzero_exit_raises(monkeypatch):
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    from perfxpert.providers.opencode_provider import OpencodeProvider
    with patch(
        "perfxpert.providers.opencode_provider.subprocess.run",
        return_value=_fake_completed(stdout="", returncode=2),
    ):
        with pytest.raises(ProviderError):
            OpencodeProvider(opencode_path="/bin/opencode").complete(
                [{"role": "user", "content": "x"}]
            )


def test_registered():
    from perfxpert.providers import registry
    import perfxpert.providers.opencode_provider  # noqa: F401
    assert "opencode" in registry.list_providers()
