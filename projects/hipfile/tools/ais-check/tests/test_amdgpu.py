"""Tests for amdgpu_supports_ais()."""

# pylint: disable=missing-function-docstring

import io


def _patch_open(monkeypatch, ais_check, *, content=None, exc=None):
    def fake_open(path, *_args, **_kwargs):
        assert path == "/proc/kallsyms"
        if exc is not None:
            raise exc
        return io.StringIO(content)

    monkeypatch.setattr(ais_check, "open", fake_open, raising=False)


def test_symbol_present(monkeypatch, ais_check):
    _patch_open(
        monkeypatch,
        ais_check,
        content="0000 t some_symbol\n0000 t kfd_ais_rw_file\n",
    )

    assert ais_check.amdgpu_supports_ais() is True


def test_symbol_absent(monkeypatch, ais_check):
    _patch_open(monkeypatch, ais_check, content="0000 t some_symbol\n0000 t other\n")

    assert ais_check.amdgpu_supports_ais() is False


def test_empty_kallsyms(monkeypatch, ais_check):
    _patch_open(monkeypatch, ais_check, content="")

    assert ais_check.amdgpu_supports_ais() is False


def test_file_not_found_warns(monkeypatch, capsys, ais_check):
    _patch_open(monkeypatch, ais_check, exc=FileNotFoundError())

    assert ais_check.amdgpu_supports_ais() is False
    assert "file not found" in capsys.readouterr().err


def test_permission_denied_warns(monkeypatch, capsys, ais_check):
    _patch_open(monkeypatch, ais_check, exc=PermissionError())

    assert ais_check.amdgpu_supports_ais() is False
    assert "permission denied" in capsys.readouterr().err
