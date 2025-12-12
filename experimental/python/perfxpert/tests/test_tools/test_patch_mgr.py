"""Tests for perfxpert.tools.patch_mgr — EXECUTION class."""

from pathlib import Path

import pytest

from perfxpert.tools import patch_mgr
from perfxpert.tools._class import ToolClass
from perfxpert.tools._safety import PathConfinementError, ShellMetacharError


# -- tool-class marker ------------------------------------------------------

def test_patch_apply_is_execution_class():
    assert patch_mgr.apply.__tool_class__ == ToolClass.EXECUTION


def test_patch_revert_is_execution_class():
    assert patch_mgr.revert.__tool_class__ == ToolClass.EXECUTION


def test_patch_verify_output_is_execution_class():
    assert patch_mgr.verify_output.__tool_class__ == ToolClass.EXECUTION


# -- patch.apply -----------------------------------------------------------

def test_apply_writes_file_and_saves_bak(tmp_path: Path):
    src = tmp_path / "kernel.cpp"
    src.write_text("int x = 1;\n")
    new_content = "int x = 2;\n"

    result = patch_mgr.apply(tmp_path, "kernel.cpp", new_content)

    assert src.read_text() == new_content
    assert (tmp_path / "kernel.cpp.bak").exists()
    assert (tmp_path / "kernel.cpp.bak").read_text() == "int x = 1;\n"
    assert result["applied"] is True
    assert result["backup_path"].endswith(".bak")


def test_apply_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(PathConfinementError):
        patch_mgr.apply(tmp_path, "../etc/passwd", "evil\n")


def test_apply_rejects_shell_metachars_in_path(tmp_path: Path):
    (tmp_path / "ok.cpp").write_text("ok\n")
    with pytest.raises(ShellMetacharError):
        patch_mgr.apply(tmp_path, "ok.cpp;rm -rf ~", "evil\n")


def test_apply_rejects_absolute_path_outside_project(tmp_path: Path):
    with pytest.raises(PathConfinementError):
        patch_mgr.apply(tmp_path, "/etc/passwd", "evil\n")


def test_apply_is_idempotent_when_called_twice(tmp_path: Path):
    src = tmp_path / "f.cpp"
    src.write_text("original\n")
    patch_mgr.apply(tmp_path, "f.cpp", "v2\n")
    # Second apply preserves FIRST backup
    patch_mgr.apply(tmp_path, "f.cpp", "v3\n")
    bak = tmp_path / "f.cpp.bak"
    assert bak.read_text() == "original\n"
    assert src.read_text() == "v3\n"


# -- patch.revert ----------------------------------------------------------

def test_revert_restores_from_bak(tmp_path: Path):
    src = tmp_path / "k.cpp"
    src.write_text("original\n")
    patch_mgr.apply(tmp_path, "k.cpp", "modified\n")
    assert src.read_text() == "modified\n"

    result = patch_mgr.revert(tmp_path, "k.cpp")

    assert src.read_text() == "original\n"
    assert result["reverted"] is True
    # backup removed after successful revert
    assert not (tmp_path / "k.cpp.bak").exists()


def test_revert_without_bak_raises(tmp_path: Path):
    (tmp_path / "k.cpp").write_text("nothing to revert\n")
    with pytest.raises(FileNotFoundError):
        patch_mgr.revert(tmp_path, "k.cpp")


def test_revert_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(PathConfinementError):
        patch_mgr.revert(tmp_path, "../passwd")


# -- patch.verify_output ---------------------------------------------------

def test_verify_output_bit_exact_passes(tmp_path: Path):
    baseline = tmp_path / "baseline.out"
    baseline.write_bytes(b"\x00\x01\x02\x03")
    new = tmp_path / "new.out"
    new.write_bytes(b"\x00\x01\x02\x03")

    r = patch_mgr.verify_output(tmp_path, "baseline.out", "new.out")
    assert r["match"] is True
    assert r["method"] == "bit_exact"


def test_verify_output_bit_exact_fails(tmp_path: Path):
    (tmp_path / "a").write_bytes(b"foo")
    (tmp_path / "b").write_bytes(b"bar")
    r = patch_mgr.verify_output(tmp_path, "a", "b")
    assert r["match"] is False


def test_verify_output_allclose_passes_on_npy(tmp_path: Path):
    np = pytest.importorskip("numpy")
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0 + 1e-9, 2.0, 3.0 - 1e-9, 4.0])
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    r = patch_mgr.verify_output(tmp_path, "a.npy", "b.npy", tolerance=1e-6)
    assert r["match"] is True
    assert r["method"] == "np_allclose"


def test_verify_output_confines_paths(tmp_path: Path):
    (tmp_path / "local.out").write_bytes(b"x")
    with pytest.raises(PathConfinementError):
        patch_mgr.verify_output(tmp_path, "local.out", "/etc/passwd")
