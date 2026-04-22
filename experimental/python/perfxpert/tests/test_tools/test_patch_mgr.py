"""Tests for perfxpert.tools.patch_mgr — EXECUTION class."""

import os
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


def test_apply_rejects_symlink_swap_after_validation(tmp_path: Path, monkeypatch):
    src = tmp_path / "victim.cpp"
    src.write_text("original\n")
    outside = tmp_path.parent / "outside.cpp"
    outside.write_text("outside\n")
    original_prepare_target = patch_mgr._prepare_target

    def swap_after_validation(project_root, rel_path):
        target = original_prepare_target(project_root, rel_path)
        src.unlink()
        src.symlink_to(outside)
        return target

    monkeypatch.setattr(patch_mgr, "_prepare_target", swap_after_validation)

    with pytest.raises(PathConfinementError):
        patch_mgr.apply(tmp_path, "victim.cpp", "mutated\n")

    assert outside.read_text() == "outside\n"


def test_apply_rejects_hardlink_to_external_inode(tmp_path: Path):
    outside = tmp_path.parent / "outside-hardlink.cpp"
    outside.write_text("outside\n")
    linked = tmp_path / "linked.cpp"
    os.link(outside, linked)

    with pytest.raises(PathConfinementError, match="multiple hard links"):
        patch_mgr.apply(tmp_path, "linked.cpp", "mutated\n")

    assert outside.read_text() == "outside\n"


def test_apply_replaces_inode_if_hardlink_added_midflight(tmp_path: Path, monkeypatch):
    src = tmp_path / "victim.cpp"
    src.write_text("original\n")
    outside = tmp_path.parent / "outside-midflight.cpp"
    original_create_backup = patch_mgr._create_backup

    def link_during_backup(project_root, backup, backup_meta, original_bytes, original_mode):
        if not outside.exists():
            os.link(src, outside)
        return original_create_backup(
            project_root,
            backup,
            backup_meta,
            original_bytes,
            original_mode,
        )

    monkeypatch.setattr(patch_mgr, "_create_backup", link_during_backup)

    result = patch_mgr.apply(tmp_path, "victim.cpp", "mutated\n")

    assert result["applied"] is True
    assert src.read_text() == "mutated\n"
    assert outside.read_text() == "original\n"
    assert src.stat().st_ino != outside.stat().st_ino


def test_apply_is_idempotent_when_called_twice(tmp_path: Path):
    src = tmp_path / "f.cpp"
    src.write_text("original\n")
    patch_mgr.apply(tmp_path, "f.cpp", "v2\n")
    # Second apply preserves FIRST backup
    patch_mgr.apply(tmp_path, "f.cpp", "v3\n")
    bak = tmp_path / "f.cpp.bak"
    assert bak.read_text() == "original\n"
    assert src.read_text() == "v3\n"


def test_apply_rejects_unmanaged_preexisting_backup(tmp_path: Path):
    src = tmp_path / "f.cpp"
    src.write_text("original\n")
    (tmp_path / "f.cpp.bak").write_text("stale\n")

    with pytest.raises(FileExistsError, match="unexpected existing backup"):
        patch_mgr.apply(tmp_path, "f.cpp", "v2\n")


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


def test_revert_restores_missing_target_from_bak(tmp_path: Path):
    src = tmp_path / "k.cpp"
    src.write_text("original\n")
    src.chmod(0o755)
    patch_mgr.apply(tmp_path, "k.cpp", "modified\n")
    src.unlink()

    result = patch_mgr.revert(tmp_path, "k.cpp")

    assert result["reverted"] is True
    assert src.read_text() == "original\n"
    assert src.stat().st_mode & 0o777 == 0o755
    assert not (tmp_path / "k.cpp.bak").exists()


def test_revert_restores_original_mode_when_target_still_exists(tmp_path: Path):
    src = tmp_path / "k.cpp"
    src.write_text("original\n")
    src.chmod(0o644)
    patch_mgr.apply(tmp_path, "k.cpp", "modified\n")
    src.chmod(0o777)

    result = patch_mgr.revert(tmp_path, "k.cpp")

    assert result["reverted"] is True
    assert src.read_text() == "original\n"
    assert src.stat().st_mode & 0o777 == 0o644


def test_revert_without_bak_raises(tmp_path: Path):
    (tmp_path / "k.cpp").write_text("nothing to revert\n")
    with pytest.raises(FileNotFoundError):
        patch_mgr.revert(tmp_path, "k.cpp")


def test_revert_rejects_unmanaged_backup_metadata(tmp_path: Path):
    src = tmp_path / "k.cpp"
    src.write_text("current\n")
    (tmp_path / "k.cpp.bak").write_text("stale\n")
    (tmp_path / "k.cpp.bak.meta").write_text('{"mode": 420}')

    with pytest.raises(ValueError, match="unexpected backup metadata"):
        patch_mgr.revert(tmp_path, "k.cpp")


def test_revert_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(PathConfinementError):
        patch_mgr.revert(tmp_path, "../passwd")


def test_revert_rejects_symlink_swap_after_validation(tmp_path: Path, monkeypatch):
    src = tmp_path / "k.cpp"
    src.write_text("original\n")
    patch_mgr.apply(tmp_path, "k.cpp", "modified\n")
    outside = tmp_path.parent / "outside.cpp"
    outside.write_text("outside\n")
    original_prepare_target = patch_mgr._prepare_target

    def swap_after_validation(project_root, rel_path):
        target = original_prepare_target(project_root, rel_path)
        src.unlink()
        src.symlink_to(outside)
        return target

    monkeypatch.setattr(patch_mgr, "_prepare_target", swap_after_validation)

    with pytest.raises(PathConfinementError):
        patch_mgr.revert(tmp_path, "k.cpp")

    assert outside.read_text() == "outside\n"


def test_revert_rejects_hardlinked_target(tmp_path: Path):
    src = tmp_path / "k.cpp"
    src.write_text("original\n")
    patch_mgr.apply(tmp_path, "k.cpp", "modified\n")
    src.unlink()
    outside = tmp_path.parent / "outside-hardlink-revert.cpp"
    outside.write_text("outside\n")
    os.link(outside, src)

    with pytest.raises(PathConfinementError, match="multiple hard links"):
        patch_mgr.revert(tmp_path, "k.cpp")

    assert outside.read_text() == "outside\n"


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


def test_verify_output_allclose_passes_on_npz(tmp_path: Path):
    np = pytest.importorskip("numpy")
    np.savez(tmp_path / "a.npz", foo=np.array([1.0, 2.0]), bar=np.array([3.0, 4.0]))
    np.savez(
        tmp_path / "b.npz",
        foo=np.array([1.0 + 1e-9, 2.0]),
        bar=np.array([3.0, 4.0 - 1e-9]),
    )

    r = patch_mgr.verify_output(tmp_path, "a.npz", "b.npz", tolerance=1e-6)

    assert r["match"] is True
    assert r["method"] == "np_allclose"


def test_verify_output_mixed_numpy_formats_falls_back_to_mismatch(tmp_path: Path):
    np = pytest.importorskip("numpy")
    np.savez(tmp_path / "a.npz", foo=np.array([1.0, 2.0]))
    np.save(tmp_path / "b.npy", np.array([1.0, 2.0]))

    r = patch_mgr.verify_output(tmp_path, "a.npz", "b.npy", tolerance=1e-6)

    assert r["match"] is False
    assert r["method"] == "bit_exact"


def test_verify_output_confines_paths(tmp_path: Path):
    (tmp_path / "local.out").write_bytes(b"x")
    with pytest.raises(PathConfinementError):
        patch_mgr.verify_output(tmp_path, "local.out", "/etc/passwd")


def test_verify_output_allows_hardlinked_read_only_compare(tmp_path: Path):
    baseline = tmp_path / "baseline.out"
    baseline.write_bytes(b"ok")
    new = tmp_path / "new.out"
    os.link(baseline, new)

    result = patch_mgr.verify_output(tmp_path, "baseline.out", "new.out")

    assert result["match"] is True
    assert result["method"] == "bit_exact"


def test_verify_output_rejects_symlink_swap_after_validation(tmp_path: Path, monkeypatch):
    baseline = tmp_path / "baseline.out"
    new = tmp_path / "new.out"
    outside = tmp_path.parent / "outside.out"
    baseline.write_bytes(b"ok")
    new.write_bytes(b"ok")
    outside.write_bytes(b"outside")
    original_prepare_target = patch_mgr._prepare_target

    def swap_after_validation(project_root, rel_path):
        target = original_prepare_target(project_root, rel_path)
        if rel_path == "new.out":
            new.unlink()
            new.symlink_to(outside)
        return target

    monkeypatch.setattr(patch_mgr, "_prepare_target", swap_after_validation)

    with pytest.raises(PathConfinementError):
        patch_mgr.verify_output(tmp_path, "baseline.out", "new.out")
