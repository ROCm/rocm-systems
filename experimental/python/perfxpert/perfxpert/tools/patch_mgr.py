"""patch_mgr — patch.apply / revert / verify_output.

EXECUTION class. Never register with MCP server (§5.8).

All three tools route through `perfxpert.tools._safety` for path confinement
and shell-metachar denial.
"""

from __future__ import annotations

import errno
import io
import json
import os
import stat
from pathlib import Path
from typing import Any, Dict

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools._safety import (
    PathConfinementError,
    confine_to_project_root,
    reject_shell_metachars,
)


_BACKUP_SUFFIX = ".bak"
_BACKUP_META_SUFFIX = ".meta"
_OPEN_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_OPEN_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_OPEN_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _prepare_target(project_root: Path, rel_path: str) -> Path:
    """Run all §5.8 checks on the target path, return resolved Path."""
    reject_shell_metachars(rel_path)
    return confine_to_project_root(Path(project_root), rel_path)


def _relative_target(project_root: Path, target: Path) -> tuple[Path, Path]:
    root = Path(project_root).resolve(strict=True)
    try:
        rel_target = target.relative_to(root)
    except ValueError as e:
        raise PathConfinementError(
            f"path {target} is outside project root {root}"
        ) from e
    if not rel_target.parts:
        raise PathConfinementError("target path must reference a file beneath project root")
    return root, rel_target


def _raise_confined_open_error(target: Path, err: OSError) -> None:
    if err.errno in {errno.ELOOP, errno.ENOTDIR}:
        raise PathConfinementError(
            f"path changed after validation or traversed a symlink: {target}"
        ) from err
    raise err


def _open_confined_parent_dir(project_root: Path, target: Path) -> tuple[int, str]:
    root, rel_target = _relative_target(project_root, target)
    dir_fd = os.open(str(root), os.O_RDONLY | _OPEN_DIRECTORY | _OPEN_CLOEXEC)
    try:
        for part in rel_target.parts[:-1]:
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | _OPEN_DIRECTORY | _OPEN_NOFOLLOW | _OPEN_CLOEXEC,
                    dir_fd=dir_fd,
                )
            except OSError as err:
                _raise_confined_open_error(target, err)
            os.close(dir_fd)
            dir_fd = next_fd
        return dir_fd, rel_target.name
    except Exception:
        os.close(dir_fd)
        raise


def _ensure_regular_file(fd: int, target: Path, *, require_single_link: bool) -> None:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise PathConfinementError(f"target is not a regular file: {target}")
    if require_single_link and st.st_nlink > 1:
        os.close(fd)
        raise PathConfinementError(
            f"target has multiple hard links and cannot be mutated safely: {target}"
        )


def _open_confined_file(
    project_root: Path,
    target: Path,
    flags: int,
    mode: int = 0o666,
    *,
    require_single_link: bool = True,
) -> int:
    dir_fd, name = _open_confined_parent_dir(project_root, target)
    try:
        open_flags = flags | _OPEN_NOFOLLOW | _OPEN_CLOEXEC
        if flags & os.O_CREAT:
            fd = os.open(name, open_flags, mode, dir_fd=dir_fd)
        else:
            fd = os.open(name, open_flags, dir_fd=dir_fd)
    except OSError as err:
        _raise_confined_open_error(target, err)
    finally:
        os.close(dir_fd)
    _ensure_regular_file(fd, target, require_single_link=require_single_link)
    return fd


def _read_fd_bytes(fd: int) -> bytes:
    with os.fdopen(os.dup(fd), "rb", closefd=True) as handle:
        handle.seek(0)
        return handle.read()


def _read_confined_bytes(project_root: Path, target: Path) -> bytes:
    fd = _open_confined_file(
        project_root,
        target,
        os.O_RDONLY,
        require_single_link=False,
    )
    try:
        return _read_fd_bytes(fd)
    finally:
        os.close(fd)


def _backup_meta_path(backup: Path) -> Path:
    return backup.with_suffix(backup.suffix + _BACKUP_META_SUFFIX)


def _rewrite_fd_bytes(fd: int, data: bytes) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    written = 0
    while written < len(data):
        written += os.write(fd, data[written:])
    os.fsync(fd)


def _replace_confined_file(project_root: Path, target: Path, data: bytes, mode: int) -> None:
    dir_fd, name = _open_confined_parent_dir(project_root, target)
    temp_name = None
    fd = None
    try:
        for attempt in range(100):
            candidate = f".{name}.perfxpert-tmp-{os.getpid()}-{attempt}"
            try:
                fd = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _OPEN_NOFOLLOW | _OPEN_CLOEXEC,
                    mode,
                    dir_fd=dir_fd,
                )
                temp_name = candidate
                break
            except FileExistsError:
                continue
        if fd is None or temp_name is None:
            raise RuntimeError(f"could not allocate confined temp file for {target}")
        _rewrite_fd_bytes(fd, data)
        os.fchmod(fd, mode)
        os.close(fd)
        fd = None
        os.replace(temp_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
        os.close(dir_fd)


def _create_backup(
    project_root: Path,
    backup: Path,
    backup_meta: Path,
    original_bytes: bytes,
    original_mode: int,
) -> None:
    try:
        backup_fd = _open_confined_file(
            project_root,
            backup,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            original_mode,
        )
    except FileExistsError:
        existing_fd = _open_confined_file(project_root, backup, os.O_RDONLY)
        os.close(existing_fd)
        try:
            meta = _read_backup_meta(project_root, backup_meta)
        except (FileNotFoundError, ValueError) as e:
            raise FileExistsError(f"unexpected existing backup for {backup}") from e
        if meta.get("perfxpert_backup") is not True:
            raise FileExistsError(f"unexpected existing backup for {backup}")
        return
    try:
        _rewrite_fd_bytes(backup_fd, original_bytes)
    finally:
        os.close(backup_fd)
    meta_fd = None
    try:
        meta_fd = _open_confined_file(
            project_root,
            backup_meta,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        _rewrite_fd_bytes(
            meta_fd,
            json.dumps(
                {"perfxpert_backup": True, "mode": original_mode},
                sort_keys=True,
            ).encode("utf-8"),
        )
    except Exception:
        try:
            _unlink_confined(project_root, backup)
        except FileNotFoundError:
            pass
        raise
    finally:
        if meta_fd is not None:
            os.close(meta_fd)


def _read_backup_meta(project_root: Path, backup_meta: Path) -> dict[str, Any]:
    meta = json.loads(_read_confined_bytes(project_root, backup_meta).decode("utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"backup metadata must be an object: {backup_meta}")
    if meta.get("perfxpert_backup") is not True:
        raise ValueError(f"unexpected backup metadata for {backup_meta}")
    if not isinstance(meta.get("mode"), int):
        raise ValueError(f"backup metadata missing integer mode: {backup_meta}")
    return meta


def _unlink_confined(project_root: Path, target: Path) -> None:
    dir_fd, name = _open_confined_parent_dir(project_root, target)
    try:
        os.unlink(name, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


@tool_class(ToolClass.EXECUTION)
def apply(project_root: Path, rel_path: str, new_content: str) -> Dict[str, Any]:
    """Replace file contents; write `.bak` of prior content if no backup exists.

    Args:
        project_root: project root directory. Anchors path confinement.
        rel_path: file path relative to project_root. Must NOT contain `..`,
                  symlinks escaping root, or shell metachars.
        new_content: full new file content (text mode).

    Returns:
        {"applied": True, "backup_path": str, "target_path": str}

    Raises:
        PathConfinementError, ShellMetacharError — if the path violates §5.8.
        FileNotFoundError — if the target does not exist.
    """
    target = _prepare_target(project_root, rel_path)
    backup = target.with_suffix(target.suffix + _BACKUP_SUFFIX)
    backup_meta = _backup_meta_path(backup)
    target_fd = _open_confined_file(project_root, target, os.O_RDONLY)
    try:
        target_mode = stat.S_IMODE(os.fstat(target_fd).st_mode)
        original_bytes = _read_fd_bytes(target_fd)
        # Preserve the FIRST backup across repeated apply() calls.
        _create_backup(project_root, backup, backup_meta, original_bytes, target_mode)
    finally:
        os.close(target_fd)
    _replace_confined_file(project_root, target, new_content.encode("utf-8"), target_mode)
    return {
        "applied": True,
        "backup_path": str(backup),
        "target_path": str(target),
    }


@tool_class(ToolClass.EXECUTION)
def revert(project_root: Path, rel_path: str) -> Dict[str, Any]:
    """Restore file from its `.bak` backup; remove backup after restore.

    Raises:
        FileNotFoundError if no `.bak` exists for this path.
        PathConfinementError, ShellMetacharError on §5.8 violation.
    """
    target = _prepare_target(project_root, rel_path)
    backup = target.with_suffix(target.suffix + _BACKUP_SUFFIX)
    backup_meta = _backup_meta_path(backup)
    backup_fd = _open_confined_file(project_root, backup, os.O_RDONLY)
    try:
        original_bytes = _read_fd_bytes(backup_fd)
    finally:
        os.close(backup_fd)
    backup_mode = int(_read_backup_meta(project_root, backup_meta)["mode"])

    try:
        target_fd = _open_confined_file(project_root, target, os.O_RDONLY)
    except FileNotFoundError:
        pass
    else:
        try:
            os.fstat(target_fd)
        finally:
            os.close(target_fd)
    _replace_confined_file(project_root, target, original_bytes, backup_mode)
    _unlink_confined(project_root, backup)
    _unlink_confined(project_root, backup_meta)
    return {
        "reverted": True,
        "target_path": str(target),
    }


@tool_class(ToolClass.EXECUTION)
def verify_output(
    project_root: Path,
    baseline_rel: str,
    new_rel: str,
    *,
    tolerance: float | None = None,
) -> Dict[str, Any]:
    """Compare two output files — bit-exact or np.allclose.

    Method selected by file extension:
    - `.npy` / `.npz` → `numpy.load` + `np.allclose(..., rtol=tolerance)`
    - everything else → `bytes ==`

    `tolerance=None` means bit-exact even for numpy arrays.

    Returns:
        {"match": bool, "method": "bit_exact"|"np_allclose", "reason": str}
    """
    baseline = _prepare_target(project_root, baseline_rel)
    new = _prepare_target(project_root, new_rel)

    # Numeric path for .npy/.npz
    if baseline.suffix == new.suffix and baseline.suffix in {".npy", ".npz"} and tolerance is not None:
        import numpy as np  # imported lazily; test guards with importorskip
        a = np.load(io.BytesIO(_read_confined_bytes(project_root, baseline)))
        b = np.load(io.BytesIO(_read_confined_bytes(project_root, new)))
        if baseline.suffix == ".npz":
            a_keys = sorted(a.files)
            b_keys = sorted(b.files)
            if a_keys != b_keys:
                return {
                    "match": False,
                    "method": "np_allclose",
                    "reason": f"npz keys differ: {a_keys!r} != {b_keys!r}",
                }
            match = all(
                bool(np.allclose(a[key], b[key], rtol=tolerance, atol=tolerance))
                for key in a_keys
            )
        else:
            match = bool(np.allclose(a, b, rtol=tolerance, atol=tolerance))
        return {
            "match": match,
            "method": "np_allclose",
            "reason": f"np.allclose(rtol={tolerance}) = {match}",
        }

    # Bit-exact path
    match = _read_confined_bytes(project_root, baseline) == _read_confined_bytes(
        project_root, new
    )
    return {
        "match": match,
        "method": "bit_exact",
        "reason": "byte-equal" if match else "byte-unequal",
    }
