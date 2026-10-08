# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pin PATH and ROCm prefix variables to the extracted tarball."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import NoReturn


from .command import log


def _split_env_list(value: str) -> list[str]:
    """Split a PATH-like variable. CMake prefixes may use ':' or ';'."""
    parts: list[str] = []
    for chunk in value.replace(";", os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _resolved(path: Path) -> Path | None:
    try:
        return path.resolve()
    except OSError:
        return None


def _is_under(path: Path, root: Path) -> bool:
    if path == root:
        return True
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _looks_like_rocm_prefix(prefix: Path) -> bool:
    """True when ``prefix`` is a ROCm/TheRock install root, not a random directory."""
    name = str(prefix).lower()
    if "therock" in name or "/opt/rocm" in name or prefix.name == "rocm":
        return True
    return any(
        (prefix / rel).exists()
        for rel in (
            "bin/hipcc",
            "bin/rocminfo",
            ".info/version",
            "share/therock",
        )
    )


def _prefix_of_entry(entry: str) -> list[Path]:
    """Candidate install roots for one PATH / CMAKE_PREFIX_PATH entry."""
    p = Path(entry)
    out = [p]
    if p.name in {"bin", "lib", "lib64", "include"}:
        out.append(p.parent)
        if p.parent.name == "llvm":
            out.append(p.parent.parent)
    elif p.name == "pkgconfig" and p.parent.name in {"lib", "lib64"}:
        out.append(p.parent.parent)
    return out


def _foreign_rocm_roots(base_env: dict, rocm_dir: Path) -> list[Path]:
    """Other ROCm/TheRock prefixes already present in the caller's environment.

    AAC6 nodes often have a host TheRock (for example 10.1) on PATH via
    HIP_PATH, CMAKE_PREFIX_PATH, or a leftover ``therock-tarball`` checkout.
    Those must not stay visible to CMake: Packages.cmake prepends
    ``$ENV{CMAKE_PREFIX_PATH}`` ahead of ``-DCMAKE_PREFIX_PATH``.
    """
    own = _resolved(rocm_dir)
    raw: list[str] = []
    for key in ("ROCM_PATH", "HIP_PATH", "HSA_PATH", "ROCM_HOME", "HIP_CLANG_PATH"):
        val = base_env.get(key, "").strip()
        if val:
            raw.append(val)
    for key in (
        "CMAKE_PREFIX_PATH",
        "CMAKE_MODULE_PATH",
        "PATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "PKG_CONFIG_PATH",
    ):
        raw.extend(_split_env_list(base_env.get(key, "")))

    roots: list[Path] = []
    seen: set[str] = set()
    for entry in raw:
        for candidate in _prefix_of_entry(entry):
            resolved = _resolved(candidate)
            if resolved is None or not resolved.exists():
                continue
            if own is not None and _is_under(resolved, own):
                continue
            key = str(resolved)
            if key in seen or not _looks_like_rocm_prefix(resolved):
                continue
            seen.add(key)
            roots.append(resolved)
    # Keep the install root only. bin/ and lib/ under it are the same leftover.
    roots.sort(key=lambda p: len(p.parts))
    shortest: list[Path] = []
    for root in roots:
        if any(_is_under(root, earlier) for earlier in shortest):
            continue
        shortest.append(root)
    return shortest


def _scrub_env_list(
    value: str, foreign: list[Path], rocm_dir: Path
) -> tuple[list[str], list[str]]:
    """Drop entries that live under a foreign ROCm prefix. Return (kept, dropped)."""
    own = _resolved(rocm_dir)
    kept: list[str] = []
    dropped: list[str] = []
    for entry in _split_env_list(value):
        resolved = _resolved(Path(entry))
        under_own = own is not None and resolved is not None and _is_under(resolved, own)
        under_foreign = resolved is not None and any(
            _is_under(resolved, root) for root in foreign
        )
        # A stale therock path that no longer exists still has to come off PATH.
        named_therock = "therock" in entry.lower() and not under_own
        if under_foreign or named_therock:
            dropped.append(entry)
        else:
            kept.append(entry)
    return kept, dropped


def make_rocm_env(base_env: dict, rocm_dir: Path) -> dict:
    """Environment that builds and runs against ``rocm_dir`` only.

    Overrides the ROCm prefix variables and strips every other ROCm/TheRock
    prefix out of the search paths. Prepending the tarball is not enough:
    HIP_PATH and CMAKE_PREFIX_PATH are absolute, and a host ``therock-tarball``
    left on PATH is still searched by CMake's find_program/find_package.
    """
    env = dict(base_env)
    foreign = _foreign_rocm_roots(base_env, rocm_dir)
    dropped: list[str] = []

    def scrub(key: str) -> list[str]:
        kept, gone = _scrub_env_list(env.get(key, ""), foreign, rocm_dir)
        dropped.extend(gone)
        return kept

    path = scrub("PATH")
    ld_library_path = scrub("LD_LIBRARY_PATH")
    library_path = scrub("LIBRARY_PATH")
    pkg_config = scrub("PKG_CONFIG_PATH")
    module_path = scrub("CMAKE_MODULE_PATH")

    rocm = str(rocm_dir)
    bins = [str(rocm_dir / "bin"), str(rocm_dir / "llvm" / "bin")]
    libs = [
        str(rocm_dir / "lib"),
        str(rocm_dir / "lib64"),
        str(rocm_dir / "llvm" / "lib"),
    ]
    pkg_dirs = [
        str(rocm_dir / "lib" / "pkgconfig"),
        str(rocm_dir / "lib64" / "pkgconfig"),
    ]

    env["ROCM_PATH"] = rocm
    env["HIP_PATH"] = rocm
    env["HSA_PATH"] = rocm
    # Replace, do not prepend. Packages.cmake does
    #   CMAKE_PREFIX_PATH = $ENV{CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}
    # so a host TheRock left in the environment is searched first.
    env["CMAKE_PREFIX_PATH"] = rocm
    llvm_bin = rocm_dir / "llvm" / "bin"
    if llvm_bin.is_dir():
        env["HIP_CLANG_PATH"] = str(llvm_bin)
    else:
        env.pop("HIP_CLANG_PATH", None)
    env["ROCPROFSYS_CI"] = "ON"

    # Make git abort quickly instead of hanging forever when a compute node has no
    # egress (the common cluster case). Abort if <1 KB/s for 30s.
    env.setdefault("GIT_HTTP_LOW_SPEED_LIMIT", "1000")
    env.setdefault("GIT_HTTP_LOW_SPEED_TIME", "300")

    env["PATH"] = os.pathsep.join(bins + path).rstrip(os.pathsep)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + ld_library_path).rstrip(os.pathsep)
    env["LIBRARY_PATH"] = os.pathsep.join(libs + library_path).rstrip(os.pathsep)
    env["PKG_CONFIG_PATH"] = os.pathsep.join(pkg_dirs + pkg_config).rstrip(os.pathsep)
    if module_path:
        env["CMAKE_MODULE_PATH"] = os.pathsep.join(module_path)
    else:
        env.pop("CMAKE_MODULE_PATH", None)

    if foreign or dropped:
        log(
            "Pinned ROCm env to "
            f"{rocm}; ignoring leftover ROCm/TheRock at: "
            + ", ".join(str(root) for root in foreign)
        )
        if dropped:
            log(
                "Removed leftover entries from search paths: "
                + ", ".join(dict.fromkeys(dropped))
            )
    return env
