#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
run_amdsmi_dual_copy_test.py
============================

The system DEB/RPM installs the amdsmi Python module twice: once into the
interpreter's site-packages (so a bare ``import amdsmi`` works) and once under
``share/amd_smi`` (captured by the TheRock artifact flow and used by downstream
tools that ``sys.path.insert(ROCM_PATH + "/share/amd_smi")``). Both copies are
necessary -- TheRock ships only ``/opt/rocm`` and cannot reach the ``/usr``
site-packages tree, so a redirector/symlink is not an option.

Two independent copies can silently drift (a partial upgrade, a stray edit),
after which ``import amdsmi`` resolves to a different version depending on
sys.path order. This guard asserts the two installed copies are byte-identical
so any drift fails a build instead of shipping.

The bundled ``libamd_smi*.so`` is excluded from both install trees, so only the
Python sources are compared.
"""

import argparse
import filecmp
import os
import sys
from pathlib import Path
from typing import List, Tuple


def _py_files(root: Path) -> List[Path]:
    return sorted(p.relative_to(root) for p in root.rglob("*.py"))


def compare_trees(tree_a: Path, tree_b: Path) -> Tuple[List[str], List[str]]:
    """Compare the .py files of two module trees.

    Returns ``(missing, differing)``: relative paths present in one tree but not
    the other, and relative paths whose contents differ. Empty lists mean the
    two trees are byte-identical.
    """
    files_a = set(_py_files(tree_a))
    files_b = set(_py_files(tree_b))

    missing = sorted(str(p) for p in files_a.symmetric_difference(files_b))
    differing = []
    for rel in sorted(files_a & files_b):
        if not filecmp.cmp(tree_a / rel, tree_b / rel, shallow=False):
            differing.append(str(rel))
    return missing, differing


def _rocm_share_dir() -> Path:
    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm"
    return Path(rocm_path) / "share" / "amd_smi" / "amdsmi"


def _sitelib_dir() -> Path:
    import amdsmi

    return Path(amdsmi.__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sitelib-tree",
        type=Path,
        default=None,
        help="site-packages amdsmi dir (default: resolved from `import amdsmi`).",
    )
    parser.add_argument(
        "--share-tree",
        type=Path,
        default=None,
        help="share/amd_smi amdsmi dir (default: ROCM_PATH/share/amd_smi/amdsmi).",
    )
    args = parser.parse_args()

    share_tree = (args.share_tree or _rocm_share_dir()).resolve()
    try:
        sitelib_tree = (args.sitelib_tree or _sitelib_dir()).resolve()
    except ImportError:
        sys.exit("could not `import amdsmi`; install the package or pass --sitelib-tree")

    if not share_tree.is_dir():
        sys.exit(f"share/amd_smi copy not found at {share_tree}")
    if not sitelib_tree.is_dir():
        sys.exit(f"site-packages copy not found at {sitelib_tree}")

    print(f"site-packages : {sitelib_tree}")
    print(f"share/amd_smi : {share_tree}")

    missing, differing = compare_trees(sitelib_tree, share_tree)
    if missing or differing:
        msg = ["the two installed amdsmi copies have drifted:"]
        if missing:
            msg.append("  only in one tree: " + ", ".join(missing))
        if differing:
            msg.append("  differing contents: " + ", ".join(differing))
        sys.exit("\n".join(msg))

    print("PASS: site-packages and share/amd_smi copies are byte-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
