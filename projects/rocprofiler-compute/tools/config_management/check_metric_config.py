#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: check the metric library and views, and the files made from them.

Runs the metric library, analysis view and membw tree spec validators, then
regenerates the metric lock (metric_lock.py) and the metric description files
(metric_description_manager.py) and reports the generated files that change. With
--fix it also writes them, so pre-commit fails until the developer reviews and stages
their diff: that staging confirms the library or view edit was intended.
"""

import argparse
import filecmp
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

TOOLS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_DIR.parent
sys.path[:0] = [str(TOOLS_DIR), str(PROJECT_ROOT / "src")]

import validate_membw_tree_spec
import validate_metric_library
import validate_views
from config_management import metric_lock
from config_management.metric_description_manager import DOCS_DIR, docs_file, sync_arch

from rocprof_compute_soc.metric_library import loader


def regenerate(lock_dir: Path, docs_dir: Path) -> None:
    """Write the metric lock and the metric description files from the shipped
    metric library and views, and remove the description files of other archs."""
    metric_lock.write_lock(lock_dir, loader.default_metric_library_dir())
    archs = sorted(loader.METRIC_LIBRARY_PARENT_ARCH)
    expected = {docs_file(arch, docs_dir) for arch in archs}
    for path in docs_dir.glob("*_metrics.yaml"):
        if path not in expected:
            path.unlink()
    for arch in archs:
        sync_arch(arch, docs_dir)


def regenerated_copies(generated_dirs: Sequence[Path], copy_root: Path) -> List[Path]:
    """Copy the lock and description directories under copy_root and regenerate the
    copies; return the copies, in the order of generated_dirs."""
    copies = [copy_root / str(index) for index in range(len(generated_dirs))]
    for directory, copy in zip(generated_dirs, copies):
        shutil.copytree(directory, copy)
    regenerate(*copies)
    return copies


def stale_files(
    generated_dirs: Sequence[Path], copies: Sequence[Path]
) -> List[Tuple[Path, Path]]:
    """Return each generated file that differs from its regenerated copy, or that
    exists on one side only, with the path of its copy."""
    return [
        (directory / name, copy / name)
        for directory, copy in zip(generated_dirs, copies)
        for name in _differing_names(filecmp.dircmp(directory, copy))
    ]


def write_stale_files(stale: Sequence[Tuple[Path, Path]]) -> None:
    """Replace each stale file with its regenerated copy, or remove it if the copy
    has none."""
    for path, copy in stale:
        if copy.exists():
            shutil.copyfile(copy, path)
        else:
            path.unlink()


def main() -> int:
    """Entry point. Returns 1 if a validator fails or, without --fix, if a generated
    file is stale."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fix", action="store_true", help="Write the generated files that change"
    )
    args = parser.parse_args()

    validators = (validate_metric_library, validate_views, validate_membw_tree_spec)
    if any([validator.main() != 0 for validator in validators]):
        return 1
    generated_dirs = (metric_lock.LOCK_DIR, DOCS_DIR)
    with tempfile.TemporaryDirectory() as copy_root:
        stale = stale_files(
            generated_dirs, regenerated_copies(generated_dirs, Path(copy_root))
        )
        if args.fix:
            write_stale_files(stale)
    if not stale:
        return 0
    if args.fix:
        print("Regenerated; review and stage their diff:")
    else:
        print("Stale; run tools/config_management/check_metric_config.py --fix:")
    for path, _ in stale:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    return 0 if args.fix else 1


def _differing_names(comparison: filecmp.dircmp) -> List[str]:
    """Return the file names that differ between two flat directories."""
    names = comparison.left_only + comparison.right_only + comparison.diff_files
    _match, mismatch, errors = filecmp.cmpfiles(
        comparison.left, comparison.right, comparison.same_files, shallow=False
    )
    return sorted(names + mismatch + errors)


if __name__ == "__main__":
    sys.exit(main())
