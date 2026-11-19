#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: check the analysis view files.

For every arch in METRIC_LIBRARY_PARENT_ARCH, check that:

- it has view files, named by panel id, one per panel, with known fields and render
  types, every row names its metric or its columns' metrics (with a label), and the
  views tree holds only metric library arch directories;
- every metric a view shows is in the arch's metric library, with a measured peak where
  the row asks for one, and a row whose table shows Min, Max or Std Dev reads no
  summed collectable without its per-dispatch partner;
- the views show every metric of the arch's library, and each measured peak in a row
  that asks for it.
"""

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from rocprof_compute_soc.metric_library import loader
from utils.analysis_views import (
    read_view_files,
    view_metric_errors,
    view_schema_errors,
    views_from_files,
)


def validate() -> List[str]:
    """Validate every arch's views against its metric library. Return error
    messages."""
    errors = [
        f"[{path.name}] is not a metric library arch directory"
        for path in sorted(config.analysis_views_dir.iterdir())
        if path.name not in loader.METRIC_LIBRARY_PARENT_ARCH
    ]
    for arch in loader.METRIC_LIBRARY_PARENT_ARCH:
        errors.extend(_arch_errors(arch))
    return errors


def main() -> int:
    """Entry point. Returns 0 on success, 1 on validation failure."""
    errors = validate()
    if errors:
        print("Analysis view validation failed:\n")
        for error in errors:
            print(f"  ERROR: {error}")
        print(f"\n{len(errors)} error(s) found.")
        return 1
    print("Analysis view validation passed.")
    return 0


def _arch_errors(arch: str) -> List[str]:
    """Report what is wrong with one arch's view files."""
    documents = read_view_files(arch)
    if not documents:
        return [f"[{arch}] is in METRIC_LIBRARY_PARENT_ARCH but has no view files"]
    schema_errors = view_schema_errors({
        str(path.relative_to(config.analysis_views_dir)): document
        for path, document in documents.items()
    })
    if schema_errors:
        return schema_errors
    library = loader.MetricLibrary.load(loader.default_metric_library_dir(), arch)
    return [
        f"[{arch}] {error}"
        for error in view_metric_errors(views_from_files(documents), library.metrics)
    ]


if __name__ == "__main__":
    sys.exit(main())
