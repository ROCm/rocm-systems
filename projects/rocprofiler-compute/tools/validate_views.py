#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: check the analysis view files.

For every arch in METRIC_LIBRARY_PARENT_ARCH, check that:

- its view files have known fields, render types and CLI styles, and the views tree
  holds only metric library arch directories;
- every metric a view shows is in the arch's metric library, with a measured peak where
  the row asks for one.

test_analysis_views_sync.py, next to the metric library sync test, checks that the
views keep the analysis configs' structure.
"""

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocprof_compute_soc.metric_library import loader
from utils.analysis_views import (
    default_views_dir,
    read_view_files,
    view_metric_errors,
    view_schema_errors,
)


def validate(views_dir: Path, metric_dir: Path) -> List[str]:
    """Validate every arch's views. Return error messages."""
    errors = [
        f"[{path.relative_to(views_dir)}] is not a metric library arch directory"
        for path in sorted(Path(views_dir).iterdir())
        if path.name not in loader.METRIC_LIBRARY_PARENT_ARCH
    ]
    for arch in loader.METRIC_LIBRARY_PARENT_ARCH:
        documents = read_view_files(arch, views_dir)
        schema_errors = view_schema_errors({
            str(path.relative_to(views_dir)): document
            for path, document in documents.items()
        })
        if schema_errors:
            errors.extend(schema_errors)
            continue
        metrics = loader.MetricLibrary.load(metric_dir, arch).metrics
        errors.extend(
            f"[{arch}] {error}"
            for error in view_metric_errors(list(documents.values()), metrics)
        )
    return errors


def main() -> int:
    """Entry point. Returns 0 on success, 1 on validation failure."""
    errors = validate(default_views_dir(), loader.default_metric_library_dir())
    if errors:
        print("Analysis view validation failed:\n")
        for error in errors:
            print(f"  ERROR: {error}")
        print(f"\n{len(errors)} error(s) found.")
        return 1
    print("Analysis view validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
