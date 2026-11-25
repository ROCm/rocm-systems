# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the metric lock (tools/config_management/metric_lock.py) and the
hook that keeps it current (check_metric_config.py).

The hook checks that the shipped lock is current; these tests cover what that check
takes on trust: a metric's lock entry follows inheritance, and the hook reports and
rewrites every generated file that is out of date.
"""

import shutil

from config_management.check_metric_config import (
    regenerate,
    regenerated_copies,
    stale_files,
    write_stale_files,
)
from config_management.metric_lock import arch_lock

from rocprof_compute_soc.metric_library import loader

# A gfx90a metric that the archs inheriting gfx90a's library do not redefine.
PARENT_ARCH = "gfx90a"
INHERITED_METRIC = "compute.bf16_ops"
ADDED_COUNTER = "TA_BUSY"


def test_a_parent_edit_reaches_the_lock_of_every_arch_that_inherits_it(tmp_path):
    """Editing a gfx90a metric changes its formula and counters in the lock of every
    arch whose library chain has gfx90a, and in no other lock."""
    metric_dir = tmp_path / "metric_library"
    shutil.copytree(loader.default_metric_library_dir(), metric_dir)
    library = loader.MetricLibrary.load(metric_dir, PARENT_ARCH)
    formula = library.metrics[INHERITED_METRIC]["formula"]
    edited_formula = f"{formula} + SUM({ADDED_COUNTER})"
    compute_file = metric_dir / PARENT_ARCH / "compute.yaml"
    text = compute_file.read_text(encoding="utf-8")
    compute_file.write_text(text.replace(formula, edited_formula), encoding="utf-8")

    for arch in loader.METRIC_LIBRARY_PARENT_ARCH:
        entry = arch_lock(metric_dir, arch).get(INHERITED_METRIC, {})
        inherits = PARENT_ARCH in loader.arch_chain(arch)
        assert (entry.get("formula") == edited_formula) == inherits, arch
        assert (ADDED_COUNTER in entry.get("counters", [])) == inherits, arch


def test_stale_generated_files_are_reported_and_rewritten(tmp_path):
    """A changed, a removed and an extra lock file, and the description file of an
    arch without a metric library, are each reported, and --fix rewrites the
    generated files to the regenerated ones."""
    generated_dirs = (tmp_path / "metric_lock", tmp_path / "metrics")
    regenerate(*generated_dirs)
    lock_dir, docs_dir = generated_dirs
    (lock_dir / "gfx942.yaml").write_text("compute.bf16_ops: {}\n", encoding="utf-8")
    (lock_dir / "gfx950.yaml").unlink()
    (lock_dir / "gfx000.yaml").write_text("", encoding="utf-8")
    (docs_dir / "gfx000_metrics.yaml").write_text("", encoding="utf-8")

    copies = regenerated_copies(generated_dirs, tmp_path / "copies")
    stale = stale_files(generated_dirs, copies)
    assert [path for path, _ in stale] == [
        lock_dir / "gfx000.yaml",
        lock_dir / "gfx942.yaml",
        lock_dir / "gfx950.yaml",
        docs_dir / "gfx000_metrics.yaml",
    ]
    write_stale_files(stale)
    assert stale_files(generated_dirs, copies) == []
