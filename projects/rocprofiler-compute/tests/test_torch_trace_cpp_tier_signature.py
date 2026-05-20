# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for C++ RecordFunction tier signature detection.

Pins the contract that detect_cpp_tier_signature() correctly identifies
C++ tier markers via the leaf sentinels aten:0, aten.nested:0,
autograd.engine:0, autograd.bwd:0 (and never matches the Python tier's
python.dispatch:0). Used by test_torch_trace_coverage to gate the
strict C++ tier assertion.

CI: registered with ctest; not in test_categories.yaml so it runs only
    on the full ctest invocation (and on the coverage job). No GPU,
    ctest timeout 60 s.
Coverage: included in the coverage XML.
Why: detection lives in coverage_utils and is consumed by the strict
    C++ tier assertion; pinned separately so a regression in the
    sentinel scan can't slip through the GPU-only coverage test.
"""

import importlib.util

import common  # noqa: F401  -- adds src/ to sys.path
import pandas as pd
import pytest

# torch_trace_coverage_utils imports torch unconditionally. Guard the
# symbol import so this module still loads on CPU-only hosts; each test
# below picks up the require_torch fixture via the module-level
# pytestmark and skips per-test when torch is unavailable. A module-level
# pytest.skip(..., allow_module_level=True) would collect zero items and
# make pytest exit with code 5 ("no tests collected"), which CTest reads
# as a test failure.
if importlib.util.find_spec("torch") is not None:
    from torch_trace_coverage_utils import (  # noqa: E402
        C_TIER_BACKWARD_SENTINELS,
        C_TIER_LEAF_CONTEXT_SENTINELS,
        OpEntry,
        compare_single_op,
        cpp_tier_signature_in_markers,
        detect_cpp_tier_signature,
        format_cpp_tier_signature_report,
    )

pytestmark = pytest.mark.usefixtures("require_torch")


def _marker_df(*function_cells):
    """Helper: build a DataFrame with the single Function column the
    detection inspects. The real rocprof-compute CSV has additional
    columns; the detection ignores them so the test fixture stays minimal."""
    return pd.DataFrame({"Function": list(function_cells)})


# -- Sentinel-set audit ------------------------------------------------------


def test_sentinel_set_matches_cpp_source():
    """The C++ tier emits exactly these four leaf context labels (see
    src/utils/roctx_recordfn/roctx_recordfn.cpp:default_leaf_context).
    A drift between the .cpp emitter and this constant would silently
    blind the coverage test to half the C++ tier outputs; this
    assertion makes the set the single source of truth from the test
    side."""
    assert C_TIER_LEAF_CONTEXT_SENTINELS == (
        "aten:0",
        "aten.nested:0",
        "autograd.engine:0",
        "autograd.bwd:0",
    )


def test_backward_subset_is_strictly_backward():
    """The backward subset must exclude the forward sentinels.
    The strict C++ tier check in test_random_operator_kernel_coverage
    keys its backward axis off this subset to assert that autograd
    flow specifically (not just forward dispatch) reached the C++
    tier. Including aten:0 here would make that axis fire on a
    forward-only run."""
    assert set(C_TIER_BACKWARD_SENTINELS) == {
        "autograd.engine:0",
        "autograd.bwd:0",
    }
    assert "aten:0" not in C_TIER_BACKWARD_SENTINELS
    assert "aten.nested:0" not in C_TIER_BACKWARD_SENTINELS


def test_python_tier_sentinel_is_not_in_cpp_set():
    """The Python tier's no-user-frame sentinel must not collide with
    any C++ tier sentinel; otherwise the detection would mark a Python
    tier run as C++ tier-active. This is the structural defence
    against an analyzer that conflates the two tiers."""
    assert "python.dispatch:0" not in C_TIER_LEAF_CONTEXT_SENTINELS


# -- cpp_tier_signature_in_markers: basic positive / negative ---------------


def test_signature_absent_on_empty_dataframe():
    """No marker rows -> (False, {}); the detection must not blow up
    on an empty CSV (rocprof-compute can produce empty trace CSVs
    when a workload crashes very early)."""
    cpp, counts = cpp_tier_signature_in_markers(_marker_df())
    assert cpp is False
    assert counts == {}


def test_signature_absent_when_function_column_missing():
    """A marker CSV without a Function column degrades to the same
    empty result. Tracks parse_roctx_markers' tolerance for missing
    columns -- we don't want a schema drift to turn into an exception
    at the coverage assertion line."""
    df = pd.DataFrame({"OtherColumn": ["x", "y"]})
    cpp, counts = cpp_tier_signature_in_markers(df)
    assert cpp is False
    assert counts == {}


def test_signature_present_for_each_individual_sentinel():
    """Each sentinel in isolation must trigger detection. This is the
    1-of-N coverage that ensures a future regression that drops the
    emitter for one specific sentinel still surfaces here."""
    for sentinel in C_TIER_LEAF_CONTEXT_SENTINELS:
        df = _marker_df(f"some.op:#1@{sentinel}")
        cpp, counts = cpp_tier_signature_in_markers(df)
        assert cpp is True, f"{sentinel} alone should trigger detection"
        assert counts == {sentinel: 1}, f"expected exactly one {sentinel}, saw {counts}"


def test_signature_counts_multiple_occurrences_in_one_cell():
    """Nested chains accumulate sentinels (a 4-deep marker can carry
    several aten.nested:0 segments). The counts must reflect every
    occurrence so the report can distinguish a workload with rich
    nested structure from one with shallow markers."""
    df = _marker_df(
        "nn.Module.A.forward/aten::view/aten::add:"
        "#1@x.py:1/#1@aten.nested:0/#1@aten.nested:0",
    )
    cpp, counts = cpp_tier_signature_in_markers(df)
    assert cpp is True
    assert counts == {"aten.nested:0": 2}


def test_signature_counts_across_multiple_rows():
    """Aggregation across rows must sum, not pick one. Without this
    the per-sentinel counts in the stdout report would be
    misleading-low on real workloads with thousands of marker rows."""
    df = _marker_df(
        "torch.mm:#1@aten:0",
        "torch.add:#1@aten:0",
        "node:#1@autograd.bwd:0",
    )
    cpp, counts = cpp_tier_signature_in_markers(df)
    assert cpp is True
    assert counts == {"aten:0": 2, "autograd.bwd:0": 1}


# -- False-positive defences ------------------------------------------------


def test_signature_does_not_match_sentinel_substring_in_filepath():
    """A file path that happens to contain 'aten:0' as a SUBSTRING
    (without the preceding '@') must NOT be detected as a C++ tier
    sentinel. The matcher uses '@<sentinel>' precisely to avoid this
    class of false positive; this test pins the contract.

    Concretely: a Python tier marker like
        "some.op:#1@my_aten:0_dir/file.py:42"
    contains the bytes 'aten:0' but not the '@aten:0' delimiter."""
    df = _marker_df("some.op:#1@my_aten:0_dir/file.py:42")
    cpp, counts = cpp_tier_signature_in_markers(df)
    assert cpp is False, (
        f"substring match must not trigger sentinel detection; saw {counts}"
    )
    assert counts == {}


def test_signature_ignores_python_tier_sentinel():
    """A pure Python tier run (every marker carries python.dispatch:0)
    must come back as cpp_tier_active=False. This is the headline
    negative-case test: a 100% Python-tier output should be
    unambiguously distinguishable from the C++ tier."""
    df = _marker_df(
        "torch.ops.aten.add:#1@python.dispatch:0",
        "torch.ops.aten.mm:#1@python.dispatch:0",
        "nn.Module.Linear.forward:#1@workload.py:42",
    )
    cpp, counts = cpp_tier_signature_in_markers(df)
    assert cpp is False
    assert counts == {}


def test_signature_handles_non_string_cells():
    """Marker CSV can contain NaN in the Function column for rows
    that ROCTX emitted without a name (rare but observed in stress
    workloads). The detection must skip them without raising."""
    import numpy as np

    df = pd.DataFrame({"Function": ["op:#1@aten:0", np.nan, None, 42]})
    cpp, counts = cpp_tier_signature_in_markers(df)
    assert cpp is True
    assert counts == {"aten:0": 1}


# -- detect_cpp_tier_signature: workload-dir entry point --------------------


def test_detect_returns_empty_when_workload_dir_has_no_csv(tmp_path):
    """No marker CSV in the dir -> (False, {}). The coverage test
    branches on this to skip the assertion gracefully when
    rocprof-compute crashed before emitting trace output."""
    cpp, counts = detect_cpp_tier_signature(str(tmp_path))
    assert cpp is False
    assert counts == {}


def test_detect_reads_marker_csv_and_detects_signature(tmp_path):
    """Smoke test that detect_cpp_tier_signature actually wires
    through _read_roctx_marker_dataframe to the disk-side CSV. A
    refactor that moves the CSV reading to a different path would
    silently regress this without coverage."""
    csv_path = tmp_path / "subdir" / "0001_marker_api_trace.csv"
    csv_path.parent.mkdir(parents=True)
    df = pd.DataFrame({
        "Function": [
            "torch.mm:#1@aten:0",
            "node:#1@autograd.bwd:0",
        ],
        "Correlation_ID": [1, 2],
    })
    df.to_csv(csv_path, index=False)
    cpp, counts = detect_cpp_tier_signature(str(tmp_path))
    assert cpp is True
    assert counts == {"aten:0": 1, "autograd.bwd:0": 1}


def test_detect_concatenates_multiple_csvs(tmp_path):
    """Real workloads emit one marker CSV per profiled rank /
    iteration; detect_cpp_tier_signature must sum across them.
    Concretely: a multi-rank run that emits the C++ sentinel only in
    rank 0's CSV must still come back as cpp_tier_active=True."""
    for i, fn in enumerate([
        "torch.add:#1@python.dispatch:0",
        "torch.mm:#1@aten:0",
    ]):
        path = tmp_path / f"rank{i}_marker_api_trace.csv"
        pd.DataFrame({"Function": [fn]}).to_csv(path, index=False)
    cpp, counts = detect_cpp_tier_signature(str(tmp_path))
    assert cpp is True
    assert counts == {"aten:0": 1}


# -- format_cpp_tier_signature_report: report rendering ---------------------


def test_report_when_cpp_tier_absent():
    """Absent-tier report must call the state out unambiguously.
    Test by substring rather than equality so wording can evolve;
    the actionable token is 'NONE' and 'Python TorchDispatchMode'."""
    lines = format_cpp_tier_signature_report(False, {})
    assert len(lines) == 1
    assert "NONE" in lines[0]
    assert "Python" in lines[0]


def test_report_when_cpp_tier_full():
    """Forward + backward sentinels both present -> FULL.
    Counts must appear so the user can sanity-check the magnitude."""
    counts = {
        "aten:0": 100,
        "aten.nested:0": 250,
        "autograd.engine:0": 30,
        "autograd.bwd:0": 30,
    }
    lines = format_cpp_tier_signature_report(True, counts)
    rendered = "\n".join(lines)
    assert "FULL" in rendered
    assert "100" in rendered  # aten:0 count
    assert "30" in rendered  # backward count


def test_compare_single_op_fails_when_kernels_do_not_overlap():
    """Non-empty ROCTX kernels that share nothing with profiler ground
    truth must FAIL (regression guard for the intersection check)."""
    op = OpEntry("torch.ops.aten.add", "aten", schema=None)
    ground_truth = {
        "torch.ops.aten.add": {
            "cuda_kernels": ["kernel_from_profiler"],
        },
    }
    outcome = compare_single_op(
        op,
        ground_truth,
        roctx_marker_names={"torch.ops.aten.add"},
        roctx_kernels_map={
            "torch.ops.aten.add": {"unrelated_roctx_kernel"},
        },
    )
    assert outcome.status == "fail"
    assert "overlap" in outcome.reason.lower()


def test_compare_single_op_passes_when_kernels_overlap():
    op = OpEntry("torch.ops.aten.add", "aten", schema=None)
    ground_truth = {
        "torch.ops.aten.add": {
            "cuda_kernels": ["vectorized_elementwise_kernel"],
        },
    }
    outcome = compare_single_op(
        op,
        ground_truth,
        roctx_marker_names={"torch.ops.aten.add"},
        roctx_kernels_map={
            "torch.ops.aten.add": {
                "vectorized_elementwise_kernel",
                "extra_kernel",
            },
        },
    )
    assert outcome.status == "pass"


def test_report_when_cpp_tier_forward_only():
    """Forward sentinels present but no autograd.* -> FORWARD-ONLY.
    Must include a hint that this is degraded if the workload should
    have triggered backward; otherwise a user might misread it as
    healthy."""
    counts = {"aten:0": 50, "aten.nested:0": 75}
    lines = format_cpp_tier_signature_report(True, counts)
    rendered = "\n".join(lines)
    assert "FORWARD-ONLY" in rendered
    assert "backward" in rendered.lower()
