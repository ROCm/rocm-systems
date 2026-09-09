# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Kernel-count leak guard for src/device/symmetric/generate.py.

The symmetric generator emits sym_kernels_host.cc containing:
  * `extern int const ncclSymkKernelCount = N;`
  * `void* ncclSymkKernelList[] = { (void*)ncclSymkDevKernel_..., ... nullptr };`
    -- exactly one authoritative entry per kernel.

Like test_kernel_counts.py for the main generator, this CPU-only test (no GPU,
no built library) guards against unintended kernel growth in three layers --
per-dimension value-sets (root cause), per-collective counts (where), and grand
total (net effect) -- and keeps PARSER breakage distinct from baseline movement.

Baselines seeded from origin/develop @ 4a99ef1f9c (develop, post-AllGatherV).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# tests/ -> kernel-count/ -> test/ -> rccl root
RCCL_ROOT = Path(__file__).resolve().parents[3]
GENERATE_PY = RCCL_ROOT / "src" / "device" / "symmetric" / "generate.py"

# ---------------------------------------------------------------------------
# Baselines (origin/develop @ 4a99ef1f9c (develop, post-AllGatherV)).
# If a change moves these numbers, update them here AND explain in the PR
# description WHY. Do not blind-update.
# ---------------------------------------------------------------------------
EXPECTED_TOTAL = 42
EXPECTED_PER_COLL = {
    "AllGather": 2,
    "AllReduce": 10,
    "ReduceScatter": 30,
}
EXPECTED_DIMS = {
    "coll": {"AllGather", "AllReduce", "ReduceScatter"},
    "algo": {"LL", "ST", "AGxLL_R", "RSxLD_AGxST", "LD", "RailA2A_LsaLD"},
    "red": {"sum", "avg"},
    "ty": {"f32", "f16", "bf16", "f8e4m3", "f8e5m2"},
}

# Anchors for parsing kernel names. Reductions carry a trailing _<red>_<ty>;
# non-reductions (AllGather) carry only _<algo>. Algorithm tokens themselves
# contain underscores (RSxLD_AGxST, RailA2A_LsaLD), so names are parsed by
# anchoring the known coll at the front and the known red+ty at the end -- never
# by a naive underscore split.
_KNOWN_COLLS = ("AllReduce", "ReduceScatter", "AllGather")
_REDUCTION_COLLS = {"AllReduce", "ReduceScatter"}
_REDS = ("sum", "avg")
_TYS = ("f32", "f16", "bf16", "f8e4m3", "f8e5m2")

_COUNT_RE = re.compile(r"ncclSymkKernelCount\s*=\s*(\d+)\s*;")
_LIST_BLOCK_RE = re.compile(r"ncclSymkKernelList\[\]\s*=\s*\{(.*?)\bnullptr\b", re.DOTALL)
_CNAME_RE = re.compile(r"\(void\*\)\s*(ncclSymkDevKernel_\w+)")
_PREFIX = "ncclSymkDevKernel_"

DIMENSIONS = ("coll", "algo", "red", "ty")


def parse_kernel_name(cname):
    """Parse a ncclSymkDevKernel_* name into {coll, algo, red, ty} via anchoring."""
    if not cname.startswith(_PREFIX):
        raise ValueError("unexpected kernel symbol %r" % cname)
    body = cname[len(_PREFIX):]

    coll = None
    for c in _KNOWN_COLLS:
        if body == c or body.startswith(c + "_"):
            coll = c
            break
    if coll is None:
        raise ValueError("cannot identify collective in %r" % cname)

    rest = body[len(coll):].lstrip("_")

    if coll in _REDUCTION_COLLS:
        for red in _REDS:
            for ty in _TYS:
                suffix = "_%s_%s" % (red, ty)
                if rest.endswith(suffix):
                    algo = rest[: -len(suffix)]
                    if not algo:
                        raise ValueError("empty algo parsed from %r" % cname)
                    return {"coll": coll, "algo": algo, "red": red, "ty": ty}
        raise ValueError("cannot anchor red/ty suffix in reduction kernel %r" % cname)

    if not rest:
        raise ValueError("empty algo parsed from %r" % cname)
    return {"coll": coll, "algo": rest, "red": None, "ty": None}


def diff_report(exp_total, act_total, exp_per_coll, act_per_coll, exp_dims, act_dims):
    """Combined total/per-collective/per-dimension diff, or None if all match."""
    lines = []
    if act_total != exp_total:
        lines.append("total %d -> %d (%+d)" % (exp_total, act_total, act_total - exp_total))

    coll_lines = []
    for coll in sorted(set(exp_per_coll) | set(act_per_coll)):
        e = exp_per_coll.get(coll, 0)
        a = act_per_coll.get(coll, 0)
        if e != a:
            coll_lines.append("    %s %d -> %d (%+d)" % (coll, e, a, a - e))
    if coll_lines:
        lines.append("per-collective delta:")
        lines.extend(coll_lines)

    for dim in sorted(set(exp_dims) | set(act_dims)):
        e = exp_dims.get(dim, set())
        a = act_dims.get(dim, set())
        gained = a - e
        lost = e - a
        if gained or lost:
            parts = []
            if gained:
                parts.append("gained %s" % sorted(gained))
            if lost:
                parts.append("lost %s" % sorted(lost))
            lines.append("dimension '%s' %s" % (dim, ", ".join(parts)))

    if not lines:
        return None
    action = (
        "ACTION: if intentional, update the EXPECTED_* constants in "
        "test_symmetric_kernels.py AND explain in the PR description WHY the "
        "kernel count changed (binary-size / build-time impact). Do not blind-update."
    )
    return "\n".join(["Symmetric kernel count changed:"] + lines + [action])


def _per_coll(records):
    counts = {}
    for r in records:
        counts[r["coll"]] = counts.get(r["coll"], 0) + 1
    return counts


def _dims(records):
    # red/ty are None for non-reduction kernels; exclude those from the value-set.
    return {dim: {r[dim] for r in records if r[dim] is not None} for dim in DIMENSIONS}


@pytest.fixture(scope="session")
def sym_host(tmp_path_factory):
    if not GENERATE_PY.exists():
        pytest.fail("symmetric generate.py not found: %s" % GENERATE_PY)
    d = tmp_path_factory.mktemp("sym")
    subprocess.run(
        [sys.executable, str(GENERATE_PY), str(d)],
        check=True,
        capture_output=True,
        text=True,
    )
    with open(os.path.join(str(d), "sym_kernels_host.cc")) as f:
        return f.read()


def _count_literal(host):
    m = _COUNT_RE.search(host)
    assert m is not None, (
        "parser integrity: could not find 'ncclSymkKernelCount = N;' -- "
        "sym_kernels_host.cc format likely changed (this is NOT a count change)"
    )
    return int(m.group(1))


def _list_cnames(host):
    block = _LIST_BLOCK_RE.search(host)
    assert block is not None, (
        "parser integrity: could not find ncclSymkKernelList[] block -- "
        "sym_kernels_host.cc format likely changed (this is NOT a count change)"
    )
    cnames = _CNAME_RE.findall(block.group(1))
    assert cnames, "parser integrity: ncclSymkKernelList[] parsed but no entries found"
    return cnames


@pytest.mark.symmetric_generator
def test_count_literal_matches_list_matches_records_matches_baseline(sym_host):
    # Chain each equality with its own message so a break points at the exact link.
    count_literal = _count_literal(sym_host)
    cnames = _list_cnames(sym_host)
    records = [parse_kernel_name(c) for c in cnames]

    assert count_literal == len(cnames), (
        "ncclSymkKernelCount (%d) != number of ncclSymkKernelList entries (%d)"
        % (count_literal, len(cnames))
    )
    assert len(cnames) == len(records), (
        "parsed %d records from %d list entries" % (len(records), len(cnames))
    )
    assert count_literal == EXPECTED_TOTAL, (
        "ncclSymkKernelCount %d != expected %d" % (count_literal, EXPECTED_TOTAL)
    )


@pytest.mark.symmetric_generator
def test_per_collective_and_dimension_baselines(sym_host):
    records = [parse_kernel_name(c) for c in _list_cnames(sym_host)]
    report = diff_report(
        EXPECTED_TOTAL, len(records),
        EXPECTED_PER_COLL, _per_coll(records),
        EXPECTED_DIMS, _dims(records),
    )
    assert report is None, report


# --- anchored name parser: valid cases --------------------------------------
@pytest.mark.symmetric_generator
@pytest.mark.parametrize("cname,expected", [
    ("ncclSymkDevKernel_ReduceScatter_RSxLD_AGxST_avg_f8e4m3",
     {"coll": "ReduceScatter", "algo": "RSxLD_AGxST", "red": "avg", "ty": "f8e4m3"}),
    ("ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_bf16",
     {"coll": "ReduceScatter", "algo": "RailA2A_LsaLD", "red": "sum", "ty": "bf16"}),
    ("ncclSymkDevKernel_AllGather_ST",
     {"coll": "AllGather", "algo": "ST", "red": None, "ty": None}),
    ("ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f32",
     {"coll": "AllReduce", "algo": "AGxLL_R", "red": "sum", "ty": "f32"}),
])
def test_parse_valid_names(cname, expected):
    assert parse_kernel_name(cname) == expected


# --- anchored name parser: error branches (must raise, never mis-parse) -----
@pytest.mark.symmetric_generator
@pytest.mark.parametrize("cname", [
    "ncclFooBar_AllGather_ST",                       # missing prefix
    "ncclSymkDevKernel_Scatter_ST",                  # unknown collective
    "ncclSymkDevKernel_ReduceScatter_RSxLD_AGxST",   # reduction w/o anchored red/ty
    "ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f4",    # unknown type
    "ncclSymkDevKernel_AllGather",                    # empty algo
])
def test_parse_invalid_names_raise(cname):
    with pytest.raises(ValueError):
        parse_kernel_name(cname)


# --- diagnostic helper unit tests -------------------------------------------
@pytest.mark.diagnostics
def test_diff_report_none_when_identical():
    assert diff_report(42, 42, {"A": 42}, {"A": 42}, {"ty": {"f32"}}, {"ty": {"f32"}}) is None


@pytest.mark.diagnostics
def test_diff_report_reports_total_coll_gained_lost_together():
    report = diff_report(
        42, 47,
        {"AllReduce": 10, "ReduceScatter": 30}, {"AllReduce": 15, "ReduceScatter": 30},
        {"ty": {"f32"}}, {"ty": {"f4"}},
    )
    assert "total 42 -> 47 (+5)" in report
    assert "AllReduce 10 -> 15 (+5)" in report
    assert "gained ['f4']" in report
    assert "lost ['f32']" in report


@pytest.mark.diagnostics
def test_diff_report_new_collective_from_zero():
    report = diff_report(
        42, 44, {"AllReduce": 42}, {"AllReduce": 42, "AllToAll": 2},
        {"coll": {"AllReduce"}}, {"coll": {"AllReduce", "AllToAll"}},
    )
    assert "AllToAll 0 -> 2 (+2)" in report
    assert "gained ['AllToAll']" in report
