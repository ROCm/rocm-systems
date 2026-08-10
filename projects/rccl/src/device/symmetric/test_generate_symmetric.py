#!/usr/bin/env python3
"""Kernel-count guard for src/device/symmetric/generate.py.

The symmetric generator enumerates the ncclSymk* device kernels and emits
sym_kernels_host.cc containing:
  * `extern int const ncclSymkKernelCount = N;`
  * `void* ncclSymkKernelList[] = { (void*)ncclSymkDevKernel_..., ... nullptr };`
    -- exactly one authoritative entry per kernel.

Like test_generate_kernel_counts.py for the main generator, this CPU-only test
(no GPU, no built library) guards against unintended kernel growth in three
layers -- per-dimension value-sets (root cause), per-collective counts (where),
and grand total (net effect) -- and keeps PARSER breakage distinct from
legitimate baseline movement.

Baselines seeded from origin/develop @ 40d3fae2dfa077769d45bac77a16e849c72d89ea.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Dict, List, Optional, Set

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATE_PY = os.path.join(HERE, "generate.py")

# ---------------------------------------------------------------------------
# Baselines (origin/develop @ 40d3fae2dfa077769d45bac77a16e849c72d89ea).
# ---------------------------------------------------------------------------
EXPECTED_TOTAL = 42
EXPECTED_PER_COLL = {
    "AllGather": 2,
    "AllReduce": 10,
    "ReduceScatter": 30,
}
EXPECTED_DIMS: Dict[str, Set[str]] = {
    "coll": {"AllGather", "AllReduce", "ReduceScatter"},
    "algo": {"LL", "ST", "AGxLL_R", "RSxLD_AGxST", "LD", "RailA2A_LsaLD"},
    "red":  {"sum", "avg"},
    "ty":   {"f32", "f16", "bf16", "f8e4m3", "f8e5m2"},
}

# Anchors for parsing kernel names. Reductions carry a trailing _<red>_<ty>;
# non-reductions (AllGather) carry only _<algo>. Algorithm tokens themselves
# contain underscores (RSxLD_AGxST, RailA2A_LsaLD), so names are parsed by
# anchoring the known coll at the front and the known red+ty at the end -- never
# by a naive underscore split.
_KNOWN_COLLS = ("AllReduce", "ReduceScatter", "AllGather")  # longest-first not needed; matched via "_" boundary
_REDUCTION_COLLS = {"AllReduce", "ReduceScatter"}
_REDS = ("sum", "avg")
_TYS = ("f32", "f16", "bf16", "f8e4m3", "f8e5m2")

_COUNT_RE = re.compile(r"ncclSymkKernelCount\s*=\s*(\d+)\s*;")
_LIST_BLOCK_RE = re.compile(r"ncclSymkKernelList\[\]\s*=\s*\{(.*?)\bnullptr\b", re.DOTALL)
_CNAME_RE = re.compile(r"\(void\*\)\s*(ncclSymkDevKernel_\w+)")
_PREFIX = "ncclSymkDevKernel_"

_DIMENSIONS = ("coll", "algo", "red", "ty")


def parse_kernel_name(cname: str) -> Dict[str, Optional[str]]:
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

    rest = body[len(coll):].lstrip("_")  # what follows "<coll>_"

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

    # Non-reduction: remainder is the whole algorithm.
    if not rest:
        raise ValueError("empty algo parsed from %r" % cname)
    return {"coll": coll, "algo": rest, "red": None, "ty": None}


def diff_report(
    exp_total: int,
    act_total: int,
    exp_per_coll: Dict[str, int],
    act_per_coll: Dict[str, int],
    exp_dims: Dict[str, Set[str]],
    act_dims: Dict[str, Set[str]],
) -> Optional[str]:
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
        "symmetric/test_generate_symmetric.py AND explain in the PR description WHY "
        "the kernel count changed (binary-size / build-time impact). Do not blind-update."
    )
    return "\n".join(["Symmetric kernel count changed:"] + lines + [action])


def _generate(tmpdir: str) -> None:
    subprocess.run(
        [sys.executable, GENERATE_PY, tmpdir],
        check=True,
        capture_output=True,
        text=True,
    )


def _per_coll(records: List[Dict[str, Optional[str]]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["coll"]] = counts.get(r["coll"], 0) + 1
    return counts


def _dims(records: List[Dict[str, Optional[str]]]) -> Dict[str, Set[str]]:
    # red/ty are None for non-reduction kernels; exclude those from the value-set.
    return {dim: {r[dim] for r in records if r[dim] is not None} for dim in _DIMENSIONS}


class SymmetricGeneratorKernelCountTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # generate.py is co-located with this test; its absence is a real error,
        # not a reason to skip (a skipped ctest reports PASS and hides breakage).
        assert os.path.exists(GENERATE_PY), "generate.py not found next to test: %s" % GENERATE_PY
        cls._dir = tempfile.mkdtemp(prefix="rccl_symkcount_")
        # Register cleanup right after mkdtemp so a failure in _generate() still
        # removes the temp dir (tearDownClass is skipped when setUpClass raises).
        cls.addClassCleanup(shutil.rmtree, cls._dir, ignore_errors=True)
        _generate(cls._dir)
        with open(os.path.join(cls._dir, "sym_kernels_host.cc")) as f:
            cls.host = f.read()

    def _kernel_count_literal(self):
        m = _COUNT_RE.search(self.host)
        self.assertIsNotNone(
            m,
            "parser integrity: could not find 'ncclSymkKernelCount = N;' -- "
            "sym_kernels_host.cc format likely changed (this is NOT a count change)",
        )
        return int(m.group(1))

    def _list_cnames(self):
        block = _LIST_BLOCK_RE.search(self.host)
        self.assertIsNotNone(
            block,
            "parser integrity: could not find ncclSymkKernelList[] block -- "
            "sym_kernels_host.cc format likely changed (this is NOT a count change)",
        )
        cnames = _CNAME_RE.findall(block.group(1))
        self.assertTrue(
            cnames,
            "parser integrity: ncclSymkKernelList[] parsed but no kernel entries found",
        )
        return cnames

    def test_count_literal_matches_list_matches_records_matches_baseline(self):
        # Chain each equality with its own message so a break points at the exact
        # link (correction #3).
        count_literal = self._kernel_count_literal()
        cnames = self._list_cnames()
        records = [parse_kernel_name(c) for c in cnames]

        self.assertEqual(
            count_literal, len(cnames),
            "ncclSymkKernelCount (%d) != number of ncclSymkKernelList entries (%d)"
            % (count_literal, len(cnames)),
        )
        self.assertEqual(
            len(cnames), len(records),
            "parsed %d records from %d list entries" % (len(records), len(cnames)),
        )
        self.assertEqual(
            count_literal, EXPECTED_TOTAL,
            "ncclSymkKernelCount %d != expected %d" % (count_literal, EXPECTED_TOTAL),
        )

    def test_per_collective_and_dimension_baselines(self):
        records = [parse_kernel_name(c) for c in self._list_cnames()]
        report = diff_report(
            EXPECTED_TOTAL, len(records),
            EXPECTED_PER_COLL, _per_coll(records),
            EXPECTED_DIMS, _dims(records),
        )
        self.assertIsNone(report, report)


class SymmetricNameParsingTest(unittest.TestCase):
    """Anchored parser must survive underscore-bearing algorithm names."""

    def test_reduction_name_with_underscore_algo(self):
        self.assertEqual(
            parse_kernel_name("ncclSymkDevKernel_ReduceScatter_RSxLD_AGxST_avg_f8e4m3"),
            {"coll": "ReduceScatter", "algo": "RSxLD_AGxST", "red": "avg", "ty": "f8e4m3"},
        )

    def test_reduction_name_raila2a(self):
        self.assertEqual(
            parse_kernel_name("ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_bf16"),
            {"coll": "ReduceScatter", "algo": "RailA2A_LsaLD", "red": "sum", "ty": "bf16"},
        )

    def test_non_reduction_name(self):
        self.assertEqual(
            parse_kernel_name("ncclSymkDevKernel_AllGather_ST"),
            {"coll": "AllGather", "algo": "ST", "red": None, "ty": None},
        )

    def test_allreduce_not_confused_with_allgather(self):
        rec = parse_kernel_name("ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f32")
        self.assertEqual(rec["coll"], "AllReduce")
        self.assertEqual(rec["algo"], "AGxLL_R")

    # The parser-integrity guarantee ("raise, never silently mis-parse a renamed
    # symbol") only holds if the ValueError branches actually fire. Exercise them.
    def test_missing_prefix_raises(self):
        self.assertRaises(ValueError, parse_kernel_name, "ncclFooBar_AllGather_ST")

    def test_unknown_collective_raises(self):
        self.assertRaises(ValueError, parse_kernel_name, "ncclSymkDevKernel_Scatter_ST")

    def test_reduction_without_anchored_red_ty_raises(self):
        # Known reduction coll but no trailing _<red>_<ty> to anchor on.
        self.assertRaises(
            ValueError, parse_kernel_name, "ncclSymkDevKernel_ReduceScatter_RSxLD_AGxST")

    def test_reduction_with_unknown_type_raises(self):
        self.assertRaises(
            ValueError, parse_kernel_name, "ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f4")

    def test_empty_algo_raises(self):
        # Non-reduction collective with no algorithm token at all.
        self.assertRaises(ValueError, parse_kernel_name, "ncclSymkDevKernel_AllGather")


class SymmetricDiffReportHelperTest(unittest.TestCase):
    def test_none_when_identical(self):
        self.assertIsNone(diff_report(
            42, 42, {"A": 42}, {"A": 42}, {"ty": {"f32"}}, {"ty": {"f32"}}))

    def test_reports_total_coll_gained_lost_together(self):
        report = diff_report(
            42, 47,
            {"AllReduce": 10, "ReduceScatter": 30}, {"AllReduce": 15, "ReduceScatter": 30},
            {"ty": {"f32"}}, {"ty": {"f4"}},
        )
        self.assertIn("total 42 -> 47 (+5)", report)
        self.assertIn("AllReduce 10 -> 15 (+5)", report)
        self.assertIn("gained ['f4']", report)
        self.assertIn("lost ['f32']", report)

    def test_new_collective_shows_as_delta_from_zero(self):
        report = diff_report(
            42, 44, {"AllReduce": 42}, {"AllReduce": 42, "AllToAll": 2},
            {"coll": {"AllReduce"}}, {"coll": {"AllReduce", "AllToAll"}})
        self.assertIn("AllToAll 0 -> 2 (+2)", report)
        self.assertIn("gained ['AllToAll']", report)


if __name__ == "__main__":
    unittest.main()
