#!/usr/bin/env python3
"""Kernel-count guard for src/device/generate.py (the main combinatorial generator).

Purpose
-------
A "kernel leak" is an unintended growth in the set of device kernels baked into
librccl -- e.g. adding a datatype/protocol/unroll, or relaxing a func_validate
rule so an existing axis multiplies out further. Each extra kernel costs binary
size and (device-linker) build time, and such growth otherwise merges unnoticed.

This test runs generate.py CPU-only (no GPU, no built library) and asserts the
generated kernel set against hardcoded baselines in three complementary layers:

  1. per-dimension value-sets   -- names the culprit AXIS when a new
                                   type/proto/unroll/algo/etc. appears (root cause);
  2. per-collective counts       -- says WHERE growth landed; catches rule
                                   relaxations that add no new axis value (e.g.
                                   widening func_validate, or adding a coll to
                                   ll128_reg_variant_colls, which doubles its LL128
                                   kernels using only existing axis values);
  3. grand total                 -- the net effect a reviewer reads at a glance.

When any layer moves, the test fails with a combined diff and REQUIRES the author
to update the EXPECTED_* constants below AND justify the change in the PR
description. Do not blind-update the numbers.

Both shipping configurations are guarded: ENABLE_ROCSHMEM OFF (default) and ON
(adds the AlltoAllGda / AlltoAllvGda GDA collectives).

Baselines seeded from origin/develop @ 40d3fae2dfa077769d45bac77a16e849c72d89ea.

Note on emitted vs declared values
-----------------------------------
The value-sets below are the values ACTUALLY EMITTED into the specialized
kernels, not the raw all_* lists in generate.py. In particular i32/i64 never
appear as distinct kernel types: equivalent_primary() folds signed sum/prod/
minmax onto the unsigned representative, so int32_t/int64_t are absent. Seed from
observed output, never from all_tys.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Dict, Optional, Set

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATE_PY = os.path.join(HERE, "generate.py")

# ---------------------------------------------------------------------------
# Baselines (origin/develop @ 40d3fae2dfa077769d45bac77a16e849c72d89ea).
# ---------------------------------------------------------------------------
EXPECTED_TOTAL_OFF = 6102
EXPECTED_PER_COLL_OFF = {
    "AllGather": 48,
    "AllReduce": 3840,
    "AlltoAllPivot": 6,
    "Broadcast": 24,
    "Reduce": 726,
    "ReduceScatter": 1452,
    "SendRecv": 6,
}

EXPECTED_TOTAL_ON = 6114
EXPECTED_PER_COLL_ON = {
    "AllGather": 48,
    "AllReduce": 3840,
    "AlltoAllGda": 6,
    "AlltoAllPivot": 6,
    "AlltoAllvGda": 6,
    "Broadcast": 24,
    "Reduce": 726,
    "ReduceScatter": 1452,
    "SendRecv": 6,
}

# Per-dimension value-sets. Only `coll` differs between OFF/ON (the two GDA
# collectives); every other axis is identical, so it is shared. These are the
# spellings emitted by DEFINE_ncclDevFunc: redop as Func*, ty as the C++ type.
_COMMON_DIMS: Dict[str, Set[str]] = {
    "algo":     {"RING", "TREE", "PAT"},
    "proto":    {"LL", "LL128", "SIMPLE"},
    "redop":    {"FuncSum", "FuncProd", "FuncMinMax", "FuncPreMulSum", "FuncSumPostDiv"},
    "ty": {
        "int8_t", "uint8_t", "uint32_t", "uint64_t", "half", "float",
        "double", "hip_bfloat16", "rccl_float8", "rccl_bfloat8",
    },
    "acc":      {"0", "1"},
    "pipeline": {"0", "1"},
    "unroll":   {"1", "2", "4", "8", "16", "32"},
    "reg":      {"0", "1", "2"},
}
EXPECTED_DIMS_OFF = dict(_COMMON_DIMS, coll=set(EXPECTED_PER_COLL_OFF))
EXPECTED_DIMS_ON = dict(_COMMON_DIMS, coll=set(EXPECTED_PER_COLL_ON))

_DIMENSIONS = ("coll", "algo", "proto", "redop", "ty", "acc", "pipeline", "unroll", "reg")

# One DEFINE_ncclDevFunc(...) is emitted per specialized kernel. It carries every
# dimension explicitly, so we parse those fields rather than the mangled sym
# suffix (which omits reg when reg==0, making field positions variable).
_DEFINE_RE = re.compile(
    r"DEFINE_ncclDevFunc\(\s*"
    r"(?P<sym>\w+)\s*,\s*"
    r"ncclFunc(?P<coll>\w+)\s*,\s*"
    r"(?P<redop>\w+)\s*,\s*"
    r"(?P<ty>\w+)\s*,\s*"
    r"NCCL_ALGO_(?P<algo>\w+)\s*,\s*"
    r"NCCL_PROTO_(?P<proto>\w+)\s*,\s*"
    r"(?P<acc>\d+)\s*,\s*"
    r"(?P<pipeline>\d+)\s*,\s*"
    r"(?P<unroll>\d+)\s*,\s*"
    r"(?P<reg>\d+)\s*\)"
)


# ---------------------------------------------------------------------------
# Pure diagnostic helper (unit-tested directly below with synthetic records, so
# the diff formatting is verified without having to perturb the generator).
# ---------------------------------------------------------------------------
def diff_report(
    label: str,
    exp_total: int,
    act_total: int,
    exp_per_coll: Dict[str, int],
    act_per_coll: Dict[str, int],
    exp_dims: Dict[str, Set[str]],
    act_dims: Dict[str, Set[str]],
) -> Optional[str]:
    """Return a combined human-readable diff, or None if everything matches.

    Reports total delta, per-collective deltas, and per-dimension gained/lost
    values together so a single failure explains net effect, location, and root
    cause at once.
    """
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
        "test_generate_kernel_counts.py AND explain in the PR description WHY the "
        "kernel count changed (binary-size / build-time impact). Do not blind-update."
    )
    return "\n".join(["Kernel count changed (%s):" % label] + lines + [action])


def _generate(tmpdir: str, rocshmem: str) -> None:
    """Run the main generator into tmpdir for the given rocSHMEM setting.

    argv layout matches src/CMakeLists.txt:
      gensrc, IFC, <unused>, local_gpu_only, rocshmem, ONLY_FUNCS
    local_gpu_only=OFF keeps the count deterministic and GPU-free (no rocminfo);
    an empty ONLY_FUNCS makes the generator use its real default shipping set.
    """
    subprocess.run(
        [sys.executable, GENERATE_PY, tmpdir, "OFF", "OFF", "OFF", rocshmem, ""],
        check=True,
        capture_output=True,
        text=True,
    )


def _parse_config(test, tmpdir):
    """Parse specialized/*.cpp; enforce exactly one DEFINE per file.

    Returns (records, num_files). The per-file check keeps PARSER breakage (regex
    no longer matches the emitted format) distinct from a legitimate baseline
    move: a file with zero matches fails here with a "format changed" message
    rather than silently lowering the count.
    """
    spec_dir = os.path.join(tmpdir, "specialized")
    test.assertTrue(os.path.isdir(spec_dir), "generator produced no specialized/ dir")
    files = sorted(glob.glob(os.path.join(spec_dir, "*.cpp")))
    test.assertTrue(files, "generator produced no specialized kernel files")

    records = []
    for fp in files:
        with open(fp) as f:
            text = f.read()
        matches = list(_DEFINE_RE.finditer(text))
        test.assertEqual(
            len(matches), 1,
            "parser integrity: expected exactly one DEFINE_ncclDevFunc in %s, found "
            "%d -- generate.py output format likely changed; update _DEFINE_RE "
            "(this is NOT a kernel-count change)" % (os.path.basename(fp), len(matches)),
        )
        records.append(matches[0].groupdict())
    return records, len(files)


def _per_coll(records):
    counts = {}
    for r in records:
        counts[r["coll"]] = counts.get(r["coll"], 0) + 1
    return counts


def _dims(records):
    return {dim: {r[dim] for r in records} for dim in _DIMENSIONS}


class MainGeneratorKernelCountTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")
        cls._dir_off = tempfile.mkdtemp(prefix="rccl_kcount_off_")
        cls._dir_on = tempfile.mkdtemp(prefix="rccl_kcount_on_")
        _generate(cls._dir_off, "OFF")
        _generate(cls._dir_on, "ON")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir_off, ignore_errors=True)
        shutil.rmtree(cls._dir_on, ignore_errors=True)

    def _check(self, tmpdir, label, exp_total, exp_per_coll, exp_dims):
        records, num_files = _parse_config(self, tmpdir)
        # Parser integrity: one DEFINE per file => records track files exactly.
        self.assertEqual(
            num_files, len(records),
            "parser integrity: %d specialized files but %d parsed kernels" % (num_files, len(records)),
        )
        report = diff_report(
            label,
            exp_total, len(records),
            exp_per_coll, _per_coll(records),
            exp_dims, _dims(records),
        )
        self.assertIsNone(report, report)

    def test_rocshmem_off(self):
        self._check(self._dir_off, "rocshmem=OFF", EXPECTED_TOTAL_OFF,
                    EXPECTED_PER_COLL_OFF, EXPECTED_DIMS_OFF)

    def test_rocshmem_on(self):
        self._check(self._dir_on, "rocshmem=ON", EXPECTED_TOTAL_ON,
                    EXPECTED_PER_COLL_ON, EXPECTED_DIMS_ON)

    def test_rocshmem_on_adds_only_gda_collectives(self):
        # Sanity: ON is a strict superset of OFF, differing solely by the two GDA
        # collectives. Guards against ON accidentally perturbing other kernels.
        self.assertEqual(EXPECTED_TOTAL_ON - EXPECTED_TOTAL_OFF,
                         EXPECTED_PER_COLL_ON["AlltoAllGda"] + EXPECTED_PER_COLL_ON["AlltoAllvGda"])
        self.assertEqual(set(EXPECTED_PER_COLL_ON) - set(EXPECTED_PER_COLL_OFF),
                         {"AlltoAllGda", "AlltoAllvGda"})


class DiffReportHelperTest(unittest.TestCase):
    """Verifies the diagnostic helper directly (correction #5): no generator run,
    no fragile dummy-type injection -- just synthetic old/new records."""

    def test_none_when_identical(self):
        self.assertIsNone(diff_report(
            "x", 10, 10, {"A": 5, "B": 5}, {"A": 5, "B": 5}, {"ty": {"x"}}, {"ty": {"x"}}))

    def test_reports_total_coll_and_gained_and_lost_together(self):
        report = diff_report(
            "x",
            10, 12,
            {"A": 5, "B": 5}, {"A": 7, "B": 5},
            {"ty": {"old"}}, {"ty": {"new"}},
        )
        self.assertIsNotNone(report)
        self.assertIn("total 10 -> 12 (+2)", report)
        self.assertIn("A 5 -> 7 (+2)", report)
        self.assertIn("gained ['new']", report)
        self.assertIn("lost ['old']", report)
        self.assertIn("PR description", report)

    def test_new_collective_shows_as_delta_from_zero(self):
        report = diff_report(
            "x", 5, 6, {"A": 5}, {"A": 5, "GDA": 1},
            {"coll": {"A"}}, {"coll": {"A", "GDA"}})
        self.assertIn("GDA 0 -> 1 (+1)", report)
        self.assertIn("gained ['GDA']", report)


if __name__ == "__main__":
    unittest.main()
