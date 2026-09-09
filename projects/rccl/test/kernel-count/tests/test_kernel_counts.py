# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Kernel-count leak guard for src/device/generate.py (the main combinatorial
device-kernel generator).

A "kernel leak" is an unintended growth in the set of device kernels baked into
librccl -- e.g. adding a datatype/protocol/unroll, or relaxing a func_validate
rule so an existing axis multiplies out further. Each extra kernel costs binary
size and (device-linker) build time, and such growth otherwise merges unnoticed.

This runs generate.py CPU-only (no GPU, no built library) and asserts the
generated kernel set against hardcoded baselines in three complementary layers:

  1. per-dimension value-sets  -- names the culprit AXIS when a new
                                  type/proto/unroll/algo appears (root cause);
  2. per-collective counts      -- says WHERE growth landed; catches rule
                                  relaxations that add no new axis value (e.g. a
                                  new coll in ll128_reg_variant_colls, which
                                  doubles its LL128 kernels with existing values);
  3. grand total                -- the net effect a reviewer reads at a glance.

Both shipping configurations are guarded: ENABLE_ROCSHMEM OFF (default) and ON
(adds the AlltoAllGda / AlltoAllvGda GDA collectives).

Baselines seeded from origin/develop @ 4a99ef1f9c (develop, post-AllGatherV).

Note on emitted vs declared values: the value-sets are the values ACTUALLY
EMITTED into the specialized kernels, not the raw all_* lists in generate.py. In
particular i32/i64 never appear as distinct kernel types: equivalent_primary()
folds signed sum/prod/minmax onto the unsigned representative, so int32_t/int64_t
are absent. Seed from observed output, never from all_tys.
"""

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# tests/ -> kernel-count/ -> test/ -> rccl root
RCCL_ROOT = Path(__file__).resolve().parents[3]
GENERATE_PY = RCCL_ROOT / "src" / "device" / "generate.py"

# ---------------------------------------------------------------------------
# Baselines (origin/develop @ 4a99ef1f9c (develop, post-AllGatherV)).
# If a change moves these numbers, update them here AND explain in the PR
# description WHY the kernel count changed (binary-size / build-time impact).
# Do not blind-update.
# ---------------------------------------------------------------------------
EXPECTED = {
    "OFF": {
        "total": 6123,
        "per_coll": {
            "AllGather": 48,
            "AllGatherV": 18,
            "AllReduce": 3840,
            "AlltoAllPivot": 6,
            "Broadcast": 24,
            "Reduce": 726,
            "ReduceScatter": 1452,
            # 6 legacy-LL (reg=0) + 3 LL128 (reg=1, unroll 1/2/4, gfx942/950).
            "SendRecv": 9,
        },
    },
    "ON": {
        "total": 6135,
        "per_coll": {
            "AllGather": 48,
            "AllGatherV": 18,
            "AllReduce": 3840,
            "AlltoAllGda": 6,
            "AlltoAllPivot": 6,
            "AlltoAllvGda": 6,
            "Broadcast": 24,
            "Reduce": 726,
            "ReduceScatter": 1452,
            # 6 legacy-LL (reg=0) + 3 LL128 (reg=1, unroll 1/2/4, gfx942/950).
            "SendRecv": 9,
        },
    },
}

# Per-dimension value-sets. Only `coll` differs between OFF/ON (the two GDA
# collectives); every other axis is identical, so it is shared. Spellings match
# DEFINE_ncclDevFunc: redop as Func*, ty as the C++ type.
COMMON_DIMS = {
    "algo": {"RING", "TREE", "PAT"},
    "proto": {"LL", "LL128", "SIMPLE"},
    "redop": {"FuncSum", "FuncProd", "FuncMinMax", "FuncPreMulSum", "FuncSumPostDiv"},
    "ty": {
        "int8_t", "uint8_t", "uint32_t", "uint64_t", "half", "float",
        "double", "hip_bfloat16", "rccl_float8", "rccl_bfloat8",
    },
    "acc": {"0", "1"},
    "pipeline": {"0", "1"},
    "unroll": {"1", "2", "4", "8", "16", "32"},
    "reg": {"0", "1", "2"},
}
DIMENSIONS = ("coll", "algo", "proto", "redop", "ty", "acc", "pipeline", "unroll", "reg")

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


def _run_generator(out_dir, rocshmem):
    """Run the main generator into out_dir for the given rocSHMEM setting.

    argv layout matches src/CMakeLists.txt:
      gensrc, IFC, <unused>, local_gpu_only, rocshmem, ONLY_FUNCS
    local_gpu_only=OFF keeps the count deterministic and GPU-free (no rocminfo)
    and yields the full MULTI-ARCH superset of kernels (arch-guarded variants
    included); a BUILD_LOCAL_GPU_TARGET_ONLY=ON build legitimately emits fewer.
    An empty ONLY_FUNCS makes the generator use its real default shipping set.
    """
    if not GENERATE_PY.exists():
        # Explicit failure, never a skip: a co-located generator that vanished is
        # a real breakage. (pytest.fail also survives `python -O`, unlike assert.)
        pytest.fail("generate.py not found next to source tree: %s" % GENERATE_PY)
    subprocess.run(
        [sys.executable, str(GENERATE_PY), str(out_dir), "OFF", "OFF", "OFF", rocshmem, ""],
        check=True,
        capture_output=True,
        text=True,
    )


def _parse(out_dir):
    """Parse specialized/*.cpp; enforce exactly one DEFINE per file.

    Returns (records, num_files, manifest_count). The per-file check keeps PARSER
    breakage (regex no longer matches the emitted format) distinct from a
    legitimate baseline move: a file with zero matches fails here with a "format
    changed" message rather than silently lowering the count. manifest_count is an
    INDEPENDENT tally from specialized_files.txt (generate.py's device-linker
    contract, built from a different code path than the .cpp files).
    """
    spec_dir = os.path.join(out_dir, "specialized")
    assert os.path.isdir(spec_dir), "generator produced no specialized/ dir"
    files = sorted(glob.glob(os.path.join(spec_dir, "*.cpp")))
    assert files, "generator produced no specialized kernel files"

    records = []
    for fp in files:
        with open(fp) as f:
            text = f.read()
        matches = list(_DEFINE_RE.finditer(text))
        assert len(matches) == 1, (
            "parser integrity: expected exactly one DEFINE_ncclDevFunc in %s, found "
            "%d -- generate.py output format likely changed; update _DEFINE_RE "
            "(this is NOT a kernel-count change)" % (os.path.basename(fp), len(matches))
        )
        records.append(matches[0].groupdict())

    manifest = os.path.join(out_dir, "specialized_files.txt")
    assert os.path.isfile(manifest), (
        "generator did not write specialized_files.txt (device-linker contract)"
    )
    with open(manifest) as f:
        manifest_count = sum(1 for line in f if line.strip())

    return records, len(files), manifest_count


def _per_coll(records):
    counts = {}
    for r in records:
        counts[r["coll"]] = counts.get(r["coll"], 0) + 1
    return counts


def _dims(records):
    return {dim: {r[dim] for r in records} for dim in DIMENSIONS}


def _expected_dims(per_coll):
    return dict(COMMON_DIMS, coll=set(per_coll))


def diff_report(label, exp_total, act_total, exp_per_coll, act_per_coll, exp_dims, act_dims):
    """Return a combined total/per-collective/per-dimension diff, or None if all
    match. Reports net effect, location, and root cause together."""
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
        "ACTION: if intentional, update the EXPECTED constants in "
        "test_kernel_counts.py AND explain in the PR description WHY the kernel "
        "count changed (binary-size / build-time impact). Do not blind-update."
    )
    return "\n".join(["Kernel count changed (%s):" % label] + lines + [action])


@pytest.fixture(scope="session")
def generated(tmp_path_factory):
    """Run the generator once per rocSHMEM setting and parse the output."""
    out = {}
    for rocshmem in ("OFF", "ON"):
        d = tmp_path_factory.mktemp("main_%s" % rocshmem.lower())
        _run_generator(str(d), rocshmem)
        records, num_files, manifest_count = _parse(str(d))
        out[rocshmem] = {
            "records": records,
            "num_files": num_files,
            "manifest_count": manifest_count,
        }
    return out


@pytest.mark.main_generator
@pytest.mark.parametrize("rocshmem", ["OFF", "ON"])
def test_kernel_count_baselines(generated, rocshmem):
    data = generated[rocshmem]
    exp = EXPECTED[rocshmem]
    report = diff_report(
        "rocshmem=%s" % rocshmem,
        exp["total"], len(data["records"]),
        exp["per_coll"], _per_coll(data["records"]),
        _expected_dims(exp["per_coll"]), _dims(data["records"]),
    )
    assert report is None, report


@pytest.mark.main_generator
@pytest.mark.parametrize("rocshmem", ["OFF", "ON"])
def test_parser_integrity(generated, rocshmem):
    # Independent cross-check: the .cpp-parsed record count must agree with
    # specialized_files.txt (a separate generator code path). Distinct from a
    # baseline move so a format change reads as "parser broke", not "count fell".
    data = generated[rocshmem]
    assert data["manifest_count"] == len(data["records"]), (
        "generator contract: specialized_files.txt lists %d kernels but %d were "
        "parsed from specialized/*.cpp" % (data["manifest_count"], len(data["records"]))
    )


@pytest.mark.main_generator
def test_rocshmem_on_is_off_plus_gda(generated):
    # Generator-backed (not just constant self-consistency): ON must equal OFF
    # plus exactly the two GDA collectives, with every shared collective's count
    # unchanged. Catches ON accidentally perturbing non-GDA kernels.
    off = _per_coll(generated["OFF"]["records"])
    on = _per_coll(generated["ON"]["records"])
    assert set(on) - set(off) == {"AlltoAllGda", "AlltoAllvGda"}
    assert set(off) - set(on) == set()
    for coll, n in off.items():
        assert on.get(coll) == n, "collective %s changed between rocSHMEM OFF and ON" % coll
    assert sum(on.values()) - sum(off.values()) == on["AlltoAllGda"] + on["AlltoAllvGda"]


# --- diagnostic helper unit tests (no generator run) ------------------------
@pytest.mark.diagnostics
def test_diff_report_none_when_identical():
    assert diff_report("x", 10, 10, {"A": 5}, {"A": 5}, {"ty": {"x"}}, {"ty": {"x"}}) is None


@pytest.mark.diagnostics
def test_diff_report_reports_total_coll_gained_lost_together():
    report = diff_report(
        "x", 10, 12, {"A": 5, "B": 5}, {"A": 7, "B": 5},
        {"ty": {"old"}}, {"ty": {"new"}},
    )
    assert report is not None
    assert "total 10 -> 12 (+2)" in report
    assert "A 5 -> 7 (+2)" in report
    assert "gained ['new']" in report
    assert "lost ['old']" in report
    assert "PR description" in report


@pytest.mark.diagnostics
def test_diff_report_new_collective_from_zero():
    report = diff_report(
        "x", 5, 6, {"A": 5}, {"A": 5, "GDA": 1},
        {"coll": {"A"}}, {"coll": {"A", "GDA"}},
    )
    assert "GDA 0 -> 1 (+1)" in report
    assert "gained ['GDA']" in report
