# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Guard: every LL flag-poll load in librccl.so keeps its cache-policy bits.

See ../README.md for why the bits matter and the environment variables.
"""

import functools
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ll_isa  # noqa: E402

pytestmark = pytest.mark.isa

# tests/ -> ll-cache-bits/ -> test/ -> rccl root
RCCL_ROOT = Path(__file__).resolve().parents[3]
# install.sh builds into build/release, or build/debug with --debug.
_DEFAULT_BUILD = next((d for d in (RCCL_ROOT / "build" / "release",
                                   RCCL_ROOT / "build" / "debug")
                       if (d / "librccl.so").exists()),
                      RCCL_ROOT / "build" / "release")
RCCL_BUILD = Path(os.environ.get("RCCL_BUILD", _DEFAULT_BUILD))
RCCL_LIB = Path(os.environ.get("RCCL_LIB", RCCL_BUILD / "librccl.so"))
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
REQUIRED = os.environ.get("RCCL_LL_GUARD_REQUIRED") == "1"

# ---------------------------------------------------------------------------
# Modifiers every LL poll load must carry, per arch, as printed by
# llvm-objdump. Each is the disassembler's spelling of the load's cache bits.
# Extend this table (after checking the arch's ISA guide) when RCCL starts
# building for a new arch.
# ---------------------------------------------------------------------------
EXPECTED_POLL_MODIFIERS = {
    # CDNA3 / CDNA4: NT=1 -- non-temporal, so the poll never re-hits a stale
    # cached line.
    "gfx942": {"nt"},
    "gfx950": {"nt"},
    # gfx1250: system scope (RCCL_LL_FIFO_SYS_SCOPE in rccl_ptr.h).
    "gfx1250": {"scope:SCOPE_SYS"},
}

MAX_REPORTED = 10


def _skip(reason):
    # In CI a skip reads as a pass, so the guard must not quietly not run.
    if REQUIRED:
        pytest.fail(reason + " [RCCL_LL_GUARD_REQUIRED=1]")
    pytest.skip(reason)


@functools.lru_cache(maxsize=None)
def _toolchain():
    tc = ll_isa.Toolchain(ROCM_PATH)
    missing = tc.missing()
    if missing:
        _skip(f"ROCm LLVM tools not found in {tc.bindir}: {missing} "
              "(set ROCM_PATH)")
    return tc


@pytest.fixture(scope="session")
def fatbin(tmp_path_factory):
    if not RCCL_LIB.is_file():
        _skip(f"librccl.so not found at {RCCL_LIB} "
              "(build RCCL, or set RCCL_BUILD / RCCL_LIB)")
    out = str(tmp_path_factory.mktemp("fatbin") / "librccl.hip_fatbin")
    try:
        _toolchain().extract_fatbin(str(RCCL_LIB.resolve()), out)
    except subprocess.CalledProcessError as e:
        # e.g. a packaged lib whose device code was split out (kpack).
        _skip(f"no device code in {RCCL_LIB}: {e.stderr.strip()}")
    return out


@pytest.fixture(scope="session")
def targets(fatbin):
    found = _toolchain().targets(fatbin)
    assert found, f"no amdgcn device code found in {RCCL_LIB}"
    if not set(found) & set(EXPECTED_POLL_MODIFIERS):
        _skip(f"{RCCL_LIB} has none of the checked arches "
              f"{sorted(EXPECTED_POLL_MODIFIERS)}, only {sorted(found)}")
    return found


@pytest.fixture(scope="session")
def disassemble(fatbin, targets, tmp_path_factory):
    @functools.lru_cache(maxsize=None)
    def _disassemble(tid):
        work = tmp_path_factory.mktemp("co")
        code_object = str(work / "code.co")
        _toolchain().unbundle(fatbin, tid, code_object)
        return _toolchain().candidate_disassembly(code_object, str(work))

    return _disassemble


def test_unchecked_arches(targets):
    # Skipped, not failed, so a default all-arch build still runs the guard for
    # the arches it does know. The skip reason keeps the gap visible.
    unknown = sorted(set(targets) - set(EXPECTED_POLL_MODIFIERS))
    if unknown:
        pytest.skip(
            f"no EXPECTED_POLL_MODIFIERS entry for {unknown}: their LL poll "
            "loads are not checked. To add one, disassemble the poll loads and "
            "confirm in the arch's ISA guide that the bits bypass stale "
            "cached copies."
        )


@pytest.mark.parametrize("arch", sorted(EXPECTED_POLL_MODIFIERS))
def test_ll_poll_loads_keep_cache_bits(arch, targets, disassemble):
    if arch not in targets:
        pytest.skip(f"{RCCL_LIB} was not built for {arch}")
    want = EXPECTED_POLL_MODIFIERS[arch]
    for tid in targets[arch]:
        name = tid.split("--", 1)[1]
        lines = disassemble(tid)
        polls = list(ll_isa.find_polls(lines))

        # A guard that finds nothing passes vacuously.
        unmatched = ll_isa.unpolled_proto_ll(lines, polls)
        assert polls, f"{name}: found no LL flag-poll loads; update ll_isa.py."
        assert not unmatched, (
            f"{name}: found no LL flag-poll load in {len(unmatched)} ProtoLL "
            f"function(s), e.g. {unmatched[:3]}. The poll's instruction shape "
            "changed (or it was split into narrower loads); update ll_isa.py."
        )

        bad = [p for p in polls if not want <= set(p.modifiers)]
        report = "\n".join(f"  {p.insn}\n      in {p.func}"
                           for p in bad[:MAX_REPORTED])
        assert not bad, (
            f"{name}: {len(bad)} of {len(polls)} LL flag-poll loads lack "
            f"{sorted(want)}. Without them the receiver can spin on a stale "
            f"cached copy of the FIFO line and hang. First "
            f"{min(len(bad), MAX_REPORTED)}:\n{report}"
        )
