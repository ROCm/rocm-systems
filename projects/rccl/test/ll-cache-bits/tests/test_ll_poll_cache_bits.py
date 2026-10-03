# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Guard: every LL flag-poll load in librccl.so keeps its cache-policy bits.

The LL protocol's receiver re-reads one 16-byte FIFO line until its flags
change (src/device/prims_ll.h). A peer GPU or NIC writes that line, so the load
must not be served from a stale cached copy. The source only *asks* for this
(__builtin_nontemporal_load, or a system-scope load on some arches); the
compiler decides the bits that end up in the instruction. If a compiler update,
a code change, or a new arch drops them, everything still builds and the first
symptom is a collective that hangs on real hardware.

This test reads the device ISA out of a built librccl.so, finds every LL poll
load by its shape (see ll_isa.py), and asserts it carries the expected bits for
its arch. It needs a built RCCL and the ROCm LLVM tools, but no GPU.

  RCCL_LIB    path to librccl.so  (default: $RCCL_BUILD/librccl.so)
  RCCL_BUILD  RCCL build dir      (default: <rccl>/build/{release,debug})
  ROCM_PATH   ROCm install root   (default: /opt/rocm)
"""

import functools
import os
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

# ---------------------------------------------------------------------------
# Modifiers every LL poll load must carry, per arch, as printed by
# llvm-objdump. Each is the disassembler's spelling of the load's cache bits.
# Seeded from librccl.so built at origin/develop; extend this table (after
# checking the arch's ISA guide) when RCCL starts building for a new arch.
# ---------------------------------------------------------------------------
EXPECTED_POLL_MODIFIERS = {
    # CDNA3 / CDNA4: NT=1 -- non-temporal, so the poll never re-hits a stale
    # cached line.
    "gfx942": {"nt"},
    "gfx950": {"nt"},
}

MAX_REPORTED = 10


@functools.lru_cache(maxsize=None)
def _toolchain():
    tc = ll_isa.Toolchain(ROCM_PATH)
    missing = tc.missing()
    if missing:
        pytest.skip(f"ROCm LLVM tools not found in {tc.bindir}: {missing} "
                    "(set ROCM_PATH)")
    return tc


@pytest.fixture(scope="session")
def fatbin(tmp_path_factory):
    if not RCCL_LIB.is_file():
        pytest.skip(f"librccl.so not found at {RCCL_LIB} "
                    "(build RCCL, or set RCCL_BUILD / RCCL_LIB)")
    out = str(tmp_path_factory.mktemp("fatbin") / "librccl.hip_fatbin")
    _toolchain().extract_fatbin(str(RCCL_LIB.resolve()), out)
    return out


@pytest.fixture(scope="session")
def targets(fatbin):
    found = _toolchain().targets(fatbin)
    assert found, f"no amdgcn device code found in {RCCL_LIB}"
    return found


@pytest.fixture(scope="session")
def disassemble(fatbin, targets, tmp_path_factory):
    @functools.lru_cache(maxsize=None)
    def _disassemble(arch):
        work = tmp_path_factory.mktemp(arch)
        code_object = str(work / f"{arch}.co")
        _toolchain().unbundle(fatbin, targets[arch], code_object)
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
    lines = disassemble(arch)
    polls = list(ll_isa.find_polls(lines))

    # A guard that finds nothing passes vacuously. Every Primitives<ProtoLL>
    # instantiation receives through the poll loop, so each must show at
    # least one; if not, the detector no longer matches what the compiler
    # emits and must be updated before this test means anything.
    polled = {p.func for p in polls}
    unmatched = [f for f in ll_isa.functions(lines)
                 if ll_isa.PROTO_LL_RE.search(f) and f not in polled]
    assert polls and not unmatched, (
        f"{arch}: found no LL flag-poll load in {len(unmatched)} ProtoLL "
        f"function(s), e.g. {unmatched[:3]}. The poll's instruction shape "
        "changed (or it was split into narrower loads); update ll_isa.py."
    )

    want = EXPECTED_POLL_MODIFIERS[arch]
    bad = [p for p in polls if not want <= set(p.modifiers)]
    report = "\n".join(f"  {p.insn}\n      in {p.func}"
                       for p in bad[:MAX_REPORTED])
    assert not bad, (
        f"{arch}: {len(bad)} of {len(polls)} LL flag-poll loads lack "
        f"{sorted(want)}. Without them the receiver can spin on a stale "
        f"cached copy of the FIFO line and hang. First "
        f"{min(len(bad), MAX_REPORTED)}:\n{report}"
    )
