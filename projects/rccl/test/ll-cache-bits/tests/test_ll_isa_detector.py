# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Unit tests for the LL poll-load detector in ll_isa.py.

These run on hand-written disassembly, so they need no GPU, no ROCm and no
built RCCL. They pin down that the detector finds a poll by its shape
regardless of cache bits -- the property the real-binary guard relies on to
catch a poll that lost them.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ll_isa  # noqa: E402

pytestmark = pytest.mark.detector

# Trimmed from the gfx942 disassembly of
# runRing<float, FuncSum<float>, ProtoLL<...>> in librccl.so.
GFX942_POLL = """\
000000000797f500 <_ZN12_GLOBAL__N_17runRingIf7FuncSumIfE7ProtoLLLi0ELi0ELi4ELi0ELi0EEEviiP15ncclDevWorkColl>:
	v_lshl_add_u64 v[12:13], v[118:119], 4, v[134:135]          // 00000798072C:
	global_load_dwordx4 v[18:21], v[12:13], off{mods}           // 000007980734:
	v_cmp_eq_u32_e32 vcc, 0, v23                                // 00000798073C:
	s_waitcnt vmcnt(0)                                          // 000007980740:
	v_cmp_ne_u32_e64 s[16:17], v19, v149                        // 000007980744:
	v_cmp_ne_u32_e64 s[18:19], v21, v149                        // 00000798074C:
	s_or_b64 s[16:17], s[16:17], s[18:19]                       // 000007980754:
	s_cbranch_execz 65497                                       // 000007980760:
"""


def polls(text):
    return list(ll_isa.find_polls(text.splitlines()))


def test_finds_poll_with_nt():
    found = polls(GFX942_POLL.format(mods=" nt"))
    assert len(found) == 1
    assert found[0].modifiers == ["nt"]
    assert "ProtoLL" in found[0].func


def test_finds_poll_that_lost_its_cache_bits():
    # The whole point: a poll with no cache bits must still be found, so the
    # guard can report it instead of silently counting one poll fewer.
    found = polls(GFX942_POLL.format(mods=""))
    assert len(found) == 1
    assert found[0].modifiers == []


def test_swapped_compare_operands_and_sgpr_flag():
    text = """\
0000 <f>:
	global_load_b128 v[4:7], v[0:1], off th:TH_LOAD_NT
	s_wait_loadcnt 0x0
	v_cmp_ne_u32_e32 vcc_lo, s9, v5
	v_cmp_ne_u32_e64 s0, s9, v7
"""
    found = polls(text)
    assert [p.modifiers for p in found] == [["th:TH_LOAD_NT"]]


def test_one_flag_compared_is_not_a_poll():
    text = GFX942_POLL.format(mods=" nt").replace(
        "v_cmp_ne_u32_e64 s[18:19], v21, v149", "s_nop 0")
    assert polls(text) == []


def test_flags_compared_to_different_values_is_not_a_poll():
    text = GFX942_POLL.format(mods=" nt").replace("v21, v149", "v21, v150")
    assert polls(text) == []


def test_compare_after_next_load_belongs_to_that_load():
    text = """\
0000 <f>:
	global_load_dwordx4 v[0:3], v[8:9], off nt
	global_load_dwordx4 v[4:7], v[10:11], off
	v_cmp_ne_u32_e64 s[0:1], v1, v20
	v_cmp_ne_u32_e64 s[2:3], v3, v20
"""
    assert polls(text) == []


@pytest.mark.parametrize("insn,mods", [
    ("global_load_dwordx4 v[18:21], v[12:13], off nt", ["nt"]),
    ("global_load_dwordx4 v[0:3], v4, s[0:1] offset:16 sc0 sc1",
     ["offset:16", "sc0", "sc1"]),
    ("flat_load_dwordx4 v[0:3], v[4:5] glc slc", ["glc", "slc"]),
    ("global_load_b128 v[0:3], v[4:5], off th:TH_LOAD_NT scope:SCOPE_SYS",
     ["th:TH_LOAD_NT", "scope:SCOPE_SYS"]),
    ("global_load_dwordx4 v[0:3], v[4:5], off", []),
])
def test_modifiers(insn, mods):
    assert ll_isa.modifiers(insn) == mods
