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


def test_compares_after_inlined_abort_check():
    # The inlined abort check can sit between the load and the compares.
    text = GFX942_POLL.format(mods=" nt").replace(
        "\ts_waitcnt vmcnt(0)", "\ts_waitcnt vmcnt(0)\n" + "\ts_nop 0\n" * 30)
    assert len(polls(text)) == 1


def test_one_flag_compared_is_not_a_poll():
    text = GFX942_POLL.format(mods=" nt").replace(
        "v_cmp_ne_u32_e64 s[18:19], v21, v149", "s_nop 0")
    assert polls(text) == []


def test_flags_compared_to_different_values_is_not_a_poll():
    text = GFX942_POLL.format(mods=" nt").replace("v21, v149", "v21, v150")
    assert polls(text) == []


def test_eq_compares_and_sys_scope():
    # gfx1250 shape: flags checked with v_cmp_eq, load at system scope.
    text = """\
0000 <f>:
	global_load_b128 v[2:5], v[8:9], off scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_cmp_eq_u32_e32 vcc_lo, s10, v3
	v_cmp_eq_u32_e64 s0, s10, v5
"""
    assert [p.modifiers for p in polls(text)] == [["scope:SCOPE_SYS"]]


def test_unpolled_lists_ll_functions_without_a_poll():
    text = GFX942_POLL.format(mods=" nt") + """\
0001 <_Z6runLLA7ProtoLLLi0EE>:
	global_load_dwordx4 v[0:3], v[8:9], off nt
0002 <_Z6runLL128A10ProtoLL128Li0EE>:
0003 <_ZN3dda6common18ddaAllReduceFlatLLIfLi0EEEvv>:
0004 <_ZN3dda6common21ddaAllReduceFlatLL128IfLi0EEEvv>:
0005 <_Z30ncclSymkDevKernel_AllGather_LL21ncclSymkDevWorkArgs4K>:
0006 <_Z30ncclSymkDevKernel_AllGather_ST21ncclSymkDevWorkArgs4K>:
0007 <other>:
"""
    lines = text.splitlines()
    assert ll_isa.unpolled(lines, polls(text)) == [
        "_Z6runLLA7ProtoLLLi0EE",
        "_ZN3dda6common18ddaAllReduceFlatLLIfLi0EEEvv",
        "_Z30ncclSymkDevKernel_AllGather_LL21ncclSymkDevWorkArgs4K",
    ]


def test_batched_loads_with_chained_compares():
    # LLA2A shape: loads issued back to back, flag2 compared against flag1.
    text = """\
0000 <f>:
	global_load_dwordx4 v[16:19], v[32:33], off sc0 sc1
	global_load_dwordx4 v[12:15], v[34:35], off sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u32_e32 vcc, v17, v3
	v_cmp_eq_u32_e64 s[4:5], v19, v17
	v_cmp_eq_u32_e32 vcc, v13, v3
	v_cmp_eq_u32_e64 s[4:5], v15, v13
"""
    assert [p.modifiers for p in polls(text)] == [["sc0", "sc1"]] * 2


def test_flags_compared_through_copies():
    # DDA shape: the flags are copied before they are compared.
    text = """\
0000 <f>:
	global_load_dwordx4 v[56:59], v[34:35], off sc0 sc1
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v8, v57
	v_mov_b32_e32 v12, v59
	v_cmp_eq_u32_e64 s[20:21], s33, v8
	v_cmp_eq_u32_e64 s[22:23], s33, v12
"""
    assert len(polls(text)) == 1


def test_compares_after_flags_are_reloaded_belong_to_the_new_load():
    text = """\
0000 <f>:
	global_load_dwordx4 v[0:3], v[8:9], off nt
	global_load_dwordx4 v[0:3], v[10:11], off
	v_cmp_ne_u32_e64 s[0:1], v1, v20
	v_cmp_ne_u32_e64 s[2:3], v3, v20
"""
    assert [p.modifiers for p in polls(text)] == [[]]


def test_split_bundles(tmp_path):
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    data = (b"CCOB" + b"\0" * (ll_isa.BUNDLE_ALIGN - 4) + magic + b"x" * 8
            + magic)  # an unaligned magic is not a bundle start
    fatbin = tmp_path / "fb"
    fatbin.write_bytes(data)
    paths = ll_isa.split_bundles(str(fatbin), str(tmp_path))
    assert [Path(p).read_bytes()[:4] for p in paths] == [b"CCOB", magic[:4]]


def test_compares_in_next_function_are_not_a_poll():
    text = """\
0000 <f>:
	global_load_dwordx4 v[0:3], v[8:9], off nt
0001 <g>:
	v_cmp_ne_u32_e64 s[0:1], v1, v20
	v_cmp_ne_u32_e64 s[2:3], v3, v20
"""
    assert polls(text) == []


def test_targets_keeps_every_feature_variant(monkeypatch):
    tc = ll_isa.Toolchain("/nonexistent")
    ids = ["hip-amdgcn-amd-amdhsa--gfx950:xnack-",
           "hip-amdgcn-amd-amdhsa--gfx950:xnack+"]
    listing = "host-x86_64-unknown-linux-gnu-\n" + "\n".join(ids)
    monkeypatch.setattr(tc, "run", lambda *args: listing)
    assert tc.targets("fatbin") == {"gfx950": ids}


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
