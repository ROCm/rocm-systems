# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from amdisa.codegen.execute.rj_arith_vopd import gen_vopd_execution
from amdisa.isa_profile import Rdna3Profile, Rdna4Profile


@pytest.mark.parametrize(
    'arch,profile', [('rdna3', Rdna3Profile()), ('rdna4', Rdna4Profile())]
)
def test_dual_routes_every_slot_and_removes_old_execution(arch, profile):
    body = gen_vopd_execution(arch, profile.vopd_slot_ops, "return;")
    assert 'execute_dual_pair' in body
    wave64_guard = body.split('if (wf.wf_size() != 32) {', 1)[1].split('}', 1)[0]
    assert 'return;' in wave64_guard
    assert 'UnimplementedInst' not in wave64_guard
    dx9 = body.split('case kVopdMulDx9ZeroF32:', 1)[1].split('case ', 1)[0]
    assert 'arithmetic_flags()' in dx9
    assert len(profile.vopd_slot_ops) == 17
    for op in profile.vopd_slot_ops:
        assert f'case k{op.enum_name}:' in body
        assert f'return rj_arith_{op.mnemonic}' in body
    for legacy in ('execute_slot', 'uses_src_neg_modifier', 'apply_neg', 'bitop2'):
        assert legacy not in body
    assert 'try_execute_vopd' not in body
    assert 'fp_mode::arithmetic' not in body
    assert 'gfx12_dot' not in body and 'gfx11_dot' not in body
    assert 'slot.src2_imm' in body
    assert 'slot.neg' not in body
    assert 'dot2_src_needs_half_replication' in body
    assert 'std::copy_n(acc.data, 32, result)' in body
    assert gen_vopd_execution('cdna5', profile.vopd_slot_ops, '') is None
