# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from amdisa.codegen import CodeGenerator
from amdisa.codegen.config import CodegenConfig
from amdisa.codegen.execute.rj_arith_vopd import gen_vopd_execution
from amdisa.isa_profile import Rdna3Profile, Rdna4Profile, Cdna5Profile


@pytest.mark.parametrize(
    'arch,profile',
    [('rdna3', Rdna3Profile()), ('rdna4', Rdna4Profile()), ('cdna5', Cdna5Profile())],
)
def test_dual_routes_every_slot_and_removes_old_execution(
    tmp_path, monkeypatch, arch, profile
):
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name=arch,
        generated_dir_name=arch,
        cpp_namespace=arch,
        profile=profile,
        operand_types={'OPR_SRC'},
    )
    codegen.out_path = str(tmp_path)
    codegen.config = CodegenConfig()
    codegen.gen_vopd()
    header = (tmp_path / arch / 'vopd.h').read_text()
    body = (tmp_path / arch / 'vopd_exec.cpp').read_text()
    assert 'execute_dual_pair' in body
    wave64_guard = body.split('if (wf.wf_size() != 32) {', 1)[1].split('}', 1)[0]
    assert 'return;' in wave64_guard
    assert ('InvalidInstructionEncoding' in wave64_guard) == (arch != 'rdna3')
    assert 'UnimplementedInst' not in wave64_guard
    dx9 = body.split('case kVopdMulDx9ZeroF32:', 1)[1].split('case ', 1)[0]
    assert 'arithmetic_flags()' in dx9
    assert len(profile.vopd_slot_ops) == (26 if arch == 'cdna5' else 17)
    for op in profile.vopd_slot_ops:
        assert f'case k{op.enum_name}:' in body
        assert f'return rj_arith_{op.mnemonic}' in body
    for legacy in (
        'execute_slot',
        'uses_src_neg_modifier',
        'apply_neg',
        'Vopd::bitop2',
    ):
        assert legacy not in header
        assert legacy not in body
    assert 'try_execute_vopd' not in body
    assert 'fp_mode::arithmetic' not in body
    assert 'gfx12_dot' not in body and 'gfx11_dot' not in body
    assert 'slot.src2_imm' in body
    assert ('slot.neg' in body) == (arch == 'cdna5')
    assert ('dot2_src_needs_half_replication' in body) == (arch != 'cdna5')
    assert 'std::copy_n(acc.data, 32, result)' in body
    assert gen_vopd_execution('cdna4', profile.vopd_slot_ops, '') is None
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
    )
    checked_in = (root / arch / 'vopd_exec.cpp').read_text()

    def execution(text):
        return re.sub(
            r'\s+',
            '',
            text.split('void Vopd::execute_impl', 1)[1]
            .split('// namespace', 1)[0]
            .rstrip('} \n'),
        )

    assert execution(body) == execution(checked_in)
    monkeypatch.setattr(
        type(profile), 'split_execution_sources', property(lambda self: False)
    )
    codegen.gen_vopd()
    monolithic = (tmp_path / arch / 'vopd.cpp').read_text()
    assert (
        '#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vopd_exec.h"' in monolithic
    )
    assert '#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"' in monolithic
    assert 'execute_dual_pair' in monolithic


def test_dual_checked_in_execution_matches_generator():
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
    )
    for arch in ('rdna3', 'rdna4', 'cdna5'):
        text = (root / arch / 'vopd_exec.cpp').read_text()
        assert text.count('return rj_arith_v_dual_') == (26 if arch == 'cdna5' else 17)
        assert 'execute_dual_pair' in text
        assert 'try_execute_vopd' not in text
    assert 'execute_dual_pair<2>' in (root / 'cdna5/vopd_exec.cpp').read_text()
