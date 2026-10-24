# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed BF16-to-FP4 conversion keeps the lane and invalid-scale guards."""

import os
from pathlib import Path
from types import SimpleNamespace

from amdisa.__main__ import _run
from amdisa.codegen.execute.vector_special import gen_cvt_scalef32


def _mrisa_dir() -> Path:
    default = (
        Path(__file__).resolve().parents[6] / 'shared' / 'machine-readable-isa' / 'isa'
    )
    return Path(os.environ.get('MRISA_PATH', default))


def test_packed_bf16_fp4_uses_direct_conversion_after_invalid_scale_guard():
    cpp = gen_cvt_scalef32(
        SimpleNamespace(
            op='pk_fp4_bf16',
            dst_ops=['vdst'],
            src_ops=['src0', 'src1'],
            arch_name='cdna4',
        )
    )
    assert cpp.count('util::bf16_to_fp4_e2m1_scaled_rne(') == 2
    assert 'std::ldexp' not in cpp
    assert cpp.index('if (biased_exp == 0xFFu)') < cpp.index('read_lane(src0, lane)')
    assert 'if (!(exec & (1ULL << lane))) continue;' in cpp
    assert 'uint32_t dst_byte = (inst_.op_sel >> 2) & 0x3;' in cpp
    assert '(old & mask) | (packed << (dst_byte * 8))' in cpp


def test_packed_f16_fp4_retains_floating_conversion():
    cpp = gen_cvt_scalef32(
        SimpleNamespace(
            op='pk_fp4_f16',
            dst_ops=['vdst'],
            src_ops=['src0', 'src1'],
            arch_name='cdna4',
        )
    )
    assert 'util::f16_to_f32(' in cpp
    assert 'std::ldexp' in cpp
    assert 'bf16_to_fp4_e2m1_scaled_rne' not in cpp


def test_packed_bf16_fp4_simd_probe_and_include_are_emitted_together(tmp_path):
    xml = _mrisa_dir()
    _run(
        SimpleNamespace(
            isafiles=[f'cdna4:{xml / "amdgpu_isa_cdna4.xml"}'],
            gen_isas=True,
            gen_dbt=False,
            isa_output=str(tmp_path),
            dbt_output=None,
        )
    )
    cpp = (tmp_path / 'cdna4/vop3_exec.cpp').read_text()
    assert '#include "rocjitsu/isa/arch/amdgpu/shared/mxfp4_simd.h"' in cpp
    body = cpp.split('void VCvtScalef32PkFp4Bf16Vop3::execute_impl', 1)[1]
    assert body.index(
        'try_execute_cvt_scalef32_pk_fp4_bf16_simd(inst, wf)'
    ) < body.index('for (uint32_t lane')
    f16_body = cpp.split('void VCvtScalef32PkFp4F16Vop3::execute_impl', 1)[1].split(
        '\nvoid ', 1
    )[0]
    assert 'try_execute_cvt_scalef32_pk_fp4_bf16_simd' not in f16_body
    assert 'bf16_to_fp4_e2m1_scaled_rne' not in f16_body
    assert 'std::ldexp' in f16_body
    unrelated = (tmp_path / 'cdna4/vop1_exec.cpp').read_text()
    assert 'mxfp4_simd.h' not in unrelated
