# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from amdisa.codegen.execute import ExecuteContext


def context(arch, name, dst, src):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        profile=None,
        arch_name=arch,
        dst_ops=dst,
        src_ops=src,
        enc_name="",
        is_vop3=False,
        has_abs=False,
    )


import re


import pytest


from amdisa.codegen.execute.rj_arith_scalar import EXEC_UPDATES, gen_scalar


@pytest.mark.parametrize('name', sorted(EXEC_UPDATES))
def test_exec_boolean_updates_snapshot_then_commit_in_order(name):
    wide = name.endswith('B64')
    suffix = '64' if wide else ''
    for arch in ('rdna3', 'rdna4'):
        body = gen_scalar(context(arch, name, ['d'], ['a']))
        assert f'rj_arith_{name.lower()}(' in body
        assert ('wf.exec_raw()' in body) == wide
        assert ('uint32_t(wf.exec())' in body) == (not wide)
        assert 'arithmetic_reports_fp_status' not in body
        assert '&new_exec, &scc' in body
        ordered = [
            f'read_scalar{suffix}(a)',
            'exec_mask =',
            'status != RJ_ARITH_SUCCESS',
            f'write_scalar{suffix}(d, result)',
            'wf.write_exec(new_exec)' if wide else 'wf.set_exec(new_exec)',
            'wf.write_scc(scc)',
        ]
        offsets = [body.index(x) for x in ordered]
        assert offsets == sorted(offsets)
        assert 'if (!exec' not in body
    assert gen_scalar(context('cdna5', name, ['d'], ['a'])) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_every_exec_boolean_method_uses_rj_arith(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
    )
    body = (root / arch / 'sop1_exec.cpp').read_text()
    assert len(EXEC_UPDATES) == 24
    for name in EXEC_UPDATES:
        assert len(re.findall(r'\b' + 'rj_arith_' + name.lower() + r'\(', body)) == 1
        assert 'amdgpu::execute_' + name.lower() not in body
