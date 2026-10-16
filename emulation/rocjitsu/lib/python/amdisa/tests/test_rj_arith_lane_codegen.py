# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import pytest

from amdisa.codegen.execute.rj_arith_lane import LANE_NAMES, RELATIVE_NAMES, gen_lane


@pytest.mark.parametrize('name', sorted(LANE_NAMES - RELATIVE_NAMES))
def test_lane_family_uses_rj_arith_without_optional_state(name):
    count = (
        3
        if name in ('V_PERMLANE16_B32', 'V_PERMLANEX16_B32')
        else (
            2 if '_VAR_' in name or name in ('V_READLANE_B32', 'V_WRITELANE_B32') else 1
        )
    )
    sources = ['a', 'b', 'c'][:count]
    for arch in ('rdna3', 'rdna4'):
        body = gen_lane(arch, name, ['d'], sources)
        assert 'amdgpu::arith::execute_' in body
        assert 'arithmetic_flags()' in body
        assert 'arithmetic_reports_fp_status' not in body
    assert gen_lane('cdna5', name, ['d'], sources) is None
    assert gen_lane('rdna4', name, [], sources) is None
