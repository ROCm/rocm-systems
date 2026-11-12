# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Input formats must explicitly select their denormal MODE field."""

import pytest

from amdisa.codegen.execute import input_policy


@pytest.mark.parametrize(
    'dtype,mode', [('f16', 'f16_f64'), ('f32', 'f32'), ('f64', 'f16_f64')]
)
def test_supported_input_policy_selects_mode(dtype, mode):
    expected = f'amdgpu::input_denormal::Policy::make(wf.fp_denorm_mode_{mode}())'
    assert input_policy.policy_expr(dtype) == expected
    assert input_policy.policy_decl(dtype) == f'  const auto input_policy = {expected};'


@pytest.mark.parametrize('dtype', ['bf16', 'f23', ''])
@pytest.mark.parametrize('emit', [input_policy.policy_expr, input_policy.policy_decl])
def test_unsupported_input_policy_is_rejected(dtype, emit):
    with pytest.raises(ValueError, match='Unsupported input-policy dtype'):
        emit(dtype)


def test_adding_a_layout_does_not_implicitly_select_mode(monkeypatch):
    monkeypatch.setitem(input_policy.FORMATS, 'bf16', 'BF16')
    with pytest.raises(ValueError, match="Unsupported input-policy dtype: 'bf16'"):
        input_policy.policy_expr('bf16')
