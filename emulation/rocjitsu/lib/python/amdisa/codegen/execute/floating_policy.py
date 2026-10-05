# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Instruction policies shared by scalar, SIMD and SDWA output lowering."""

# These F32 results cannot preserve output denormals, independently of MODE.
# Their SDWA forms use the same output modifiers as VOP3. SIN can produce
# subnormals and retains its separate MODE policy.
FLUSH_NEAREST_F32_OPS = frozenset(
    {'V_LOG_F32', 'V_EXP_F32', 'V_RCP_F32', 'V_RSQ_F32', 'V_SQRT_F32', 'V_COS_F32'}
)

# Evaluate and round these operations to half before output modifiers.
ROUNDED_F16_OPS = frozenset(
    {
        'V_LOG_F16',
        'V_EXP_F16',
        'V_RCP_F16',
        'V_RSQ_F16',
        'V_SQRT_F16',
        'V_SIN_F16',
        'V_COS_F16',
    }
)
