# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Instruction policies shared by scalar, SIMD and SDWA output lowering."""

# F32 LOG/EXP flush denormals and ignore guest rounding. Their SDWA forms
# use the same output modifiers as VOP3. F16 and legacy variants are distinct.
FLUSH_NEAREST_F32_OPS = frozenset({'V_LOG_F32', 'V_EXP_F32'})
