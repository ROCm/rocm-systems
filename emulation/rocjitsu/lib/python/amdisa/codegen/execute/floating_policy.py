# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Instruction policies shared by scalar, SIMD and SDWA output lowering."""

from dataclasses import dataclass

# These F32 results cannot preserve output denormals, independently of MODE.
# Their SDWA forms use the same output modifiers as VOP3. SIN can produce
# subnormals and retains its separate MODE policy.
FLUSH_NEAREST_F32_OPS = frozenset(
    {'V_LOG_F32', 'V_EXP_F32', 'V_RCP_F32', 'V_RSQ_F32', 'V_SQRT_F32', 'V_COS_F32'}
)


@dataclass(frozen=True)
class F16Operation:
    """How a VALU F16 operation treats its source and its result.

    ``calls`` are the helper spellings the semantics lowering uses for it.
    The two properties are decided separately, even where they coincide.
    """

    instruction: str
    calls: tuple[str, ...]
    # The source half is flushed under MODE at the register read, before it
    # is widened; the helper itself does not flush.
    flushed_source: bool
    # Evaluated by the transcendental unit: the result is rounded to half
    # before OMOD/CLAMP, with the TRANS-unit output policy.
    transcendental: bool


F16_OPERATIONS = (
    F16Operation(
        'V_LOG_F16', ('log', 'log2'), flushed_source=True, transcendental=True
    ),
    F16Operation(
        'V_EXP_F16', ('exp', 'exp2'), flushed_source=True, transcendental=True
    ),
    F16Operation('V_RCP_F16', ('rcp',), flushed_source=True, transcendental=True),
    F16Operation('V_RSQ_F16', ('rsq',), flushed_source=True, transcendental=True),
    F16Operation('V_SQRT_F16', ('sqrt',), flushed_source=True, transcendental=True),
    F16Operation('V_SIN_F16', ('sin',), flushed_source=True, transcendental=True),
    F16Operation('V_COS_F16', ('cos',), flushed_source=True, transcendental=True),
)

# Instructions rounded to half before output modifiers, with the TRANS policy.
ROUNDED_F16_OPS = frozenset(
    op.instruction for op in F16_OPERATIONS if op.transcendental
)

# Helper calls evaluated by the transcendental unit.
F16_TRANSCENDENTAL_CALLS = frozenset(
    call for op in F16_OPERATIONS if op.transcendental for call in op.calls
)

# Instructions whose F16 source is flushed before widening.
F16_FLUSHED_SOURCE_OPS = frozenset(
    op.instruction for op in F16_OPERATIONS if op.flushed_source
)

# Helper calls whose F16 source the caller flushes before widening.
F16_FLUSHED_SOURCE_CALLS = frozenset(
    call for op in F16_OPERATIONS if op.flushed_source for call in op.calls
)

# Integral rounding whose source honors MODE input flushing, in every format.
# TRUNC and RNDNE give a signed zero for a subnormal either way; FRACT's input
# flush is not implemented.
INPUT_FLUSHED_ROUNDING = frozenset({'ceil', 'floor'})
