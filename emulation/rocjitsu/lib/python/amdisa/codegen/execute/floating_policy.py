# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared instruction classifications for FP lowering.

F16_TRANS_OPERATIONS covers TRANS only. CEIL/FLOOR source flushing uses
INPUT_FLUSHED_ROUNDING; sema_lower.py handles arithmetic and conversions separately.
"""

from dataclasses import dataclass

# These F32 results cannot preserve output denormals, independently of MODE.
# Their SDWA forms use the same output modifiers as VOP3. SIN can produce
# subnormals and retains its separate MODE policy.
FLUSH_NEAREST_F32_OPS = frozenset(
    {'V_LOG_F32', 'V_EXP_F32', 'V_RCP_F32', 'V_RSQ_F32', 'V_SQRT_F32', 'V_COS_F32'}
)


@dataclass(frozen=True)
class F16TransOperation:
    """Source policy and helper spellings for an F16 TRANS operation.

    ``calls`` are the helper spellings the semantics lowering uses for it.
    """

    instruction: str
    calls: tuple[str, ...]
    # The source half is flushed under MODE at the register read, before it
    # is widened; the helper itself does not flush.
    flushed_source: bool


# TRANS entries only; other F16 source policies are selected outside this table.
F16_TRANS_OPERATIONS = (
    F16TransOperation('V_LOG_F16', ('log', 'log2'), flushed_source=True),
    F16TransOperation('V_EXP_F16', ('exp', 'exp2'), flushed_source=True),
    F16TransOperation('V_RCP_F16', ('rcp',), flushed_source=True),
    F16TransOperation('V_RSQ_F16', ('rsq',), flushed_source=True),
    F16TransOperation('V_SQRT_F16', ('sqrt',), flushed_source=True),
    F16TransOperation('V_SIN_F16', ('sin',), flushed_source=True),
    F16TransOperation('V_COS_F16', ('cos',), flushed_source=True),
)

# TRANS instructions selecting the TRANS output policy.
F16_TRANSCENDENTAL_OPS = frozenset(op.instruction for op in F16_TRANS_OPERATIONS)

# Helper calls evaluated by the transcendental unit.
F16_TRANSCENDENTAL_CALLS = frozenset(
    call for op in F16_TRANS_OPERATIONS for call in op.calls
)

# TRANS instructions whose F16 source is flushed before widening.
F16_FLUSHED_SOURCE_OPS = frozenset(
    op.instruction for op in F16_TRANS_OPERATIONS if op.flushed_source
)

# TRANS helper calls whose F16 source the caller flushes before widening.
F16_FLUSHED_SOURCE_CALLS = frozenset(
    call for op in F16_TRANS_OPERATIONS if op.flushed_source for call in op.calls
)

# Integral rounding whose source honors MODE input flushing, in every format.
# TRUNC and RNDNE give a signed zero for a subnormal either way; FRACT's input
# flush is not implemented.
INPUT_FLUSHED_ROUNDING = frozenset({'ceil', 'floor'})
