# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execute body generation subpackage.

Provides ``ExecuteContext`` (the unified context for execute body
generation) and ``DISPATCH`` (registry mapping semantic class names
to handler functions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from amdisa.gpuisa import Instruction
    from amdisa.isa_profile import IsaProfile
    from amdisa.semantics import InstructionSemantics


@dataclass
class ExecuteContext:
    """Read-only context passed to execute body generators.

    Bundles all per-instruction state needed by the extracted generator
    functions so they share a uniform ``handler(ctx) -> str`` signature.
    """

    inst: Instruction
    sem: InstructionSemantics
    dst_ops: list[str]
    src_ops: list[str]
    profile: IsaProfile
    enc_name: str
    is_vop3: bool
    has_abs: bool
    opsel_exprs: tuple[str, str] = ('', '')
    op_sel_hi_2_expr: str = ''
    arch_name: str = ''
    enc_field_names: set[str] = field(default_factory=set)
    encoding_map: dict | None = None
    supports_dpp: bool = False
    supports_dpp8: bool = False
    supports_sdwa: bool = False
    result_writer: str | None = None
    uses_vop3_half_selectors: bool = False

    @property
    def cls(self) -> str:
        return self.sem.semantic_class

    @property
    def op(self) -> str | None:
        return self.sem.operation

    @property
    def dtype(self) -> str | None:
        return self.sem.data_type

    @property
    def scc(self) -> str | None:
        return self.sem.sets_scc

    @property
    def cond(self) -> str | None:
        return self.sem.branch_condition

    @property
    def cmpx_writes_vcc(self) -> bool:
        return self.profile.cmpx_writes_vcc


# Registry: semantic_class -> handler(ctx) -> str
# Populated by _register_handlers() below.
DISPATCH: dict[str, Callable[[ExecuteContext], str]] = {}


def _register_handlers() -> None:
    """Populate DISPATCH with handlers from the extracted modules."""
    from amdisa.codegen.execute.vector_cmp import (
        gen_vector_cmpx,
    )
    from amdisa.codegen.execute.vector_special import (
        gen_vector_movrel,
        gen_vector_swaprel,
        gen_vector_sat_pack,
        gen_vector_perm_pk16,
        gen_vector_dot2c_bf16,
        gen_vector_bitop3,
        gen_vector_cvt_pk,
        gen_cvt_fp8,
    )
    from amdisa.codegen.execute.packed import (
        gen_pk_binop,
        gen_pk_ternary,
        gen_pk_binop_f32,
        gen_pk_ternary_f32,
        gen_pk_ternary_f64,
        gen_pk_binop_u64,
        gen_pk_binop_f64,
        gen_pk_lshl_add_u64,
    )
    from amdisa.codegen.execute.matrix import (
        gen_accvgpr_read,
        gen_accvgpr_write,
        gen_mfma,
    )

    # Vector ALU — vector_unary, pseudo_scalar_unary, vector_binop, and
    # vector_ternary are handled by the SemaAST pipeline (_SEMA_CLASSES).
    # pseudo_scalar_unary retains its VALU expression/encoding contract but
    # uses ExecModel.SCALAR because the ISA says V_S_* ignores EXEC.

    # Relational CMPX keeps the legacy fallback; CLASS is handled by rj-arith.
    DISPATCH['vector_cmpx'] = lambda c: gen_vector_cmpx(
        c.src_ops,
        c.op,
        c.dtype,
        c.cmpx_writes_vcc,
        c.is_vop3,
        c.dst_ops,
        c.has_abs,
        c.result_writer,
    )
    # Vector special
    DISPATCH['vector_movrel'] = lambda c: gen_vector_movrel(
        c.dst_ops,
        c.src_ops,
        c.op,
        c.profile.uses_vgpr_msb_indexing,
        c.supports_dpp,
        c.supports_dpp8,
        c.supports_sdwa,
        c.profile.dpp_bound_ctrl_applies_to_inactive_sources,
        # Older profiles preserve the legacy bit-copy treatment of modifiers.
        is_vop3=c.is_vop3
        and c.arch_name in ('rdna3', 'rdna4', 'cdna3', 'cdna4', 'cdna5'),
        stage_after_destination=c.arch_name == 'rdna3_5',
        # Preserve legacy CDNA bit-move modifiers; RDNA treats signs as FP32.
        apply_dpp_source_modifiers=c.arch_name in ('rdna3', 'rdna4'),
    )
    DISPATCH['vector_swaprel'] = lambda c: gen_vector_swaprel(
        c.dst_ops,
        c.src_ops,
        c.profile.uses_vgpr_msb_indexing,
    )
    DISPATCH['vector_sat_pack'] = lambda c: gen_vector_sat_pack(
        c.dst_ops,
        c.src_ops,
        c.op,
        true16_dst=c.is_vop3 and c.profile.uses_true16_vop3_opsel,
    )
    DISPATCH['vector_perm_pk16'] = lambda c: gen_vector_perm_pk16(
        c.dst_ops, c.src_ops, c.op, c.profile.uses_vgpr_msb_indexing
    )
    from .rj_arith_vector import gen_integer_dot
    from .rj_arith_float import gen_float_dot32

    DISPATCH['vector_dot'] = lambda c: (
        gen_integer_dot(c) if c.dtype == 'i32' else gen_float_dot32(c)
    )
    DISPATCH['vector_dot2c_bf16'] = lambda c: gen_vector_dot2c_bf16(
        c.dst_ops, c.src_ops
    )
    DISPATCH['vector_bitop3'] = lambda c: gen_vector_bitop3(
        c.dst_ops,
        c.src_ops,
        c.dtype,
        (
            c.opsel_exprs[0]
            if c.arch_name == 'cdna5' and c.is_vop3 and c.dtype == 'b16'
            else None
        ),
    )
    DISPATCH['vector_cvt_pk'] = lambda c: gen_vector_cvt_pk(
        c.dst_ops,
        c.src_ops,
        c.cls,
        c.op,
        opsel=(
            'inst_.opsel'
            if c.is_vop3 and 'opsel' in c.enc_field_names
            else 'inst_.op_sel' if c.is_vop3 and 'op_sel' in c.enc_field_names else '0u'
        ),
        dtype=c.dtype,
        is_vop3=c.is_vop3,
        has_abs=c.has_abs,
        fp8_format_select=(
            'inst_.clamp'
            if c.cls == 'vector_cvt_pk'
            and c.op in ('fp8_f32', 'fp8_f16')
            and c.is_vop3
            and c.arch_name == 'cdna5'
            else None
        ),
        arch_name=c.arch_name,
    )
    DISPATCH['cvt_fp8'] = lambda c: gen_cvt_fp8(c)

    # Packed
    DISPATCH['pk_binop'] = lambda c: gen_pk_binop(
        c.dst_ops,
        c.src_ops,
        c.op,
        c.dtype,
        opsel_exprs=c.opsel_exprs,
    )
    DISPATCH['pk_ternary'] = lambda c: gen_pk_ternary(
        c.dst_ops,
        c.src_ops,
        c.op,
        c.dtype,
        op_sel_hi_2_expr=c.op_sel_hi_2_expr,
        opsel_exprs=c.opsel_exprs,
    )
    DISPATCH['pk_binop_f32'] = lambda c: gen_pk_binop_f32(
        c.dst_ops,
        c.src_ops,
        c.op,
        opsel_exprs=c.opsel_exprs,
        use_cdna5_helpers=c.arch_name == 'cdna5',
    )
    DISPATCH['pk_ternary_f32'] = lambda c: gen_pk_ternary_f32(
        c.dst_ops,
        c.src_ops,
        c.op,
        op_sel_hi_2_expr=c.op_sel_hi_2_expr,
        opsel_exprs=c.opsel_exprs,
        use_cdna5_helpers=c.arch_name == 'cdna5',
    )
    DISPATCH['pk_binop_u64'] = lambda c: gen_pk_binop_u64(c.dst_ops, c.src_ops, c.op)
    DISPATCH['pk_binop_f64'] = lambda c: gen_pk_binop_f64(c.dst_ops, c.src_ops, c.op)
    DISPATCH['pk_ternary_f64'] = lambda c: gen_pk_ternary_f64(
        c.dst_ops, c.src_ops, c.op
    )
    DISPATCH['pk_lshl_add_u64'] = lambda c: gen_pk_lshl_add_u64(c.dst_ops, c.src_ops)
    # DOT: every semantic class routes here, so the generator keeps no
    # dot2_/dot4_/dot8_ fallback. test_dot_dispatch_covers_every_semantic_class
    # fails if a new class arrives without an entry.
    for _cls in ('dot2_f32_f16', 'dot2_f32_bf16'):
        DISPATCH[_cls] = gen_float_dot32
    from amdisa.codegen.execute.rj_arith_float import gen_half_dot

    for _cls in ('dot2_f16_f16', 'dot2_bf16_bf16'):
        DISPATCH[_cls] = gen_half_dot
    for _cls in (
        'dot2_i32_i16',
        'dot2_u32_u16',
        'dot4_i32_i8',
        'dot4_i32_iu8',
        'dot4_u32_u8',
        'dot8_i32_i4',
        'dot8_i32_iu4',
        'dot8_u32_u4',
    ):
        DISPATCH[_cls] = gen_integer_dot
    DISPATCH['dot4_f32_fp8'] = gen_float_dot32

    # Matrix
    DISPATCH['accvgpr_read'] = lambda c: gen_accvgpr_read(c.dst_ops, c.src_ops)
    DISPATCH['accvgpr_write'] = lambda c: gen_accvgpr_write(c.dst_ops, c.src_ops)
    DISPATCH['mfma'] = gen_mfma


_register_handlers()
