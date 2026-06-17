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
    from amdisa.codegen.execute.scalar import (
        gen_scalar_cmpk,
    )
    from amdisa.codegen.execute.vector_cmp import (
        gen_vector_cmp_class,
        gen_vector_cmpx,
    )
    from amdisa.codegen.execute.vector_special import (
        gen_vector_mbcnt,
        gen_vector_movrel,
        gen_vector_mad_64_32,
        gen_vector_mad_32_16,
        gen_vector_div_fixup,
        gen_vector_div_scale,
        gen_vector_div_fmas,
        gen_vector_dot,
        gen_vector_dot2c_bf16,
        gen_vector_bitop3,
        gen_vector_permlane_swap,
        gen_vector_permlane,
        gen_vector_permlane64,
        gen_vector_cvt_pk,
        gen_vector_cvt_scale,
        gen_cvt_fp8,
        gen_cvt_scalef32,
    )
    from amdisa.codegen.execute.packed import (
        gen_pk_binop,
        gen_pk_ternary,
        gen_pk_binop_f32,
        gen_pk_ternary_f32,
        gen_pk_mov_b32,
        gen_mad_mix_f32,
        gen_mad_mix_lo_hi,
        gen_mad_mix_bf16,
        gen_dot2,
        gen_dot4,
        gen_dot8,
    )
    from amdisa.codegen.execute.matrix import (
        gen_accvgpr_read,
        gen_accvgpr_write,
        gen_mfma,
    )

    # Scalar ALU — scalar_unary, scalar_binop, scalar_cmp, scalar_bitcmp,
    # scalar_saveexec now handled by SemaAST pipeline (_SEMA_CLASSES).
    DISPATCH['scalar_cmpk'] = lambda c: gen_scalar_cmpk(
        c.dst_ops, c.src_ops, c.op, c.dtype
    )

    # Vector ALU — vector_unary, vector_binop, vector_ternary now handled
    # by SemaAST pipeline (_SEMA_CLASSES).

    # Vector compare — vector_cmp, vector_add_co handled by SemaAST pipeline
    # (_SEMA_CLASSES). vector_cmp_class is NOT: the SemaAST lowering mangles the
    # operand (a bit_cast<float>(static_cast<uint32_t>(...)) value round-trip)
    # and always classifies in f32, so the dedicated per-type generator here
    # (correct f16/f32/f64 qnan masks, full 64-bit f64 read) owns it instead.
    DISPATCH['vector_cmpx'] = lambda c: gen_vector_cmpx(
        c.src_ops, c.op, c.dtype, c.cmpx_writes_vcc, c.is_vop3, c.dst_ops, c.has_abs
    )
    DISPATCH['vector_cmp_class'] = lambda c: gen_vector_cmp_class(
        c.dst_ops, c.src_ops, c.dtype, False, False, c.is_vop3, c.has_abs
    )
    DISPATCH['vector_cmpx_class'] = lambda c: gen_vector_cmp_class(
        c.dst_ops, c.src_ops, c.dtype, True, c.cmpx_writes_vcc, c.is_vop3, c.has_abs
    )

    # Vector special
    DISPATCH['vector_mbcnt'] = lambda c: gen_vector_mbcnt(c.dst_ops, c.src_ops, c.op)
    DISPATCH['vector_movrel'] = lambda c: gen_vector_movrel(
        c.dst_ops, c.src_ops, c.op, c.profile.uses_vgpr_msb_indexing
    )
    DISPATCH['vector_mad_64_32'] = lambda c: gen_vector_mad_64_32(
        c.dst_ops, c.src_ops, c.dtype
    )
    DISPATCH['vector_mad_32_16'] = lambda c: gen_vector_mad_32_16(
        c.dst_ops, c.src_ops, c.dtype
    )
    DISPATCH['vector_div_fixup'] = lambda c: gen_vector_div_fixup(
        c.dst_ops, c.src_ops, c.dtype, c.is_vop3, c.has_abs
    )
    DISPATCH['vector_div_scale'] = lambda c: gen_vector_div_scale(
        c.dst_ops, c.src_ops, c.dtype, c.is_vop3, c.has_abs
    )
    DISPATCH['vector_div_fmas'] = lambda c: gen_vector_div_fmas(
        c.dst_ops, c.src_ops, c.dtype, c.is_vop3, c.has_abs
    )
    DISPATCH['vector_dot'] = lambda c: gen_vector_dot(
        c.dst_ops, c.src_ops, c.op, c.dtype
    )
    DISPATCH['vector_dot2c_bf16'] = lambda c: gen_vector_dot2c_bf16(
        c.dst_ops, c.src_ops
    )
    DISPATCH['vector_bitop3'] = lambda c: gen_vector_bitop3(
        c.dst_ops, c.src_ops, c.dtype
    )
    DISPATCH['vector_permlane16'] = lambda c: gen_vector_permlane(
        c.dst_ops, c.src_ops, c.op, cross=False, op_sel_expr=c.opsel_exprs[0]
    )
    DISPATCH['vector_permlanex16'] = lambda c: gen_vector_permlane(
        c.dst_ops, c.src_ops, c.op, cross=True, op_sel_expr=c.opsel_exprs[0]
    )
    DISPATCH['vector_permlane16_swap'] = lambda c: gen_vector_permlane_swap(
        c.dst_ops, c.src_ops, stride=16
    )
    DISPATCH['vector_permlane32_swap'] = lambda c: gen_vector_permlane_swap(
        c.dst_ops, c.src_ops, stride=32
    )
    DISPATCH['vector_permlane64'] = lambda c: gen_vector_permlane64(
        c.dst_ops, c.src_ops
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
    )
    DISPATCH['vector_cvt_scale'] = lambda c: gen_vector_cvt_scale(
        c.dst_ops, c.src_ops, c.cls, c.op
    )
    DISPATCH['cvt_fp8'] = lambda c: gen_cvt_fp8(c)
    DISPATCH['cvt_scalef32'] = lambda c: gen_cvt_scalef32(c)

    # Packed
    DISPATCH['pk_binop'] = lambda c: gen_pk_binop(
        c.dst_ops, c.src_ops, c.op, c.dtype, opsel_exprs=c.opsel_exprs
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
        use_gfx1250_helpers=c.arch_name == 'gfx1250',
    )
    DISPATCH['pk_ternary_f32'] = lambda c: gen_pk_ternary_f32(
        c.dst_ops,
        c.src_ops,
        c.op,
        op_sel_hi_2_expr=c.op_sel_hi_2_expr,
        opsel_exprs=c.opsel_exprs,
        use_gfx1250_helpers=c.arch_name == 'gfx1250',
    )
    DISPATCH['pk_mov_b32'] = lambda c: gen_pk_mov_b32(
        c.dst_ops,
        c.src_ops,
        opsel_exprs=c.opsel_exprs,
        use_gfx1250_helpers=c.arch_name == 'gfx1250',
    )
    DISPATCH['mad_mix_f32'] = lambda c: gen_mad_mix_f32(
        c.dst_ops,
        c.src_ops,
        op_sel_hi_2_expr=c.op_sel_hi_2_expr,
        opsel_exprs=c.opsel_exprs,
        use_gfx1250_helpers=c.arch_name == 'gfx1250',
    )
    DISPATCH['mad_mixlo_f16'] = lambda c: gen_mad_mix_lo_hi(
        c.dst_ops,
        c.src_ops,
        is_lo=True,
        op_sel_hi_2_expr=c.op_sel_hi_2_expr,
        opsel_exprs=c.opsel_exprs,
        use_gfx1250_helpers=c.arch_name == 'gfx1250',
    )
    DISPATCH['mad_mixhi_f16'] = lambda c: gen_mad_mix_lo_hi(
        c.dst_ops,
        c.src_ops,
        is_lo=False,
        op_sel_hi_2_expr=c.op_sel_hi_2_expr,
        opsel_exprs=c.opsel_exprs,
        use_gfx1250_helpers=c.arch_name == 'gfx1250',
    )
    DISPATCH['mad_mix_f32_bf16'] = lambda c: gen_mad_mix_bf16(
        c.dst_ops,
        c.src_ops,
        result='f32',
        op_sel_hi_2_expr=c.op_sel_hi_2_expr,
        opsel_exprs=c.opsel_exprs,
        use_gfx1250_helpers=c.arch_name == 'gfx1250',
    )
    DISPATCH['mad_mixlo_bf16'] = lambda c: gen_mad_mix_bf16(
        c.dst_ops,
        c.src_ops,
        result='lo',
        op_sel_hi_2_expr=c.op_sel_hi_2_expr,
        opsel_exprs=c.opsel_exprs,
        use_gfx1250_helpers=c.arch_name == 'gfx1250',
    )
    DISPATCH['mad_mixhi_bf16'] = lambda c: gen_mad_mix_bf16(
        c.dst_ops,
        c.src_ops,
        result='hi',
        op_sel_hi_2_expr=c.op_sel_hi_2_expr,
        opsel_exprs=c.opsel_exprs,
        use_gfx1250_helpers=c.arch_name == 'gfx1250',
    )
    DISPATCH['dot2'] = lambda c: gen_dot2(
        c.dst_ops, c.src_ops, c.cls, opsel_exprs=c.opsel_exprs
    )
    DISPATCH['dot4'] = lambda c: gen_dot4(c.dst_ops, c.src_ops, c.cls)
    DISPATCH['dot8'] = lambda c: gen_dot8(c.dst_ops, c.src_ops, c.cls)

    # Matrix
    DISPATCH['accvgpr_read'] = lambda c: gen_accvgpr_read(c.dst_ops, c.src_ops)
    DISPATCH['accvgpr_write'] = lambda c: gen_accvgpr_write(c.dst_ops, c.src_ops)
    DISPATCH['mfma'] = lambda c: gen_mfma(
        c.inst, c.dst_ops, c.src_ops, arch_name=c.arch_name
    )


_register_handlers()
