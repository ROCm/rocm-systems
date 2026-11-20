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
    """Register remaining VM/matrix/paired-word handlers after rj-arith routing."""
    from amdisa.codegen.execute.vector_special import (
        gen_vector_movrel,
        gen_vector_swaprel,
    )
    from amdisa.codegen.execute.packed import (
        gen_pk_binop_u64,
        gen_pk_binop_f64,
        gen_pk_ternary_f64,
        gen_pk_lshl_add_u64,
    )
    from amdisa.codegen.execute.matrix import gen_mfma

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

    DISPATCH['pk_binop_u64'] = lambda c: gen_pk_binop_u64(c.dst_ops, c.src_ops, c.op)

    DISPATCH['pk_binop_f64'] = lambda c: gen_pk_binop_f64(c.dst_ops, c.src_ops, c.op)

    DISPATCH['pk_ternary_f64'] = lambda c: gen_pk_ternary_f64(
        c.dst_ops, c.src_ops, c.op
    )

    DISPATCH['pk_lshl_add_u64'] = lambda c: gen_pk_lshl_add_u64(c.dst_ops, c.src_ops)

    DISPATCH['mfma'] = gen_mfma


_register_handlers()
