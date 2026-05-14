# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lower SemaAST to C++ execute_impl bodies.

Walks a :class:`~amdisa.sema_ast.SemaBlock`'s expression tree and emits
C++ code implementing the instruction's behavior in the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
)
from amdisa.sema_helpers import (
    HELPER_REGISTRY,
    HelperTreatment,
)


class RegClass(Enum):
    SGPR = auto()
    VGPR = auto()
    ACC_VGPR = auto()


@dataclass
class LoweringContext:
    """Immutable context for one lowering invocation."""

    exec_model: ExecModel
    operand_reg_classes: dict[tuple[str, int], RegClass] = field(
        default_factory=dict,
    )
    indent: int = 1
    declared: set[str] = field(default_factory=set)
    is_lhs: bool = False


_INFIX_OPS: dict[SemaNodeKind, str] = {
    SemaNodeKind.ADD: '+', SemaNodeKind.SUB: '-',
    SemaNodeKind.MUL: '*', SemaNodeKind.DIV: '/',
    SemaNodeKind.MOD: '%',
    SemaNodeKind.AND: '&', SemaNodeKind.OR: '|',
    SemaNodeKind.XOR: '^', SemaNodeKind.SHL: '<<',
    SemaNodeKind.SHR: '>>',
    SemaNodeKind.LAND: '&&', SemaNodeKind.LOR: '||',
    SemaNodeKind.EQ: '==', SemaNodeKind.NE: '!=',
    SemaNodeKind.LT: '<', SemaNodeKind.GT: '>',
    SemaNodeKind.LE: '<=', SemaNodeKind.GE: '>=',
}

_CONTEXT_READS: dict[str, str] = {
    'SCC': 'wf.scc()',
    'VCC': 'wf.vcc()',
    'EXEC': 'wf.exec()',
    'EXEC_LO': 'static_cast<uint32_t>(wf.exec())',
    'M0': 'wf.m0()',
    'laneId': 'lane',
}

_CONTEXT_WRITES: dict[str, str] = {
    'SCC': 'wf.write_scc',
    'VCC': 'wf.write_vcc',
    'EXEC': 'wf.write_exec',
}

_STD_MATH: dict[SemaNodeKind, str] = {
    SemaNodeKind.SQRT: 'std::sqrt',
    SemaNodeKind.SIN: 'std::sin',
    SemaNodeKind.COS: 'std::cos',
    SemaNodeKind.LOG2: 'std::log2',
    SemaNodeKind.FLOOR: 'std::floor',
    SemaNodeKind.TRUNC: 'std::trunc',
}


def lower_sema_block(block: SemaBlock, ctx: LoweringContext | None = None) -> str:
    """Lower a SemaBlock to a C++ execute_impl body string.

    Args:
        block: The instruction's semantic block.
        ctx: Optional lowering context. If None, one is created from the
            block's execution model.

    Returns:
        Multi-line C++ string for the execute_impl body (indented with 2-space).
    """
    if ctx is None:
        ctx = LoweringContext(exec_model=block.pragma)

    if block.is_empty:
        return '  (void)wf;'

    body_lines = _lower_stmt(block.body, ctx)

    if ctx.exec_model == ExecModel.VECTOR:
        wrapped = []
        wrapped.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        wrapped.append('    if (!(wf.exec() & (1ULL << lane))) continue;')
        for line in body_lines:
            wrapped.append('  ' + line)
        wrapped.append('  }')
        return '\n'.join(wrapped)

    return '\n'.join(body_lines)


def _indent(ctx: LoweringContext) -> str:
    return '  ' * ctx.indent


def _lower_stmt(node: SemaNode, ctx: LoweringContext) -> list[str]:
    """Lower a statement node to C++ lines."""
    kind = node.kind

    if kind == SemaNodeKind.SEQ:
        lines: list[str] = []
        for child in node.children:
            lines.extend(_lower_stmt(child, ctx))
        return lines

    if kind == SemaNodeKind.ASSIGN:
        return _lower_assign(node, ctx)

    if kind in (SemaNodeKind.ADD_ASSIGN, SemaNodeKind.SUB_ASSIGN):
        op = '+=' if kind == SemaNodeKind.ADD_ASSIGN else '-='
        lhs = _lower_expr(node.children[0], ctx)
        rhs = _lower_expr(node.children[1], ctx)
        return [f'{_indent(ctx)}{lhs} {op} {rhs};']

    if kind == SemaNodeKind.IF:
        return _lower_if(node, ctx)

    if kind == SemaNodeKind.FOR:
        return _lower_for(node, ctx)

    if kind == SemaNodeKind.WHILE:
        cond = _lower_expr(node.children[0], ctx)
        lines = [f'{_indent(ctx)}while ({cond}) {{']
        inner_ctx = LoweringContext(
            exec_model=ctx.exec_model,
            operand_reg_classes=ctx.operand_reg_classes,
            indent=ctx.indent + 1,
            declared=ctx.declared,
        )
        lines.extend(_lower_stmt(node.children[1], inner_ctx))
        lines.append(f'{_indent(ctx)}}}')
        return lines

    if kind == SemaNodeKind.BREAK:
        return [f'{_indent(ctx)}break;']

    if kind == SemaNodeKind.CONTINUE:
        return [f'{_indent(ctx)}continue;']

    if kind == SemaNodeKind.RETURN:
        val = _lower_expr(node.children[0], ctx)
        return [f'{_indent(ctx)}return {val};']

    if kind == SemaNodeKind.COMMENT:
        if node.children and node.children[0].lit_value:
            return [f'{_indent(ctx)}// {node.children[0].lit_value}']
        return []

    if kind == SemaNodeKind.DECLARE:
        return _lower_declare(node, ctx)

    if kind == SemaNodeKind.PRAGMA:
        return []

    if kind == SemaNodeKind.EVAL:
        if node.children and node.children[0].lit_value:
            return [f'{_indent(ctx)}// eval: {node.children[0].lit_value}']
        return []

    # Expression used as statement (e.g., standalone .call)
    expr = _lower_expr(node, ctx)
    return [f'{_indent(ctx)}{expr};']


def _lower_assign(node: SemaNode, ctx: LoweringContext) -> list[str]:
    """Lower an assignment statement."""
    lhs_node = node.children[0]
    rhs_node = node.children[1]

    # Context ID write: SCC, VCC, EXEC
    if lhs_node.kind == SemaNodeKind.ID and lhs_node.id_name in _CONTEXT_WRITES:
        rhs = _lower_expr(rhs_node, ctx)
        writer = _CONTEXT_WRITES[lhs_node.id_name]
        return [f'{_indent(ctx)}{writer}({rhs});']

    # Destination operand write: .instoperand(D, N)
    if _is_dst_operand(lhs_node):
        return _lower_dst_write(lhs_node, rhs_node, ctx)

    # ARRAYSLICE on LHS: bit-range insert
    if lhs_node.kind == SemaNodeKind.ARRAYSLICE:
        target = _lower_expr(lhs_node.children[0], ctx)
        hi = _lower_expr(lhs_node.children[1], ctx)
        lo = _lower_expr(lhs_node.children[2], ctx)
        rhs = _lower_expr(rhs_node, ctx)
        return [f'{_indent(ctx)}util::insert_bits({target}, {hi}, {lo}, {rhs});']

    # ARRAYDEREF on LHS: memory write
    if lhs_node.kind == SemaNodeKind.ARRAYDEREF and lhs_node.children:
        arr = lhs_node.children[0]
        while arr.kind == SemaNodeKind.CAST and arr.children:
            arr = arr.children[0]
        if arr.kind == SemaNodeKind.ID and arr.id_name in ('MEM', 'LDS'):
            idx = _lower_expr(lhs_node.children[1], ctx)
            rhs = _lower_expr(rhs_node, ctx)
            elem_ty = rhs_node.ty.cpp_type if rhs_node.ty else 'uint32_t'
            if arr.id_name == 'LDS':
                return [f'{_indent(ctx)}wf.lds().write<{elem_ty}>({idx}, {rhs});']
            if ctx.exec_model == ExecModel.SCALAR:
                return [f'{_indent(ctx)}wf.scalar_mem().write<{elem_ty}>({idx}, {rhs});']
            return [f'{_indent(ctx)}wf.vmem().write<{elem_ty}>({idx}, {rhs});']

    # Local variable assignment
    lhs = _lower_expr(lhs_node, ctx)
    rhs = _lower_expr(rhs_node, ctx)

    if lhs_node.kind == SemaNodeKind.ID and lhs_node.id_name:
        var_name = lhs_node.id_name
        if var_name not in ctx.declared and var_name not in _CONTEXT_READS:
            cpp_ty = 'uint32_t'
            if rhs_node.ty:
                cpp_ty = rhs_node.ty.cpp_type
            elif node.ty:
                cpp_ty = node.ty.cpp_type
            ctx.declared.add(var_name)
            return [f'{_indent(ctx)}{cpp_ty} {lhs} = {rhs};']

    return [f'{_indent(ctx)}{lhs} = {rhs};']


def _lower_if(node: SemaNode, ctx: LoweringContext) -> list[str]:
    """Lower an IF node (supports 2, 3, or multi-branch elif chains)."""
    children = node.children
    lines: list[str] = []
    inner_ctx = LoweringContext(
        exec_model=ctx.exec_model,
        operand_reg_classes=ctx.operand_reg_classes,
        indent=ctx.indent + 1,
        declared=ctx.declared,
    )

    if len(children) == 2:
        cond = _lower_expr(children[0], ctx)
        lines.append(f'{_indent(ctx)}if ({cond}) {{')
        lines.extend(_lower_stmt(children[1], inner_ctx))
        lines.append(f'{_indent(ctx)}}}')
    elif len(children) == 3:
        cond = _lower_expr(children[0], ctx)
        lines.append(f'{_indent(ctx)}if ({cond}) {{')
        lines.extend(_lower_stmt(children[1], inner_ctx))
        lines.append(f'{_indent(ctx)}}} else {{')
        lines.extend(_lower_stmt(children[2], inner_ctx))
        lines.append(f'{_indent(ctx)}}}')
    else:
        # Multi-branch: treat as if/elif chain (pairs of cond, body)
        for i in range(0, len(children) - 1, 2):
            cond = _lower_expr(children[i], ctx)
            keyword = 'if' if i == 0 else '} else if'
            lines.append(f'{_indent(ctx)}{keyword} ({cond}) {{')
            lines.extend(_lower_stmt(children[i + 1], inner_ctx))
        if len(children) % 2 == 1:
            lines.append(f'{_indent(ctx)}}} else {{')
            lines.extend(_lower_stmt(children[-1], inner_ctx))
        lines.append(f'{_indent(ctx)}}}')

    return lines


def _lower_for(node: SemaNode, ctx: LoweringContext) -> list[str]:
    """Lower a FOR node with 4 children: init, cond, step, body."""
    init_lines = _lower_stmt(node.children[0], ctx)
    cond = _lower_expr(node.children[1], ctx)
    step_lines = _lower_stmt(node.children[2], ctx)
    inner_ctx = LoweringContext(
        exec_model=ctx.exec_model,
        operand_reg_classes=ctx.operand_reg_classes,
        indent=ctx.indent + 1,
        declared=ctx.declared,
    )

    init_str = '; '.join(l.strip().rstrip(';') for l in init_lines) if init_lines else ''
    step_str = '; '.join(l.strip().rstrip(';') for l in step_lines) if step_lines else ''

    lines = [f'{_indent(ctx)}for ({init_str}; {cond}; {step_str}) {{']
    lines.extend(_lower_stmt(node.children[3], inner_ctx))
    lines.append(f'{_indent(ctx)}}}')
    return lines


def _lower_declare(node: SemaNode, ctx: LoweringContext) -> list[str]:
    """Lower a DECLARE node."""
    if node.children:
        var = node.children[0]
        if var.kind == SemaNodeKind.ID and var.id_name:
            cpp_ty = var.ty.cpp_type if var.ty else 'uint32_t'
            ctx.declared.add(var.id_name)
            return [f'{_indent(ctx)}{cpp_ty} {var.id_name} = 0;']
    return []


def _lower_expr(node: SemaNode, ctx: LoweringContext) -> str:
    """Lower an expression node to a C++ expression string."""
    kind = node.kind

    if kind == SemaNodeKind.LIT:
        return _lower_lit(node)

    if kind == SemaNodeKind.ID:
        return _lower_id(node, ctx)

    if kind in _INFIX_OPS:
        lhs = _lower_expr(node.children[0], ctx)
        rhs = _lower_expr(node.children[1], ctx)
        op = _INFIX_OPS[kind]
        return f'({lhs} {op} {rhs})'

    if kind == SemaNodeKind.UNORD_NE:
        lhs = _lower_expr(node.children[0], ctx)
        rhs = _lower_expr(node.children[1], ctx)
        return f'util::unordered_ne({lhs}, {rhs})'

    if kind == SemaNodeKind.POW:
        lhs = _lower_expr(node.children[0], ctx)
        rhs = _lower_expr(node.children[1], ctx)
        return f'std::pow({lhs}, {rhs})'

    if kind == SemaNodeKind.FPOW:
        lhs = _lower_expr(node.children[0], ctx)
        rhs = _lower_expr(node.children[1], ctx)
        return f'std::pow({lhs}, {rhs})'

    if kind == SemaNodeKind.LDEXP:
        val = _lower_expr(node.children[0], ctx)
        exp = _lower_expr(node.children[1], ctx)
        return f'std::ldexp({val}, {exp})'

    if kind in _STD_MATH:
        arg = _lower_expr(node.children[0], ctx)
        return f'{_STD_MATH[kind]}({arg})'

    if kind == SemaNodeKind.FRACT:
        arg = _lower_expr(node.children[0], ctx)
        return f'util::fract({arg})'

    if kind == SemaNodeKind.FMA:
        a = _lower_expr(node.children[0], ctx)
        b = _lower_expr(node.children[1], ctx)
        c = _lower_expr(node.children[2], ctx)
        return f'std::fma({a}, {b}, {c})'

    if kind == SemaNodeKind.BITNEG:
        arg = _lower_expr(node.children[0], ctx)
        return f'(~{arg})'

    if kind == SemaNodeKind.BOOLNEG:
        arg = _lower_expr(node.children[0], ctx)
        return f'(!{arg})'

    if kind == SemaNodeKind.ABS:
        arg = _lower_expr(node.children[0], ctx)
        return f'std::abs({arg})'

    if kind == SemaNodeKind.UMINUS:
        arg = _lower_expr(node.children[0], ctx)
        return f'(-{arg})'

    if kind == SemaNodeKind.UPLUS:
        arg = _lower_expr(node.children[0], ctx)
        return f'(+{arg})'

    if kind == SemaNodeKind.SIGN:
        arg = _lower_expr(node.children[0], ctx)
        return f'util::sign({arg})'

    if kind == SemaNodeKind.SIGNEXT:
        arg = _lower_expr(node.children[0], ctx)
        return f'util::sign_extend({arg})'

    if kind == SemaNodeKind.SIGNEXT_FROM_BIT:
        if len(node.children) == 2:
            val = _lower_expr(node.children[0], ctx)
            bit = _lower_expr(node.children[1], ctx)
            return f'util::sign_extend_from_bit({val}, {bit})'
        arg = _lower_expr(node.children[0], ctx)
        return f'util::sign_extend_from_bit({arg})'

    if kind == SemaNodeKind.CAST:
        return _lower_cast(node, ctx)

    if kind == SemaNodeKind.INSTOPERAND:
        return _lower_instoperand_read(node, ctx)

    if kind == SemaNodeKind.ARRAYDEREF:
        return _lower_arrayderef(node, ctx)

    if kind == SemaNodeKind.ARRAYSLICE:
        val = _lower_expr(node.children[0], ctx)
        hi = _lower_expr(node.children[1], ctx)
        lo = _lower_expr(node.children[2], ctx)
        return f'util::extract_bits({val}, {hi}, {lo})'

    if kind == SemaNodeKind.ARRAYSLICESIZE:
        val = _lower_expr(node.children[0], ctx)
        off = _lower_expr(node.children[1], ctx)
        cnt = _lower_expr(node.children[2], ctx)
        return f'util::extract_bits_sized({val}, {off}, {cnt})'

    if kind == SemaNodeKind.FIELDDEREF:
        obj = _lower_expr(node.children[0], ctx)
        if len(node.children) > 1:
            field_name = _lower_expr(node.children[1], ctx)
            return f'{obj}.{field_name}'
        return obj

    if kind == SemaNodeKind.BITCAT:
        args = [_lower_expr(c, ctx) for c in node.children]
        if len(args) == 2:
            lo_bits = node.children[1].ty.size if node.children[1].ty else 32
            return f'util::bitcat({args[0]}, {args[1]}, {lo_bits})'
        return f'util::bitcat_n({", ".join(args)})'

    if kind == SemaNodeKind.CALL:
        return _lower_call(node, ctx)

    if kind == SemaNodeKind.TERNARY:
        cond = _lower_expr(node.children[0], ctx)
        then = _lower_expr(node.children[1], ctx)
        else_ = _lower_expr(node.children[2], ctx)
        return f'({cond} ? {then} : {else_})'

    if kind == SemaNodeKind.CONS_ARRAY:
        elems = ', '.join(_lower_expr(c, ctx) for c in node.children)
        return f'{{{elems}}}'

    if kind == SemaNodeKind.SUM:
        arr = _lower_expr(node.children[0], ctx)
        if len(node.children) > 1:
            cnt = _lower_expr(node.children[1], ctx)
            return f'util::sum({arr}, {cnt})'
        return f'util::sum({arr})'

    if kind == SemaNodeKind.WITHIN:
        val = _lower_expr(node.children[0], ctx)
        lo = _lower_expr(node.children[1], ctx)
        if len(node.children) >= 3:
            hi = _lower_expr(node.children[2], ctx)
            return f'({val} >= {lo} && {val} <= {hi})'
        return f'({val} >= {lo})'

    if kind == SemaNodeKind.EXPONENT:
        arg = _lower_expr(node.children[0], ctx)
        return f'util::extract_exponent({arg})'

    if kind == SemaNodeKind.MANTISSA:
        arg = _lower_expr(node.children[0], ctx)
        return f'util::extract_mantissa({arg})'

    if kind == SemaNodeKind.LAMBDA:
        if node.children:
            return _lower_expr(node.children[-1], ctx)
        return '/* lambda */'

    if kind == SemaNodeKind.ASSIGN:
        lhs = _lower_expr(node.children[0], ctx)
        rhs = _lower_expr(node.children[1], ctx)
        return f'({lhs} = {rhs})'

    if kind in (SemaNodeKind.IF, SemaNodeKind.FOR, SemaNodeKind.WHILE,
                 SemaNodeKind.SEQ, SemaNodeKind.DECLARE, SemaNodeKind.BREAK,
                 SemaNodeKind.CONTINUE, SemaNodeKind.RETURN,
                 SemaNodeKind.COMMENT, SemaNodeKind.PRAGMA, SemaNodeKind.EVAL,
                 SemaNodeKind.ADD_ASSIGN, SemaNodeKind.SUB_ASSIGN,
                 SemaNodeKind.SWITCH, SemaNodeKind.CASE, SemaNodeKind.DEFAULT):
        return f'/* stmt-in-expr: {kind.name} */'

    raise ValueError(f'Unhandled SemaNodeKind in lowering: {kind.name}')


def _lower_lit(node: SemaNode) -> str:
    """Lower a literal value with appropriate C++ suffix."""
    val = node.lit_value or '0'
    if node.ty:
        if node.ty.base == 'F' and node.ty.size == 32:
            if '.' not in val and 'e' not in val.lower():
                return f'{val}.0f'
            return f'{val}f'
        if node.ty.base == 'F' and node.ty.size == 64:
            if '.' not in val and 'e' not in val.lower():
                return f'{val}.0'
            return val
        if node.ty.size > 32:
            if node.ty.base == 'I':
                return f'{val}LL'
            return f'{val}ULL'
    return val


def _lower_id(node: SemaNode, ctx: LoweringContext) -> str:
    """Lower an identifier reference."""
    name = node.id_name or ''
    if name in _CONTEXT_READS:
        return _CONTEXT_READS[name]
    return name


def _lower_cast(node: SemaNode, ctx: LoweringContext) -> str:
    """Lower a .cast node to static_cast or std::bit_cast."""
    inner = node.children[0]
    target = node.cast_target
    if target is None:
        return _lower_expr(inner, ctx)

    inner_expr = _lower_expr(inner, ctx)
    cpp_ty = target.cpp_type

    if _is_reinterpret(inner, target):
        return f'std::bit_cast<{cpp_ty}>({inner_expr})'

    return f'static_cast<{cpp_ty}>({inner_expr})'


def _is_reinterpret(node: SemaNode, target: SemaType | None = None) -> bool:
    """Check if a cast requires bit reinterpretation (different type bases)."""
    if node.kind == SemaNodeKind.INSTOPERAND:
        src_base = node.ty.base if node.ty else 'B'
        tgt_base = target.base if target else 'B'
        return src_base != tgt_base and src_base == 'B'
    if node.kind == SemaNodeKind.CAST and node.children:
        return _is_reinterpret(node.children[0], target)
    return False


def _is_dst_operand(node: SemaNode) -> bool:
    """Check if a node is a destination operand write."""
    if node.kind == SemaNodeKind.INSTOPERAND:
        if node.children and node.children[0].kind == SemaNodeKind.ID:
            return node.children[0].id_name == 'D'
    if node.kind == SemaNodeKind.CAST and node.children:
        return _is_dst_operand(node.children[0])
    return False


def _get_operand_index(node: SemaNode) -> int:
    """Extract operand index from .instoperand(S/D, N)."""
    target = node
    while target.kind == SemaNodeKind.CAST and target.children:
        target = target.children[0]
    if target.kind == SemaNodeKind.INSTOPERAND and len(target.children) >= 2:
        idx_node = target.children[1]
        if idx_node.kind == SemaNodeKind.LIT and idx_node.lit_value:
            return int(idx_node.lit_value)
    return 0


def _lower_instoperand_read(node: SemaNode, ctx: LoweringContext) -> str:
    """Lower .instoperand(S, N) to a read expression."""
    if len(node.children) < 2:
        return '/* malformed instoperand */'

    tag_node = node.children[0]
    idx_node = node.children[1]
    tag = tag_node.id_name or 'S'
    idx = idx_node.lit_value or '0'

    reg_class = ctx.operand_reg_classes.get((tag, int(idx)))
    if reg_class == RegClass.SGPR or ctx.exec_model == ExecModel.SCALAR:
        return f'inst.src{idx}.read_scalar(wf)'

    return f'inst.src{idx}.read_lane(wf, lane)'


def _lower_dst_write(
    lhs_node: SemaNode, rhs_node: SemaNode, ctx: LoweringContext,
) -> list[str]:
    """Lower a destination operand write."""
    idx = _get_operand_index(lhs_node)
    rhs = _lower_expr(rhs_node, ctx)

    if ctx.exec_model == ExecModel.SCALAR:
        return [f'{_indent(ctx)}inst.dst{idx}.write_scalar(wf, {rhs});']

    return [f'{_indent(ctx)}inst.dst{idx}.write_lane(wf, lane, {rhs});']


def _lower_arrayderef(node: SemaNode, ctx: LoweringContext) -> str:
    """Lower .arrayderef — memory read or bit index."""
    if len(node.children) < 2:
        return '/* malformed arrayderef */'

    array_node = node.children[0]
    index_node = node.children[1]

    array_expr = _lower_expr(array_node, ctx)
    index_expr = _lower_expr(index_node, ctx)

    if array_node.kind == SemaNodeKind.ID:
        name = array_node.id_name or ''
        if name == 'MEM':
            elem_ty = node.ty.cpp_type if node.ty else 'uint32_t'
            if ctx.exec_model == ExecModel.SCALAR:
                return f'wf.scalar_mem().read<{elem_ty}>({index_expr})'
            return f'wf.vmem().read<{elem_ty}>({index_expr})'
        if name == 'LDS':
            elem_ty = node.ty.cpp_type if node.ty else 'uint32_t'
            return f'wf.lds().read<{elem_ty}>({index_expr})'

    # Bit index (VCC/EXEC bitmask access)
    return f'(({array_expr} >> {index_expr}) & 1)'


def _lower_call(node: SemaNode, ctx: LoweringContext) -> str:
    """Lower a .call node through the helper registry."""
    callee = node.call_name or ''
    args = [_lower_expr(c, ctx) for c in node.children[1:]]
    args_str = ', '.join(args)

    entry = HELPER_REGISTRY.get(callee)
    if entry is None:
        return f'{callee}({args_str})'

    treatment, cpp_name = entry
    if treatment == HelperTreatment.OPAQUE_NOP:
        return f'/* {callee} -- no-op */'
    if treatment == HelperTreatment.INLINE_CPP and cpp_name:
        return f'{cpp_name}({args_str})'
    if treatment == HelperTreatment.RECURSIVE:
        return f'/* recursive: {callee}({args_str}) */'

    return f'{callee}({args_str})'
