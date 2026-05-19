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


@dataclass(frozen=True)
class OperandBinding:
    """Named operand with register class and width."""

    name: str
    reg_class: RegClass
    bit_width: int = 32


@dataclass
class OperandMap:
    """Maps INSTOPERAND(S/D, index) to named operand bindings."""

    src_bindings: dict[int, OperandBinding] = field(default_factory=dict)
    dst_bindings: dict[int, OperandBinding] = field(default_factory=dict)

    def src(self, idx: int) -> OperandBinding | None:
        return self.src_bindings.get(idx)

    def dst(self, idx: int) -> OperandBinding | None:
        return self.dst_bindings.get(idx)

    @staticmethod
    def from_operand_names(
        src_ops: list[str],
        dst_ops: list[str],
        exec_model: ExecModel,
        dtype: str | None = None,
        src_width: int | None = None,
        dst_width: int | None = None,
    ) -> OperandMap:
        is_64 = dtype in ('b64', 'i64', 'u64', 'f64')
        is_scalar = exec_model == ExecModel.SCALAR
        reg = RegClass.SGPR if is_scalar else RegClass.VGPR
        default_width = 64 if is_64 else 32
        sw = src_width if src_width is not None else default_width
        dw = dst_width if dst_width is not None else default_width
        src_b = {i: OperandBinding(name, reg, sw) for i, name in enumerate(src_ops)}
        dst_b = {i: OperandBinding(name, reg, dw) for i, name in enumerate(dst_ops)}
        return OperandMap(src_bindings=src_b, dst_bindings=dst_b)


@dataclass
class LoweringContext:
    """Context for one lowering invocation."""

    exec_model: ExecModel
    operand_map: OperandMap | None = None
    indent: int = 1
    declared: set[str] = field(default_factory=set)
    is_lhs: bool = False
    vcc_var: str = 'vcc'
    vcc_read: str | None = None
    vcc_dst: str | None = None


_INFIX_OPS: dict[SemaNodeKind, str] = {
    SemaNodeKind.ADD: '+',
    SemaNodeKind.SUB: '-',
    SemaNodeKind.MUL: '*',
    SemaNodeKind.DIV: '/',
    SemaNodeKind.MOD: '%',
    SemaNodeKind.AND: '&',
    SemaNodeKind.OR: '|',
    SemaNodeKind.XOR: '^',
    SemaNodeKind.SHL: '<<',
    SemaNodeKind.SHR: '>>',
    SemaNodeKind.LAND: '&&',
    SemaNodeKind.LOR: '||',
    SemaNodeKind.EQ: '==',
    SemaNodeKind.NE: '!=',
    SemaNodeKind.LT: '<',
    SemaNodeKind.GT: '>',
    SemaNodeKind.LE: '<=',
    SemaNodeKind.GE: '>=',
}

_CONTEXT_READS: dict[str, str] = {
    'SCC': 'wf.read_scc()',
    'VCC': 'wf.vcc()',
    'EXEC': 'wf.exec()',
    'EXEC_LO': 'static_cast<uint32_t>(wf.exec())',
    'M0': 'wf.m0()',
    'laneId': 'lane',
}

_CONTEXT_WRITES: dict[str, str] = {
    'SCC': 'wf.write_scc',
    'VCC': 'wf.write_vcc',
    'EXEC': 'wf.set_exec',
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
        writes_vcc = _writes_vcc(block.body)
        wrapped = []
        wrapped.append('  uint64_t exec = wf.exec();')
        if writes_vcc:
            vcc_init = _vcc_init_expr(ctx)
            wrapped.append(f'  uint64_t vcc = {vcc_init};')
        wrapped.append('  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {')
        wrapped.append('    if (!(exec & (1ULL << lane))) continue;')
        for line in body_lines:
            wrapped.append('  ' + line)
        wrapped.append('  }')
        if writes_vcc:
            vcc_write = _vcc_write_stmt(ctx)
            wrapped.append(f'  {vcc_write}')
        return '\n'.join(wrapped)

    return '\n'.join(body_lines)


_VCC_WRITING_CALLS = frozenset(
    {
        'add_co',
        'sub_co',
        'addc_co',
        'subbc_co',
    }
)


def _writes_vcc(node: SemaNode) -> bool:
    """Check if the AST writes to VCC."""
    if node.kind == SemaNodeKind.ASSIGN and node.children:
        lhs = node.children[0]
        if lhs.kind == SemaNodeKind.ARRAYDEREF and lhs.children:
            arr = lhs.children[0]
            while arr.kind == SemaNodeKind.CAST and arr.children:
                arr = arr.children[0]
            if arr.kind == SemaNodeKind.ID and arr.id_name == 'VCC':
                return True
    if node.kind == SemaNodeKind.CALL and node.call_name in _VCC_WRITING_CALLS:
        return True
    for child in node.children or ():
        if _writes_vcc(child):
            return True
    return False


def _vcc_init_expr(ctx: LoweringContext) -> str:
    """Return the C++ expression to initialize the vcc local variable."""
    if ctx.vcc_dst:
        return 'wf.vcc()'
    if ctx.operand_map:
        dst = ctx.operand_map.dst(0)
        if dst:
            return f'{dst.name}.read_scalar64(wf)'
    return 'wf.vcc()'


def _vcc_write_stmt(ctx: LoweringContext) -> str:
    """Return the C++ statement to write back the vcc local variable."""
    if ctx.vcc_dst and ctx.vcc_dst != '__vcc__':
        return f'{ctx.vcc_dst}.write_scalar64(wf, vcc);'
    if ctx.vcc_dst == '__vcc__':
        return 'wf.set_vcc(vcc);'
    if ctx.operand_map:
        dst = ctx.operand_map.dst(0)
        if dst:
            return f'{dst.name}.write_scalar64(wf, vcc);'
    return 'wf.set_vcc(vcc);'


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
            operand_map=ctx.operand_map,
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

    # Source operand write (for swap): .instoperand(S, N)
    if _is_src_operand_write(lhs_node):
        return _lower_src_write(lhs_node, rhs_node, ctx)

    # ARRAYSLICE on LHS: bit-range insert
    if lhs_node.kind == SemaNodeKind.ARRAYSLICE:
        target = _lower_expr(lhs_node.children[0], ctx)
        hi = _lower_expr(lhs_node.children[1], ctx)
        lo = _lower_expr(lhs_node.children[2], ctx)
        rhs = _lower_expr(rhs_node, ctx)
        return [f'{_indent(ctx)}util::insert_bits({target}, {hi}, {lo}, {rhs});']

    # ARRAYDEREF on LHS: bitmask or memory write
    if lhs_node.kind == SemaNodeKind.ARRAYDEREF and lhs_node.children:
        arr = lhs_node.children[0]
        while arr.kind == SemaNodeKind.CAST and arr.children:
            arr = arr.children[0]
        if arr.kind == SemaNodeKind.ID and arr.id_name in ('VCC', 'EXEC'):
            idx = _lower_expr(lhs_node.children[1], ctx)
            rhs = _lower_expr(rhs_node, ctx)
            ind = _indent(ctx)
            return [
                f'{ind}if ({rhs})',
                f'{ind}  {ctx.vcc_var} |= (1ULL << {idx});',
                f'{ind}else',
                f'{ind}  {ctx.vcc_var} &= ~(1ULL << {idx});',
            ]
        if arr.kind == SemaNodeKind.ID and arr.id_name in ('MEM', 'LDS'):
            idx = _lower_expr(lhs_node.children[1], ctx)
            rhs = _lower_expr(rhs_node, ctx)
            elem_ty = rhs_node.ty.cpp_type if rhs_node.ty else 'uint32_t'
            if arr.id_name == 'LDS':
                return [f'{_indent(ctx)}wf.lds().write<{elem_ty}>({idx}, {rhs});']
            if ctx.exec_model == ExecModel.SCALAR:
                return [
                    f'{_indent(ctx)}wf.scalar_mem().write<{elem_ty}>({idx}, {rhs});'
                ]
            return [f'{_indent(ctx)}wf.vmem().write<{elem_ty}>({idx}, {rhs});']

    # Local variable assignment
    lhs = _lower_expr(lhs_node, ctx)
    rhs = _lower_expr(rhs_node, ctx)

    if lhs_node.kind == SemaNodeKind.ID and lhs_node.id_name:
        var_name = lhs_node.id_name
        if var_name not in ctx.declared and var_name not in _CONTEXT_READS:
            cpp_ty = 'uint32_t'
            if lhs_node.ty:
                cpp_ty = lhs_node.ty.cpp_type
            elif rhs_node.ty:
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
        operand_map=ctx.operand_map,
        indent=ctx.indent + 1,
        declared=ctx.declared,
    )

    if len(children) == 2:
        cond = _lower_expr(children[0], ctx)
        body_lines = _lower_stmt(children[1], inner_ctx)
        if len(body_lines) == 1:
            lines.append(f'{_indent(ctx)}if ({cond})')
            lines.append(body_lines[0])
        else:
            lines.append(f'{_indent(ctx)}if ({cond}) {{')
            lines.extend(body_lines)
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
        operand_map=ctx.operand_map,
        indent=ctx.indent + 1,
        declared=ctx.declared,
    )

    init_str = (
        '; '.join(l.strip().rstrip(';') for l in init_lines) if init_lines else ''
    )
    step_str = (
        '; '.join(l.strip().rstrip(';') for l in step_lines) if step_lines else ''
    )

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
        return f'[&]() {{ auto v = {arg}; return v - std::floor(v); }}()'

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
        if node.ty and node.ty.base == 'I':
            return (
                f'[&]() {{ int{node.ty.size}_t v = {arg};'
                f' return static_cast<uint{node.ty.size}_t>'
                f'(v < 0 ? (0u - static_cast<uint{node.ty.size}_t>(v))'
                f' : static_cast<uint{node.ty.size}_t>(v)); }}()'
            )
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
            bit = node.children[1].lit_value or '0'
            shift = 31 - int(bit)
            return f'static_cast<int32_t>(static_cast<int32_t>({val} << {shift}) >> {shift})'
        arg = _lower_expr(node.children[0], ctx)
        return f'static_cast<int32_t>({arg})'

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
        return f'{cond} ? {then} : {else_}'

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

    if kind in (
        SemaNodeKind.IF,
        SemaNodeKind.FOR,
        SemaNodeKind.WHILE,
        SemaNodeKind.SEQ,
        SemaNodeKind.DECLARE,
        SemaNodeKind.BREAK,
        SemaNodeKind.CONTINUE,
        SemaNodeKind.RETURN,
        SemaNodeKind.COMMENT,
        SemaNodeKind.PRAGMA,
        SemaNodeKind.EVAL,
        SemaNodeKind.ADD_ASSIGN,
        SemaNodeKind.SUB_ASSIGN,
        SemaNodeKind.SWITCH,
        SemaNodeKind.CASE,
        SemaNodeKind.DEFAULT,
    ):
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

    if inner.ty and (inner.ty == target or inner.ty.cpp_type == target.cpp_type):
        if inner.kind != SemaNodeKind.INSTOPERAND or target.base in ('B', 'U', 'I'):
            return inner_expr

    cpp_ty = target.cpp_type

    if target.base == 'F' and target.size == 16:
        return f'util::f16_to_f32(static_cast<uint16_t>({inner_expr}))'
    if target.base == 'BF' and target.size == 16:
        return f'util::bf16_to_f32(static_cast<uint16_t>({inner_expr}))'

    if _is_reinterpret(inner, target):
        return f'std::bit_cast<{cpp_ty}>({inner_expr})'

    return f'static_cast<{cpp_ty}>({inner_expr})'


def _is_reinterpret(node: SemaNode, target: SemaType | None = None) -> bool:
    """Check if a cast requires bit reinterpretation.

    INSTOPERAND reads always return uint32_t/uint64_t. Any cast to a
    float type (F16/F32/F64/BF16) on an INSTOPERAND needs bit_cast,
    not static_cast, because we're reinterpreting register bits as float.
    """
    if target is None:
        return False
    if node.kind == SemaNodeKind.INSTOPERAND:
        if target.base in ('F', 'BF', 'FP'):
            return True
        return False
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


def _is_src_operand_write(node: SemaNode) -> bool:
    """Check if a node is a source operand used as write target (e.g., swap)."""
    if node.kind == SemaNodeKind.INSTOPERAND:
        if node.children and node.children[0].kind == SemaNodeKind.ID:
            return node.children[0].id_name == 'S'
    if node.kind == SemaNodeKind.CAST and node.children:
        return _is_src_operand_write(node.children[0])
    return False


def _lower_src_write(
    lhs_node: SemaNode,
    rhs_node: SemaNode,
    ctx: LoweringContext,
) -> list[str]:
    """Lower a write to a source operand (used by vector_swap)."""
    idx = _get_operand_index(lhs_node)
    rhs = _lower_expr(rhs_node, ctx)
    binding = ctx.operand_map.src(idx) if ctx.operand_map else None
    if binding:
        name = binding.name
        if binding.bit_width == 64:
            return [f'{_indent(ctx)}{name}.write_lane64(wf, lane, {rhs});']
        return [f'{_indent(ctx)}{name}.write_lane(wf, lane, {rhs});']
    return [f'{_indent(ctx)}inst.src{idx}.write_lane(wf, lane, {rhs});']


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
    idx = int(idx_node.lit_value or '0')

    binding = None
    if ctx.operand_map:
        if tag == 'D':
            binding = ctx.operand_map.dst(idx)
        else:
            binding = ctx.operand_map.src(idx)

    if binding:
        name = binding.name
        if binding.reg_class == RegClass.SGPR or ctx.exec_model == ExecModel.SCALAR:
            if binding.bit_width == 64:
                return f'{name}.read_scalar64(wf)'
            return f'{name}.read_scalar(wf)'
        if binding.bit_width == 64:
            return f'{name}.read_lane64(wf, lane)'
        return f'{name}.read_lane(wf, lane)'

    if ctx.exec_model == ExecModel.SCALAR:
        return f'inst.src{idx}.read_scalar(wf)'
    return f'inst.src{idx}.read_lane(wf, lane)'


def _get_operand_dtype(node: SemaNode) -> SemaType | None:
    """Get the semantic type from a destination operand node."""
    target = node
    while target.kind == SemaNodeKind.CAST and target.children:
        if target.cast_target:
            return target.cast_target
        target = target.children[0]
    if target.kind == SemaNodeKind.INSTOPERAND:
        return target.ty
    return None


def _rhs_is_float_expr(node: SemaNode) -> int:
    """Return 32 or 64 if the expression produces a float C++ type, else 0.

    Checks if the RHS expression evaluates to a float in C++, meaning the
    dst write needs bit_cast<uint32_t> to store it as a register value.
    """
    if node.ty and node.ty.base in ('F', 'BF') and node.ty.size in (32, 64):
        if node.kind in (
            SemaNodeKind.ADD,
            SemaNodeKind.SUB,
            SemaNodeKind.MUL,
            SemaNodeKind.DIV,
            SemaNodeKind.FMA,
            SemaNodeKind.SQRT,
            SemaNodeKind.SIN,
            SemaNodeKind.COS,
            SemaNodeKind.ABS,
            SemaNodeKind.LOG2,
            SemaNodeKind.FLOOR,
            SemaNodeKind.TRUNC,
            SemaNodeKind.FRACT,
            SemaNodeKind.LDEXP,
            SemaNodeKind.UMINUS,
            SemaNodeKind.UPLUS,
            SemaNodeKind.ID,
        ):
            return node.ty.size
        if node.kind == SemaNodeKind.CALL:
            return node.ty.size
    if node.kind == SemaNodeKind.TERNARY:
        return _rhs_is_float_expr(node.children[1])
    return 0


def _lower_dst_write(
    lhs_node: SemaNode,
    rhs_node: SemaNode,
    ctx: LoweringContext,
) -> list[str]:
    """Lower a destination operand write."""
    idx = _get_operand_index(lhs_node)
    rhs = _lower_expr(rhs_node, ctx)

    needs_bitcast = _rhs_is_float_expr(rhs_node)
    lhs_ty = _get_operand_dtype(lhs_node)
    if lhs_ty and lhs_ty.base == 'F' and lhs_ty.size == 16:
        rhs = f'util::f32_to_f16({rhs})'
    elif lhs_ty and lhs_ty.base == 'BF' and lhs_ty.size == 16:
        rhs = f'util::f32_to_bf16({rhs})'
    elif lhs_ty and lhs_ty.size == 16 and lhs_ty.base in ('I', 'U'):
        cpp = lhs_ty.cpp_type
        rhs = (
            f'static_cast<uint32_t>(static_cast<uint16_t>(static_cast<{cpp}>({rhs})))'
            if cpp == 'int16_t'
            else f'static_cast<uint32_t>(static_cast<{cpp}>({rhs}))'
        )
    elif needs_bitcast == 32:
        rhs = f'std::bit_cast<uint32_t>({rhs})'
    elif needs_bitcast == 64:
        rhs = f'std::bit_cast<uint64_t>({rhs})'

    binding = None
    if ctx.operand_map:
        binding = ctx.operand_map.dst(idx)

    if binding:
        name = binding.name
        if binding.reg_class == RegClass.SGPR or ctx.exec_model == ExecModel.SCALAR:
            if binding.bit_width == 64:
                return [f'{_indent(ctx)}{name}.write_scalar64(wf, {rhs});']
            return [f'{_indent(ctx)}{name}.write_scalar(wf, {rhs});']
        if binding.bit_width == 64:
            return [f'{_indent(ctx)}{name}.write_lane64(wf, lane, {rhs});']
        return [f'{_indent(ctx)}{name}.write_lane(wf, lane, {rhs});']

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
    if (
        array_node.kind == SemaNodeKind.ID
        and array_node.id_name == 'VCC'
        and ctx.vcc_read is not None
    ):
        return f'(({ctx.vcc_read} >> {index_expr}) & 1)'
    return f'(({array_expr} >> {index_expr}) & 1)'


_INLINE_UNARY_OPS: dict[str, str] = {
    'rcp': 'amdgpu::transcendental::rcp_f32({0})',
    'rcp_iflag': 'amdgpu::transcendental::rcp_f32({0})',
    'rsq': 'amdgpu::transcendental::rsq_f32({0})',
    'sqrt': 'amdgpu::transcendental::sqrt_f32({0})',
    'sin': 'amdgpu::transcendental::sin_f32({0})',
    'cos': 'amdgpu::transcendental::cos_f32({0})',
    'log': 'amdgpu::transcendental::log_f32({0})',
    'exp': 'amdgpu::transcendental::exp_f32({0})',
    'rcp_f64': 'amdgpu::transcendental::rcp_f64({0})',
    'rsq_f64': 'amdgpu::transcendental::rsq_f64({0})',
    'sqrt_f64': 'amdgpu::transcendental::sqrt_f64({0})',
    'abs': '[&]() {{ int32_t v = static_cast<int32_t>({0});'
    ' return static_cast<uint32_t>'
    '(v < 0 ? (0u - static_cast<uint32_t>(v))'
    ' : static_cast<uint32_t>(v)); }}()',
    'rndne': 'std::nearbyint({0})',
    'ceil': 'std::ceil({0})',
    'exp2': 'amdgpu::transcendental::exp_f32({0})',
    'bcnt': 'static_cast<uint32_t>(std::popcount({0}))',
    'bcnt1': 'static_cast<uint32_t>(std::popcount({0}))',
    'bcnt0': 'static_cast<uint32_t>(std::popcount(~{0}))',
    'bfrev': '[&]() {{ uint32_t s = {0}; uint32_t r = 0;'
    ' for (int i = 0; i < 32; ++i) r |= ((s >> i) & 1) << (31 - i);'
    ' return r; }}()',
    'brev': '[&]() {{ uint32_t s = {0}; uint32_t r = 0;'
    ' for (int i = 0; i < 32; ++i) r |= ((s >> i) & 1) << (31 - i);'
    ' return r; }}()',
    'ffbl': '[&]() {{ auto s = {0};'
    ' return s == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countr_zero(s)); }}()',
    'ffbh_u32': '[&]() {{ auto s = {0};'
    ' return s == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countl_zero(s)); }}()',
    'ff0': '[&]() {{ auto s = ~static_cast<uint32_t>({0});'
    ' return s == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countr_zero(s)); }}()',
    'ff1': '[&]() {{ auto s = static_cast<uint32_t>({0});'
    ' return s == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countr_zero(s)); }}()',
    'flbit': '[&]() {{ auto s = static_cast<uint32_t>({0});'
    ' return s == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countl_zero(s)); }}()',
    'flbit_i32': '[&]() {{ int32_t sv = static_cast<int32_t>({0});'
    ' uint32_t abs_val = sv < 0 ? ~static_cast<uint32_t>(sv)'
    ' : static_cast<uint32_t>(sv);'
    ' return abs_val == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countl_zero(abs_val)); }}()',
    'flbit_i32_i64': '[&]() {{ int64_t sv = static_cast<int64_t>({0});'
    ' uint64_t abs_val = sv < 0 ? ~static_cast<uint64_t>(sv)'
    ' : static_cast<uint64_t>(sv);'
    ' return abs_val == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countl_zero(abs_val)); }}()',
    'ctz': '[&]() {{ auto s = static_cast<uint64_t>({0});'
    ' return s == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countr_zero(s)); }}()',
    'cls': '[&]() {{ int32_t sv = static_cast<int32_t>({0});'
    ' if (sv == 0 || sv == -1) return 32u;'
    ' uint32_t u = sv < 0 ? ~static_cast<uint32_t>(sv) : static_cast<uint32_t>(sv);'
    ' return static_cast<uint32_t>(std::countl_zero(u)) - 1u; }}()',
    'wqm': '[&]() {{ uint32_t s = {0}; uint32_t r = 0;'
    ' for (int i = 0; i < 8; ++i)'
    ' if (s & (0xFu << (i * 4))) r |= (0xFu << (i * 4));'
    ' return r; }}()',
    'clz': '[&]() {{ auto s = static_cast<uint32_t>({0});'
    ' return s == 0 ? 32u : static_cast<uint32_t>(std::countl_zero(s)); }}()',
    'clz64': '[&]() {{ auto s = static_cast<uint64_t>({0});'
    ' return s == 0 ? 64u : static_cast<uint32_t>(std::countl_zero(s)); }}()',
    'cvt_hi_f32_f16': 'std::bit_cast<uint32_t>(util::f16_to_f32(static_cast<uint16_t>(({0}) >> 16)))',
    'brev64': '[&]() {{ uint64_t s = {0}; uint64_t r = 0;'
    ' for (int i = 0; i < 64; ++i) r |= ((s >> i) & 1ULL) << (63 - i);'
    ' return r; }}()',
    'wqm64': '[&]() {{ uint64_t s = {0}; uint64_t r = 0;'
    ' for (int i = 0; i < 16; ++i)'
    ' if (s & (0xFULL << (i * 4))) r |= (0xFULL << (i * 4));'
    ' return r; }}()',
    'quadmask64': '[&]() {{ uint64_t s = {0}; uint64_t r = 0;'
    ' for (int i = 0; i < 16; ++i)'
    ' if (s & (0xFULL << (i * 4))) r |= (1ULL << i);'
    ' return r; }}()',
    'ff064': '[&]() {{ auto s = ~static_cast<uint64_t>({0});'
    ' return s == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countr_zero(s)); }}()',
    'ff164': '[&]() {{ auto s = static_cast<uint64_t>({0});'
    ' return s == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countr_zero(s)); }}()',
    'flbit64': '[&]() {{ auto s = static_cast<uint64_t>({0});'
    ' return s == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countl_zero(s)); }}()',
    'ctz64': '[&]() {{ auto s = static_cast<uint64_t>({0});'
    ' return s == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countr_zero(s)); }}()',
    'bcnt164': 'static_cast<uint32_t>(std::popcount(static_cast<uint64_t>({0})))',
    'bcnt064': 'static_cast<uint32_t>(std::popcount(~static_cast<uint64_t>({0})))',
    'quadmask': '[&]() {{ uint32_t s = {0}; uint32_t r = 0;'
    ' for (int i = 0; i < 8; ++i)'
    ' if (s & (0xFu << (i * 4))) r |= (1u << i);'
    ' return r; }}()',
    'bitset0': '({0} & ~(1u << ({0} & 31u)))',
    'bitset1': '({0} | (1u << ({0} & 31u)))',
    'v_readfirstlane': '{0}',
    'ffbh_i32': '[&]() {{ auto s = static_cast<int32_t>({0});'
    ' uint32_t a = s < 0 ? ~static_cast<uint32_t>(s) : static_cast<uint32_t>(s);'
    ' return a == 0 ? static_cast<uint32_t>(-1)'
    ' : static_cast<uint32_t>(std::countl_zero(a)); }}()',
    'cls_i32': '[&]() {{ auto s = static_cast<int32_t>({0});'
    ' uint32_t a = s < 0 ? ~static_cast<uint32_t>(s) : static_cast<uint32_t>(s);'
    ' return a == 0 ? 31u : static_cast<uint32_t>(std::countl_zero(a)) - 1; }}()',
    'frexp_exp_f32': '[&]() {{ float s = {0}; int exp = 0;'
    ' if (s != 0.0f && !std::isnan(s) && !std::isinf(s)) std::frexp(s, &exp);'
    ' return static_cast<uint32_t>(exp); }}()',
    'frexp_exp_f16': '[&]() {{ float s = {0}; int exp = 0;'
    ' if (s != 0.0f && !std::isnan(s) && !std::isinf(s)) std::frexp(s, &exp);'
    ' return static_cast<uint32_t>(exp); }}()',
    'frexp_exp_f64': '[&]() {{ double s = {0};'
    ' int exp = 0;'
    ' if (s != 0.0 && !std::isnan(s) && !std::isinf(s)) std::frexp(s, &exp);'
    ' return static_cast<uint32_t>(exp); }}()',
    'frexp_mant_f64': '[&]() {{ double s = {0};'
    ' int exp = 0;'
    ' return std::frexp(s, &exp); }}()',
    'frexp_mant_f32': '[&]() {{ int e; return std::frexp(static_cast<float>({0}), &e); }}()',
    'log2': 'amdgpu::transcendental::log_f32({0})',
    'cvt_f32_i32': 'std::bit_cast<uint32_t>(static_cast<float>(static_cast<int32_t>({0})))',
    'cvt_f32_u32': 'std::bit_cast<uint32_t>(static_cast<float>({0}))',
    'cvt_i32_f32': '[&]() -> uint32_t {{ float s = std::bit_cast<float>(static_cast<uint32_t>({0}));'
    ' if (std::isnan(s)) return 0u;'
    ' if (s >= 2147483648.0f) return static_cast<uint32_t>(INT32_MAX);'
    ' if (s < -2147483648.0f) return static_cast<uint32_t>(INT32_MIN);'
    ' return static_cast<uint32_t>(static_cast<int32_t>(s)); }}()',
    'cvt_u32_f32': '[&]() -> uint32_t {{ float s = std::bit_cast<float>(static_cast<uint32_t>({0}));'
    ' if (std::isnan(s) || s < 0.0f) return 0u;'
    ' if (s >= 4294967296.0f) return UINT32_MAX;'
    ' return static_cast<uint32_t>(s); }}()',
    'cvt_f16_f32': 'util::f32_to_f16(std::bit_cast<float>(static_cast<uint32_t>({0})))',
    'cvt_f32_f16': 'std::bit_cast<uint32_t>(util::f16_to_f32(static_cast<uint16_t>({0})))',
    'cvt_f32_bf16': 'std::bit_cast<uint32_t>(util::bf16_to_f32(static_cast<uint16_t>({0})))',
    'cvt_f32_fp8': 'std::bit_cast<uint32_t>(util::fp8_e4m3_to_f32(static_cast<uint8_t>({0})))',
    'cvt_f32_bf8': 'std::bit_cast<uint32_t>(util::bf8_e5m2_to_f32(static_cast<uint8_t>({0})))',
    'cvt_f64_i32': 'std::bit_cast<uint64_t>(static_cast<double>(static_cast<int32_t>({0})))',
    'cvt_f64_u32': 'std::bit_cast<uint64_t>(static_cast<double>({0}))',
    'cvt_i32_f64': '[&]() -> uint32_t {{ double s = std::bit_cast<double>(static_cast<uint64_t>({0}));'
    ' if (std::isnan(s)) return 0;'
    ' if (s >= 2147483648.0) return static_cast<uint32_t>(INT32_MAX);'
    ' if (s < -2147483648.0) return static_cast<uint32_t>(INT32_MIN);'
    ' return static_cast<uint32_t>(static_cast<int32_t>(s)); }}()',
    'cvt_u32_f64': '[&]() -> uint32_t {{ double s = std::bit_cast<double>(static_cast<uint64_t>({0}));'
    ' if (std::isnan(s) || s < 0.0) return 0u;'
    ' if (s >= 4294967296.0) return UINT32_MAX;'
    ' return static_cast<uint32_t>(s); }}()',
    'cvt_f64_f32': 'std::bit_cast<uint64_t>(static_cast<double>(std::bit_cast<float>(static_cast<uint32_t>({0}))))',
    'cvt_f32_f64': 'std::bit_cast<uint32_t>(static_cast<float>(std::bit_cast<double>(static_cast<uint64_t>({0}))))',
    'cvt_f16_u16': 'util::f32_to_f16(static_cast<float>(static_cast<uint16_t>({0})))',
    'cvt_f16_i16': 'util::f32_to_f16(static_cast<float>(static_cast<int16_t>({0} & 0xFFFF)))',
    'cvt_u16_f16': '[&]() -> uint32_t {{ float s = util::f16_to_f32(static_cast<uint16_t>({0}));'
    ' if (std::isnan(s) || s < 0.0f) return 0u;'
    ' if (s >= 65536.0f) return static_cast<uint32_t>(UINT16_MAX);'
    ' return static_cast<uint32_t>(static_cast<uint16_t>(s)); }}()',
    'cvt_i16_f16': '[&]() -> uint32_t {{ float s = util::f16_to_f32(static_cast<uint16_t>({0}));'
    ' if (std::isnan(s)) return 0u;'
    ' if (s >= 32768.0f) return static_cast<uint32_t>(static_cast<uint16_t>(INT16_MAX));'
    ' if (s < -32768.0f) return static_cast<uint32_t>(static_cast<uint16_t>(INT16_MIN));'
    ' return static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(s))); }}()',
    'cvt_i32_i16': 'static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>({0} & 0xFFFF)))',
    'cvt_u32_u16': '({0} & 0xFFFFu)',
    'cvt_f32_ubyte0': 'std::bit_cast<uint32_t>(static_cast<float>({0} & 0xFFu))',
    'cvt_f32_ubyte1': 'std::bit_cast<uint32_t>(static_cast<float>(({0} >> 8) & 0xFFu))',
    'cvt_f32_ubyte2': 'std::bit_cast<uint32_t>(static_cast<float>(({0} >> 16) & 0xFFu))',
    'cvt_f32_ubyte3': 'std::bit_cast<uint32_t>(static_cast<float>(({0} >> 24) & 0xFFu))',
    'cvt_rpi_i32_f32': '[&]() -> uint32_t {{ float s = std::bit_cast<float>(static_cast<uint32_t>({0}));'
    ' float r = std::ceil(s - 0.5f);'
    ' if (std::isnan(r)) return 0u;'
    ' if (r >= 2147483648.0f) return static_cast<uint32_t>(INT32_MAX);'
    ' if (r < -2147483648.0f) return static_cast<uint32_t>(INT32_MIN);'
    ' return static_cast<uint32_t>(static_cast<int32_t>(r)); }}()',
    'cvt_flr_i32_f32': '[&]() -> uint32_t {{ float s = std::bit_cast<float>(static_cast<uint32_t>({0}));'
    ' float r = std::floor(s);'
    ' if (std::isnan(r)) return 0u;'
    ' if (r >= 2147483648.0f) return static_cast<uint32_t>(INT32_MAX);'
    ' if (r < -2147483648.0f) return static_cast<uint32_t>(INT32_MIN);'
    ' return static_cast<uint32_t>(static_cast<int32_t>(r)); }}()',
    'cvt': '{0}',
}

_INLINE_BINARY_OPS: dict[str, str] = {
    'mul_hi': '[&]() {{ auto a = static_cast<uint64_t>({0});'
    ' auto b = static_cast<uint64_t>({1});'
    ' return static_cast<uint32_t>((a * b) >> 32); }}()',
    'util::mulhi': '[&]() {{ auto a = static_cast<uint64_t>({0});'
    ' auto b = static_cast<uint64_t>({1});'
    ' return static_cast<uint32_t>((a * b) >> 32); }}()',
    'mulhi': '[&]() {{ auto a = static_cast<uint64_t>({0});'
    ' auto b = static_cast<uint64_t>({1});'
    ' return static_cast<uint32_t>((a * b) >> 32); }}()',
    'util::arithmetic_shr': '[&]() {{ auto v = static_cast<int32_t>({0});'
    ' return static_cast<uint32_t>(v >> ({1} & 31u)); }}()',
    'util::arithmetic_shr_i16': '[&]() {{ auto v = static_cast<int16_t>({0});'
    ' return static_cast<uint32_t>(static_cast<uint16_t>(v >> ({1} & 15u))); }}()',
    'bitset0': '({0} & ~(1u << ({0} & 31u)))',
    'bitset1': '({0} | (1u << ({0} & 31u)))',
    'bfm': '(((1u << ({0} & 31u)) - 1u) << ({1} & 31u))',
    'bfm64': '[&]() {{ uint64_t cnt = {0} & 63u; uint64_t off = {1} & 63u;'
    ' return cnt == 0 ? 0ULL : ((1ULL << cnt) - 1ULL) << off; }}()',
    'ashr': '[&]() {{ auto v = static_cast<int32_t>({0});'
    ' return static_cast<uint32_t>(v >> ({1})); }}()',
    'ashr_i64': '[&]() {{ auto v = static_cast<int64_t>({0});'
    ' return static_cast<uint64_t>(v >> ({1} & 63u)); }}()',
    'mul_legacy': '[&]() {{ auto a = {0}; auto b = {1};'
    ' return (a == 0.0f || b == 0.0f) ? 0.0f : a * b; }}()',
    'mul_hi_u24': '[&]() {{ auto a = static_cast<uint64_t>({0} & 0x00FFFFFFu);'
    ' auto b = static_cast<uint64_t>({1} & 0x00FFFFFFu);'
    ' return static_cast<uint32_t>((a * b) >> 32); }}()',
    'mul_hi_i24': '[&]() {{ auto a = static_cast<int64_t>(static_cast<int32_t>({0} << 8) >> 8);'
    ' auto b = static_cast<int64_t>(static_cast<int32_t>({1} << 8) >> 8);'
    ' return static_cast<uint32_t>(static_cast<uint64_t>((a * b) >> 32)); }}()',
    'mul_u24': '(({0} & 0x00FFFFFFu) * ({1} & 0x00FFFFFFu))',
    'mul_i24': '[&]() {{ auto a = static_cast<int32_t>({0} << 8) >> 8;'
    ' auto b = static_cast<int32_t>({1} << 8) >> 8;'
    ' return static_cast<uint32_t>(a * b); }}()',
    'mul_lo_u16': 'static_cast<uint32_t>(static_cast<uint16_t>('
    'static_cast<uint16_t>({0}) * static_cast<uint16_t>({1})))',
    'std::min': 'std::min({0}, {1})',
    'std::max': 'std::max({0}, {1})',
    'std::fmin': 'std::fmin({0}, {1})',
    'std::fmax': 'std::fmax({0}, {1})',
    'is_ordered': '(!std::isnan({0}) && !std::isnan({1}))',
    'is_unordered': '(std::isnan({0}) || std::isnan({1}))',
    'fp_class_test': '[&]() -> bool {{'
    ' float s0 = {0};'
    ' uint32_t mask = static_cast<uint32_t>({1}); bool match = false;'
    ' if ((mask & 0x001u) && std::isnan(s0)'
    ' && (std::bit_cast<uint32_t>(s0) & 0x00400000u) == 0) match = true;'
    ' if ((mask & 0x002u) && std::isnan(s0)'
    ' && (std::bit_cast<uint32_t>(s0) & 0x00400000u) != 0) match = true;'
    ' if ((mask & 0x004u) && std::isinf(s0) && s0 < 0) match = true;'
    ' if ((mask & 0x008u) && std::isnormal(s0) && s0 < 0) match = true;'
    ' if ((mask & 0x010u) && !std::isnormal(s0) && !std::isinf(s0)'
    ' && !std::isnan(s0) && s0 != 0.0f && std::signbit(s0)) match = true;'
    ' if ((mask & 0x020u) && s0 == 0.0f && std::signbit(s0)) match = true;'
    ' if ((mask & 0x040u) && s0 == 0.0f && !std::signbit(s0)) match = true;'
    ' if ((mask & 0x080u) && !std::isnormal(s0) && !std::isinf(s0)'
    ' && !std::isnan(s0) && s0 != 0.0f && !std::signbit(s0)) match = true;'
    ' if ((mask & 0x100u) && std::isnormal(s0) && s0 > 0) match = true;'
    ' if ((mask & 0x200u) && std::isinf(s0) && s0 > 0) match = true;'
    ' return match; }}()',
    'fp_class_test_f64': '[&]() -> bool {{'
    ' double s0 = {0};'
    ' uint32_t mask = static_cast<uint32_t>({1}); bool match = false;'
    ' if ((mask & 0x001u) && std::isnan(s0)'
    ' && (std::bit_cast<uint64_t>(s0) & 0x0008000000000000ULL) == 0) match = true;'
    ' if ((mask & 0x002u) && std::isnan(s0)'
    ' && (std::bit_cast<uint64_t>(s0) & 0x0008000000000000ULL) != 0) match = true;'
    ' if ((mask & 0x004u) && std::isinf(s0) && s0 < 0) match = true;'
    ' if ((mask & 0x008u) && std::isnormal(s0) && s0 < 0) match = true;'
    ' if ((mask & 0x010u) && !std::isnormal(s0) && !std::isinf(s0)'
    ' && !std::isnan(s0) && s0 != 0.0 && std::signbit(s0)) match = true;'
    ' if ((mask & 0x020u) && s0 == 0.0 && std::signbit(s0)) match = true;'
    ' if ((mask & 0x040u) && s0 == 0.0 && !std::signbit(s0)) match = true;'
    ' if ((mask & 0x080u) && !std::isnormal(s0) && !std::isinf(s0)'
    ' && !std::isnan(s0) && s0 != 0.0 && !std::signbit(s0)) match = true;'
    ' if ((mask & 0x100u) && std::isnormal(s0) && s0 > 0) match = true;'
    ' if ((mask & 0x200u) && std::isinf(s0) && s0 > 0) match = true;'
    ' return match; }}()',
    'add_co': '[&]() {{ uint64_t w = static_cast<uint64_t>({0})'
    ' + static_cast<uint64_t>({1});'
    ' if (w > 0xFFFFFFFFULL) vcc |= (1ULL << lane);'
    ' else vcc &= ~(1ULL << lane);'
    ' return static_cast<uint32_t>(w); }}()',
    'sub_co': '[&]() {{ uint32_t a = {0}, b = {1};'
    ' if (a < b) vcc |= (1ULL << lane);'
    ' else vcc &= ~(1ULL << lane);'
    ' return a - b; }}()',
    'addc': '[&]() {{ uint64_t w = static_cast<uint64_t>({0})'
    ' + static_cast<uint64_t>({1})'
    ' + static_cast<uint64_t>(wf.read_scc());'
    ' wf.write_scc(w > 0xFFFFFFFFULL);'
    ' return static_cast<uint32_t>(w); }}()',
    'subb': '[&]() {{ uint32_t a = {0}, b = {1};'
    ' uint32_t cin = wf.read_scc() ? 1u : 0u;'
    ' wf.write_scc(static_cast<uint64_t>(a) < static_cast<uint64_t>(b) + cin);'
    ' return a - b - cin; }}()',
    'lshl1_add': '[&]() {{ uint64_t w = (static_cast<uint64_t>({0}) << 1u) + static_cast<uint64_t>({1});'
    ' wf.write_scc(w > 0xFFFFFFFFULL);'
    ' return static_cast<uint32_t>(w); }}()',
    'lshl2_add': '[&]() {{ uint64_t w = (static_cast<uint64_t>({0}) << 2u) + static_cast<uint64_t>({1});'
    ' wf.write_scc(w > 0xFFFFFFFFULL);'
    ' return static_cast<uint32_t>(w); }}()',
    'lshl3_add': '[&]() {{ uint64_t w = (static_cast<uint64_t>({0}) << 3u) + static_cast<uint64_t>({1});'
    ' wf.write_scc(w > 0xFFFFFFFFULL);'
    ' return static_cast<uint32_t>(w); }}()',
    'lshl4_add': '[&]() {{ uint64_t w = (static_cast<uint64_t>({0}) << 4u) + static_cast<uint64_t>({1});'
    ' wf.write_scc(w > 0xFFFFFFFFULL);'
    ' return static_cast<uint32_t>(w); }}()',
    'pack_ll': '(({0} & 0xFFFFu) | (({1} & 0xFFFFu) << 16))',
    'pack_lh': '(({0} & 0xFFFFu) | ({1} & 0xFFFF0000u))',
    'pack_hh': '((({0} >> 16) & 0xFFFFu) | ({1} & 0xFFFF0000u))',
    'bfe': '[&]() {{ uint32_t base = {0}, field = {1};'
    ' uint32_t offset = field & 31u;'
    ' uint32_t width = (field >> 16) & 127u;'
    ' if (width == 0) return 0u;'
    ' uint32_t mask = width >= 32 ? ~0u : ((1u << width) - 1u);'
    ' return (base >> offset) & mask; }}()',
    'bfe_i32': '[&]() {{ uint32_t base = {0}, field = {1};'
    ' uint32_t offset = field & 31u;'
    ' uint32_t width = (field >> 16) & 127u;'
    ' if (width == 0) return 0u;'
    ' uint32_t mask = width >= 32 ? ~0u : ((1u << width) - 1u);'
    ' uint32_t extracted = (base >> offset) & mask;'
    ' if (width < 32 && (extracted & (1u << (width - 1))))'
    '   extracted |= ~mask;'
    ' return extracted; }}()',
    'bfe64': '[&]() {{ uint64_t base = {0};'
    ' uint32_t field = static_cast<uint32_t>({1});'
    ' uint32_t offset = field & 63u;'
    ' uint32_t width = (field >> 16) & 127u;'
    ' if (width == 0) return static_cast<uint64_t>(0);'
    ' uint64_t mask = width >= 64 ? ~0ULL : ((1ULL << width) - 1ULL);'
    ' return (base >> offset) & mask; }}()',
    'bfe_i64': '[&]() {{ uint64_t base = {0};'
    ' uint32_t field = static_cast<uint32_t>({1});'
    ' uint32_t offset = field & 63u;'
    ' uint32_t width = (field >> 16) & 127u;'
    ' if (width == 0) return static_cast<int64_t>(0);'
    ' uint64_t mask = width >= 64 ? ~0ULL : ((1ULL << width) - 1ULL);'
    ' uint64_t extracted = (base >> offset) & mask;'
    ' if (width < 64 && (extracted & (1ULL << (width - 1))))'
    '   extracted |= ~mask;'
    ' return static_cast<int64_t>(extracted); }}()',
    'ABSDIFF': '[&]() {{ auto a = static_cast<int64_t>(static_cast<int32_t>({0}));'
    ' auto b = static_cast<int64_t>(static_cast<int32_t>({1}));'
    ' return static_cast<uint32_t>(a > b ? a - b : b - a); }}()',
    'pack_b32_f16': '(({0} & 0xFFFFu) | (({1} & 0xFFFFu) << 16))',
    'v_readlane': '{0}',
}

_INLINE_TERNARY_OPS: dict[str, str] = {
    'min3': 'std::fmin(std::fmin({0}, {1}), {2})',
    'max3': 'std::fmax(std::fmax({0}, {1}), {2})',
    'med3': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' return std::fmax(std::fmin(std::fmax(a, b), c), std::fmin(a, b)); }}()',
    'lerp_u8': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' uint32_t r = 0;'
    ' for (int i = 0; i < 4; ++i) {{'
    ' uint8_t ab = (a >> (i*8)) & 0xFF;'
    ' uint8_t bb = (b >> (i*8)) & 0xFF;'
    ' uint8_t cb = (c >> (i*8)) & 0xFF;'
    ' r |= static_cast<uint32_t>(ab + ((bb - ab) * cb + 128) / 256) << (i*8);'
    ' }} return r; }}()',
    'add3': '({0} + {1} + {2})',
    'or3': '({0} | {1} | {2})',
    'xor3': '({0} ^ {1} ^ {2})',
    'xad': '(({0} ^ {1}) + {2})',
    'lshl_add': '(({0} << {1}) + {2})',
    'lshl_or': '(({0} << {1}) | {2})',
    'add_lshl': '(({0} + {1}) << {2})',
    'and_or': '(({0} & {1}) | {2})',
    'bfi': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' return (a & b) | (~a & c); }}()',
    'bfm': '(((1u << ({0} & 31u)) - 1u) << ({1} & 31u))',
    'alignbit': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' uint64_t val = (static_cast<uint64_t>(a) << 32) | b;'
    ' return static_cast<uint32_t>(val >> (c & 31u)); }}()',
    'alignbyte': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' uint64_t val = (static_cast<uint64_t>(a) << 32) | b;'
    ' return static_cast<uint32_t>(val >> ((c & 3u) * 8u)); }}()',
    'sad_u8': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' uint32_t r = c;'
    ' for (int i = 0; i < 4; ++i) {{'
    ' uint8_t ab = (a >> (i*8)) & 0xFF; uint8_t bb = (b >> (i*8)) & 0xFF;'
    ' r += (ab > bb) ? (ab - bb) : (bb - ab);'
    ' }} return r; }}()',
    'sad_hi_u8': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' uint32_t r = 0;'
    ' for (int i = 0; i < 4; ++i) {{'
    ' uint8_t ab = (a >> (i*8)) & 0xFF; uint8_t bb = (b >> (i*8)) & 0xFF;'
    ' r += (ab > bb) ? (ab - bb) : (bb - ab);'
    ' }} return (r << 16) + c; }}()',
    'sad_u16': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' uint16_t al = a & 0xFFFF; uint16_t ah = (a >> 16) & 0xFFFF;'
    ' uint16_t bl = b & 0xFFFF; uint16_t bh = (b >> 16) & 0xFFFF;'
    ' return c + ((al > bl) ? (al - bl) : (bl - al))'
    ' + ((ah > bh) ? (ah - bh) : (bh - ah)); }}()',
    'sad_u32': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' return c + ((a > b) ? (a - b) : (b - a)); }}()',
    'msad_u8': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' uint32_t r = c;'
    ' for (int i = 0; i < 4; ++i) {{'
    ' uint8_t rb = (a >> (i*8)) & 0xFF;'
    ' if (rb != 0) {{'
    ' uint8_t sb = (b >> (i*8)) & 0xFF;'
    ' r += (rb > sb) ? (rb - sb) : (sb - rb);'
    ' }}}} return r; }}()',
    'perm': '[&]() {{ auto a={0}; auto b={1}; auto c={2};'
    ' uint32_t r = 0; uint64_t src = (static_cast<uint64_t>(a) << 32) | b;'
    ' for (int i = 0; i < 4; ++i) {{'
    ' uint8_t sel = (c >> (i*8)) & 0xFF;'
    ' uint8_t byte = (sel < 8) ? static_cast<uint8_t>((src >> (sel*8)) & 0xFF)'
    ' : (sel == 0xC) ? 0u : (sel == 0xD) ? 0xFFu : 0u;'
    ' r |= static_cast<uint32_t>(byte) << (i*8);'
    ' }} return r; }}()',
    'minimum3': 'std::fmin(std::fmin({0}, {1}), {2})',
    'maximum3': 'std::fmax(std::fmax({0}, {1}), {2})',
    'maxmin': 'std::fmin(std::fmax({0}, {1}), {2})',
    'minmax': 'std::fmax(std::fmin({0}, {1}), {2})',
    'maxmin_num': 'std::fmin(std::fmax({0}, {1}), {2})',
    'minmax_num': 'std::fmax(std::fmin({0}, {1}), {2})',
    'maximumminimum': 'std::fmin(std::fmax({0}, {1}), {2})',
    'minimummaximum': 'std::fmax(std::fmin({0}, {1}), {2})',
    'addc_co': '[&]() {{ uint64_t w = static_cast<uint64_t>({0})'
    ' + static_cast<uint64_t>({1})'
    ' + static_cast<uint64_t>({2});'
    ' if (w > 0xFFFFFFFFULL) vcc |= (1ULL << lane);'
    ' else vcc &= ~(1ULL << lane);'
    ' return static_cast<uint32_t>(w); }}()',
    'dot2c': '[&]() {{ uint32_t a = {0}, b = {1};'
    ' float c = std::bit_cast<float>(static_cast<uint32_t>({2}));'
    ' float lo = util::f16_to_f32(static_cast<uint16_t>(a))'
    ' * util::f16_to_f32(static_cast<uint16_t>(b));'
    ' float hi = util::f16_to_f32(static_cast<uint16_t>(a >> 16))'
    ' * util::f16_to_f32(static_cast<uint16_t>(b >> 16));'
    ' return std::bit_cast<uint32_t>(lo + hi + c); }}()',
    'dot4c': '[&]() {{ uint32_t a = {0}, b = {1}; int32_t c = static_cast<int32_t>({2});'
    ' int32_t sum = c;'
    ' for (int i = 0; i < 4; ++i)'
    '   sum += static_cast<int32_t>(static_cast<int8_t>((a >> (i*8)) & 0xFF))'
    '        * static_cast<int32_t>(static_cast<int8_t>((b >> (i*8)) & 0xFF));'
    ' return static_cast<uint32_t>(sum); }}()',
    'dot8c': '[&]() {{ uint32_t a = {0}, b = {1}; int32_t c = static_cast<int32_t>({2});'
    ' int32_t sum = c;'
    ' for (int i = 0; i < 8; ++i) {{'
    '   int32_t av = static_cast<int32_t>((a >> (i*4)) & 0xF);'
    '   if (av & 8) av |= ~0xF;'
    '   int32_t bv = static_cast<int32_t>((b >> (i*4)) & 0xF);'
    '   if (bv & 8) bv |= ~0xF;'
    '   sum += av * bv; }}'
    ' return static_cast<uint32_t>(sum); }}()',
    'mad_32_16': '[&]() {{ int32_t a = static_cast<int32_t>(static_cast<int16_t>({0}));'
    ' int32_t b = static_cast<int32_t>(static_cast<int16_t>({1}));'
    ' int32_t c = static_cast<int32_t>({2});'
    ' return static_cast<uint32_t>(a * b + c); }}()',
    'v_writelane': '(lane == {2}) ? {1} : {0}',
    'util::bfe': '[&]() {{ uint32_t base = {0};'
    ' uint32_t offset = {1};'
    ' uint32_t width = {2};'
    ' if (width == 0) return 0u;'
    ' uint32_t mask = width >= 32 ? ~0u : ((1u << width) - 1u);'
    ' return (base >> offset) & mask; }}()',
    'subbc_co': '[&]() {{ uint64_t a = static_cast<uint64_t>({0}),'
    ' b = static_cast<uint64_t>({1}),'
    ' c = static_cast<uint64_t>({2});'
    ' if (a < b + c) vcc |= (1ULL << lane);'
    ' else vcc &= ~(1ULL << lane);'
    ' return static_cast<uint32_t>(a - b - c); }}()',
    'mad_u24': '(({0} & 0x00FFFFFFu) * ({1} & 0x00FFFFFFu) + {2})',
    'mad_i24': '[&]() {{ auto a = static_cast<int32_t>({0} << 8) >> 8;'
    ' auto b = static_cast<int32_t>({1} << 8) >> 8;'
    ' return static_cast<uint32_t>(a * b + static_cast<int32_t>({2})); }}()',
    'bfe_u': '[&]() {{ uint32_t src={0}; uint32_t off={1} & 31u; uint32_t w={2} & 31u;'
    ' auto b = static_cast<int32_t>({1} << 8) >> 8;'
    ' return static_cast<uint32_t>(a * b + static_cast<int32_t>({2})); }}()',
    'bfe_u': '[&]() {{ uint32_t src={0}; uint32_t off={1} & 31u; uint32_t w={2} & 31u;'
    ' if (w == 0) return 0u;'
    ' uint32_t mask = (w >= 32) ? ~0u : ((1u << w) - 1u);'
    ' return (src >> off) & mask; }}()',
    'bfe_i': '[&]() -> uint32_t {{ int32_t src=static_cast<int32_t>({0});'
    ' uint32_t off={1} & 31u; uint32_t w={2} & 31u;'
    ' if (w == 0) return 0u; int32_t val = (src >> off) & ((1 << w) - 1);'
    ' if (val & (1 << (w-1))) val |= -(1 << w);'
    ' return static_cast<uint32_t>(val); }}()',
    'cubeid': '[&]() {{ auto x={0}; auto y={1}; auto z={2};'
    ' float ax=std::fabs(x), ay=std::fabs(y), az=std::fabs(z);'
    ' if (ax >= ay && ax >= az) return x >= 0 ? 0.0f : 1.0f;'
    ' if (ay >= ax && ay >= az) return y >= 0 ? 2.0f : 3.0f;'
    ' return z >= 0 ? 4.0f : 5.0f; }}()',
    'cubesc': '[&]() {{ auto x={0}; auto y={1}; auto z={2};'
    ' float ax=std::fabs(x), ay=std::fabs(y), az=std::fabs(z);'
    ' if (ax >= ay && ax >= az) return x >= 0 ? z : -z;'
    ' if (ay >= ax && ay >= az) return x;'
    ' return z >= 0 ? -x : x; }}()',
    'cubetc': '[&]() {{ auto x={0}; auto y={1}; auto z={2};'
    ' float ax=std::fabs(x), ay=std::fabs(y), az=std::fabs(z);'
    ' if (ax >= ay && ax >= az) return -y;'
    ' if (ay >= ax && ay >= az) return y >= 0 ? -z : z;'
    ' return -y; }}()',
    'cubema': '[&]() {{ auto x={0}; auto y={1}; auto z={2};'
    ' float ax=std::fabs(x), ay=std::fabs(y), az=std::fabs(z);'
    ' return 2.0f * std::fmax(ax, std::fmax(ay, az)); }}()',
}


def _lower_call(node: SemaNode, ctx: LoweringContext) -> str:
    """Lower a .call node through inline ops, helper registry, or VOP3 modifiers."""
    callee = node.call_name or ''

    if callee == 'apply_src_mod':
        return _lower_apply_src_mod(node, ctx)
    if callee == 'apply_omod':
        return _lower_apply_omod(node, ctx)
    if callee == 'apply_clamp':
        return _lower_apply_clamp(node, ctx)

    args = [_lower_expr(c, ctx) for c in node.children[1:]]
    args_str = ', '.join(args)

    if len(args) == 1 and callee in _INLINE_UNARY_OPS:
        return _INLINE_UNARY_OPS[callee].format(args[0])
    if len(args) == 2 and callee in _INLINE_BINARY_OPS:
        return _INLINE_BINARY_OPS[callee].format(args[0], args[1])
    if len(args) == 3 and callee in _INLINE_TERNARY_OPS:
        return _INLINE_TERNARY_OPS[callee].format(args[0], args[1], args[2])

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


def _lower_apply_src_mod(node: SemaNode, ctx: LoweringContext) -> str:
    """Lower apply_src_mod CALL to inline VOP3 NEG/ABS code.

    VOP3 modifiers always interpret bits as float. The lambda bit_casts
    the input to float, applies abs/neg, and returns float (or double
    for F64 operations).

    CALL children: [ID('apply_src_mod'), src_expr, LIT(src_idx),
                     LIT(has_neg), LIT(has_abs)]
    """
    if len(node.children) < 5:
        return _lower_expr(node.children[1], ctx) if len(node.children) > 1 else '0'

    src_expr = _lower_expr(node.children[1], ctx)
    src_idx = node.children[2].lit_value or '0'
    has_neg = node.children[3].lit_value == '1'
    has_abs = node.children[4].lit_value == '1'

    if not has_neg and not has_abs:
        return src_expr

    src_child = node.children[1]
    src_ty = src_child.ty
    if src_ty and src_ty.base in ('I', 'U'):
        return src_expr

    is_64 = node.ty and node.ty.size == 64
    fp_type = 'double' if is_64 else 'float'

    src_is_raw = src_child.kind == SemaNodeKind.INSTOPERAND or (
        src_child.kind == SemaNodeKind.CAST
        and src_child.children
        and src_child.children[0].kind == SemaNodeKind.INSTOPERAND
        and src_child.ty
        and src_child.ty.base == 'B'
    )
    if src_is_raw and node.ty and node.ty.size in (32, 64):
        init = f'std::bit_cast<{fp_type}>({src_expr})'
    else:
        init = src_expr
    parts = [f'[&]() {{ {fp_type} sv = {init};']
    if has_abs:
        parts.append(f' if (inst_.abs & (1u << {src_idx})) sv = std::fabs(sv);')
    if has_neg:
        parts.append(f' if (inst_.neg & (1u << {src_idx})) sv = -sv;')
    parts.append(' return sv; }()')
    return ''.join(parts)


def _lower_apply_omod(node: SemaNode, ctx: LoweringContext) -> str:
    """Lower apply_omod CALL to inline VOP3 OMOD code.

    Operates in float domain: bit_cast input to float/double, apply omod,
    returns float/double.

    CALL children: [ID('apply_omod'), rhs_expr]
    """
    if len(node.children) < 2:
        return '0'
    rhs = _lower_expr(node.children[1], ctx)
    is_f64 = node.ty and node.ty.size == 64
    fp_type = 'double' if is_f64 else 'float'
    suffix = '' if is_f64 else 'f'
    return (
        f'[&]() {{ {fp_type} v = {rhs};'
        f' if (inst_.omod == 1) v *= 2.0{suffix};'
        f' else if (inst_.omod == 2) v *= 4.0{suffix};'
        f' else if (inst_.omod == 3) v *= 0.5{suffix};'
        f' return v; }}()'
    )


def _lower_apply_clamp(node: SemaNode, ctx: LoweringContext) -> str:
    """Lower apply_clamp CALL to inline VOP3 CLAMP code.

    Operates in float domain: bit_cast input to float/double, apply clamp,
    returns float/double. The final bit_cast back to uint32_t/uint64_t
    happens at the destination write site.

    CALL children: [ID('apply_clamp'), rhs_expr]
    """
    if len(node.children) < 2:
        return '0'
    rhs = _lower_expr(node.children[1], ctx)
    is_f64 = node.ty and node.ty.size == 64
    fp_type = 'double' if is_f64 else 'float'
    suffix = '' if is_f64 else 'f'
    return (
        f'[&]() {{ {fp_type} v = {rhs};'
        f' if (inst_.clamp) v = std::clamp(v, 0.0{suffix}, 1.0{suffix});'
        f' return v; }}()'
    )
