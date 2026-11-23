# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""AST transformation and expression-building helpers, and the written form of metric
equations."""

import ast
import re
from typing import Dict, FrozenSet, List, Set, Union

from utils.logger import console_warning
from utils.utils_common import SUPPORTED_FIELD
from utils.utils_counter_defs import AMMOLITE_VAR_RE, SUPPORTED_DENOM

RAW_PMC_COLUMN_RE = re.compile(r"raw_pmc_df\[['\"]([^'\"]+)['\"]\]")
# Python has no $ in names, so a $name variable is written with this prefix in Python.
VARIABLE_PREFIX = "ammolite__"
# Functions whose constant factor the written form moves in front of the call:
# SUM(x * 2) is written 2 * SUM(x).
AGGREGATION_FUNCTIONS = frozenset({"SUM", "MIN", "MAX", "AVG"})
# Text and binding strength of each operator of an equation. The binding strength
# decides where the written form needs parentheses.
BINARY_OPERATORS = {
    ast.Add: ("+", 1),
    ast.Sub: ("-", 1),
    ast.Mult: ("*", 2),
    ast.Div: ("/", 2),
}
# A number of an equation, such as 128, 0.5 or 1e9.
NUMBER_RE = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
# Marks a per-channel template such as TCC_HIT[::_1]. Analyze expands the template
# once for each channel, so normalize_equation() leaves it as written.
CHANNEL_PLACEHOLDER = "::_"


class InvalidExpressionError(ValueError):
    """Raised when a metric formula contains disallowed AST constructs."""


# ast.Attribute is deliberately omitted — it enables dunder-chain escapes.
ALLOWED_AST_NODES: FrozenSet[type] = frozenset({
    ast.Module,
    ast.Expr,
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Tuple,
    ast.List,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.Call,
    ast.keyword,
    ast.Subscript,
    ast.Slice,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitOr,
    ast.BitAnd,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
    ast.Invert,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.Gt,
    ast.LtE,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.And,
    ast.Or,
})

SUPPORTED_CALL: Dict[str, str] = {
    # If the below has a single arg, like(expr), it is an aggr,
    # in which case it turns into a pandas function.
    # If it has args like a list [], it turns into a Python function.
    "MIN": "to_min",
    "MAX": "to_max",
    # simple aggr
    "AVG": "to_avg",
    "MEDIAN": "to_median",
    "STD": "to_std",
    # functions apply to whole column of df or a single value
    "TO_INT": "to_int",
    "SUM": "to_sum",
    # Support the below with 2 inputs
    "ROUND": "to_round",
    "QUANTILE": "to_quantile",
    "MOD": "to_mod",
    # String concatenation operation
    "CONCAT": "to_concat",
}


class CodeTransformer(ast.NodeTransformer):
    """Python AST visitor to transform user equation strings to df format."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if type(node) not in ALLOWED_AST_NODES:
            raise InvalidExpressionError(
                f"Disallowed expression element: {type(node).__name__}"
            )
        return super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.Call:
        if isinstance(node.func, ast.Name):
            if node.func.id not in SUPPORTED_CALL:
                raise InvalidExpressionError(f"Unsupported call: {node.func.id}")
            self.generic_visit(node)
            node.func.id = SUPPORTED_CALL[node.func.id]
        else:
            self.generic_visit(node)
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.Expr:
        self.generic_visit(node)

        if isinstance(node.body, ast.Constant):
            raise InvalidExpressionError(
                "IF body must contain a df['column'] expression, not a literal"
            )

        new_node = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=node.body, attr="where", ctx=ast.Load()),
                args=[node.test, node.orelse],
                keywords=[],
            )
        )
        return new_node

    # NB:
    # visit_Name is for replacing HW counter to its df expr. In this way, we
    # could support any HW counter names, which is easier than regex.
    #
    # There are 2 limitations:
    #   - It is not straightforward to support types other than simple column
    #     in df, such as [], (). If we need to support those, have to implement
    #     in correct way or work around.
    #   - The 'raw_pmc_df' is hack code. For other data sources, like wavefront
    #     data,We need to think about template or pass it as a parameter.
    def visit_Name(self, node: ast.Name) -> Union[ast.Name, ast.Subscript]:
        self.generic_visit(node)
        if (not node.id.startswith(VARIABLE_PREFIX)) and (
            not node.id in SUPPORTED_CALL
        ):
            return ast.Subscript(
                value=ast.Name(id="raw_pmc_df", ctx=ast.Load()),
                slice=ast.Constant(value=node.id),
                ctx=ast.Load(),
            )
        return node


def transform_expression(
    ast_node: ast.AST,
    source: str,
) -> bool:
    """Run CodeTransformer on *ast_node*, warning on invalid formulas.

    Returns True on success, False when the expression was rejected.
    """
    transformer = CodeTransformer()
    try:
        transformer.visit(ast_node)
    except InvalidExpressionError as exc:
        console_warning(f"Invalid metric expression '{source}': {exc}")
        return False
    return True


def build_eval_string(equation: str) -> str:
    """
    Convert user defined equation string to eval executable string.
    For example,
        input:
            100 * SUM(SQ_ACTIVE_INST_SCA) / SUM(GRBM_GUI_ACTIVE * $numCU)
        output:
            100 * to_sum(
                raw_pmc_df["SQ_ACTIVE_INST_SCA"]
            ) / to_sum(
                raw_pmc_df["GRBM_GUI_ACTIVE"] *
                ammolite__numCU
            )
        input:
            SUM(TCC_EA_RDREQ_LEVEL_31) / SUM(TCC_EA_RDREQ_31)
        output:
            to_sum(
                raw_pmc_df["TCC_EA_RDREQ_LEVEL_31"]
            ) / to_sum(
                raw_pmc_df["TCC_EA_RDREQ_31"]
            )
        We can not handle the below for now:
        input:
            AVG(
                (
                    0
                    if (TCC_EA_RDREQ_31 == 0)
                    else (
                        TCC_EA_RDREQ_LEVEL_31 /
                        TCC_EA_RDREQ_31
                    )
                )
            )
        But potential workaround is:
        output:
            to_avg(
                raw_pmc_df["TCC_EA_RDREQ_31"].where(
                    raw_pmc_df["TCC_EA_RDREQ_31"] == 0,
                    raw_pmc_df["TCC_EA_RDREQ_LEVEL_31"] /
                    raw_pmc_df["TCC_EA_RDREQ_31"]
                )
            )
    """
    if not equation:
        return ""

    equation_string = str(equation)

    # build-in variable starts with '$', python can not handle it.
    # replace '$' with 'ammolite__'.
    equation_string = equation_string.replace("$", VARIABLE_PREFIX)

    # convert equation string to intermediate expression in df array format
    ast_node = ast.parse(equation_string)
    if not transform_expression(ast_node, equation):
        return ""

    equation_string = ast.unparse(ast_node)

    # correct column name/label in df with [], such as TCC_HIT[0],
    # the target is df['TCC_HIT[0]']
    equation_string = re.sub(r"\'\]\[(\d+)\]", r"[\g<1>]']", equation_string)

    return equation_string


def columns_read(eval_string: str, build_in_vars: Dict[str, str]) -> Set[str]:
    """Return the raw_pmc_df columns an eval string reads. Columns that its
    built-in variables read count too."""
    columns = set(RAW_PMC_COLUMN_RE.findall(eval_string))
    pending = set(AMMOLITE_VAR_RE.findall(eval_string))
    seen: Set[str] = set()
    while pending:
        name = pending.pop()
        seen.add(name)
        if name not in build_in_vars:
            continue
        builtin_string = build_eval_string(build_in_vars[name])
        columns.update(RAW_PMC_COLUMN_RE.findall(builtin_string))
        pending.update(set(AMMOLITE_VAR_RE.findall(builtin_string)) - seen)
    return columns


def update_denominator_string(equation: str, normal_unit: str) -> str:
    """Update $denom in equation with runtime normalization unit."""
    if not equation:
        return ""

    equation_string = str(equation)
    if normal_unit in SUPPORTED_DENOM.keys():
        equation_string = re.sub(
            r"\$denom",
            SUPPORTED_DENOM[normal_unit],
            equation_string,
        )
    return equation_string


def update_normal_unit_string(equation: str, normal_unit: str) -> str:
    """
    Update $normUnit in equation with runtime normalization unit.
    It is string replacement for display only.
    """

    # TODO: We might want to do it for subtitle contains $normUnit
    if not equation:
        return ""

    return re.sub(
        r"\((?P<PREFIX>\w*)\s+\+\s+(\$normUnit\))",
        rf"\g<PREFIX> {re.sub('_', ' ', normal_unit)}",
        str(equation),
    )


def build_metric_value_string(
    dfs: dict,
    dfs_type: dict,
    normal_unit: str,
) -> None:
    """Apply the real eval string to its field in the metric_table df."""
    for table_id, df in dfs.items():
        if dfs_type[table_id] == "metric_table":
            for expr in df.columns:
                if expr in SUPPORTED_FIELD:
                    # NB: apply all build-in before building the whole string
                    df[expr] = df[expr].apply(
                        update_denominator_string,
                        normal_unit=normal_unit,
                    )

                    # NB: there should be a faster way to do with single apply
                    if not df.empty:
                        for i in range(df.shape[0]):
                            row_idx_label = df.index.to_list()[i]
                            if expr.lower() != "alias":
                                df.at[row_idx_label, expr] = build_eval_string(
                                    df.at[row_idx_label, expr],
                                )

                elif expr.lower() == "unit" or expr.lower() == "units":
                    df[expr] = df[expr].apply(
                        update_normal_unit_string,
                        normal_unit=normal_unit,
                    )


def normalize_equation(equation: str) -> str:
    """Return a metric equation in the written form of the YAML files, which
    format_equation() describes. A per-channel template stays as written.

    Raises ValueError if the text is not an equation.
    """
    if CHANNEL_PLACEHOLDER in equation:
        return equation
    return format_equation(parse_equation(equation))


def parse_equation(equation: str) -> ast.expr:
    """Return the syntax tree of a metric equation.

    An equation has numbers, names, $name variables, function calls, the operators
    + - * / and a leading minus sign. In the tree a $name variable is a name that
    starts with $, and each number keeps its text in the equation as its text
    attribute. Raises ValueError if the text is not an equation.
    """
    source = " ".join(equation.replace("$", VARIABLE_PREFIX).split())
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise InvalidExpressionError(f"{error.msg} in {equation!r}") from error
    return _EquationReader(source).visit(tree.body)


def format_equation(tree: ast.expr) -> str:
    """Write the syntax tree of a metric equation in the written form of the YAML
    files: each constant factor of an aggregation function in front of the call, one
    space around each operator, and only the parentheses that the order of
    evaluation needs."""
    return _write_equation(_factor_constants(tree))


def _factor_constants(node: ast.expr) -> ast.expr:
    """Return a new tree with the constant factors of the aggregation functions in
    front of their calls."""
    if isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(node.op, _factor_constants(node.operand))
    if isinstance(node, ast.BinOp):
        return ast.BinOp(
            _factor_constants(node.left), node.op, _factor_constants(node.right)
        )
    if isinstance(node, ast.Call):
        args = [_factor_constants(arg) for arg in node.args]
        return _factor_constant(node.func.id, args)
    return node


def _factor_constant(function: str, args: List[ast.expr]) -> ast.expr:
    """Return the call function(args), as 2 * SUM(x) when it is SUM(x * 2), and the
    same for the other aggregation functions.

    Numbers are never negative here, since a minus sign is a unary operation, so the
    factor does not change MIN or MAX.
    """
    call = ast.Call(ast.Name(function, ast.Load()), args, [])
    if function not in AGGREGATION_FUNCTIONS or len(args) != 1:
        return call
    product = args[0]
    if not isinstance(product, ast.BinOp) or not isinstance(product.op, ast.Mult):
        return call
    for factor, rest in ((product.left, product.right), (product.right, product.left)):
        if isinstance(factor, ast.Constant):
            return ast.BinOp(factor, ast.Mult(), _factor_constant(function, [rest]))
    return call


def _write_equation(
    node: ast.expr, parent_binding: int = 0, is_right: bool = False
) -> str:
    """Write a syntax tree as an equation with only the parentheses it needs.

    A binary operation needs them when it binds less strongly than its parent
    operation. A right operand also keeps them when it binds as strongly, such as
    a + (b + c), so the order of evaluation stays the same.
    """
    if isinstance(node, ast.Constant):
        return node.text
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp):
        operand_text = _write_equation(node.operand)
        # -(a + b) keeps its parentheses: -a + b is a different equation
        if isinstance(node.operand, ast.BinOp):
            operand_text = f"({operand_text})"
        return f"-{operand_text}"
    if isinstance(node, ast.BinOp):
        operator, binding = BINARY_OPERATORS[type(node.op)]
        left_text = _write_equation(node.left, binding)
        right_text = _write_equation(node.right, binding, is_right=True)
        operation_text = f"{left_text} {operator} {right_text}"
        if binding < parent_binding or (binding == parent_binding and is_right):
            return f"({operation_text})"
        return operation_text
    args_text = ", ".join(_write_equation(arg) for arg in node.args)
    return f"{node.func.id}({args_text})"


class _EquationReader(ast.NodeTransformer):
    """Check that a Python syntax tree is a metric equation, and give its $name
    variables and numbers their text in the equation."""

    def __init__(self, source: str) -> None:
        self.source = source

    def generic_visit(self, node: ast.AST) -> ast.AST:
        text = ast.get_source_segment(self.source, node).replace(VARIABLE_PREFIX, "$")
        raise InvalidExpressionError(f"{text!r} is not part of an equation")

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        if type(node.op) not in BINARY_OPERATORS:
            return self.generic_visit(node)
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.UnaryOp:
        if not isinstance(node.op, ast.USub):
            return self.generic_visit(node)
        node.operand = self.visit(node.operand)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        if (
            not isinstance(node.func, ast.Name)
            or VARIABLE_PREFIX in node.func.id
            or node.keywords
            or not node.args
        ):
            return self.generic_visit(node)
        node.args = [self.visit(arg) for arg in node.args]
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        name = node.id
        if name.startswith(VARIABLE_PREFIX):
            name = "$" + name[len(VARIABLE_PREFIX) :]
        # A $ inside a name, as in a$b
        if VARIABLE_PREFIX in name:
            return self.generic_visit(node)
        node.id = name
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.expr:
        text = ast.get_source_segment(self.source, node)
        # Python reads the names None, True and False as constants
        if node.value is None or isinstance(node.value, bool):
            return ast.Name(text, ast.Load())
        if not NUMBER_RE.fullmatch(text):
            return self.generic_visit(node)
        # The equation's text of the number, such as 1e9 rather than 1000000000.0
        node.text = text
        return node
