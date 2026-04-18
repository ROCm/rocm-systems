# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""AST transformation and expression-building helpers."""

from __future__ import annotations

import ast
import re

import astunparse

from utils import schema
from utils.utils_common import SUPPORTED_FIELD
from utils.utils_counter_defs import SUPPORTED_DENOM

SUPPORTED_CALL: dict[str, str] = {
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
    # Concat operation from the memory chart "active cus"
    "CONCAT": "to_concat",
    # Threshold-based clamping for multi-pass profiling noise
    "NOISE_CLAMP": "to_noise_clamp",
}


class CodeTransformer(ast.NodeTransformer):
    """Python AST visitor to transform user equation strings to df format."""

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in SUPPORTED_CALL:
                node.func.id = SUPPORTED_CALL[node.func.id]
            else:
                raise Exception("Unknown call:", node.func.id)
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.Expr:
        self.generic_visit(node)

        if isinstance(node.body, ast.Constant):
            raise Exception(
                "Don't support body of IF with number only! Has to be expr with "
                "df['column']."
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
    def visit_Name(self, node: ast.Name) -> ast.Name | ast.Subscript:
        self.generic_visit(node)
        if (not node.id.startswith("ammolite__")) and (not node.id in SUPPORTED_CALL):
            return ast.Subscript(
                value=ast.Name(id="raw_pmc_df", ctx=ast.Load()),
                slice=ast.Constant(value=node.id),
                ctx=ast.Load(),
            )
        return node


def build_eval_string(equation: str, coll_level: str, config: dict) -> str:
    """
    Convert user defined equation string to eval executable string.
    For example,
        input:
            100 * SUM(SQ_ACTIVE_INST_SCA) / SUM(GRBM_GUI_ACTIVE * $numCU)
        output:
            100 * to_sum(
                raw_pmc_df["pmc_perf"]["SQ_ACTIVE_INST_SCA"]
            ) / to_sum(
                raw_pmc_df["pmc_perf"]["GRBM_GUI_ACTIVE"] *
                numCU
            )
        input:
            SUM(TCC_EA_RDREQ_LEVEL_31) / SUM(TCC_EA_RDREQ_31)
        output:
            to_sum(
                raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_LEVEL_31"]
            ) / to_sum(
                raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"]
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
                raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"].where(
                    raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"] == 0,
                    raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_LEVEL_31"] /
                    raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"]
                )
            )
    """
    if coll_level is None:
        raise Exception("Error: coll_level can not be None.")

    if not equation:
        return ""

    equation_string = str(equation)

    # build-in variable starts with '$', python can not handle it.
    # replace '$' with 'ammolite__'.
    equation_string = re.sub(r"\$", "ammolite__", equation_string)

    # convert equation string to intermediate expression in df array format
    ast_node = ast.parse(equation_string)
    transformer = CodeTransformer()
    transformer.visit(ast_node)

    equation_string = astunparse.unparse(ast_node)

    # correct column name/label in df with [], such as TCC_HIT[0],
    # the target is df['TCC_HIT[0]']
    equation_string = re.sub(r"\'\]\[(\d+)\]", r"[\g<1>]']", equation_string)

    # apply coll_level
    if config.get("format_rocprof_output") == "rocpd":
        # Replace SQ_ACCUM_PREV_HIRES with coll_level_ACCUM then ignore coll_level df
        equation_string = re.sub(
            "SQ_ACCUM_PREV_HIRES", f"{coll_level}_ACCUM", equation_string
        )
        equation_string = re.sub(
            r"raw_pmc_df",
            f"raw_pmc_df['{schema.PMC_PERF_FILE_PREFIX}']",
            equation_string,
        )
    else:
        # Use pmc_perf.csv for all counters
        equation_string = re.sub(
            r"raw_pmc_df",
            f"raw_pmc_df['{schema.PMC_PERF_FILE_PREFIX}']",
            equation_string,
        )
        # Use coll_level csv for SQ_ACCUM_PREV_HIRES counter only
        equation_string = re.sub(
            rf"raw_pmc_df['{schema.PMC_PERF_FILE_PREFIX}']['SQ_ACCUM_PREV_HIRES']",
            f"raw_pmc_df['{coll_level}']['SQ_ACCUM_PREV_HIRES']",
            equation_string,
        )
    return equation_string


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
    ).capitalize()


def gen_counter_list(formula: str) -> tuple[bool, list[str]]:
    function_filter = {
        "MIN": None,
        "MAX": None,
        "AVG": None,
        "ROUND": None,
        "TO_INT": None,
        "GB": None,
        "STD": None,
        "GFLOP": None,
        "GOP": None,
        "OP": None,
        "CU": None,
        "NC": None,
        "UC": None,
        "CC": None,
        "RW": None,
        "GIOP": None,
        "GFLOPs": None,
        "CONCAT": None,
        "MOD": None,
    }

    built_in_counter = [
        "LDS_Per_Workgroup",
        "Grid_Size",
        "Workgroup_Size",
        "Arch_VGPR",
        "Accum_VGPR",
        "SGPR",
        "Scratch_Per_Workitem",
        "Start_Timestamp",
        "End_Timestamp",
    ]

    visited = False
    counters: list[str] = []
    if not isinstance(formula, str):
        return visited, counters
    try:
        tree = ast.parse(
            formula
            .replace("$normUnit", "SQ_WAVES")
            .replace("$denom", "SQ_WAVES")
            .replace(
                "$numActiveCUs",
                "TO_INT(MIN((((ROUND(AVG(((4 * SQ_BUSY_CU_CYCLES) / "
                "$GRBM_GUI_ACTIVE_PER_XCD})), 0) / $maxWavesPerCU) * 8) + "
                "MIN(MOD(ROUND(AVG(((4 * SQ_BUSY_CU_CYCLES) / "
                "$GRBM_GUI_ACTIVE_PER_XCD)), 0), $maxWavesPerCU), 8)), $numCU))",
            )
            .replace("$", "")
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                val = (
                    str(node.id)[:-4] if str(node.id).endswith("_sum") else str(node.id)
                )
                if val.isupper() and val not in function_filter:
                    counters.append(val)
                    visited = True
                if val in built_in_counter:
                    visited = True
    except Exception:
        pass

    return visited, counters


def build_metric_value_string(
    dfs: dict,
    dfs_type: dict,
    normal_unit: str,
    profiling_config: dict,
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
                                    df.at[row_idx_label, "coll_level"],
                                    profiling_config,
                                )

                elif expr.lower() == "unit" or expr.lower() == "units":
                    df[expr] = df[expr].apply(
                        update_normal_unit_string,
                        normal_unit=normal_unit,
                    )
