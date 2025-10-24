##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import argparse
import ast
import json
import re
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import astunparse
import numpy as np
import pandas as pd

from utils import schema
from utils.logger import console_debug, console_error, console_warning, demarcate
from utils.specs import MachineSpecs

# ------------------------------------------------------------------------------
# Internal global definitions

# NB:
# Ammolite is unique gemstone from the Rocky Mountains.
# "ammolite__" is a special internal prefix to mark build-in global variables
# calculated or parsed from raw data sources. Its range is only in this file.
# Any other general prefixes string, like "buildin__", might be used by the
# editor. Whenever change it to a new one, replace all appearances in this file.

# 001 is ID of pmc_kernel_top.csv table
PMC_KERNEL_TOP_TABLE_ID: int = 1

# Build-in $denom defined in mongodb query:
#       "denom": {
#              "$switch" : {
#                 "branches": [
#                    {
#                         "case":  { "$eq": [ $normUnit, "per Wave"]} ,
#                         "then":  "&SQ_WAVES"
#                    },
#                    {
#                         "case":  { "$eq": [ $normUnit, "per Cycle"]} ,
#                         "then":  "&GRBM_GUI_ACTIVE"
#                    },
#                    {
#                         "case":  { "$eq": [ $normUnit, "per Sec"]} ,
#                         "then":  {"$divide":[{"$subtract": ["&End_Timestamp",
#                                                              "&Start_Timestamp" ]},
#                                              1000000000]}
#              }
#       }
SUPPORTED_DENOM: dict[str, str] = {
    "per_wave": "SQ_WAVES",
    "per_cycle": "$GRBM_GUI_ACTIVE_PER_XCD",
    "per_second": "((End_Timestamp - Start_Timestamp) / 1000000000)",
    "per_kernel": "1",
}

# Build-in defined in mongodb variables:
BUILD_IN_VARS: dict[str, str] = {
    "GRBM_GUI_ACTIVE_PER_XCD": "(GRBM_GUI_ACTIVE / $num_xcd)",
    "GRBM_COUNT_PER_XCD": "(GRBM_COUNT / $num_xcd)",
    "GRBM_SPI_BUSY_PER_XCD": "(GRBM_SPI_BUSY / $num_xcd)",
    "numActiveCUs": "TO_INT(MIN((((ROUND(AVG(((4 * SQ_BUSY_CU_CYCLES) / \
        $GRBM_GUI_ACTIVE_PER_XCD)), 0) / $max_waves_per_cu) * 8) + \
        MIN(MOD(ROUND(AVG(((4 * SQ_BUSY_CU_CYCLES) / \
        $GRBM_GUI_ACTIVE_PER_XCD)), 0), $max_waves_per_cu), 8)), $cu_per_gpu))",
    "kernelBusyCycles": "ROUND(AVG((((End_Timestamp - Start_Timestamp) / \
        1000) * $max_sclk)), 0)",
    "hbmBandwidth": "($max_mclk / 1000 * 32 * $num_hbm_channels)",
}

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
}

PC_SAMPLING_NOT_ISSUE_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"

# ------------------------------------------------------------------------------


def to_min(*args: Any) -> Union[float, None]:
    if len(args) == 1 and isinstance(args[0], pd.Series):
        return args[0].min()
    elif min(args) is None:
        return np.nan
    else:
        return min(args)


def to_max(*args: Any) -> Union[float, np.ndarray, None]:
    if len(args) == 1 and isinstance(args[0], pd.Series):
        return args[0].max()
    elif len(args) == 2 and (
        isinstance(args[0], pd.Series) or isinstance(args[1], pd.Series)
    ):
        return np.maximum(args[0], args[1])
    elif max(args) == None:
        return np.nan
    else:
        return max(args)


def to_avg(
    a: Union[pd.Series, np.ndarray, list, int, float, str, np.number, None],
) -> Union[float, np.floating, None]:
    if a is None:
        return np.nan
    elif isinstance(a, pd.Series):
        if a.empty:
            return np.nan
        elif np.isnan(a).all():
            return np.nan
        else:
            return a.mean()
    elif isinstance(a, (np.ndarray, list)):
        arr = np.array(a)
        if arr.size == 0:
            return np.nan
        elif np.isnan(arr).all():
            return np.nan
        else:
            return np.nanmean(arr)
    elif isinstance(a, (int, float, np.number)):
        if np.isnan(a):
            return np.nan
        else:
            return float(a)
    elif isinstance(a, str):
        if not a:
            return np.nan
        return float(a)
    else:
        raise Exception(f"to_avg: unsupported type: {type(a)}")


def to_median(a: Union[pd.Series, None]) -> Union[float, None]:
    if a is None:
        return None
    elif isinstance(a, pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return a.median()
    else:
        raise Exception("to_median: unsupported type.")


def to_std(a: pd.Series) -> float:
    if isinstance(a, pd.Series):
        return a.std()
    else:
        raise Exception("to_std: unsupported type.")


def to_int(
    a: Union[int, float, str, np.integer, pd.Series, None],
) -> Union[int, pd.Series, None]:
    if a is None:
        return None
    elif isinstance(a, (int, float, np.integer)):
        return int(a)
    elif isinstance(a, pd.Series):
        return a.astype(int)
    elif isinstance(a, str):
        return int(a)
    else:
        raise Exception("to_int: unsupported type.")


def to_sum(a: Union[pd.Series, None]) -> Union[float, None]:
    if a is None:
        return np.nan
    elif np.isnan(a).all():
        return np.nan
    elif a.empty:
        return np.nan
    elif isinstance(a, pd.Series):
        return a.sum()
    else:
        raise Exception("to_sum: unsupported type.")


def to_round(a: Union[pd.Series, float], b: int) -> Union[pd.Series, float]:
    if isinstance(a, pd.Series):
        return a.round(b)
    else:
        return round(a, b)


def to_quantile(a: Union[pd.Series, None], b: float) -> Union[float, None]:
    if a is None:
        return None
    elif isinstance(a, pd.Series):
        return a.quantile(b)
    else:
        raise Exception("to_quantile: unsupported type.")


def to_mod(
    a: Union[pd.Series, float], b: Union[pd.Series, float]
) -> Union[pd.Series, float]:
    if isinstance(a, pd.Series):
        return a.mod(b)
    else:
        return a % b


def to_concat(a: Any, b: Any) -> str:  # noqa: ANN401
    return str(a) + str(b)


class CodeTransformer(ast.NodeTransformer):
    """
    Python AST visitor to transform user defined equation string to df format
    """

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
    def visit_Name(self, node: ast.Name) -> Union[ast.Name, ast.Subscript]:
        self.generic_visit(node)
        if (not node.id.startswith("ammolite__")) and (not node.id in SUPPORTED_CALL):
            return ast.Subscript(
                value=ast.Name(id="raw_pmc_df", ctx=ast.Load()),
                slice=ast.Constant(value=node.id),
                ctx=ast.Load(),
            )

        return node


class MetricEvaluator:
    """Encapsulates metric evaluation logic and eliminates global variables."""

    def __init__(
        self,
        raw_pmc_df: Union[pd.DataFrame, dict],
        sys_vars: dict[str, Any],
        empirical_peaks: dict[str, Any],
    ) -> None:
        self.raw_pmc_df = raw_pmc_df
        self.sys_vars = sys_vars
        self.empirical_peaks = empirical_peaks
        self._prepare_df_cache()

    def _prepare_df_cache(self) -> None:
        """Prepare cached dataframe access for performance."""
        if isinstance(self.raw_pmc_df, dict):
            self.df_cache = {
                f"raw_pmc_df_{key}": self.raw_pmc_df[key]
                for key in self.raw_pmc_df.keys()
            }
        elif isinstance(self.raw_pmc_df, pd.DataFrame):
            raw_pmc_df_keys = set(self.raw_pmc_df.columns.get_level_values(0))
            self.df_cache = {
                f"raw_pmc_df_{key}": self.raw_pmc_df[key] for key in raw_pmc_df_keys
            }
        else:
            raise ValueError(f'Unknown `raw_pmc_df` type: "{type(self.raw_pmc_df)}".')

    def eval_expression(self, expr: str) -> Union[str, float, int]:
        """Evaluate a single expression with proper local context."""
        try:
            # Optimize dataframe access by replacing dict notation with dir_path
            # variable access
            opt_expr = re.sub(r"raw_pmc_df\['(.*?)'\]", r"raw_pmc_df_\1", expr)

            # Create comprehensive local context
            local_expr_context = {}
            local_expr_context.update(self.df_cache)
            local_expr_context.update(self.sys_vars)
            local_expr_context.update(self.empirical_peaks)

            # Add utility functions to local context
            local_expr_context.update({
                "to_min": to_min,
                "to_max": to_max,
                "to_avg": to_avg,
                "to_median": to_median,
                "to_std": to_std,
                "to_int": to_int,
                "to_sum": to_sum,
                "to_round": to_round,
                "to_quantile": to_quantile,
                "to_mod": to_mod,
                "to_concat": to_concat,
            })

            eval_result = eval(
                compile(opt_expr, "<string>", "eval"),
                {},
                local_expr_context,
            )

            if np.isnan(eval_result):
                return ""
            else:
                return eval_result

        except (TypeError, NameError, KeyError) as exception:
            if "empirical_peak" in str(exception):
                console_warning(
                    f"Missing empirical peak data: {exception}. Using empty value."
                )
                return ""
            else:
                return ""

        except AttributeError as attribute_error:
            if str(attribute_error) == "'NoneType' object has no attribute 'get'":
                return ""
            else:
                console_error("analysis", str(attribute_error))
                return ""


def build_eval_string(equation: str, coll_level: str, config: dict) -> str:
    """
    Convert user defined equation string to eval executable string.
    For example,
        input:
            AVG(100  * SQ_ACTIVE_INST_SCA / ( GRBM_GUI_ACTIVE * $numCU ))
        output:
            to_avg(
                100 * raw_pmc_df["pmc_perf"]["SQ_ACTIVE_INST_SCA"] /
                (
                    raw_pmc_df["pmc_perf"]["GRBM_GUI_ACTIVE"] *
                    numCU
                )
            )
        input:
            AVG(
                (
                    TCC_EA_RDREQ_LEVEL_31 / TCC_EA_RDREQ_31
                )
                if (TCC_EA_RDREQ_31 != 0)
                else (0)
            )
        output:
            to_avg(
                (
                    raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_LEVEL_31"] /
                    raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"]
                ).where(
                    raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"] != 0,
                    0
                )
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
        equation_string = re.sub(
            r"raw_pmc_df", f"raw_pmc_df['{coll_level}']", equation_string
        )
    return equation_string


def update_denominator_string(equation: str, normal_unit: str) -> str:
    """
    Update $denom in equation with runtime normalization unit.
    """
    if not equation:
        return ""

    equation_string = str(equation)

    if normal_unit in SUPPORTED_DENOM.keys():
        equation_string = re.sub(
            r"\$denom", SUPPORTED_DENOM[normal_unit], equation_string
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
    counters = []
    if not isinstance(formula, str):
        return visited, counters
    try:
        tree = ast.parse(
            formula.replace("$normUnit", "SQ_WAVES")
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


def calc_builtin_var(var: Union[int, str], sys_info: pd.Series) -> int:  # type: ignore[return]
    """
    Calculate build-in variable based on sys_info:
    """
    if isinstance(var, int):
        return var
    elif isinstance(var, str) and var.startswith("$total_l2_chan"):
        return int(sys_info.total_l2_chan)
    else:
        console_error(f'Built-in var "{var}" is not supported')


@demarcate
def build_dfs(
    arch_configs: schema.ArchConfig,
    filter_metrics: Optional[list[str]],
    sys_info: pd.Series,
) -> None:
    """
    - Build dataframe for each type of data source within each panel.
      Each dataframe will be used as a template to load data with each run later.
      For now, support "metric_table" and "raw_csv_table". Otherwise, put an empty df.
    - Collect/build metric_list to suport customrized metrics profiling.
    """

    # TODO: more error checking for filter_metrics!!
    simple_box = {
        "Min": ["MIN(", ")"],
        "Q1": ["QUANTILE(", ", 0.25)"],
        "Median": ["MEDIAN(", ")"],
        "Q3": ["QUANTILE(", ", 0.75)"],
        "Max": ["MAX(", ")"],
    }

    dfs = {}
    metric_list = {}
    dfs_type = {}
    metric_counters = {}

    for panel_id, panel in arch_configs.panel_configs.items():
        for data_source in panel["data source"]:
            for type, data_config in data_source.items():
                if (
                    type == "metric_table"
                    and "metric" in data_config
                    and "placeholder_range" in data_config["metric"]
                ):
                    new_metrics = {}
                    if sys_info is not None:
                        # NB: support single placeholder for now!!
                        p_range = data_config["metric"].pop("placeholder_range")
                        metric, metric_expr = data_config["metric"].popitem()

                        for p, r in p_range.items():
                            # NB: We have to resolve placeholder range first if it
                            #   is a build-in var. It will be too late to do it in
                            #   eval_metric(). This is the only reason we need
                            #   sys_info at this stage.
                            var = calc_builtin_var(r, sys_info)
                            for i in range(var):
                                new_key = metric.replace(p, str(i))
                                new_val = {}
                                for k, v in metric_expr.items():
                                    new_val[k] = metric_expr[k].replace(p, str(i))
                                    new_metrics[new_key] = new_val

                    data_config["metric"] = new_metrics

    for panel_id, panel in arch_configs.panel_configs.items():
        for data_source in panel["data source"]:
            for type, data_config in data_source.items():
                if type == "metric_table":
                    headers = ["Metric_ID"]
                    data_source_idx = str(data_config["id"] // 100)

                    if data_source_idx != 0 or (
                        filter_metrics and data_source_idx in filter_metrics
                    ):
                        metric_list[data_source_idx] = panel["title"]
                    if (
                        "cli_style" in data_config
                        and data_config["cli_style"] == "simple_box"
                    ):
                        headers.append(data_config["header"]["metric"])
                        for k in simple_box.keys():
                            headers.append(k)

                        for key, tile in data_config["header"].items():
                            if key != "metric" and key != "expr":
                                headers.append(tile)
                    else:
                        headers.append(data_config["header"]["metric"])
                        for key, tile in data_config["header"].items():
                            if key != "metric":
                                headers.append(tile)

                    headers.append("coll_level")

                    # Only add Metrics Description column if it is defined in the panel
                    if "metrics_description" in panel:
                        headers.append("Description")

                    df = pd.DataFrame(columns=headers)

                    if not data_config["metric"]:
                        data_source_idx = (
                            f"{data_config['id'] // 100}.{data_config['id'] % 100}"
                        )
                        metric_list[data_source_idx] = data_config["title"]

                    for i, (key, entries) in enumerate(data_config["metric"].items()):
                        data_source_idx = (
                            f"{data_config['id'] // 100}.{data_config['id'] % 100}"
                        )
                        metric_idx = f"{data_source_idx}.{i}"
                        eqn_content = []

                        if (
                            (not filter_metrics)
                            or (
                                metric_idx in filter_metrics
                            )  # no filter  # metric in filter
                            or
                            # the whole table in filter
                            (data_source_idx in filter_metrics)
                            or
                            # the whole IP block in filter
                            (str(panel_id // 100) in filter_metrics)
                        ):
                            values = [metric_idx, key]

                            metric_list[data_source_idx] = data_config["title"]

                            if (
                                "cli_style" in data_config
                                and data_config["cli_style"] == "simple_box"
                            ):
                                for k, v in entries.items():
                                    if k == "expr":
                                        for bv in simple_box.values():
                                            values.append(bv[0] + v + bv[1])
                                    else:
                                        if k not in {"coll_level", "alias"}:
                                            values.append(v)
                            else:
                                for k, v in entries.items():
                                    if k not in {"coll_level", "alias"}:
                                        values.append(v)
                                        eqn_content.append(v)

                            if "alias" in entries.keys():
                                values.append(entries["alias"])

                            values.append(
                                entries.get("coll_level", schema.PMC_PERF_FILE_PREFIX)
                            )

                            if "metrics_description" in panel:
                                values.append(panel["metrics_description"].get(key, ""))

                            df_new_row = pd.DataFrame([values], columns=headers)
                            df = pd.concat([df, df_new_row])

                        # collect metric_list
                        metric_list[metric_idx] = key

                        # generate mapping of counters and metrics
                        filtered_counters = {}
                        formula_visited = False

                        for formula in eqn_content:
                            if formula is not None and formula != "None":
                                visited, counters = gen_counter_list(formula)
                                if visited:
                                    formula_visited = True
                                for counter in counters:
                                    filtered_counters[counter] = None

                        if filtered_counters or formula_visited:
                            metric_counters[key] = list(filtered_counters)

                    df.set_index("Metric_ID", inplace=True)
                elif type == "raw_csv_table":
                    data_source_idx = str(data_config["id"] // 100)
                    if (
                        (not filter_metrics)
                        or (data_source_idx == "0")  # no filter
                        or (data_source_idx in filter_metrics)
                    ):
                        if "columnwise" in data_config and data_config["columnwise"]:
                            df = pd.DataFrame(
                                [data_config["source"]], columns=["from_csv_columnwise"]
                            )
                        else:
                            df = pd.DataFrame(
                                [data_config["source"]], columns=["from_csv"]
                            )
                        metric_list[data_source_idx] = panel["title"]
                    else:
                        df = pd.DataFrame()
                elif type == "pc_sampling_table":
                    data_source_idx = str(data_config["id"] // 100)
                    df = pd.DataFrame(
                        [data_config["source"]], columns=["from_pc_sampling"]
                    )
                    metric_list[data_source_idx] = panel["title"]
                else:
                    df = pd.DataFrame()

                dfs[data_config["id"]] = df
                dfs_type[data_config["id"]] = type

    setattr(arch_configs, "dfs", dfs)
    setattr(arch_configs, "metric_list", metric_list)
    setattr(arch_configs, "dfs_type", dfs_type)
    setattr(arch_configs, "metric_counters", metric_counters)


def build_metric_value_string(
    dfs: dict, dfs_type: dict, normal_unit: str, profiling_config: dict
) -> None:
    """
    Apply the real eval string to its field in the metric_table df.
    """

    for id, df in dfs.items():
        if dfs_type[id] == "metric_table":
            for expr in df.columns:
                if expr in schema.SUPPORTED_FIELD:
                    # NB: apply all build-in before building the whole string
                    df[expr] = df[expr].apply(
                        update_denominator_string, normal_unit=normal_unit
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
                        update_normal_unit_string, normal_unit=normal_unit
                    )


def create_empirical_peaks_dict(empirical_peaks_df: pd.DataFrame) -> dict[str, float]:
    """Create empirical peaks dictionary"""
    empirical_peaks = {}

    if not empirical_peaks_df.empty:
        peak_data_row = empirical_peaks_df.iloc[0]
        for col in empirical_peaks_df.columns:
            empirical_peaks[f"ammolite__{col}_empirical_peak"] = peak_data_row[col]
    else:
        peak_names = [
            "FP16Flops",
            "FP32Flops",
            "FP64Flops",
            "MFMAF64Flops",
            "MFMAF32Flops",
            "MFMAF16Flops",
            "MFMABF16Flops",
            "MFMAF8Flops",
            "MFMAI8Ops",
            "HBMBw",
            "L2Bw",
            "L1Bw",
            "LDSBw",
            "MFMA_FLOPs_F6F4",
        ]
        # initialize peaks to 0
        for peak_name in peak_names:
            empirical_peaks[f"ammolite__{peak_name}_empirical_peak"] = 0

    return empirical_peaks


def create_sys_vars(sys_info: pd.Series) -> dict[str, Union[int, float]]:
    """Create variables from sys.info."""
    sys_vars_collection = {}

    sys_vars_config = [
        ("se_per_gpu", int, "se_per_gpu"),
        ("pipes_per_gpu", int, "pipes_per_gpu"),
        ("cu_per_gpu", int, "cu_per_gpu"),
        ("simd_per_cu", int, "simd_per_cu"),
        ("sqc_per_gpu", int, "sqc_per_gpu"),
        ("lds_banks_per_cu", int, "lds_banks_per_cu"),
        ("cur_sclk", float, "cur_sclk"),
        ("cur_mclk", float, "cur_mclk"),
        ("max_mclk", float, "max_mclk"),
        ("max_sclk", float, "max_sclk"),
        ("max_waves_per_cu", int, "max_waves_per_cu"),
        ("num_hbm_channels", float, "num_hbm_channels"),
        ("num_xcd", int, "num_xcd"),
        ("wave_size", int, "wave_size"),
    ]

    for var_name, var_type, attr_name in sys_vars_config:
        variable_value = var_type(getattr(sys_info, attr_name))
        if np.isnan(variable_value) or variable_value == 0:
            console_warning(
                f"{attr_name} is not available in sysinfo.csv, please provide the "
                "correct value using --specs-correction"
            )
        sys_vars_collection[f"ammolite__{var_name}"] = variable_value

    # Special case for total_l2_chan
    total_l2_channel_count = calc_builtin_var("$total_l2_chan", sys_info)
    if np.isnan(total_l2_channel_count) or total_l2_channel_count == 0:
        console_warning(
            "total_l2_chan is not available in sysinfo.csv, please provide the correct "
            "value using --specs-correction"
        )
    sys_vars_collection["ammolite__total_l2_chan"] = total_l2_channel_count

    return sys_vars_collection


def calc_builtin_vars(
    raw_pmc_df: Union[pd.DataFrame, dict], config: dict
) -> dict[str, Optional[Union[str, float, int]]]:
    """Calculate built-in variables"""
    # TODO: fix all $normUnit in Unit column or title
    # build and eval all derived build-in global variables
    builtin_vars_collection = {}

    # First pass: calculate per-XCD values
    for variable_key, variable_value in BUILD_IN_VARS.items():
        if "PER_XCD" not in variable_key:
            continue

        # NB: assume all built-in vars from pmc_perf.csv for now
        eval_string = build_eval_string(
            variable_value, schema.PMC_PERF_FILE_PREFIX, config
        )
        try:
            # Create temporary evaluator for this calculation
            temporary_evaluator = MetricEvaluator(raw_pmc_df, {}, {})
            calculation_result = temporary_evaluator.eval_expression(eval_string)
            builtin_vars_collection[f"ammolite__{variable_key}"] = calculation_result
        except (TypeError, NameError, KeyError, AttributeError):
            builtin_vars_collection[f"ammolite__{variable_key}"] = None

    # Second pass: calculate remaining variables that depend on per-XCD values
    for variable_key, variable_value in BUILD_IN_VARS.items():
        if "PER_XCD" in variable_key:
            continue

        eval_string = build_eval_string(
            variable_value, schema.PMC_PERF_FILE_PREFIX, config
        )
        try:
            temporary_evaluator = MetricEvaluator(
                raw_pmc_df, builtin_vars_collection, {}
            )
            calculation_result = temporary_evaluator.eval_expression(eval_string)
            builtin_vars_collection[f"ammolite__{variable_key}"] = calculation_result
        except (TypeError, NameError, KeyError, AttributeError):
            builtin_vars_collection[f"ammolite__{variable_key}"] = None

    return builtin_vars_collection


@demarcate
def eval_metric(
    dfs: dict,
    dfs_type: dict,
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: Union[pd.DataFrame, dict],
    debug: bool,
    config: dict,
) -> None:
    """
    Execute the expr string for each metric in the df.
    """

    # confirm no illogical counter values (only consider non-roofline runs)
    roof_only_run = sys_info.ip_blocks == "roofline"
    if (
        (not roof_only_run)
        and hasattr(raw_pmc_df.get("pmc_perf", {}), "GRBM_GUI_ACTIVE")
        and (raw_pmc_df["pmc_perf"]["GRBM_GUI_ACTIVE"] == 0).any()
    ):
        console_warning("Dectected GRBM_GUI_ACTIVE == 0")
        console_error("Hauting execution for warning above.")

    sys_vars = create_sys_vars(sys_info)
    empirical_peaks = create_empirical_peaks_dict(empirical_peaks_df)
    builtin_vars = calc_builtin_vars(raw_pmc_df, config)
    sys_vars.update(builtin_vars)

    # Create metric evaluator
    metric_evaluator = MetricEvaluator(raw_pmc_df, sys_vars, empirical_peaks)

    exprs_to_eval = []

    # Hmmm... apply + lambda should just work
    # df['Value'] = df['Value'].apply(
    #     lambda s: eval(
    #         compile(str(s), '<string>', 'eval')
    #     )
    # )
    for df_id, df in dfs.items():
        if dfs_type[df_id] == "metric_table":
            for row_id, row in df.iterrows():
                for expr in df.columns:
                    if expr in schema.SUPPORTED_FIELD and expr.lower() != "alias":
                        if row[expr]:
                            exprs_to_eval.append((df_id, row_id, expr, row[expr]))

                            if debug:
                                debug_evaluate_metrics(
                                    expr, row[expr], metric_evaluator, raw_pmc_df
                                )
                        else:
                            # If not insert nan, the whole col might be treated
                            # as string but not nubmer if there is NONE
                            row[expr] = ""

    for df_id, row_id, col, expr in exprs_to_eval:
        eval_result = metric_evaluator.eval_expression(expr)
        dfs[df_id].loc[row_id, col] = eval_result


def debug_evaluate_metrics(
    expr: str,
    row_expr: str,
    metric_evaluator: MetricEvaluator,
    raw_pmc_df: Union[pd.DataFrame, dict],
) -> None:
    """Debug helper for expression evaluation."""
    print("~" * 40 + "\nExpression:")
    print(f"{expr} = {row_expr}")
    print("Inputs:")

    # Show matched variables
    matched_vars = re.findall(r"ammolite__\w+", row_expr)
    if matched_vars:
        for vars in matched_vars:
            if vars in metric_evaluator.sys_vars:
                print(f"Var {vars}: {metric_evaluator.sys_vars[vars]}")
            elif vars in metric_evaluator.empirical_peaks:
                print(f"Var {vars}: {metric_evaluator.empirical_peaks[vars]}")
            else:
                print(f"Var {vars}: [not found]")

    # Show matched columns
    matched_cols = re.findall(r"raw_pmc_df\['\w+'\]\['\w+'\]", row_expr)
    if matched_cols:
        for cols in matched_cols:
            col_match = re.match(r"raw_pmc_df\['(\w+)'\]\['(\w+)'\]", cols)
            try:
                if isinstance(raw_pmc_df, dict) and col_match.group(1) in raw_pmc_df:
                    column_data = raw_pmc_df[col_match.group(1)][
                        col_match.group(2)
                    ].to_list()
                    print(f"{cols}: {column_data}")
            except KeyError as key_error:
                console_warning(
                    f"Skipping entry. Encountered a missing key: {key_error}"
                )

    print("\nOutput:")
    try:
        eval_result = metric_evaluator.eval_expression(row_expr)
        print(eval_result)
        print("~" * 40)
    except Exception as e:
        console_warning(f"Debug evaluation failed: {e}")
        print("~" * 40)


@demarcate
def apply_filters(
    workload: schema.Workload, dir_path: str, is_gui: bool, debug: bool
) -> pd.DataFrame:
    """
    Apply user's filters to the raw_pmc df.
    """

    # TODO: error out properly if filters out of bound
    filtered_df = workload.raw_pmc

    # Apply node filter
    if workload.filter_nodes:
        filtered_df = filtered_df.loc[
            filtered_df[schema.PMC_PERF_FILE_PREFIX]["Node"]
            .astype(str)
            .isin([workload.filter_gpu_ids])
        ]
        if filtered_df.empty:
            console_error("analysis", f"{workload.filter_nodes} is invalid")

    # Apply GPU ID filter
    if workload.filter_gpu_ids:
        filtered_df = filtered_df.loc[
            filtered_df[schema.PMC_PERF_FILE_PREFIX]["GPU_ID"]
            .astype(str)
            .isin([workload.filter_gpu_ids])
        ]
        if filtered_df.empty:
            console_error("analysis", f"{workload.filter_gpu_ids} is an invalid gpu-id")

    # Apply kernel filter
    # NB:
    # Kernel id is unique!
    # We pick up kernel names from kerne ids first.
    # Then filter valid entries with kernel names.
    if workload.filter_kernel_ids:
        filtered_df = apply_kernel_filter(filtered_df, workload, dir_path)

    # Apply dispatch filter
    if workload.filter_dispatch_ids:
        filtered_df = apply_dispatch_filter(filtered_df, workload)

    if debug:
        print("~" * 40, "\nraw pmc df info:\n")
        print(workload.raw_pmc.info())
        print("~" * 40, "\nfiltered pmc df info:")
        print(filtered_df.info())

    return filtered_df


def apply_kernel_filter(
    df: pd.DataFrame, workload: schema.Workload, dir_path_path: str
) -> pd.DataFrame:
    """Apply kernel ID or name filters."""
    if all(isinstance(kernel_id, int) for kernel_id in workload.filter_kernel_ids):
        # Handle integer kernel IDs
        kernels_dataframe = pd.read_csv(Path(dir_path_path) / "pmc_kernel_top.csv")

        # Validate kernel IDs
        for kernel_id in workload.filter_kernel_ids:
            if kernel_id >= len(kernels_dataframe["Kernel_Name"]):
                console_error(
                    f"{kernel_id} is an invalid kernel id. "
                    "Please enter an id between 0-"
                    f"{len(kernels_dataframe['Kernel_Name']) - 1}"
                )

        # Extract kernel names and mark selected kernels with "*"
        # TODO: fix it for unaligned comparison
        selected_kernels = []
        kernel_top_dataframe = workload.dfs[PMC_KERNEL_TOP_TABLE_ID]
        kernel_top_dataframe["S"] = ""

        for kernel_id in workload.filter_kernel_ids:
            selected_kernels.append(kernel_top_dataframe.loc[kernel_id, "Kernel_Name"])
            kernel_top_dataframe.loc[kernel_id, "S"] = "*"

        if selected_kernels:
            df = df.loc[
                df[schema.PMC_PERF_FILE_PREFIX]["Kernel_Name"].isin(selected_kernels)
            ]

    elif all(isinstance(kernel_id, str) for kernel_id in workload.filter_kernel_ids):
        # Handle string kernel names
        cleaned_dataframe = df[schema.PMC_PERF_FILE_PREFIX]["Kernel_Name"].apply(
            lambda kernel_name: (
                kernel_name.strip() if isinstance(kernel_name, str) else kernel_name
            )
        )
        df = df.loc[cleaned_dataframe.isin(workload.filter_kernel_ids)]
    else:
        console_error(
            "analyze",
            "Mixing kernel indices and string filters is not currently supported",
        )

    return df


def apply_dispatch_filter(df: pd.DataFrame, workload: schema.Workload) -> pd.DataFrame:
    """Apply dispatch ID filters."""
    # NB: support ignoring the 1st n dispatched execution by '> n'
    #     The better way may be parsing python slice string
    for dispatch_id in workload.filter_dispatch_ids:
        if int(dispatch_id) >= len(df):  # subtract 2 bc of the two header rows
            console_error("analysis", f"{dispatch_id} is an invalid dispatch id.")

    if (
        isinstance(workload.filter_dispatch_ids[0], str)
        and ">" in workload.filter_dispatch_ids[0]
    ):
        dispatch_match = re.match(r"\> (\d+)", workload.filter_dispatch_ids[0])
        df = df[
            df[schema.PMC_PERF_FILE_PREFIX]["Dispatch_ID"]
            > int(dispatch_match.group(1))
        ]
    else:
        selected_dispatches = [
            int(dispatch_str) for dispatch_str in workload.filter_dispatch_ids
        ]
        df = df.loc[selected_dispatches]

    return df


def find_key_recursively(
    data: Union[dict, list], search_key: str
) -> Union[list, dict, None]:
    """
    Recursively search for the search_key in the given data
    (which can be a dict or list).
    If the key is found, returns the value as a DataFrame.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == search_key:
                return value
            elif isinstance(value, (dict, list)):
                result = find_key_recursively(value, search_key)
                if result:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = find_key_recursively(item, search_key)
            if result:
                return result
    return None  # Return None if the key was not found


def search_key_in_json(file_path: Path, search_key: str) -> Union[list, dict, None]:
    # FIXME:
    #   Load the entire JSON into memory.
    #   Should not use for large file.
    with open(file_path) as file:
        data = json.load(file)
        found = find_key_recursively(data, search_key)
        if found is None:
            console_error(f'Key "{search_key}" not found in the JSON file.')
        return found


def search_pc_sampling_record(
    records: Union[list[dict], dict],
) -> Optional[list[tuple]]:
    """
    Search PC sampling records.

    Group by (code_object_id, code_object_offset, inst_index), and aggregate
    counts, stall reasons, and dispatch IDs.

    Returns:
        A sorted list of tuples:
        (
            code_object_id,
            code_object_offset,
            inst_index,
            total_count,
            count_issued,
            count_stalled,
            sorted_stall_reasons,
            sorted_dispatch_ids,
        )
    """

    if not records:
        console_warning("PC sampling: no pc sampling record found!")
        return None

    # records should always be a list of dict
    if isinstance(records, dict):
        records = [records]

    rocp_inst_not_issued_prefix_len = len(PC_SAMPLING_NOT_ISSUE_PREFIX)

    stall_reason_keys = {
        "NONE": 0,
        # No instruction available in the instruction cache.
        "NO_INSTRUCTION_AVAILABLE": 0,
        "ALU_DEPENDENCY": 0,  # ALU dependency not resolved.
        "WAITCNT": 0,
        "INTERNAL_INSTRUCTION": 0,  # Wave executes an internal instruction.
        "BARRIER_WAIT": 0,
        "ARBITER_NOT_WIN": 0,  # The instruction did not win the arbiter.
        "ARBITER_WIN_EX_STALL": 0,
        # Arbiter issued an instruction, but the execution pipe
        # pushed it back from execution.
        "OTHER_WAIT": 0,
        # Other types of wait (e.g., wait for XNACK acknowledgment).
        "SLEEP_WAIT": 0,
        "LAST": 0,
    }

    grouped_data: dict[tuple, list] = {}

    for item in records:
        record = item.get("record", {})
        pc_info = record.get("pc", {})

        code_object_id = pc_info.get("code_object_id")
        code_object_offset = pc_info.get("code_object_offset")
        inst_index = item.get("inst_index")
        dispatch_id = record.get("dispatch_id")

        if None in (code_object_id, code_object_offset, inst_index):
            continue

        key = (code_object_id, code_object_offset, inst_index)

        snapshot = record.get("snapshot", {})
        issued = record.get("wave_issued", False)

        if key not in grouped_data:
            grouped_data[key] = [0, 0, 0, {}, set()]

        entry = grouped_data[key]

        # Update counts
        entry[0] += 1  # total_count
        if issued:
            entry[1] += 1  # count_issued
        else:
            entry[2] += 1  # count_stalled
            stall_reason = snapshot.get("stall_reason")
            if stall_reason and len(stall_reason) > rocp_inst_not_issued_prefix_len:
                reason_key = stall_reason[rocp_inst_not_issued_prefix_len:]
                if reason_key in stall_reason_keys:
                    entry[3][reason_key] = entry[3].get(reason_key, 0) + 1

        # Add dispatch_id if valid
        if dispatch_id is not None:
            entry[4].add(dispatch_id)

    if not grouped_data:
        console_warning("PC sampling: no pc sampling record found!")
        return None

    # Prepare sorted output list
    sorted_counts = sorted(
        [
            (
                code_object_id,
                code_object_offset,
                inst_index,
                info[0],  # total_count
                info[1],  # count_issued
                info[2],  # count_stalled
                sorted(
                    ((k, v) for k, v in info[3].items() if v > 0),
                    key=lambda item: item[1],
                    reverse=True,
                ),  # sorted stall reasons
                sorted(info[4]),  # sorted dispatch_ids list
            )
            for (
                code_object_id,
                code_object_offset,
                inst_index,
            ), info in grouped_data.items()
        ],
        key=lambda x: (x[0], x[1], x[2]),
    )

    return sorted_counts


@demarcate
def load_pc_sampling_data_per_kernel(
    method: str,
    file_name: Path,
    csv_file_name: Path,
    kernel_name: str,
    sorting_type: str,
) -> pd.DataFrame:
    """
    Load PC sampling raw data from json file with given method and kernel name,
    count pc sampling and sort it in the order of compiled asm and associate with
    kernel source code if available,
    then return df.

    :param method: "host_trap" or "stochastic".
    :type method: str
    :param file_name: The pc sampling json file.
    :type file_name: Path
    :param kernel_name: The kernel name to be filtered out.
    :type kernel_name: str
    :param sorting_type: "offset" or "count".
    :type sorting_type: str
    :return: The counted and reordering pc sampling info.
    :rtype: pd.DataFrame:
    """
    # Load kernel trace CSV with kernel info
    kernel_trace_df = pd.read_csv(
        csv_file_name, usecols=["Dispatch_Id", "Kernel_Id", "Kernel_Name"]
    )
    console_debug(
        f"PC sampling: loaded kernel trace with {len(kernel_trace_df)} entries"
    )

    # Filter kernels matching requested kernel_name
    matching_kernels = kernel_trace_df[kernel_trace_df["Kernel_Name"] == kernel_name]
    if matching_kernels.empty:
        console_warning(f"PC sampling: cannot find kernel '{kernel_name}' in CSV")
        return pd.DataFrame()

    # Extract raw PC sampling records from JSON
    pc_sample_key_loc = (
        search_key_in_json(file_name, "pc_sample_host_trap")
        if method == "host_trap"
        else search_key_in_json(file_name, "pc_sample_stochastic")
    )

    if not pc_sample_key_loc:
        console_warning("PC sampling: can not find pc sample.")
        return pd.DataFrame()

    # Get processed sampling data grouped by (code_object_id, offset, inst_index)
    records = search_pc_sampling_record(pc_sample_key_loc)
    if not records:
        console_warning("PC sampling: no records found in PC sampling data.")
        return pd.DataFrame()

    # Flatten records by dispatch_id to create one row per dispatch ID
    rows = []
    for (
        code_object_id,
        offset,
        inst_index,
        count,
        count_issued,
        count_stalled,
        stall_reasons,
        dispatch_ids,
    ) in records:
        for dispatch_id in dispatch_ids:
            rows.append({
                "dispatch_id": dispatch_id,
                "code_object_id": code_object_id,
                "offset": offset,
                "inst_index": inst_index,
                "count": count,
                "count_issued": count_issued,
                "count_stalled": count_stalled,
                "stall_reason": stall_reasons,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        console_warning("PC sampling: no records found after flattening dispatch IDs.")
        return df

    # Map dispatch_id to kernel info (Kernel_Id and Kernel_Name)
    dispatch_to_kernel = kernel_trace_df.set_index("Dispatch_Id")[
        ["Kernel_Id", "Kernel_Name"]
    ]

    # Map dispatch_id to kernel info (Kernel_Id and Kernel_Name)
    df["kernel_id"] = df["dispatch_id"].map(dispatch_to_kernel["Kernel_Id"])
    df["kernel_name"] = df["dispatch_id"].map(dispatch_to_kernel["Kernel_Name"])

    # Drop dispatch_id
    df.drop(columns=["dispatch_id"], inplace=True)

    def merge_stall_reasons(
        stall_reason_series: list[Optional[list[tuple[str, int]]]],
    ) -> list[tuple[str, int]]:
        """
        Function to merge stall_reason lists (list of dicts -> merged & sorted dict)
        """
        merged_counts = {}

        for entry in stall_reason_series:
            if not entry:
                continue
            # Each entry is a list of (key, count) tuples
            for k, v in entry:
                if v > 0:
                    merged_counts[k] = merged_counts.get(k, 0) + v

        # Return sorted list of tuples by descending count
        return sorted(merged_counts.items(), key=lambda item: item[1], reverse=True)

    # Group and aggregate
    df = df.groupby(["code_object_id", "offset", "kernel_id"], as_index=False).agg({
        "inst_index": "first",
        "count": "sum",
        "count_issued": "sum",
        "count_stalled": "sum",
        "stall_reason": merge_stall_reasons,
        "kernel_name": "first",
    })

    # Filter DataFrame to only include rows matching the requested kernel_name
    df = df[df["kernel_name"] == kernel_name]

    # Convert offset column to hex string for display, keep original numeric for sorting
    df["offset"] = df["offset"].apply(lambda x: hex(x))

    # Load PC sampling instructions from JSON (if available)
    pc_sample_instructions = search_key_in_json(file_name, "pc_sample_instructions")
    df["instruction"] = (
        df["inst_index"].apply(
            lambda x: pc_sample_instructions[x]
            if x < len(pc_sample_instructions)
            else None
        )
        if pc_sample_instructions
        else None
    )

    # Load source code comments (if available)
    pc_sample_comments = search_key_in_json(file_name, "pc_sample_comments")
    df["source_line"] = (
        df["inst_index"].apply(
            lambda x: f".../{Path(pc_sample_comments[x]).name}"
            if x < len(pc_sample_comments)
            else None
        )
        if pc_sample_comments
        else None
    )

    # Sorting and returning relevant columns depending on method and sorting_type
    if sorting_type == "offset":
        df_sorted = df.sort_values(by=["code_object_id", "offset"])
    elif sorting_type == "count":
        df_sorted = df.sort_values(by=["count"], ascending=False)
    else:
        console_error(
            'Error: pc sampling sorting_type must be either "offset" or "count".'
        )
        return pd.DataFrame()

    columns_to_return = (
        [
            "source_line",
            "instruction",
            "code_object_id",
            "offset",
            "count",
        ]
        if method == "host_trap"
        else [
            "source_line",
            "instruction",
            "code_object_id",
            "offset",
            "count",
            "count_issued",
            "count_stalled",
            "stall_reason",
        ]
    )

    return df_sorted[columns_to_return]
    # might support sort by stall reason in the future


@demarcate
def load_pc_sampling_data(
    workload: schema.Workload, dir_path: str, file_prefix: str, sorting_type: str
) -> pd.DataFrame:
    """
    Load PC sampling raw data, filter and sort it by specified conditions,
    then return df.
    """

    if not file_prefix or file_prefix.lower() == "none":
        return pd.DataFrame()

    pc_sampling_method = None

    # NB:
    #  - The default file name is subject to changes from rocprofv3
    #  - Prioritize stochastic
    #  - Alternatively, we could check pc_sampling_method in json
    stochastic_path = Path(dir_path) / f"{file_prefix}_pc_sampling_stochastic.csv"
    host_trap_path = Path(dir_path) / f"{file_prefix}_pc_sampling_host_trap.csv"
    json_file_path = Path(dir_path) / f"{file_prefix}_results.json"
    csv_kernel_trace_file_path = Path(dir_path) / f"{file_prefix}_kernel_trace.csv"

    if not csv_kernel_trace_file_path.exists():
        console_error(
            f"PC sampling: can not read {csv_kernel_trace_file_path}", exit=False
        )
        return pd.DataFrame()

    if stochastic_path.exists():
        pc_sampling_method = "stochastic"
        csv_file_path = stochastic_path
    elif host_trap_path.exists():
        pc_sampling_method = "host_trap"
        csv_file_path = host_trap_path
    else:
        console_warning(
            f"PC sampling: can not detect pc sampling method for {file_prefix}"
        )
        return pd.DataFrame()

    # No kernel filter, return grouped and sorted csv dir_pathectly
    if not workload.filter_kernel_ids:
        # Load instruction CSV
        df = pd.read_csv(csv_file_path)

        # Load kernel trace CSV
        kernel_trace_df = pd.read_csv(csv_kernel_trace_file_path)

        # Merge on Correlation_Id (instruction CSV) and Dispatch_Id (kernel trace CSV)
        merged_df = df.merge(
            kernel_trace_df[["Dispatch_Id", "Kernel_Name", "Kernel_Id"]],
            how="left",
            left_on="Correlation_Id",
            right_on="Dispatch_Id",
        )

        # Group by Instruction_Comment and aggregate
        grouped_counts = (
            merged_df.groupby("Instruction_Comment")
            .agg(
                count=("Instruction_Comment", "count"),
                instruction=("Instruction", "first"),
                Kernel_Id=("Kernel_Id", "first"),
                Kernel_Name=("Kernel_Name", "first"),
            )
            .reset_index()
            .rename(columns={"Instruction_Comment": "source_line"})
        )
        grouped_counts = grouped_counts[
            [
                "source_line",
                "Kernel_Name",
                "instruction",
                "count",
            ]
        ]
        grouped_counts["source_line"] = grouped_counts["source_line"].apply(
            lambda x: f".../{Path(x).name}" if isinstance(x, str) and x else x
        )

        return grouped_counts.sort_values(by="count", ascending=False)

    elif len(workload.filter_kernel_ids) > 1:
        console_error(
            "PC sampling supports single kernel only! Please specify -k with "
            "single kernel.",
            exit=False,
        )
        return pd.DataFrame()

    elif len(workload.filter_kernel_ids) == 1:
        if not json_file_path.exists():
            console_error(f"PC sampling: can not read {json_file_path}", exit=False)
            return pd.DataFrame()
        else:
            # NB:
            #   We should find better way to remove the dependency on kernel_top_table
            kernel_top_df = workload.dfs[PMC_KERNEL_TOP_TABLE_ID]
            file = Path(dir_path) / str(kernel_top_df.loc[0, "from_csv"])
            kernel_index = workload.filter_kernel_ids[0]

            kernel_df = pd.read_csv(file)

            if kernel_index >= len(kernel_df):
                console_warning(
                    f"Kernel index {kernel_index} is out of bounds. "
                    f"kernel_top CSV has only {len(kernel_df)} rows."
                )
                return pd.DataFrame()

            kernel_name = kernel_df.iloc[kernel_index]["Kernel_Name"]

            return load_pc_sampling_data_per_kernel(
                pc_sampling_method,
                json_file_path,
                csv_kernel_trace_file_path,
                kernel_name,
                sorting_type,
            )
    else:
        console_warning("PC sampling: No data")
        return pd.DataFrame()


@demarcate
def load_non_mertrics_table(
    workload: schema.Workload, dir_path: str, args: argparse.Namespace
) -> None:
    # NB:
    #   - Do pmc_kernel_top.csv loading before eval_metric because we need the
    #     kernel names.
    #   - There might be a better way/timing to load raw_csv_table.

    # NB:
    #   "from_csv", "from_csv_columnwise", and "from_pc_sampling"
    #   are 3 internal symbols converted in build_dfs() for non-metrics table.
    #   There might be better way to store these info without the orginal entry.
    tmp = {}
    for df_id, df in workload.dfs.items():
        if "from_csv" in df.columns:
            csv_file = Path(dir_path) / str(df.loc[0, "from_csv"])
            if csv_file.exists():
                tmp[df_id] = pd.read_csv(csv_file)
            else:
                console_warning(
                    f"Couldn't load {csv_file.name}. "
                    "This may result in missing analysis data."
                )
        # NB: Special case for sysinfo. Probably room for improvement in this whole
        # function design
        elif "from_csv_columnwise" in df.columns and id == 101:
            tmp[df_id] = workload.sys_info.transpose()
            # All transposed columns should be marked with a general header
            tmp[df_id].columns = ["Info"]
        elif "from_csv_columnwise" in df.columns:
            # NB:
            #   Another way might be doing transpose in tty like metric_table.
            #   But we need to figure out headers and comparison properly.
            csv_file = Path(dir_path) / str(df.loc[0, "from_csv_columnwise"])
            if csv_file.exists():
                tmp[df_id] = pd.read_csv(csv_file).transpose()
                # NB:
                #   All transposed columns should be marked with a general header,
                #   so tty could detect them and show them correctly in comparison.
                tmp[df_id].columns = ["Info"]
            else:
                console_warning(
                    f"Couldn't load {csv_file.name}. "
                    "This may result in missing analysis data."
                )
        elif "from_pc_sampling" in df.columns:
            tmp[df_id] = load_pc_sampling_data(
                workload,
                dir_path,
                df.loc[0, "from_pc_sampling"],
                args.pc_sampling_sorting_type,
            )

    workload.dfs.update(tmp)


@demarcate
def load_table_data(
    workload: schema.Workload,
    dir_path: str,
    is_gui: bool,
    args: argparse.Namespace,
    config: dict,
    skip_kernel_top: bool = False,
) -> None:
    """
    - Load data for all "raw_csv_table"
    - Load data for "pc_sampling_table"
    - Calculate mertric value for all "metric_table"
    """
    if not skip_kernel_top:
        load_non_mertrics_table(workload, dir_path, args)

    eval_metric(
        workload.dfs,
        workload.dfs_type,
        workload.sys_info.iloc[0],
        workload.roofline_peaks,
        apply_filters(workload, dir_path, is_gui, args.debug),
        args.debug,
        config,
    )


def build_comparable_columns(time_unit: str) -> list[str]:
    """
    Build comparable columns/headers for display
    """
    comparable_columns = schema.SUPPORTED_FIELD
    top_stat_base = [
        "Count",
        "Sum",
        "Mean",
        "Median",
        "Standard Deviation",
        "Description",
    ]

    for h in top_stat_base:
        comparable_columns.append(f"{h}({time_unit})")

    return comparable_columns


def correct_sys_info(mspec: MachineSpecs, specs_correction: str) -> pd.DataFrame:
    """
    Correct system spec items manually based on user-provided corrections.
    """
    # Parse key:value pairs
    pairs: dict[str, str] = {}
    for pair in specs_correction.split(","):
        if ":" in pair:
            key, value = pair.split(":", 1)
            pairs[key.strip()] = value.strip()

    # Apply corrections
    for key, value in pairs.items():
        if hasattr(mspec, key):
            setattr(mspec, key, value)
        else:
            console_error(
                "analyze", f'Invalid spec "{key}". Use --specs to see valid options'
            )
    return mspec.get_class_members()
