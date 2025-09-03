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

import ast
import json
import multiprocessing
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import astunparse  # type: ignore
import numpy as np
import pandas as pd

from utils import schema
from utils.logger import console_debug, console_error, console_warning, demarcate

# ------------------------------------------------------------------------------
# Constants and Configuration

# NB:
# Ammolite is unique gemstone from the Rocky Mountains.
# "ammolite__" is a special internal prefix to mark build-in global variables
# calculated or parsed from raw data sources. Its range is only in this file.
# Any other general prefixes string, like "buildin__", might be used by the
# editor. Whenever change it to a new one, replace all appearances in this file.

# 001 is ID of pmc_kernel_top.csv table
PMC_KERNEL_TOP_TABLE_ID = 1

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
SUPPORTED_DENOM = {
    "per_wave": "SQ_WAVES",
    "per_cycle": "$GRBM_GUI_ACTIVE_PER_XCD",
    "per_second": "((End_Timestamp - Start_Timestamp) / 1000000000)",
    "per_kernel": "1",
}

# Build-in defined in mongodb variables:
BUILD_IN_VARS = {
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

SUPPORTED_CALLS = {
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

SIMPLE_BOX_STATS = {
    "Min": ["MIN(", ")"],
    "Q1": ["QUANTILE(", ", 0.25)"],
    "Median": ["MEDIAN(", ")"],
    "Q3": ["QUANTILE(", ", 0.75)"],
    "Max": ["MAX(", ")"],
}

FUNCTION_FILTER = {
    "MIN",
    "MAX",
    "AVG",
    "ROUND",
    "TO_INT",
    "GB",
    "STD",
    "GFLOP",
    "GOP",
    "OP",
    "CU",
    "NC",
    "UC",
    "CC",
    "RW",
    "GIOP",
    "GFLOPs",
    "CONCAT",
    "MOD",
}

BUILT_IN_COUNTERS = [
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

DEFAULT_PEAKS = [
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
PC_SAMPLING_NOT_ISSUE_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"

# ------------------------------------------------------------------------------
# Utility Functions


def to_min(*args: Any) -> float:
    if len(args) == 1 and isinstance(args[0], pd.Series):
        return args[0].min()
    elif min(args) is None:
        return np.nan
    else:
        return min(args)


def to_max(*args: Any) -> float:
    if len(args) == 1 and isinstance(args[0], pd.Series):
        return args[0].max()
    elif len(args) == 2 and (
        isinstance(args[0], pd.Series) or isinstance(args[1], pd.Series)
    ):
        result = np.maximum(args[0], args[1])
        # Handle case where np.maximum returns an array
        if isinstance(result, np.ndarray):
            return float(result.item()) if result.size == 1 else float(result.max())
        return float(result)
    elif any(arg is None for arg in args):
        return np.nan
    else:
        return float(max(args))


def to_avg(a: Any) -> Optional[float]:
    if a is None:
        return np.nan
    elif isinstance(a, pd.Series):
        if a.empty or np.isnan(a).all():
            return np.nan
        else:
            return a.mean()
    elif isinstance(a, (np.ndarray, list)):
        arr = np.array(a)
        if arr.size == 0 or np.isnan(arr).all():
            return np.nan
        else:
            return float(np.nanmean(arr))
    elif isinstance(a, (int, float, np.number)):
        return np.nan if np.isnan(a) else float(a)
    else:
        raise ValueError(f"to_avg: unsupported type: {type(a)}")


def to_median(a: Any) -> Optional[float]:
    if a is None:
        return None
    elif isinstance(a, pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return a.median()
    else:
        raise ValueError("to_median: unsupported type.")


def to_std(a: Any) -> Optional[float]:
    if isinstance(a, pd.Series):
        return a.std()
    else:
        raise ValueError("to_std: unsupported type.")


def to_int(a: Any) -> Any:
    if a is None:
        return None
    elif isinstance(a, (int, float, np.integer)):
        return int(a)
    elif isinstance(a, pd.Series):
        return a.astype("Int64")
    else:
        raise ValueError("to_int: unsupported type.")


def to_sum(a: Any) -> Optional[float]:
    if (
        a is None
        or (hasattr(a, "empty") and a.empty)
        or (hasattr(a, "__iter__") and np.isnan(a).all())
    ):
        return np.nan
    elif isinstance(a, pd.Series):
        return a.sum()
    else:
        raise ValueError("to_sum: unsupported type.")


def to_round(a: Any, b: int) -> Any:
    if isinstance(a, pd.Series):
        return a.round(b)
    else:
        return round(a, b)


def to_quantile(a: Any, b: float) -> Optional[float]:
    if a is None:
        return None
    elif isinstance(a, pd.Series):
        return a.quantile(b)
    else:
        raise ValueError("to_quantile: unsupported type.")


def to_mod(a: Any, b: Any) -> Any:
    if isinstance(a, pd.Series):
        return a.mod(b)
    else:
        return a % b


def to_concat(a: Any, b: Any) -> str:
    return str(a) + str(b)


class CodeTransformer(ast.NodeTransformer):
    """
    Python AST visitor to transform user defined equation string to df format
    """

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in SUPPORTED_CALLS:
                node.func.id = SUPPORTED_CALLS[node.func.id]
            else:
                raise ValueError(f"Unknown call: {node.func.id}")
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.Expr:
        self.generic_visit(node)

        if isinstance(node.body, ast.Num):
            raise ValueError(
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
    def visit_Name(self, node):
        self.generic_visit(node)
        if (not node.id.startswith("ammolite__")) and (not node.id in SUPPORTED_CALLS):
            new_node = ast.Subscript(
                value=ast.Name(id="raw_pmc_df", ctx=ast.Load()),
                slice=ast.Constant(value=node.id),
                ctx=ast.Load(),
            )
            return new_node
        return node


def build_eval_string(equation: str, coll_level: str, config: dict[str, Any]) -> str:
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
        raise ValueError("Error: coll_level can not be None.")

    if not equation:
        return ""

    s = str(equation)

    # build-in variable starts with '$', python can not handle it.
    # replace '$' with 'ammolite__'.
    # TODO: pre-check there is no "ammolite__" in all config files.
    s = re.sub(r"\$", "ammolite__", s)

    # convert equation string to intermediate expression in df array format
    ast_node = ast.parse(s)
    transformer = CodeTransformer()
    transformer.visit(ast_node)

    s = astunparse.unparse(ast_node)

    # correct column name/label in df with [], such as TCC_HIT[0],
    # the target is df['TCC_HIT[0]']
    s = re.sub(r"\'\]\[(\d+)\]", r"[\g<1>]']", s)

    # apply coll_level
    if config.get("format_rocprof_output") == "rocpd":
        # Replace SQ_ACCUM_PREV_HIRES with coll_level_ACCUM then ignore coll_level df
        s = re.sub("SQ_ACCUM_PREV_HIRES", f"{coll_level}_ACCUM", s)
        s = re.sub(
            r"raw_pmc_df", "raw_pmc_df['" + schema.PMC_PERF_FILE_PREFIX + "']", s
        )
    else:
        s = re.sub(r"raw_pmc_df", "raw_pmc_df['" + coll_level + "']", s)

    return s


def update_denom_string(equation: str, unit: str) -> str:
    """
    Update $denom in equation with runtime normalization unit.
    """
    if not equation:
        return ""

    s = str(equation)
    if unit in SUPPORTED_DENOM:
        s = re.sub(r"\$denom", SUPPORTED_DENOM[unit], s)

    return s


def update_normunit_string(equation: str, unit: str) -> str:
    """
    Update $normUnit in equation with runtime normalization unit.
    It is string replacement for display only.
    """

    # TODO: We might want to do it for subtitle contains $normUnit
    if not equation:
        return ""

    return re.sub(
        r"\((?P<PREFIX>\w*)\s+\+\s+(\$normUnit\))",
        r"\g<PREFIX> " + re.sub("_", " ", unit),
        str(equation),
    ).capitalize()


def gen_counter_list(formula: str) -> tuple[bool, list[str]]:
    visited = False
    counters: list[str] = []

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
                if val.isupper() and val not in FUNCTION_FILTER:
                    counters.append(val)
                    visited = True
                if val in BUILT_IN_COUNTERS:
                    visited = True
    except Exception:
        pass

    return visited, counters


def calc_builtin_var(var: Any, sys_info: Any) -> Optional[int]:
    """
    Calculate build-in variable based on sys_info:
    """
    if isinstance(var, int):
        return var
    elif isinstance(var, str) and var.startswith("$total_l2_chan"):
        return sys_info.total_l2_chan
    else:
        console_error(f'Built-in var "{var}" is not supported')


def init_metric_evaluator(
    raw_pmc_df: dict[str, pd.DataFrame],
    ammolite_vars: dict[str, Any],
    empirical_peaks: dict[str, Any],
) -> None:
    if isinstance(raw_pmc_df, dict):
        raw_pmc_df_keys = set(raw_pmc_df.keys())
    elif isinstance(raw_pmc_df, pd.DataFrame):
        raw_pmc_df_keys = set(raw_pmc_df.columns.get_level_values(0))
    else:
        raise ValueError(f"Unknown `raw_pmc_df` type '{type(raw_pmc_df)}'.")

    raw_pmc_df_items = {f"raw_pmc_df_{key}": raw_pmc_df[key] for key in raw_pmc_df_keys}

    # The globals here are not shared across all processes,
    # they exist only within the subprocess's context,
    # and their lifetime ends when the process terminates.
    # The process-local globals are used for performance optimization.
    globals().update(raw_pmc_df_items)
    globals().update(ammolite_vars)
    globals().update(empirical_peaks)


def run_metric_evaluator(row_expr: str) -> Any:
    try:
        # cache dataframes of 'raw_pmc_df'
        # this may replace some KeyErrors with NameErrors
        # e.g. row_pmc_df['key'] -> row_pmc_df_key will throw NameError now
        row_expr = re.sub(r"raw_pmc_df\['(.*?)'\]", r"raw_pmc_df_\1", row_expr)
        out = eval(compile(row_expr, "<string>", "eval"))

        if isinstance(out, (int, float)) and np.isnan(out):
            return ""
        else:
            return out

    except (TypeError, NameError, KeyError) as e:
        if "empirical_peak" in str(e):
            console_warning(f"Missing empirical peak data: {e}. Using empty value.")
        return ""

    except AttributeError as ae:
        if str(ae) == "'NoneType' object has no attribute 'get'":
            return ""
        else:
            console_error("analysis", str(ae))


def build_metric_row_values(
    metric_idx: str,
    key: str,
    entries: dict[str, Any],
    data_config: dict[str, Any],
    panel: dict[str, Any],
) -> list[Any]:
    values = [metric_idx, key]

    if "simple_box" == data_config.get("cli_style"):
        for k, v in entries.items():
            if "expr" == k:
                for box_k, box_v in SIMPLE_BOX_STATS.items():
                    values.append(box_v[0] + v + box_v[1])
            else:
                if k not in {"coll_level", "alias"}:
                    values.append(v)
    else:
        for k, v in entries.items():
            if k not in {"coll_level", "alias"}:
                values.append(v)

    if "alias" in entries:
        values.append(entries["alias"])

    values.append(entries.get("coll_level", schema.PMC_PERF_FILE_PREFIX))

    if "metrics_description" in panel:
        values.append(panel["metrics_description"].get(key, ""))

    return values


def collect_metric_counters(
    entries: dict[str, Any], key: str, metric_counters: dict[str, list[str]]
) -> None:
    filter_dict: dict[str, None] = {}
    visited = False

    for formula in entries.values():
        if formula is not None and str(formula) != "None":
            is_visited, counters = gen_counter_list(str(formula))
            if is_visited:
                visited = True
            for counter in counters:
                filter_dict[counter] = None

    if filter_dict or visited:
        metric_counters[key] = list(filter_dict.keys())


def build_metric_table_df(
    panel_id: int,
    data_config: dict[str, Any],
    panel: dict[str, Any],
    filter_metrics: Optional[list[str]],
    metric_list: dict[str, str],
    metric_counters: dict[str, list[str]],
) -> pd.DataFrame:
    headers = ["Metric_ID"]
    data_source_idx = str(data_config["id"] // 100)

    if "0" != data_source_idx or (filter_metrics and data_source_idx in filter_metrics):
        metric_list[data_source_idx] = panel["title"]

    if "simple_box" == data_config.get("cli_style"):
        headers.append(data_config["header"]["metric"])
        headers.extend(SIMPLE_BOX_STATS.keys())

        for key, tile in data_config["header"].items():
            if key not in {"metric", "expr"}:
                headers.append(tile)
    else:
        headers.append(data_config["header"]["metric"])
        for key, tile in data_config["header"].items():
            if "metric" != key:
                headers.append(tile)

    headers.append("coll_level")

    if "metrics_description" in panel:
        headers.append("Description")

    df = pd.DataFrame(columns=headers)

    for i, (key, entries) in enumerate(data_config["metric"].items()):
        data_source_idx = f"{data_config['id'] // 100}.{data_config['id'] % 100}"
        metric_idx = f"{data_source_idx}.{i}"

        if (
            (not filter_metrics)
            or (metric_idx in filter_metrics)  # no filter  # metric in filter
            or
            # the whole table in filter
            (data_source_idx in filter_metrics)
            or
            # the whole IP block in filter
            (str(panel_id // 100) in filter_metrics)
        ):

            values = build_metric_row_values(
                metric_idx, key, entries, data_config, panel
            )

            df_new_row = pd.DataFrame([values], columns=headers)
            df = pd.concat([df, df_new_row])

            metric_list[data_source_idx] = data_config["title"]

        # Collect metric counters
        metric_list[metric_idx] = key
        collect_metric_counters(entries, key, metric_counters)

    df.set_index("Metric_ID", inplace=True)
    return df


def build_raw_csv_table_df(
    data_config: dict[str, Any],
    filter_metrics: Optional[list[str]],
    metric_list: dict[str, str],
) -> pd.DataFrame:
    data_source_idx = str(data_config["id"] // 100)

    if (
        not filter_metrics
        or data_source_idx == "0"
        or data_source_idx in filter_metrics
    ):
        column = "from_csv_columnwise" if data_config.get("columnwise") else "from_csv"
        df = pd.DataFrame([data_config["source"]], columns=[column])
        metric_list[data_source_idx] = data_config["title"]
        return df

    return pd.DataFrame()


def build_pc_sampling_table_df(
    data_config: dict[str, Any], metric_list: dict[str, str]
) -> pd.DataFrame:
    data_source_idx = str(data_config["id"] // 100)
    df = pd.DataFrame([data_config["source"]], columns=["from_pc_sampling"])
    metric_list[data_source_idx] = data_config["title"]
    return df


@demarcate
def build_dfs(
    arch_configs: Any, filter_metrics: Optional[list[str]], sys_info: Any
) -> None:
    """
    - Build dataframe for each type of data source within each panel.
      Each dataframe will be used as a template to load data with each run later.
      For now, support "metric_table" and "raw_csv_table". Otherwise, put an empty df.
    - Collect/build metric_list to suport customrized metrics profiling.
    """

    # TODO: more error checking for filter_metrics!!
    # if filter_metrics:
    #     for metric in filter_metrics:
    #         if not metric in avail_ip_blocks:
    #             print("{} is not a valid metric to filter".format(metric))
    #             exit(1)

    d: dict[int, pd.DataFrame] = {}
    metric_list: dict[str, str] = {}
    dfs_type: dict[int, str] = {}
    metric_counters: dict[str, list[str]] = {}

    for panel_id, panel in arch_configs.panel_configs.items():
        for data_source in panel["data source"]:
            for type_key, data_config in data_source.items():
                if (
                    "metric_table" == type_key
                    and "metric" in data_config
                    and "placeholder_range" in data_config["metric"]
                ):
                    # NB: support single placeholder for now!!
                    new_metrics: dict[str, dict[str, str]] = {}
                    p_range = data_config["metric"].pop("placeholder_range")
                    metric, metric_expr = data_config["metric"].popitem()

                    for p, r in p_range.items():
                        # NB: We have to resolve placeholder range first if it
                        #   is a build-in var. It will be too late to do it in
                        #   eval_metric(). This is the only reason we need
                        #   sys_info at this stage.
                        var = calc_builtin_var(r, sys_info)
                        if var is not None:
                            for i in range(int(var)):
                                new_key = metric.replace(p, str(i))
                                new_val = {
                                    k: v.replace(p, str(i))
                                    for k, v in metric_expr.items()
                                }
                                new_metrics[new_key] = new_val

                    data_config["metric"] = new_metrics

    # Build dataframes
    for panel_id, panel in arch_configs.panel_configs.items():
        for data_source in panel["data source"]:
            for type_key, data_config in data_source.items():
                if type_key == "metric_table":
                    df = build_metric_table_df(
                        panel_id,
                        data_config,
                        panel,
                        filter_metrics,
                        metric_list,
                        metric_counters,
                    )
                elif type_key == "raw_csv_table":
                    df = build_raw_csv_table_df(
                        data_config, filter_metrics, metric_list
                    )
                elif type_key == "pc_sampling_table":
                    df = build_pc_sampling_table_df(data_config, metric_list)
                else:
                    df = pd.DataFrame()

                d[data_config["id"]] = df
                dfs_type[data_config["id"]] = type_key

    setattr(arch_configs, "dfs", d)
    setattr(arch_configs, "metric_list", metric_list)
    setattr(arch_configs, "dfs_type", dfs_type)
    setattr(arch_configs, "metric_counters", metric_counters)


def build_metric_value_string(
    dfs: dict[int, pd.DataFrame],
    dfs_type: dict[int, str],
    normal_unit: str,
    profiling_config: dict[str, Any],
) -> None:
    """
    Apply the real eval string to its field in the metric_table df.
    """

    for table_id, df in dfs.items():
        if dfs_type[table_id] == "metric_table":
            for expr in df.columns:
                if expr in schema.SUPPORTED_FIELD:
                    # NB: apply all build-in before building the whole string
                    df[expr] = df[expr].apply(update_denom_string, unit=normal_unit)

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

                elif expr.lower() in {"unit", "units"}:
                    df[expr] = df[expr].apply(update_normunit_string, unit=normal_unit)


def eval_builtin_vars(config: dict[str, Any]) -> dict[str, Any]:

    build_in: dict[str, Any] = {}

    # First pass: per-XCD values
    for key, value in BUILD_IN_VARS.items():
        if "PER_XCD" not in key:
            continue

        s = build_eval_string(value, schema.PMC_PERF_FILE_PREFIX, config)
        try:
            build_in[f"ammolite__{key}"] = eval(compile(s, "<string>", "eval"))
        except (TypeError, KeyError, AttributeError):
            build_in[f"ammolite__{key}"] = None

    # Update globals with per-XCD values for second pass
    globals().update(build_in)

    # Second pass: other built-in variables
    for key, value in BUILD_IN_VARS.items():
        if "PER_XCD" in key:
            continue

        s = build_eval_string(value, schema.PMC_PERF_FILE_PREFIX, config)
        try:
            build_in[f"ammolite__{key}"] = eval(compile(s, "<string>", "eval"))
        except (TypeError, KeyError, AttributeError):
            build_in[f"ammolite__{key}"] = None

    return build_in


def build_ammolite_vars(
    sys_info: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    ammolite_vars: dict[str, Any] = {}

    # System info variables
    sys_vars = [
        "se_per_gpu",
        "pipes_per_gpu",
        "cu_per_gpu",
        "simd_per_cu",
        "sqc_per_gpu",
        "lds_banks_per_cu",
        "cur_sclk",
        "cur_mclk",
        "max_mclk",
        "max_sclk",
        "max_waves_per_cu",
        "num_hbm_channels",
        "num_xcd",
        "wave_size",
    ]

    for var in sys_vars:
        value = getattr(sys_info, var, None)
        if (
            value is None
            or (isinstance(value, (int, float)) and np.isnan(value))
            or value == 0
        ):
            console_warning(
                f"{var} is not available in sysinfo.csv, please provide the correct "
                "value using --specs-correction"
            )
            value = 0

        if var in {
            "se_per_gpu",
            "pipes_per_gpu",
            "cu_per_gpu",
            "simd_per_cu",
            "sqc_per_gpu",
            "lds_banks_per_cu",
            "max_waves_per_cu",
            "num_xcd",
            "wave_size",
        }:
            ammolite_vars[f"ammolite__{var}"] = int(value)
        else:
            ammolite_vars[f"ammolite__{var}"] = float(value)

    # Total L2 channels
    ammolite_vars["ammolite__total_l2_chan"] = calc_builtin_var(
        "$total_l2_chan", sys_info
    )

    # TODO: fix all $normUnit in Unit column or title
    # Build derived variables
    build_in = eval_builtin_vars(config)
    ammolite_vars.update(build_in)

    return ammolite_vars


def debug_expression(
    expr: str,
    row_expr: str,
    raw_pmc_df: dict[str, pd.DataFrame],
    empirical_peaks: dict[str, Any],
) -> None:
    print("~" * 40 + "\nExpression:")
    print(f"{expr} = {row_expr}")
    print("Inputs:")

    matched_vars = re.findall(r"ammolite__\w+", row_expr)
    if matched_vars:
        for v in matched_vars:
            try:
                value = eval(compile(v, "<string>", "eval"))
                print("Var ", v, ":", value)
            except NameError:
                if "_empirical_peak" in v:
                    if v in empirical_peaks:
                        print(
                            "Var ",
                            v,
                            ":",
                            empirical_peaks[v],
                        )
                    else:
                        print(
                            "Var ",
                            v,
                            ": [empirical peak not found]",  # noqa
                        )
                else:
                    print(
                        "Var ",
                        v,
                        ": [not available in main thread]",  # noqa
                    )

    matched_cols = re.findall(r"raw_pmc_df\['\w+'\]\['\w+'\]", row_expr)
    if matched_cols:
        for c in matched_cols:
            m = re.match(r"raw_pmc_df\['(\w+)'\]\['(\w+)'\]", c)
            if m:
                try:
                    data = raw_pmc_df[m.group(1)][m.group(2)].tolist()
                    print(f"{c}: {data}")
                except KeyError as ke:
                    console_warning(f"Skipping entry. Encountered a missing key: {ke}")

    print("\nOutput:")
    try:
        print(eval(compile(row_expr, "<string>", "eval")))
        print("~" * 40)
    except NameError as ne:
        if "empirical_peak" in str(ne):
            console_warning(
                "Skipping debug evaluation. Empirical peak variables "  # noqa
                "not available in main thread: {}".format(str(ne))  # noqa
            )
        else:
            console_warning(
                "Skipping debug evaluation. Variable not available: {}".format(  # noqa
                    str(ne)
                )
            )
        print("~" * 40)
    except (TypeError, KeyError) as e:
        console_warning(f"Skipping entry. Encountered error: {e}")
    except AttributeError as ae:
        if str(ae) == "'NoneType' object has no attribute 'get'":
            console_warning(f"Skipping entry. Encountered missing csv: {np.nan}")
        else:
            console_error("analysis", str(ae))


def collect_expressions_for_evaluation(
    df: pd.DataFrame,
    table_id: int,
    debug: bool,
    raw_pmc_df: dict[str, pd.DataFrame],
    row_expr_indexes: list[tuple[int, Any, str]],
    row_exprs: list[str],
    empirical_peaks: dict[str, Any],
) -> None:
    for idx, row in df.iterrows():
        for expr in df.columns:
            if expr in schema.SUPPORTED_FIELD and expr.lower() != "alias":
                if row[expr]:
                    row_expr_indexes.append((table_id, idx, expr))
                    row_exprs.append(row[expr])

                    if debug:
                        debug_expression(expr, row[expr], raw_pmc_df, empirical_peaks)
                else:
                    row[expr] = ""


def create_empirical_peaks_dict(empirical_peaks_df: pd.DataFrame) -> dict[str, Any]:
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


@demarcate
def eval_metric(
    dfs: dict[int, pd.DataFrame],
    dfs_type: dict[int, str],
    sys_info: Any,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: dict[str, pd.DataFrame],
    debug: bool,
    config: dict[str, Any],
) -> None:

    # confirm no illogical counter values (only consider non-roofline runs)
    roof_only_run = sys_info.ip_blocks == "roofline"
    if (
        (not roof_only_run)
        and hasattr(raw_pmc_df.get("pmc_perf", pd.DataFrame()), "GRBM_GUI_ACTIVE")
        and "GRBM_GUI_ACTIVE" in raw_pmc_df["pmc_perf"].columns
        and (raw_pmc_df["pmc_perf"]["GRBM_GUI_ACTIVE"] == 0).any()
    ):
        console_warning("Dectected GRBM_GUI_ACTIVE == 0")
        console_error("Halting  execution for warning above.")

    # Build ammolite variables from sys_info
    ammolite_vars = build_ammolite_vars(sys_info, empirical_peaks_df, config)
    empirical_peaks = create_empirical_peaks_dict(empirical_peaks_df)

    # Collect expressions to evaluate
    row_expr_indexes: list[tuple[int, Any, str]] = []
    row_exprs: list[str] = []

    for table_id, df in dfs.items():
        if dfs_type[table_id] == "metric_table":
            collect_expressions_for_evaluation(
                df,
                table_id,
                debug,
                raw_pmc_df,
                row_expr_indexes,
                row_exprs,
                empirical_peaks,
            )

    # Empirically, 16 is about as much as we need.
    processes = min(16, multiprocessing.cpu_count() // 2)

    # breakpoint()
    with multiprocessing.Pool(
        processes=processes,
        initializer=init_metric_evaluator,
        initargs=(raw_pmc_df, ammolite_vars, empirical_peaks),
    ) as pool:
        outs = pool.map(run_metric_evaluator, row_exprs)

    # update dataframes with results
    for (df_id, row, col), out in zip(row_expr_indexes, outs):
        dfs[df_id].loc[row, col] = out


@demarcate
def apply_filters(
    workload: Any, directory: str, is_gui: bool, debug: bool
) -> pd.DataFrame:
    """
    Apply user's filters to the raw_pmc df.
    """

    ret_df = workload.raw_pmc.copy()

    if workload.filter_nodes:
        ret_df = ret_df.loc[
            ret_df[schema.PMC_PERF_FILE_PREFIX]["Node"]
            .astype(str)
            .isin([workload.filter_gpu_ids])
        ]
        if ret_df.empty:
            console_error("analysis", f"{workload.filter_nodes} is invalid")

    if workload.filter_gpu_ids:
        ret_df = ret_df.loc[
            ret_df[schema.PMC_PERF_FILE_PREFIX]["GPU_ID"]
            .astype(str)
            .isin([workload.filter_gpu_ids])
        ]
        if ret_df.empty:
            console_error("analysis", f"{workload.filter_gpu_ids} is an invalid gpu-id")

    # NB:
    # Kernel id is unique!
    # We pick up kernel names from kerne ids first.
    # Then filter valid entries with kernel names.
    if workload.filter_kernel_ids:
        if all(isinstance(kid, int) for kid in workload.filter_kernel_ids):
            # Verify valid kernel filter
            kernels_df = pd.read_csv(str(Path(directory) / "pmc_kernel_top.csv"))
            for kernel_id in workload.filter_kernel_ids:
                if kernel_id >= len(kernels_df["Kernel_Name"]):
                    console_error(
                        f"{kernel_id} is an invalid kernel id. "
                        f"Please enter an id between 0-{len(kernels_df['Kernel_Name']) - 1}"
                    )

            kernels = []
            # NB: mark selected kernels with "*"
            #     TODO: fix it for unaligned comparison
            kernel_top_df = workload.dfs[PMC_KERNEL_TOP_TABLE_ID]
            kernel_top_df["S"] = ""

            for kernel_id in workload.filter_kernel_ids:
                kernels.append(kernel_top_df.loc[kernel_id, "Kernel_Name"])
                kernel_top_df.loc[kernel_id, "S"] = "*"

            if kernels:
                ret_df = ret_df.loc[
                    ret_df[schema.PMC_PERF_FILE_PREFIX]["Kernel_Name"].isin(kernels)
                ]
        elif all(isinstance(kid, str) for kid in workload.filter_kernel_ids):
            df_cleaned = ret_df[schema.PMC_PERF_FILE_PREFIX]["Kernel_Name"].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            ret_df = ret_df.loc[df_cleaned.isin(workload.filter_kernel_ids)]
        else:
            console_error(
                "analyze",
                "Mixing kernel indices and string filters is not currently supported",
            )

    if workload.filter_dispatch_ids:
        # NB: support ignoring the 1st n dispatched execution by '> n'
        #     The better way may be parsing python slice string
        for d in workload.filter_dispatch_ids:
            if int(d) >= len(ret_df):  # subtract 2 bc of the two header rows
                console_error("analysis", f"{d} is an invalid dispatch id.")

        first_filter = workload.filter_dispatch_ids[0]
        if first_filter.startswith(">"):
            m = re.match(r">\s*(\d+)", first_filter)
            if m:
                threshold = int(m.group(1))
                ret_df = ret_df[
                    ret_df[schema.PMC_PERF_FILE_PREFIX]["Dispatch_ID"] > threshold
                ]
        else:
            dispatches = [int(x) for x in workload.filter_dispatch_ids]
            ret_df = ret_df.loc[dispatches]

    if debug:
        print("~" * 40, "\nraw pmc df info:\n")
        print(workload.raw_pmc.info())
        print("~" * 40, "\nfiltered pmc df info:")
        print(ret_df.info())

    return ret_df


def find_key_recursively(data: Any, search_key: str) -> Optional[Any]:
    """
    Recursively search for the search_key in the given data
    (which can be a dict or list).
    If the key is found, returns the value as a DataFrame.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == search_key:
                # Convert JSON value to DataFrame
                return value
            elif isinstance(value, (dict, list)):
                result = find_key_recursively(value, search_key)
                if result is not None:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = find_key_recursively(item, search_key)
            if result is not None:
                return result
    return None


def search_key_in_json(file_path: str, search_key: str) -> Optional[Any]:
    # FIXME:
    #   Load the entire JSON into memory.
    #   Should not use for large file.
    with open(file_path, "r") as file:
        data = json.load(file)
        found = find_key_recursively(data, search_key)
        if found is None:
            console_error(f"Key '{search_key}' not found in the JSON file.")
        return found


def search_pc_sampling_record(
    records: list[dict[str, Any]],
) -> Optional[list[tuple[Any, ...]]]:
    """
    Search PC sampling records, and group and sort them
    """

    # NB:
    #  The field stall_reason is vailid only for HW stochastic pc sampling.

    # Todo: might save wavefront count for HW stochastic pc sampling?

    grouped_data: defaultdict[str, defaultdict[int, Any]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "count": 0,
                "count_issued": 0,
                "count_stalled": 0,
                "inst_index": None,
                "stall_reason": {
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
                },
            }
        )
    )

    rocp_inst_not_issued_prefix_len = len(PC_SAMPLING_NOT_ISSUE_PREFIX)

    # Populate grouped_data
    for item in records:
        pc_info = item["record"].get("pc", {})
        code_object_id = pc_info.get("code_object_id")
        code_object_offset = pc_info.get("code_object_offset")
        snapshot = item["record"].get("snapshot", {})
        inst_index = item.get("inst_index")
        issued = item["record"].get("wave_issued")

        # TODO: opt me
        if (
            code_object_id is not None
            and code_object_offset is not None
            and inst_index is not None
        ):
            grouped_data[code_object_id][code_object_offset]["count"] += 1
            # NB: the write here could be duplicated. If there is perf issue,
            # We might want to opt it.
            grouped_data[code_object_id][code_object_offset]["inst_index"] = inst_index

            if snapshot:
                if issued:
                    grouped_data[code_object_id][code_object_offset][
                        "count_issued"
                    ] += 1
                else:
                    grouped_data[code_object_id][code_object_offset][
                        "count_stalled"
                    ] += 1
                    stall_reason = snapshot.get("stall_reason", "")
                    if stall_reason.startswith(
                        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"
                    ):
                        reason_key = stall_reason[rocp_inst_not_issued_prefix_len:]
                        grouped_data[code_object_id][code_object_offset][
                            "stall_reason"
                        ][reason_key] += 1

    if not grouped_data:
        console_warning("PC sampling: no pc sampling record found!")
        return None

    # Convert to sorted list of tuples
    sorted_counts = sorted(
        [
            (
                code_object_id,
                info["inst_index"],
                offset,
                info["count"],
                info["count_issued"],
                info["count_stalled"],
                # For info["stall_reason"], remove the zero entries,
                # sorting the remaining items by their values in descending order
                sorted(
                    ((k, v) for k, v in info["stall_reason"].items() if v > 0),
                    key=lambda item: item[1],
                    reverse=True,
                ),
            )
            for code_object_id, offsets in grouped_data.items()
            for offset, info in offsets.items()
        ],
        key=lambda x: (
            x[0],
            x[2],
        ),  # Sort by code_object_id, then by code_object_offset
    )

    return sorted_counts


@demarcate
def load_pc_sampling_data_per_kernel(
    method: str, file_name: str, kernel_name: str, sorting_type: str
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
    kernel_info_list = search_key_in_json(file_name, "kernel_symbols") or []

    kernel_info = {}
    if kernel_info_list:
        for item in kernel_info_list:
            if (
                item["formatted_kernel_name"] == kernel_name
                or item["demangled_kernel_name"] == kernel_name
                or item["truncated_kernel_name"] == kernel_name
            ):
                kernel_info["code_object_id"] = item["code_object_id"]
                kernel_info["entry_byte_offset"] = item["kernel_code_entry_byte_offset"]
                break

    if not kernel_info:
        console_warning("PC sampling: can not find the kernel %s " % kernel_name)
        return pd.DataFrame()

    console_debug("PC sampling: kernel %s " % kernel_info)

    filtered_sorted_list = sorted(
        [
            item
            for item in kernel_info_list
            if item["code_object_id"] == kernel_info["code_object_id"]
        ],
        key=lambda x: x["kernel_code_entry_byte_offset"],
    )

    for i, item in enumerate(filtered_sorted_list):
        if item["kernel_code_entry_byte_offset"] == kernel_info["entry_byte_offset"]:
            next_index = i + 1
            if next_index < len(filtered_sorted_list):  # Ensure the next item exists
                next_item = filtered_sorted_list[next_index]
                kernel_info["potential_end_offset"] = next_item[
                    "kernel_code_entry_byte_offset"
                ]
            else:
                kernel_info["potential_end_offset"] = sys.maxsize
            break

    pc_sample_key_loc = (
        search_key_in_json(file_name, "pc_sample_host_trap")
        if method == "host_trap"
        else search_key_in_json(file_name, "pc_sample_stochastic")
    )

    if not pc_sample_key_loc:
        pc_sample_key_loc = []

    df = pd.DataFrame(
        search_pc_sampling_record(pc_sample_key_loc),
        columns=[
            "code_object_id",
            "inst_index",
            "offset",
            "count",
            "count_issued",
            "count_stalled",
            "stall_reason",
        ],
    )

    df = df[
        (df["code_object_id"] == kernel_info["code_object_id"])
        & (df["offset"] > kernel_info["entry_byte_offset"])
        & (df["offset"] < kernel_info["potential_end_offset"])
    ][
        [
            "inst_index",
            "offset",
            "count",
            "count_issued",
            "count_stalled",
            "stall_reason",
        ]
    ]

    df["offset"] = df["offset"].apply(lambda x: hex(x))

    pc_sample_instructions = (
        search_key_in_json(file_name, "pc_sample_instructions") or {}
    )

    df["instruction"] = df["inst_index"].apply(
        lambda x: pc_sample_instructions[x] if x < len(pc_sample_instructions) else None
    )

    pc_sample_comments = search_key_in_json(file_name, "pc_sample_comments") or {}

    df["source_line"] = df["inst_index"].apply(
        lambda x: (
            str(Path("...") / Path(pc_sample_comments[x]).name)
            if x < len(pc_sample_instructions)
            else None
        )
    )

    if sorting_type == "offset":
        return (
            df[["source_line", "instruction", "offset", "count"]]
            if method == "host_trap"
            else df[
                [
                    "source_line",
                    "instruction",
                    "offset",
                    "count",
                    "count_issued",
                    "count_stalled",
                    "stall_reason",
                ]
            ]
        )
    else:  # sort by "count"
        return (
            df[["source_line", "instruction", "offset", "count"]].sort_values(
                by="count", ascending=False
            )
            if method == "host_trap"
            else df[
                [
                    "source_line",
                    "instruction",
                    "offset",
                    "count",
                    "count_issued",
                    "count_stalled",
                    "stall_reason",
                ]
            ].sort_values(by="count", ascending=False)
        )
    # might support sort by stall reason in the future


@demarcate
def load_pc_sampling_data(
    workload: Any, directory: str, file_prefix: str, sorting_type: str
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

    stochastic_path = Path(directory) / f"{file_prefix}_pc_sampling_stochastic.csv"
    host_trap_path = Path(directory) / f"{file_prefix}_pc_sampling_host_trap.csv"

    if stochastic_path.exists():
        pc_sampling_method = "stochastic"
        csv_file_path = stochastic_path
    elif host_trap_path.exists():
        pc_sampling_method = "host_trap"
        csv_file_path = host_trap_path
    else:
        console_warning(
            f"PC sampling: cannot detect pc sampling method for {file_prefix}"
        )
        return pd.DataFrame()

    # No kernel filter, return grouped and sorted csv directly
    if not workload.filter_kernel_ids:
        df = pd.read_csv(str(csv_file_path))
        # Group by 'Instruction_Comment' and count occurrences
        grouped_counts = (
            df.groupby("Instruction_Comment")
            .agg(
                count=("Instruction_Comment", "count"),
                instruction=("Instruction", "first"),
            )
            .reset_index()
            .rename(columns={"Instruction_Comment": "source_line"})
        )

        grouped_counts = grouped_counts[["source_line", "instruction", "count"]]
        grouped_counts["source_line"] = grouped_counts["source_line"].apply(
            lambda x: f".../{Path(x).name}"
        )

        # Sort by the count of occurrences
        return grouped_counts.sort_values(by="count", ascending=False)

    elif len(workload.filter_kernel_ids) > 1:
        console_error(
            "PC sampling supports single kernel only! Please specify -k with "
            "single kernel."
        )
        return pd.DataFrame()

    elif len(workload.filter_kernel_ids) == 1:
        # NB: the default file name is subject to changes from rocprofv3/rocprofiler_sdk
        json_file_path = Path(directory) / f"{file_prefix}_results.json"
        if not json_file_path.exists():
            console_error(f"PC sampling: cannot read {json_file_path}")
            return pd.DataFrame()

        # NB:
        #   We should find better way to remove the dependency on kernel_top_table
        kernel_top_df = workload.dfs[PMC_KERNEL_TOP_TABLE_ID]
        file_path = Path(directory) / kernel_top_df.loc[0, "from_csv"]
        kernel_name = pd.read_csv(str(file_path)).loc[
            workload.filter_kernel_ids[0], "Kernel_Name"
        ]

        return load_pc_sampling_data_per_kernel(
            pc_sampling_method, str(json_file_path), kernel_name, sorting_type
        )
    else:
        console_warning("PC sampling: No data")
        return pd.DataFrame()


@demarcate
def load_non_mertrics_table(workload: Any, directory: str, args: Any) -> None:
    # NB:
    #   - Do pmc_kernel_top.csv loading before eval_metric because we need the
    #     kernel names.
    #   - There might be a better way/timing to load raw_csv_table.

    # NB:
    #   "from_csv", "from_csv_columnwise", and "from_pc_sampling"
    #   are 3 internal symbols converted in build_dfs() for non-metrics table.
    #   There might be better way to store these info without the orginal entry.
    tmp: dict[int, pd.DataFrame] = {}

    for table_id, df in workload.dfs.items():
        if "from_csv" in df.columns:
            file_path = Path(directory) / df.loc[0, "from_csv"]
            if file_path.exists():
                tmp[table_id] = pd.read_csv(str(file_path))
            else:
                console_warning(
                    f"Couldn't load {file_path.name}. "
                    "This may result in missing analysis data."
                )
        elif "from_csv_columnwise" in df.columns:
            if table_id == 101:
                # NB: Special case for sysinfo. Probably room for improvement in this whole
                # function design
                tmp[table_id] = workload.sys_info.transpose()
                # All transposed columns should be marked with a general header
                tmp[table_id].columns = ["Info"]
            else:
                # NB:
                #   Another way might be doing transpose in tty like metric_table.
                #   But we need to figure out headers and comparison properly.
                file_path = Path(directory) / df.loc[0, "from_csv_columnwise"]
                if file_path.exists():
                    tmp[table_id] = pd.read_csv(str(file_path)).transpose()
                    # NB:
                    #   All transposed columns should be marked with a general header,
                    #   so tty could detect them and show them correctly in comparison.
                    tmp[table_id].columns = ["Info"]
                else:
                    console_warning(
                        f"Couldn't load {file_path.name}. "
                        "This may result in missing analysis data."
                    )
        elif "from_pc_sampling" in df.columns:
            tmp[table_id] = load_pc_sampling_data(
                workload,
                directory,
                df.loc[0, "from_pc_sampling"],
                args.pc_sampling_sorting_type,
            )

    workload.dfs.update(tmp)


@demarcate
def load_table_data(
    workload: Any,
    directory: str,
    is_gui: bool,
    args: Any,
    config: dict[str, Any],
    skip_kernel_top: bool = False,
) -> None:
    """
    - Load data for all "raw_csv_table"
    - Load data for "pc_sampling_table"
    - Calculate mertric value for all "metric_table"
    """
    if not skip_kernel_top:
        load_non_mertrics_table(workload, directory, args)

    eval_metric(
        workload.dfs,
        workload.dfs_type,
        workload.sys_info.iloc[0],
        workload.roofline_peaks,
        apply_filters(workload, directory, is_gui, args.debug),
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

    for header in top_stat_base:
        comparable_columns.append(f"{header}({time_unit})")

    return comparable_columns


def correct_sys_info(mspec: Any, specs_correction: str) -> Any:
    """
    Correct system spec items manually
    """
    # TODO: more err checking for string specs_correction

    pairs = dict(re.findall(r"(\w+):\s*(\d+)", specs_correction))

    for k, v in pairs.items():
        if not hasattr(mspec, str(k)):
            console_error(
                "analyze",
                f"Invalid specs correction '{k}'. Please use --specs option "
                f"to peak valid specs",
            )
        setattr(mspec, str(k), v)
    return mspec.get_class_members()
