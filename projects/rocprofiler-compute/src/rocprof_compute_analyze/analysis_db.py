# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import ast
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional, Union

import astunparse
import numpy as np
import pandas as pd

import utils.analysis_orm as orm
from config import rocprof_compute_home
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import utils_analysis
from utils.analysis_orm import Database, get_views
from utils.logger import console_debug, console_error, console_warning, demarcate
from utils.parser import (
    BUILD_IN_VARS,
    PC_SAMPLING_NOT_ISSUE_PREFIX,
    CodeTransformer,
    to_avg,
    to_concat,
    to_int,
    to_max,
    to_median,
    to_min,
    to_mod,
    to_noise_clamp,
    to_quantile,
    to_round,
    to_std,
    to_sum,
)
from utils.roofline_calc import (
    CACHE_HIERARCHY,
    MFMA_DATATYPES,
    PEAK_OPS_DATATYPES,
    SUPPORTED_DATATYPES,
)
from utils.utils_common import get_uuid, get_version


class db_analysis(OmniAnalyze_Base):
    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()
        if self._profiling_config.get("format_rocprof_output") != "rocpd":
            console_error(
                "Creation of analysis database is only supported "
                "for profiling data with rocpd output format."
            )
        self._roofline_ceilings_per_workload = self.calc_roofline_ceilings()
        self._pc_sampling_data_per_workload = self.calc_pc_sampling_data()
        self._pmc_df_per_workload = self.calc_pmc_df_data()
        self._pmc_df_per_workload = self.apply_pmc_filters()
        self._dispatch_data_per_workload = self.calc_dispatch_data()
        (
            self._metrics_info_data_per_workload,
            self._metric_expression_data_per_workload,
        ) = self.calc_metrics_data()
        (
            self._kernel_values_data_per_workload,
            self._workload_values_data_per_workload,
        ) = self.calc_expressions()
        self._roofline_data_per_kernel, self._roofline_data_per_workload = (
            self.calc_roofline_data()
        )

    @demarcate
    def run_analysis(self) -> None:
        """Run CLI analysis."""
        super().run_analysis()

        # Initialize analysis database
        # Create db uuid
        if self.get_args().output_name:
            db_name = f"{self.get_args().output_name}.db"
        else:
            db_name = f"rocprof_compute_{get_uuid()}.db"
        Database.init(db_name)
        console_debug(f"Initialized database: {db_name}")

        # Iterate over all workloads
        for workload_path in self._runs.keys():
            # Add workload
            workload_obj = orm.Workload(
                name=workload_path.split("/")[-2],
                sub_name=workload_path.split("/")[-1],
                sys_info_extdata=self._runs[workload_path].sys_info.iloc[0].to_dict(),
                roofline_bench_extdata=self._roofline_ceilings_per_workload.get(
                    workload_path
                ),
                profiling_config_extdata=self._profiling_config,
            )
            Database.get_session().add(workload_obj)

            # Add kernel
            kernel_objs: dict[str, orm.Kernel] = {}

            for dispatch in self._dispatch_data_per_workload.get(
                workload_path, pd.DataFrame()
            ).itertuples():
                # Add kernel object and map it, if not already added
                if dispatch.kernel_name not in kernel_objs:
                    kernel_objs[dispatch.kernel_name] = orm.Kernel(
                        kernel_name=dispatch.kernel_name,
                        workload=workload_obj,
                    )
                    Database.get_session().add(kernel_objs[dispatch.kernel_name])

                # Add dispatch object and link with kernel object
                Database.get_session().add(
                    orm.Dispatch(
                        dispatch_id=dispatch.dispatch_id,
                        gpu_id=dispatch.gpu_id,
                        start_timestamp=dispatch.start_timestamp,
                        end_timestamp=dispatch.end_timestamp,
                        kernel=kernel_objs[dispatch.kernel_name],
                    )
                )

            # Add kernel-level roofline data points
            for roofline_data in self._roofline_data_per_kernel.get(
                workload_path, pd.DataFrame()
            ).itertuples():
                if roofline_data.kernel_name not in kernel_objs:
                    console_warning(
                        f"Kernel {roofline_data.kernel_name} from roofline data "
                        "not found in dispatch data. Skipping roofline entry."
                    )
                    continue
                Database.get_session().add(
                    orm.KernelRooflineData(
                        total_flops=roofline_data.total_flops,
                        l1_cache_data=roofline_data.l1_cache_data,
                        l2_cache_data=roofline_data.l2_cache_data,
                        hbm_cache_data=roofline_data.hbm_cache_data,
                        kernel=kernel_objs[roofline_data.kernel_name],
                    )
                )

            # Add workload-level roofline data
            workload_roofline = self._roofline_data_per_workload.get(workload_path)
            if workload_roofline:
                Database.get_session().add(
                    orm.WorkloadRooflineData(
                        total_flops=workload_roofline.get("total_flops"),
                        l1_cache_data=workload_roofline.get("l1_cache_data"),
                        l2_cache_data=workload_roofline.get("l2_cache_data"),
                        hbm_cache_data=workload_roofline.get("hbm_cache_data"),
                        workload=workload_obj,
                    )
                )

            # Add pc sampling data
            for pc_sample in self._pc_sampling_data_per_workload.get(
                workload_path, pd.DataFrame()
            ).itertuples():
                if pc_sample.kernel_name not in kernel_objs:
                    console_warning(
                        f"Kernel {pc_sample.kernel_name} from PC sampling data "
                        "not found in dispatch data. Skipping PC sampling entry."
                    )
                    continue
                Database.get_session().add(
                    orm.PCsampling(
                        source=pc_sample.source_line,
                        instruction=pc_sample.instruction,
                        count=pc_sample.count,
                        offset=pc_sample.offset,
                        count_issue=pc_sample.count_issued,
                        count_stall=pc_sample.count_stalled,
                        stall_reason=pc_sample.stall_reason,
                        kernel=kernel_objs[pc_sample.kernel_name],
                    )
                )

            # Add metrics and values - iterate on values, create metrics as needed
            self.run_analysis_metrics(workload_path, workload_obj, kernel_objs)

            # Add metadata
            version = get_version(rocprof_compute_home)
            Database.get_session().add(
                orm.Metadata(
                    compute_version=version["version"],
                    git_version=version["sha"],
                    schema_version=orm.SCHEMA_VERSION,
                )
            )

        # Create views
        for view_stmt in get_views():
            Database.get_session().execute(view_stmt)

        # Write database
        Database.write()
        console_debug("Completed writing database")
        console_warning(f"Created file: {db_name}")

    def run_analysis_metrics(
        self,
        workload_path: str,
        workload_obj: orm.Workload,
        kernel_objs: dict[str, orm.Kernel],
    ) -> None:
        """Add metric definitions and metric values to the database."""
        # Add metrics and values - iterate on values, create metrics as needed
        metrics_info_dict = {
            row.metric_id: row
            for row in self._metrics_info_data_per_workload.get(
                workload_path, pd.DataFrame()
            ).itertuples()
        }
        metric_objs: dict[str, orm.MetricDefinition] = {}

        for value in self._kernel_values_data_per_workload.get(
            workload_path, pd.DataFrame()
        ).itertuples():
            # Check if kernel exists
            if value.kernel_name not in kernel_objs:
                console_warning(
                    f"Kernel {value.kernel_name} from values data "
                    "not found in dispatch data. Skipping metric value."
                )
                continue

            # Create or reuse metric object
            if value.metric_id not in metric_objs:
                # Fetch metric info
                if value.metric_id not in metrics_info_dict:
                    console_warning(
                        f"Metric {value.metric_id} from values data "
                        "not found in metrics info. Skipping metric value."
                    )
                    continue
                metric_info = metrics_info_dict[value.metric_id]
                metric_objs[value.metric_id] = orm.MetricDefinition(
                    name=metric_info.name,
                    metric_id=metric_info.metric_id,
                    description=metric_info.description,
                    unit=metric_info.unit,
                    table_name=metric_info.table_name,
                    sub_table_name=metric_info.sub_table_name,
                    workload=workload_obj,
                )
                Database.get_session().add(metric_objs[value.metric_id])

            # Add kernel-level metric value
            Database.get_session().add(
                orm.KernelMetricValue(
                    metric=metric_objs[value.metric_id],
                    kernel=kernel_objs[value.kernel_name],
                    value_name=value.value_name,
                    value=value.value,
                )
            )

        # Add workload-level metric values
        for value in self._workload_values_data_per_workload.get(
            workload_path, pd.DataFrame()
        ).itertuples():
            if value.metric_id not in metric_objs:
                console_warning(
                    f"Metric {value.metric_id} from workload values data "
                    "not found in metric objects. Skipping workload metric value."
                )
                continue

            Database.get_session().add(
                orm.WorkloadMetricValue(
                    metric=metric_objs[value.metric_id],
                    workload=workload_obj,
                    value_name=value.value_name,
                    value=value.value,
                )
            )

    def calc_pmc_df_data(self) -> dict[str, pd.DataFrame]:
        pmc_df_per_workload: dict[str, pd.DataFrame] = {}
        args = self.get_args()

        for workload_path in self._runs.keys():
            pmc_df = utils_analysis.process_rocpd_csv(
                pd.read_csv(Path(workload_path) / "pmc_perf.csv")
            )

            # Create multi index df with collection level as pmc_perf
            raw_pmc = pd.concat([pmc_df], keys=["pmc_perf"], axis=1, copy=False)

            if args.spatial_multiplexing:
                raw_pmc = self.spatial_multiplex_merge_counters(raw_pmc)

            if self._profiling_config.get("iteration_multiplexing") is not None:
                raw_pmc = self.iteration_multiplex_impute_counters(
                    raw_pmc,
                    policy=self._profiling_config["iteration_multiplexing"],
                )

            pmc_df_per_workload[workload_path] = raw_pmc["pmc_perf"]

        console_debug("Collected dispatch data")
        return pmc_df_per_workload

    def calc_roofline_ceilings(self) -> dict[str, dict[str, Any]]:
        roofline_ceilings_per_workload: dict[str, dict[str, Any]] = {}

        for workload_path in self._runs.keys():
            if not (Path(workload_path) / "roofline.csv").exists():
                console_warning(f"Roofline ceilings not found for {workload_path}.")
                continue

            roofline_dict = (
                pd.read_csv(f"{workload_path}/roofline.csv").iloc[0].to_dict()
            )
            keys: list[str] = []
            for mem_level in CACHE_HIERARCHY:
                keys.append(f"{mem_level}Bw")
            for dtype in SUPPORTED_DATATYPES[
                self._runs[workload_path].sys_info.iloc[0]["gpu_arch"]
            ]:
                if dtype in PEAK_OPS_DATATYPES:
                    if dtype.startswith("F") or dtype.startswith("B"):
                        keys.append(f"{dtype}Flops")
                    elif dtype.startswith("I"):
                        keys.append(f"{dtype}Ops")
                if dtype in MFMA_DATATYPES:
                    if dtype.startswith("F") or dtype.startswith("B"):
                        # FP16 -> F16
                        dtype = dtype.replace("FP", "F")
                        keys.append(f"MFMA{dtype}Flops")
                    elif dtype.startswith("I"):
                        keys.append(f"MFMA{dtype}Ops")
            roofline_ceilings_per_workload[workload_path] = {
                key: roofline_dict[key] for key in keys if key in roofline_dict
            }

        if roofline_ceilings_per_workload:
            console_debug("Collected roofline ceilings")
        return roofline_ceilings_per_workload

    def calc_pc_sampling_data(self) -> dict[str, pd.DataFrame]:
        pc_sampling_data_per_workload: dict[str, pd.DataFrame] = {}

        for workload_path in self._runs.keys():
            if not (Path(workload_path) / "ps_file_results.json").exists():
                console_warning(f"PC sampling data not found for {workload_path}.")
                continue

            pc_sampling_data = json.loads(
                (Path(workload_path) / "ps_file_results.json").read_text()
            )
            pc_sampling_data = pc_sampling_data["rocprofiler-sdk-tool"][0]
            pc_sampling_stochastic = pc_sampling_data["buffer_records"][
                "pc_sample_stochastic"
            ]
            pc_sampling_host_trap = pc_sampling_data["buffer_records"][
                "pc_sample_host_trap"
            ]
            pc_sampling_instruction = pc_sampling_data["strings"][
                "pc_sample_instructions"
            ]
            pc_sampling_comments = pc_sampling_data["strings"]["pc_sample_comments"]
            pc_sampling_kernel_name_dict = {
                symbol["code_object_id"]: symbol["formatted_kernel_name"]
                for symbol in pc_sampling_data["kernel_symbols"]
            }

            pc_df = pd.DataFrame([
                {
                    "inst_index": pc_sample["inst_index"],
                    "code_object_id": pc_sample["record"]["pc"]["code_object_id"],
                    "code_object_offset": pc_sample["record"]["pc"][
                        "code_object_offset"
                    ],
                    "stall_reason": pc_sample["record"]
                    .get("snapshot", {})
                    .get("stall_reason"),
                    "wave_issued": pc_sample["record"].get("wave_issued"),
                }
                for pc_sample in pc_sampling_stochastic + pc_sampling_host_trap
            ])

            def custom_aggregator(
                column_name: str,
            ) -> Callable[[pd.Series], Union[int, dict[str, int], None]]:
                if column_name == "count_issued":

                    def aggregator(series: pd.Series) -> Optional[int]:
                        return None if series.isnull().all() else series.sum()

                    return aggregator
                if column_name == "count_stalled":

                    def aggregator(series: pd.Series) -> Optional[int]:
                        if series.isnull().all():
                            return None
                        return series.count() - series.sum()

                    return aggregator
                if column_name == "stall_reason":

                    def aggregator(series: pd.Series) -> Optional[dict[str, int]]:
                        if series.isnull().all():
                            return None
                        cleaned_series = series.dropna().str[
                            len(PC_SAMPLING_NOT_ISSUE_PREFIX) :
                        ]
                        return cleaned_series.value_counts().to_dict()

                    return aggregator
                raise ValueError(f"Unknown column name: {column_name}")

            grouped_df = (
                pc_df
                .groupby(["code_object_id", "code_object_offset"])
                .agg(
                    count=("code_object_id", "size"),
                    inst_index=("inst_index", "last"),
                    count_issued=("wave_issued", custom_aggregator("count_issued")),
                    count_stalled=("wave_issued", custom_aggregator("count_stalled")),
                    stall_reason=("stall_reason", custom_aggregator("stall_reason")),
                )
                .reset_index()
            )

            grouped_df["instruction"] = grouped_df["inst_index"].apply(
                lambda x: (
                    pc_sampling_instruction[x]
                    if x < len(pc_sampling_instruction)
                    else None
                )
            )
            grouped_df["source_line"] = grouped_df["inst_index"].apply(
                lambda x: (
                    pc_sampling_comments[x] if x < len(pc_sampling_comments) else None
                )
            )
            grouped_df["kernel_name"] = grouped_df["code_object_id"].apply(
                lambda x: pc_sampling_kernel_name_dict.get(x)
            )
            grouped_df = grouped_df.rename(columns={"code_object_offset": "offset"})
            grouped_df = grouped_df.drop(columns=["code_object_id", "inst_index"])

            pc_sampling_data_per_workload[workload_path] = grouped_df

        if pc_sampling_data_per_workload:
            console_debug("Collected PC sampling data")
        return pc_sampling_data_per_workload

    @staticmethod
    def evaluate(
        name: str,
        value: str,
        pmc_df: pd.DataFrame,
        sys_info: dict[str, Any],  # noqa ANN401
        parse: bool = False,
    ) -> Any:  # noqa ANN401
        if parse:
            value = re.sub(
                r"\$([0-9A-Za-z_]+)",
                lambda m: f'sys_info["{m.group(1)}"]',
                value,
            )
            ast_node = ast.parse(value)
            transformer = CodeTransformer()
            transformer.visit(ast_node)
            value = astunparse.unparse(ast_node)
            value = value.replace("raw_pmc_df", "pmc_df")
            value = value.replace("pmc_df['sys_info']", "sys_info")
        else:
            value = value.replace("raw_pmc_df['pmc_perf']", "pmc_df")
            value = re.sub(
                "ammolite__([0-9A-Za-z_]+)",
                lambda m: f'sys_info["{m.group(1)}"]',
                value,
            )
        try:
            eval_result = eval(
                compile(value, "<string>", "eval"),
                {},  # no globals
                {
                    # only locals
                    "pmc_df": pmc_df,
                    "sys_info": sys_info,
                    "to_avg": to_avg,
                    "to_concat": to_concat,
                    "to_int": to_int,
                    "to_max": to_max,
                    "to_median": to_median,
                    "to_min": to_min,
                    "to_mod": to_mod,
                    "to_quantile": to_quantile,
                    "to_round": to_round,
                    "to_std": to_std,
                    "to_sum": to_sum,
                    "to_noise_clamp": to_noise_clamp,
                },
            )

            # eval_result can be None if expression has None explicitly specified
            # Do not give warning for this case and simply return None
            if eval_result is None:
                return None

            # Only return None for scalar NA values (NaN, pd.NA)
            # For vectors/Series, return as-is to preserve shape for downstream
            # operations. Note: pd.NA is not detected as scalar by np.isscalar()
            is_scalar_na = eval_result is pd.NA or (
                np.isscalar(eval_result) and pd.isna(eval_result)
            )

            if is_scalar_na:
                # Only warn if expression doesn't have "None" as an explicit fallback
                # Expressions with .where(..., None) are expected to return NA
                if "None" not in value:
                    console_warning(
                        f"Could not evaluate expression for {name}: {value} - "
                        "likely due to missing counter data."
                    )
                return None
            else:
                return eval_result
        except Exception as e:
            console_warning(f"Failed to evaluate expression for {name}: {value} - {e}")
            return None

    @staticmethod
    def calc_builtin_vars(pmc_df: pd.DataFrame, sys_info: dict) -> pd.DataFrame:
        """Calculate built-in variables (numActiveCUs, kernelBusyCycles, etc.)"""
        # Calculate PER_XCD variables first
        for key, value in BUILD_IN_VARS.items():
            if "PER_XCD" in key:
                sys_info[key] = db_analysis.evaluate(
                    key, value, pmc_df, sys_info, parse=True
                )
        # Variable dependent on PER_XCD variables
        for key, value in BUILD_IN_VARS.items():
            if "PER_XCD" not in key:
                sys_info[key] = db_analysis.evaluate(
                    key, value, pmc_df, sys_info, parse=True
                )
        return pmc_df

    @staticmethod
    def calc_dataframe_expressions(
        pmc_df: pd.DataFrame, sys_info: dict, expression_df: pd.DataFrame
    ) -> pd.Series:
        # Calculate built-in variables
        db_analysis.calc_builtin_vars(pmc_df, sys_info)
        # Evaluate expressions while printing warnings
        return expression_df.apply(
            lambda row: db_analysis.evaluate(
                f"{row['metric_id']} - {row['value_name']}",
                row["value"],
                pmc_df,
                sys_info,
            ),
            axis=1,
        )

    def calc_expressions(
        self,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Calculate both kernel-level and workload-level metrics"""
        kernel_values_data = {}
        workload_values_data = {}

        for workload_path in self._runs.keys():
            pmc_df = self._pmc_df_per_workload[workload_path]
            expression_template = self._metric_expression_data_per_workload[
                workload_path
            ]
            sys_info = self._runs[workload_path].sys_info.iloc[0].to_dict()
            for key, value in self._roofline_ceilings_per_workload.get(
                workload_path, {}
            ).items():
                sys_info[f"{key}_empirical_peak"] = value

            # Calculate kernel-level metrics
            kernel_values_list = []

            for kernel_name, kernel_pmc_df in pmc_df.groupby("Kernel_Name"):
                kernel_expression_df = expression_template.assign(
                    kernel_name=kernel_name
                )
                kernel_expression_df["value"] = db_analysis.calc_dataframe_expressions(
                    kernel_pmc_df,
                    sys_info.copy(),
                    kernel_expression_df,
                )
                kernel_values_list.append(kernel_expression_df)

            kernel_values_data[workload_path] = (
                pd.concat(kernel_values_list, ignore_index=True)
                if kernel_values_list
                else pd.DataFrame()
            )

            # Calculate workload-level metrics (aggregate across ALL dispatches)
            workload_values_data[workload_path] = expression_template.copy()
            workload_values_data[workload_path]["value"] = (
                db_analysis.calc_dataframe_expressions(
                    pmc_df,
                    sys_info.copy(),
                    workload_values_data[workload_path],
                )
            )

        console_debug("Calculated kernel-level and workload-level metric values")
        return kernel_values_data, workload_values_data

    def calc_metrics_data(
        self,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        metrics_info_data_per_workload: dict[str, pd.DataFrame] = {}
        metric_expression_data_per_workload: dict[str, pd.DataFrame] = {}

        for workload_path in self._runs.keys():
            gfx_arch = self._runs[workload_path].sys_info.iloc[0]["gpu_arch"]
            # for example 201 -> Wavefront
            table_names_map = dict()
            for panel_config in self._arch_configs[gfx_arch].panel_configs.values():
                table_names_map[panel_config["id"]] = panel_config["title"]
                for source in panel_config["data source"]:
                    table_names_map[list(source.values())[0]["id"]] = list(
                        source.values()
                    )[0]["title"]
            # Build metric data
            non_expression_columns = [
                "Metric",
                "Channel",
                "Unit",
                "Description",
                "coll_level",
                "Type",
                "Xfer",
                "Coherency",
                "Transaction",
            ]
            metrics_info_df = pd.DataFrame([
                {
                    "name": row.get("Metric") or row["Channel"].strip(),
                    "metric_id": metric_id,
                    "description": row.get("Description"),
                    "unit": row.get("Unit"),
                    "table_name": table_names_map[int(metric_id.split(".")[0]) * 100],
                    "sub_table_name": table_names_map[
                        int(metric_id.split(".")[0]) * 100
                        + int(metric_id.split(".")[1])
                    ],
                }
                for metric_df_id, metric_df in self._arch_configs[gfx_arch].dfs.items()
                if metric_df_id
                != 402  # Skip roofline data points handled in calc_roofline_data
                if set(metric_df.columns).intersection({"Metric", "Channel"})
                for metric_id, row in metric_df.iterrows()
            ])
            expression_df = pd.DataFrame([
                {
                    "metric_id": metric_id,
                    "value_name": value_name,
                    "value": row[value_name].strip(),
                }
                for metric_df_id, metric_df in self._arch_configs[gfx_arch].dfs.items()
                if metric_df_id
                != 402  # Skip roofline data points handled in calc_roofline_data
                if set(metric_df.columns).intersection({"Metric", "Channel"})
                for metric_id, row in metric_df.iterrows()
                for value_name in metric_df.drop(
                    columns=non_expression_columns, errors="ignore"
                ).columns
            ])

            metrics_info_data_per_workload[workload_path] = metrics_info_df
            metric_expression_data_per_workload[workload_path] = expression_df

        console_debug("Collected metrics data")
        return metrics_info_data_per_workload, metric_expression_data_per_workload

    def calc_dispatch_data(self) -> dict[str, pd.DataFrame]:
        dispatch_data_per_workload: dict[str, pd.DataFrame] = {}

        for workload_path in self._runs.keys():
            dispatch_df = pd.DataFrame([
                {
                    "dispatch_id": row.Dispatch_ID,
                    "kernel_name": row.Kernel_Name,
                    "gpu_id": row.GPU_ID,
                    "start_timestamp": row.Start_Timestamp,
                    "end_timestamp": row.End_Timestamp,
                }
                for row in self._pmc_df_per_workload[workload_path].itertuples()
            ])
            dispatch_data_per_workload[workload_path] = dispatch_df

        console_debug("Calculated dispatch data")
        return dispatch_data_per_workload

    def apply_pmc_filters(self) -> dict[str, pd.DataFrame]:
        pmc_df_per_workload = self._pmc_df_per_workload.copy()

        for workload_path, pmc_df in pmc_df_per_workload.items():
            top_kernels = (
                pmc_df
                .assign(duration=pmc_df["End_Timestamp"] - pmc_df["Start_Timestamp"])
                .sort_values(by="duration", ascending=False)
                .drop_duplicates("Kernel_Name")["Kernel_Name"]
                .to_list()
            )
            # Filter gpu_ids
            if self._runs[workload_path].filter_gpu_ids:
                pmc_df = pmc_df.loc[
                    pmc_df["GPU_ID"]
                    .astype(str)
                    .isin([self._runs[workload_path].filter_gpu_ids])
                ]
            # Filter kernel_ids
            if self._runs[workload_path].filter_kernel_ids:
                pmc_df = pmc_df.loc[
                    pmc_df["Kernel_Name"].isin([
                        top_kernels[id]
                        for id in self._runs[workload_path].filter_kernel_ids
                    ])
                ]
            # Filter dispatch_ids
            if self._runs[workload_path].filter_dispatch_ids:
                if ">" in self._runs[workload_path].filter_dispatch_ids[0]:
                    m = re.match(
                        r"\> (\d+)", self._runs[workload_path].filter_dispatch_ids[0]
                    )
                    pmc_df = pmc_df[pmc_df["Dispatch_ID"] > int(m.group(1))]
                else:
                    pmc_df = pmc_df.loc[
                        pmc_df["Dispatch_ID"]
                        .astype(str)
                        .isin(self._runs[workload_path].filter_dispatch_ids)
                    ]
            pmc_df_per_workload[workload_path] = pmc_df

        console_debug("Applied analysis mode filters")
        return pmc_df_per_workload

    def calc_roofline_data(self) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
        """Calculate both kernel-level and workload-level roofline data"""
        roofline_data_per_kernel: dict[str, pd.DataFrame] = {}
        roofline_data_per_workload: dict[str, dict] = {}

        for workload_path in self._runs.keys():
            pmc_df = self._pmc_df_per_workload[workload_path].copy()
            sys_info = self._runs[workload_path].sys_info.iloc[0].to_dict()
            gfx_arch = sys_info["gpu_arch"]
            roofline_data_df = self._arch_configs[gfx_arch].dfs[402]

            if roofline_data_df.empty:
                console_warning(
                    f"Roofline data is filtered out or not found for {workload_path}."
                )
                continue

            roofline_data_expressions = dict(
                zip(roofline_data_df["Metric"], roofline_data_df["Value"])
            )
            roofline_data_expressions = {
                "total_flops": roofline_data_expressions.get(
                    "Performance (GFLOPs)", ""
                ),
                "l1_cache_data": roofline_data_expressions.get("AI L1", ""),
                "l2_cache_data": roofline_data_expressions.get("AI L2", ""),
                "hbm_cache_data": roofline_data_expressions.get("AI HBM", ""),
            }

            # Calculate kernel-level roofline data
            top_kernels = (
                pmc_df
                .assign(duration=pmc_df["End_Timestamp"] - pmc_df["Start_Timestamp"])
                .sort_values(by="duration", ascending=False)
                .drop_duplicates("Kernel_Name")["Kernel_Name"]
                .to_list()
            )

            roofline_df = pd.DataFrame([
                {
                    "kernel_name": kernel_name,
                    **{
                        metric_name: db_analysis.evaluate(
                            metric_name,
                            roofline_data_expressions[metric_name],
                            pmc_df[pmc_df["Kernel_Name"] == kernel_name],
                            sys_info,
                        )
                        for metric_name in roofline_data_expressions
                        if roofline_data_expressions[metric_name]
                    },
                }
                for kernel_name in top_kernels[: self.get_args().max_stat_num]
            ])

            roofline_data_per_kernel[workload_path] = roofline_df

            # Calculate workload-level roofline data (using full dataframe)
            workload_roofline = {
                metric_name: db_analysis.evaluate(
                    metric_name,
                    roofline_data_expressions[metric_name],
                    pmc_df,
                    sys_info,
                )
                for metric_name in roofline_data_expressions
                if roofline_data_expressions[metric_name]
            }

            roofline_data_per_workload[workload_path] = workload_roofline

        console_debug("Calculated kernel-level and workload-level roofline data")
        return roofline_data_per_kernel, roofline_data_per_workload
