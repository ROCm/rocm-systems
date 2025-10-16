##############################################################################bl
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import ast
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional, Union

import astunparse
import pandas as pd

import utils.analysis_orm as orm
from config import rocprof_compute_home
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import rocpd_data
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
from utils.utils import get_uuid, get_version


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
        self._pmc_df_per_workload = {
            workload_path: rocpd_data.process_rocpd_csv(
                pd.read_csv(Path(workload_path) / "pmc_perf.csv")
            )
            for workload_path in self._runs.keys()
        }
        self._top_kernels_per_workload = {
            workload_path: pmc_df.assign(
                duration=pmc_df["End_Timestamp"] - pmc_df["Start_Timestamp"]
            )
            .sort_values(by="duration", ascending=False)
            .drop_duplicates("Kernel_Name")["Kernel_Name"]
            .to_list()
            for workload_path, pmc_df in self._pmc_df_per_workload.items()
        }
        console_debug("Collected dispatch data")
        self._pmc_df_per_workload = self.apply_pmc_filters()
        self._dispatch_data_per_workload = self.calc_dispatch_data()
        self._metrics_info_data_per_workload, self._values_data_per_workload = (
            self.calc_metrics_data()
        )
        self._values_data_per_workload = self.calc_expressions()
        self._roofline_data_per_workload = self.calc_roofline_data()

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

        for workload_path in self._runs.keys():
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
            for pc_sample in self._pc_sampling_data_per_workload.get(
                workload_path, pd.DataFrame()
            ).itertuples():
                Database.get_session().add(
                    orm.PCsampling(
                        source=pc_sample.source_line,
                        instruction=pc_sample.instruction,
                        count=pc_sample.count,
                        kernel_name=pc_sample.kernel_name,
                        offset=pc_sample.offset,
                        count_issue=pc_sample.count_issued,
                        count_stall=pc_sample.count_stalled,
                        stall_reason=pc_sample.stall_reason,
                        workload=workload_obj,
                    )
                )
            for dispatch in self._dispatch_data_per_workload.get(
                workload_path, pd.DataFrame()
            ).itertuples():
                Database.get_session().add(
                    orm.Dispatch(
                        dispatch_id=dispatch.dispatch_id,
                        kernel_name=dispatch.kernel_name,
                        gpu_id=dispatch.gpu_id,
                        duration=dispatch.duration,
                        workload=workload_obj,
                    )
                )
            for metric in self._metrics_info_data_per_workload.get(
                workload_path, pd.DataFrame()
            ).itertuples():
                metric_obj = orm.Metric(
                    name=metric.name,
                    metric_id=metric.metric_id,
                    description=metric.description,
                    unit=metric.unit,
                    table_name=metric.table_name,
                    sub_table_name=metric.sub_table_name,
                    workload=workload_obj,
                )
                Database.get_session().add(metric_obj)
                for value in self._values_data_per_workload.get(
                    workload_path, pd.DataFrame()
                ).itertuples():
                    if value.metric_id == metric.metric_id:
                        Database.get_session().add(
                            orm.Value(
                                metric=metric_obj,
                                value_name=value.value_name,
                                value=value.value,
                            )
                        )

            for roofline_data in self._roofline_data_per_workload.get(
                workload_path, pd.DataFrame()
            ).itertuples():
                Database.get_session().add(
                    orm.RooflineData(
                        kernel_name=roofline_data.kernel_name,
                        total_flops=roofline_data.total_flops,
                        l1_cache_data=roofline_data.l1_cache_data,
                        l2_cache_data=roofline_data.l2_cache_data,
                        hbm_cache_data=roofline_data.hbm_cache_data,
                        workload=workload_obj,
                    )
                )

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
                pc_df.groupby(["code_object_id", "code_object_offset"])
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
                lambda x: pc_sampling_instruction[x]
                if x < len(pc_sampling_instruction)
                else None
            )
            grouped_df["source_line"] = grouped_df["inst_index"].apply(
                lambda x: pc_sampling_comments[x]
                if x < len(pc_sampling_comments)
                else None
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
            return eval(
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
                },
            )
        except Exception as e:
            console_warning(f"Failed to evaluate expression for {name}: {value} - {e}")
            return None

    def calc_expressions(self) -> dict[str, pd.DataFrame]:
        values_data_per_workload = self._values_data_per_workload.copy()

        for workload_path in self._runs.keys():
            pmc_df = self._pmc_df_per_workload[workload_path].copy()
            sys_info = self._runs[workload_path].sys_info.iloc[0].to_dict()
            for key, value in self._roofline_ceilings_per_workload.get(
                workload_path, {}
            ).items():
                sys_info[f"{key}_empirical_peak"] = value

            # Calculate PER_XCD variables first
            for key, value in BUILD_IN_VARS.items():
                if "PER_XCD" in key:
                    sys_info[key] = db_analysis.evaluate(
                        key, value, pmc_df, sys_info, parse=True
                    )

            # variable dependent on PER_XCD variables
            for key, value in BUILD_IN_VARS.items():
                if "PER_XCD" not in key:
                    sys_info[key] = db_analysis.evaluate(
                        key, value, pmc_df, sys_info, parse=True
                    )

            # Get name and print warning
            values_data_per_workload[workload_path]["value"] = values_data_per_workload[
                workload_path
            ].apply(
                lambda row: db_analysis.evaluate(
                    f"{row['metric_id']} - {row['value_name']}",
                    row["value"],
                    pmc_df,
                    sys_info,
                ),
                axis=1,
            )

        console_debug("Calculated metric values")
        return values_data_per_workload

    def calc_metrics_data(
        self,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        metrics_info_data_per_workload: dict[str, pd.DataFrame] = {}
        values_data_per_workload: dict[str, pd.DataFrame] = {}

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
            values_df = pd.DataFrame([
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
            values_data_per_workload[workload_path] = values_df

        console_debug("Collected metrics data")
        return metrics_info_data_per_workload, values_data_per_workload

    def calc_dispatch_data(self) -> dict[str, pd.DataFrame]:
        dispatch_data_per_workload: dict[str, pd.DataFrame] = {}

        for workload_path in self._runs.keys():
            dispatch_df = pd.DataFrame([
                {
                    "dispatch_id": row.Dispatch_ID,
                    "kernel_name": row.Kernel_Name,
                    "gpu_id": row.GPU_ID,
                    "duration": row.End_Timestamp - row.Start_Timestamp,
                }
                for row in self._pmc_df_per_workload[workload_path].itertuples()
            ])
            dispatch_data_per_workload[workload_path] = dispatch_df

        console_debug("Calculated dispatch data")
        return dispatch_data_per_workload

    def apply_pmc_filters(self) -> dict[str, pd.DataFrame]:
        pmc_df_per_workload = self._pmc_df_per_workload.copy()

        for workload_path, pmc_df in pmc_df_per_workload.items():
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
                        self._top_kernels_per_workload[workload_path][id]
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

    def calc_roofline_data(self) -> dict[str, pd.DataFrame]:
        roofline_data_per_workload: dict[str, pd.DataFrame] = {}

        for workload_path in self._runs.keys():
            pmc_df = self._pmc_df_per_workload[workload_path].copy()
            sys_info = self._runs[workload_path].sys_info.iloc[0].to_dict()
            gfx_arch = sys_info["gpu_arch"]
            roofline_data_df = self._arch_configs[gfx_arch].dfs[402]
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
                    },
                }
                for kernel_name in self._top_kernels_per_workload[workload_path][
                    : self.get_args().max_stat_num
                ]
            ])

            roofline_data_per_workload[workload_path] = roofline_df

        console_debug("Calculated roofline data points")
        return roofline_data_per_workload
