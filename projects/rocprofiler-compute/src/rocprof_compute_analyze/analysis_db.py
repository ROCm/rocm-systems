# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import ast
import re
import warnings
from pathlib import Path
from typing import Any, Optional

import astunparse
import numpy as np
import pandas as pd

import utils.analysis_orm as orm
from config import rocprof_compute_home
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from roofline.roofline_main import ROOFLINE_SUPPORTED
from utils import schema, utils_analysis
from utils.analysis_orm import Database
from utils.file_io import load_pc_sampling_results, process_pc_sampling_kernel_trace
from utils.logger import (
    console_debug,
    console_error,
    console_warning,
    demarcate,
)
from utils.metrics.aggregation import (
    calc_pct_of_peak,
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
from utils.metrics.common import ValuDualIssueDetector
from utils.metrics.expression import CodeTransformer
from utils.metrics.noise_clamper import (
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
    print_noise_clamp_summary,
    to_noise_clamp,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.pc_sampling_analysis import load_aggregated_pc_sampling
from utils.roofline_calc import (
    MATRIX_DATATYPES,
    PEAK_OPS_DATATYPES,
    SUPPORTED_DATATYPES,
)
from utils.utils_analysis import (
    PEAK_COL_PREFERENCE,
    VALUE_COL_PREFERENCE,
)
from utils.utils_common import get_uuid, get_version
from utils.utils_counter_defs import extract_counters_and_variables, get_build_in_vars


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
        pc_sampling_tool_data = (
            {path: load_pc_sampling_results(path) for path in self._runs}
            if self.pc_sampling_collected()
            else {}
        )
        self._pc_sampling_data_per_workload = self.calc_pc_sampling_data(
            pc_sampling_tool_data
        )
        self._pmc_df_per_workload = self.calc_pmc_df_data()
        self._pmc_df_per_workload = self.apply_pmc_filters()
        self._dispatch_data_per_workload = self.calc_dispatch_data(
            pc_sampling_tool_data
        )
        (
            self._metrics_info_data_per_workload,
            self._metric_expression_data_per_workload,
        ) = self.calc_metrics_data()
        (
            self._kernel_values_data_per_workload,
            self._workload_values_data_per_workload,
        ) = self.calc_expressions()
        (
            self._roofline_data_per_kernel,
            self._roofline_data_per_workload,
        ) = self.calc_roofline_data()

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
                kernel_name = getattr(roofline_data, "kernel_name", None)
                if kernel_name not in kernel_objs:
                    console_warning(
                        f"Kernel {kernel_name} from roofline data "
                        "not found in dispatch data. Skipping roofline entry."
                    )
                    continue
                Database.get_session().add(
                    orm.KernelRooflineData(
                        total_flops=getattr(roofline_data, "total_flops", None),
                        l0_cache_data=getattr(roofline_data, "l0_cache_data", None),
                        l1_cache_data=getattr(roofline_data, "l1_cache_data", None),
                        l2_cache_data=getattr(roofline_data, "l2_cache_data", None),
                        hbm_cache_data=getattr(roofline_data, "hbm_cache_data", None),
                        lds_cache_data=getattr(roofline_data, "lds_cache_data", None),
                        kernel=kernel_objs[kernel_name],
                    )
                )

            # Add workload-level roofline data
            workload_roofline = self._roofline_data_per_workload.get(workload_path)
            if workload_roofline:
                Database.get_session().add(
                    orm.WorkloadRooflineData(
                        total_flops=workload_roofline.get("total_flops"),
                        l0_cache_data=workload_roofline.get("l0_cache_data"),
                        l1_cache_data=workload_roofline.get("l1_cache_data"),
                        l2_cache_data=workload_roofline.get("l2_cache_data"),
                        hbm_cache_data=workload_roofline.get("hbm_cache_data"),
                        lds_cache_data=workload_roofline.get("lds_cache_data"),
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

        if self.get_args().output_format == "csv":
            Database.commit()
            Database.write_csv_dir(Path(db_name).with_suffix(""))
        else:
            Database.create_views()
            Database.commit()
            Database.write()

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
            if not (Path(workload_path) / "pmc_perf.csv").exists():
                continue

            pmc_df = utils_analysis.process_rocpd_csv(
                pd.read_csv(Path(workload_path) / "pmc_perf.csv")
            )

            if args.spatial_multiplexing:
                pmc_df = self.spatial_multiplex_merge_counters(pmc_df)

            if self._profiling_config.get("iteration_multiplexing") is not None:
                pmc_df = self.iteration_multiplex_impute_counters(
                    pmc_df,
                    policy=self._profiling_config["iteration_multiplexing"],
                    workload_dir=Path(workload_path),
                )

            pmc_df_per_workload[workload_path] = pmc_df

        if pmc_df_per_workload:
            console_debug("Collected dispatch data")

        return pmc_df_per_workload

    def calc_roofline_ceilings(self) -> dict[str, dict[str, Any]]:
        roofline_ceilings_per_workload: dict[str, dict[str, Any]] = {}

        for workload_path in self._runs.keys():
            sys_row = self._runs[workload_path].sys_info.iloc[0]
            gpu_arch = sys_row["gpu_arch"]

            if gpu_arch not in ROOFLINE_SUPPORTED:
                console_warning(f"Roofline not supported for {gpu_arch}.")
                continue
            if not (Path(workload_path) / "roofline.csv").exists():
                console_warning(f"Roofline ceilings not found for {workload_path}.")
                continue

            roofline_dict = (
                pd.read_csv(f"{workload_path}/roofline.csv").iloc[0].to_dict()
            )
            keys: list[str] = []

            matrix_ops_type = utils_analysis.get_matrix_ops_type(sys_row["gpu_series"])

            for mem_level in mi_gpu_specs.get_memory_levels(sys_row["gpu_model"]):
                keys.append(f"{mem_level}Bw")
            for dtype in SUPPORTED_DATATYPES[gpu_arch]:
                if dtype in PEAK_OPS_DATATYPES:
                    if dtype.startswith("F") or dtype.startswith("B"):
                        keys.append(f"{dtype}Flops")
                    elif dtype.startswith("I"):
                        keys.append(f"{dtype}Ops")
                if dtype in MATRIX_DATATYPES:
                    if dtype.startswith("F") or dtype.startswith("B"):
                        # FP16 -> F16
                        matrix_dtype = dtype.replace("FP", "F")
                        keys.append(f"{matrix_ops_type}{matrix_dtype}Flops")
                    elif dtype.startswith("I"):
                        keys.append(f"{matrix_ops_type}{dtype}Ops")
            roofline_ceilings_per_workload[workload_path] = {
                key: roofline_dict[key] for key in keys if key in roofline_dict
            }

        if roofline_ceilings_per_workload:
            console_debug("Collected roofline ceilings")
        return roofline_ceilings_per_workload

    def calc_pc_sampling_data(
        self,
        tool_data_per_workload: dict[str, Optional[dict[str, Any]]],
    ) -> dict[str, pd.DataFrame]:
        pc_sampling_data_per_workload: dict[str, pd.DataFrame] = {}

        for workload_path in self._runs.keys():
            pc_sampling_data = tool_data_per_workload.get(workload_path)
            if pc_sampling_data is None:
                console_warning(f"PC sampling data not found for {workload_path}.")
                continue

            grouped_df = load_aggregated_pc_sampling(
                pc_sampling_data,
                group_by=["code_object_id", "code_object_offset"],
                attach={"instruction", "source_line", "kernel_name"},
            )
            grouped_df = grouped_df.rename(columns={"code_object_offset": "offset"})
            grouped_df = grouped_df[
                [
                    "offset",
                    "count",
                    "count_issued",
                    "count_stalled",
                    "stall_reason",
                    "instruction",
                    "source_line",
                    "kernel_name",
                ]
            ]

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
        emit_variance_warnings: bool = False,
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
            value = value.replace("raw_pmc_df", "pmc_df")
            value = re.sub(
                "ammolite__([0-9A-Za-z_]+)",
                lambda m: f'sys_info["{m.group(1)}"]',
                value,
            )
        try:
            prev_noise_clamp_count = get_noise_clamp_warnings()["count"]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
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
            # RuntimeWarnings (e.g. divide-by-zero) are surfaced only under --verbose
            for w in caught:
                console_debug(
                    f"RuntimeWarning evaluating {name}: {value} - {w.message}"
                )

            # eval_result can be None if expression has None explicitly specified
            # Do not give warning for this case and simply return None
            if eval_result is None:
                return None

            # Only return None for scalar NA values (NaN, pd.NA, +/-inf).
            # For vectors/Series, return as-is to preserve shape for downstream
            # operations. Note: pd.NA is not detected as scalar by np.isscalar()
            is_scalar_na = eval_result is pd.NA or (
                np.isscalar(eval_result)
                and (pd.isna(eval_result) or np.isinf(eval_result))
            )

            if is_scalar_na:
                # Skip warning when None is explicit or a RuntimeWarning
                # already explained the NA
                if "None" in value:
                    console_debug(
                        f"Expression for {name}: {value} evaluated to "
                        "None - explicitly specified."
                    )
                elif not caught:
                    console_warning(
                        f"Expression for {name}: {value} evaluated to N/A "
                        "(divide-by-zero or empty counter data)."
                    )
                return None

            if (
                emit_variance_warnings
                and get_noise_clamp_warnings()["count"] > prev_noise_clamp_count
            ):
                console_warning(f"Variance corrected for metric: {name}")
            return eval_result
        except Exception as e:
            console_warning(f"Failed to evaluate expression for {name}: {value} - {e}")
            return None

    @staticmethod
    def calc_builtin_vars(
        pmc_df: pd.DataFrame,
        sys_info: dict,
        expressions: list[str],
    ) -> None:
        """Evaluate arch-specific built-in variables referenced by expressions
        (numActiveCUs, etc.). Mutates ``sys_info`` in place."""
        gpu_series = mi_gpu_specs.get_gpu_series(sys_info["gpu_arch"])
        _, expression_builtin_vars = extract_counters_and_variables(
            "\n".join(expressions), gpu_series
        )
        build_in_vars = {
            k: v
            for k, v in get_build_in_vars(gpu_series).items()
            if k in expression_builtin_vars
        }
        # Calculate PER_XCD variables first
        for key, value in build_in_vars.items():
            if "PER_XCD" in key:
                sys_info[key] = db_analysis.evaluate(
                    key, value, pmc_df, sys_info, parse=True
                )
        # Variable dependent on PER_XCD variables
        for key, value in build_in_vars.items():
            if "PER_XCD" not in key:
                sys_info[key] = db_analysis.evaluate(
                    key, value, pmc_df, sys_info, parse=True
                )

    @staticmethod
    def calc_dataframe_expressions(
        pmc_df: pd.DataFrame,
        sys_info: dict,
        expression_df: pd.DataFrame,
        emit_variance_warnings: bool = False,
    ) -> pd.Series:
        db_analysis.calc_builtin_vars(
            pmc_df,
            sys_info,
            [
                v
                for v in expression_df["value"].tolist()
                if isinstance(v, str) and v and v != "None"
            ],
        )
        return expression_df.apply(
            lambda row: db_analysis.evaluate(
                f"{row['metric_id']} - {row['value_name']}",
                row["value"],
                pmc_df,
                sys_info,
                emit_variance_warnings=emit_variance_warnings,
            ),
            axis=1,
        )

    @staticmethod
    def validate_dual_issue_metrics(
        pmc_df: pd.DataFrame,
        sys_info: dict,
        workload_values_df: pd.DataFrame,
        arch_config: schema.ArchConfig,
    ) -> None:
        """Warn when VALU metrics exceed peak in the workload-level results."""
        detector = ValuDualIssueDetector(
            gpu_arch=sys_info.get("gpu_arch", ""),
            raw_pmc_df=pmc_df,
        )

        candidates: list[tuple[str, str, str]] = []
        for df_id, df in arch_config.dfs.items():
            if arch_config.dfs_type.get(df_id) != "metric_table":
                continue
            if "Metric" not in df.columns or "Value" not in df.columns:
                continue
            if "Peak (Empirical)" in df.columns:
                peak_col = "Peak (Empirical)"
            elif "Peak" in df.columns:
                peak_col = "Peak"
            else:
                continue
            for metric_id, row in df.iterrows():
                metric_name = row.get("Metric", "")
                if metric_name in ValuDualIssueDetector.candidate_metrics:
                    candidates.append((metric_id, metric_name, peak_col))
        if not candidates:
            return

        values_by_metric_id = {
            metric_id: dict(zip(group["value_name"], group["value"]))
            for metric_id, group in workload_values_df.groupby("metric_id")
        }

        for metric_id, metric_name, peak_col in candidates:
            values = values_by_metric_id.get(metric_id)
            if values is None:
                continue
            try:
                value = float(values.get("Value", 0))
                peak = float(values.get(peak_col, 0))
            except (ValueError, TypeError):
                continue
            detector.check(metric_name, value, peak)

    def calc_expressions(
        self,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Calculate kernel-level and workload-level metrics,
        including Percent of Peak."""
        kernel_values_data = {}
        workload_values_data = {}

        for workload_path in self._pmc_df_per_workload.keys():
            pmc_df = self._pmc_df_per_workload[workload_path]
            expression_template = self._metric_expression_data_per_workload[
                workload_path
            ]
            sys_info = self._runs[workload_path].sys_info.iloc[0].to_dict()
            for key, value in self._roofline_ceilings_per_workload.get(
                workload_path, {}
            ).items():
                sys_info[f"{key}_empirical_peak"] = value

            metrics_info = self._metrics_info_data_per_workload.get(
                workload_path, pd.DataFrame(columns=["pct_of_peak", "metric_id"])
            )
            pct_of_peak_metric_ids = set(
                metrics_info.loc[metrics_info["pct_of_peak"], "metric_id"]
            )

            # Calculate kernel-level metrics
            kernel_values_list = []
            new_kernel_rows: list[dict] = []

            for kernel_name, kernel_pmc_df in pmc_df.groupby("Kernel_Name"):
                kernel_expression_df = expression_template.assign(
                    kernel_name=kernel_name
                )
                kernel_expression_df["value"] = db_analysis.calc_dataframe_expressions(
                    kernel_pmc_df,
                    sys_info.copy(),
                    kernel_expression_df,
                )
                new_kernel_rows.extend(
                    db_analysis._derive_pct_of_peak_values(
                        pct_of_peak_metric_ids, kernel_expression_df
                    )
                )
                kernel_values_list.append(kernel_expression_df)

            if kernel_values_list and new_kernel_rows:
                kernel_values_data[workload_path] = pd.concat(
                    kernel_values_list + [pd.DataFrame(new_kernel_rows)],
                    ignore_index=True,
                )
            elif kernel_values_list:
                kernel_values_data[workload_path] = pd.concat(
                    kernel_values_list, ignore_index=True
                )
            else:
                kernel_values_data[workload_path] = pd.DataFrame()

            # Variance warnings are emitted at workload-level, not per kernel.
            console_debug(f"Processing workload: {workload_path}")
            clear_noise_clamp_warnings()
            workload_expression_df = expression_template.copy()
            workload_expression_df["value"] = db_analysis.calc_dataframe_expressions(
                pmc_df,
                sys_info.copy(),
                workload_expression_df,
                emit_variance_warnings=True,
            )
            print_noise_clamp_summary()
            db_analysis.validate_dual_issue_metrics(
                pmc_df,
                sys_info,
                workload_expression_df,
                self._arch_configs[sys_info["gpu_arch"]],
            )
            new_workload_rows = db_analysis._derive_pct_of_peak_values(
                pct_of_peak_metric_ids, workload_expression_df
            )
            if new_workload_rows:
                workload_values_data[workload_path] = pd.concat(
                    [workload_expression_df, pd.DataFrame(new_workload_rows)],
                    ignore_index=True,
                )
            else:
                workload_values_data[workload_path] = workload_expression_df

        if kernel_values_data or workload_values_data:
            console_debug("Calculated kernel-level and workload-level metric values")

        return kernel_values_data, workload_values_data

    @staticmethod
    def _derive_pct_of_peak_values(
        pct_of_peak_metric_ids: set[str],
        values_df: pd.DataFrame,
    ) -> list[dict]:
        """Return new Percent of Peak rows for pct_of_peak-enabled metrics."""
        candidates = values_df[
            values_df["metric_id"].isin(pct_of_peak_metric_ids)
            & values_df["value_name"].isin([
                "Avg",
                "Value",
                "Peak",
                "Peak (Empirical)",
            ])
        ]
        new_rows = []
        for _metric_id, grp in candidates.groupby("metric_id"):
            vals = grp.set_index("value_name")["value"]
            val = next(
                (vals.get(col) for col in VALUE_COL_PREFERENCE if col in vals.index),
                None,
            )
            peak = next(
                (vals.get(col) for col in PEAK_COL_PREFERENCE if col in vals.index),
                None,
            )
            pct = calc_pct_of_peak(val, peak)
            if pct is None:
                continue
            base = grp.iloc[0].to_dict()
            base["value_name"] = "Percent of Peak"
            base["value"] = pct
            new_rows.append(base)
        return new_rows

    def calc_metrics_data(
        self,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        metrics_info_data_per_workload: dict[str, pd.DataFrame] = {}
        metric_expression_data_per_workload: dict[str, pd.DataFrame] = {}

        for workload_path in self._pmc_df_per_workload.keys():
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
                "Type",
                "Xfer",
                "Coherency",
                "Transaction",
                "Percent of Peak",
            ]
            metrics_info_df = pd.DataFrame([
                {
                    "name": row.get("Metric") or row["Channel"].strip(),
                    "metric_id": metric_id,
                    "description": row.get("Description"),
                    "unit": row.get("Unit"),
                    "pct_of_peak": row.get("Percent of Peak") is True,
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

        if metrics_info_data_per_workload or metric_expression_data_per_workload:
            console_debug("Collected metrics data")

        return metrics_info_data_per_workload, metric_expression_data_per_workload

    def calc_dispatch_data(
        self,
        tool_data_per_workload: dict[str, Optional[dict[str, Any]]],
    ) -> dict[str, pd.DataFrame]:
        dispatch_data_per_workload: dict[str, pd.DataFrame] = {}

        for workload_path in self._runs.keys():
            if self.pc_sampling_only():
                tool_data = tool_data_per_workload.get(workload_path)
                trace_df = process_pc_sampling_kernel_trace(tool_data)
                trace_df = pd.DataFrame({
                    "dispatch_id": trace_df["Dispatch_Id"],
                    "kernel_name": trace_df["Kernel_Name"],
                    "gpu_id": trace_df["GPU_ID"],
                    "start_timestamp": trace_df["Start_Timestamp"],
                    "end_timestamp": trace_df["End_Timestamp"],
                })
                dispatch_data_per_workload[workload_path] = trace_df
            else:
                dispatch_data_per_workload[workload_path] = pd.DataFrame([
                    {
                        "dispatch_id": row.Dispatch_ID,
                        "kernel_name": row.Kernel_Name,
                        "gpu_id": row.GPU_ID,
                        "start_timestamp": row.Start_Timestamp,
                        "end_timestamp": row.End_Timestamp,
                    }
                    for row in self._pmc_df_per_workload[workload_path].itertuples()
                ])

        if dispatch_data_per_workload:
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

        if pmc_df_per_workload:
            console_debug("Applied analysis mode filters")

        return pmc_df_per_workload

    def calc_roofline_data(self) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
        """Calculate both kernel-level and workload-level roofline data"""
        roofline_data_per_kernel: dict[str, pd.DataFrame] = {}
        roofline_data_per_workload: dict[str, dict] = {}

        for workload_path in self._pmc_df_per_workload.keys():
            pmc_df = self._pmc_df_per_workload[workload_path].copy()
            sys_info = self._runs[workload_path].sys_info.iloc[0].to_dict()
            gfx_arch = sys_info["gpu_arch"]
            roofline_data_df = self._arch_configs[gfx_arch].dfs.get(402)

            if roofline_data_df is None or roofline_data_df.empty:
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
                "l0_cache_data": roofline_data_expressions.get("AI L0", ""),
                "l1_cache_data": roofline_data_expressions.get("AI L1", ""),
                "l2_cache_data": roofline_data_expressions.get("AI L2", ""),
                "hbm_cache_data": roofline_data_expressions.get("AI HBM", ""),
                "lds_cache_data": roofline_data_expressions.get("AI LDS", ""),
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
