# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import ast
import re
import warnings
from pathlib import Path
from typing import Any, NamedTuple, Optional

import astunparse
import numpy as np
import pandas as pd

import utils.analysis_orm as orm
from config import rocprof_compute_home
from pc_sampling.code_object_analysis import (
    CodeObjectSymbol,
    load_code_object_disassemblies,
)
from pc_sampling.pc_sampling_analysis import (
    SOURCE_LINE_MISSING,
    InstructionLineRecord,
    load_aggregated_pc_sampling,
)
from pc_sampling.per_kernel_isa_export import (
    WorkloadIsaExport,
    export_per_kernel_isa_files,
)
from pc_sampling.source_snapshot_analysis import (
    SourceFrame,
    WorkloadSourceSnapshot,
    export_source_snapshot_files,
    parse_source_frames,
    read_source_file_digest_and_lines,
    resolve_snapshot_path,
)
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from roofline.roofline_main import ROOFLINE_SUPPORTED
from utils import schema, utils_analysis
from utils.analysis_orm import Database
from utils.file_io import (
    load_pc_sampling_results,
    process_pc_sampling_kernel_traces,
)
from utils.logger import (
    console_debug,
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
from utils.metrics.common import EVAL_BUILTINS, ValuDualIssueDetector
from utils.metrics.expression import transform_expression
from utils.metrics.noise_clamper import (
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
    print_noise_clamp_summary,
    to_noise_clamp,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.roofline_calc import (
    SUPPORTED_DATATYPES,
    OpsSupport,
)
from utils.utils_analysis import (
    PEAK_COL_PREFERENCE,
    VALUE_COL_PREFERENCE,
)
from utils.utils_common import get_uuid, get_version, normalize_filter_to_str_list
from utils.utils_counter_defs import (
    extract_counters_and_variables,
    get_build_in_vars,
)

KernelKey = str
CodeObjectKey = tuple[int, int]
KernelSymbolKey = tuple[int, int, str]  # (pid, code_object_id, kernel_name)


def filter_dispatch_frame(
    dispatch_frame: pd.DataFrame,
    filter_gpu_ids: Optional[list[str]],
    filter_kernel_ids: Optional[list[int]],
    filter_dispatch_ids: Optional[list[str]],
) -> pd.DataFrame:
    """Apply the analysis mode filters to one frame of dispatch rows.

    The frame carries the profiler's column names, so both the counter frame
    and the PC-sampling trace can be filtered by the same rules.
    """
    top_kernels = (
        dispatch_frame
        .assign(
            duration=dispatch_frame["End_Timestamp"] - dispatch_frame["Start_Timestamp"]
        )
        .sort_values(by="duration", ascending=False)
        .drop_duplicates("Kernel_Name")["Kernel_Name"]
        .to_list()
    )
    if filter_gpu_ids:
        dispatch_frame = dispatch_frame.loc[
            dispatch_frame["GPU_ID"]
            .astype(str)
            .isin(normalize_filter_to_str_list(filter_gpu_ids))
        ]
    if filter_kernel_ids:
        dispatch_frame = dispatch_frame.loc[
            dispatch_frame["Kernel_Name"].isin([
                top_kernels[kernel_id] for kernel_id in filter_kernel_ids
            ])
        ]
    if filter_dispatch_ids:
        if ">" in filter_dispatch_ids[0]:
            dispatch_bound = re.match(r"\>\s*(\d+)", filter_dispatch_ids[0])
            dispatch_frame = dispatch_frame[
                dispatch_frame["Dispatch_ID"] > int(dispatch_bound.group(1))
            ]
        else:
            dispatch_frame = dispatch_frame.loc[
                dispatch_frame["Dispatch_ID"].astype(str).isin(filter_dispatch_ids)
            ]
    return dispatch_frame


class MetricInfoRow(NamedTuple):
    name: str
    metric_id: str
    description: Optional[str]
    unit: Optional[str]
    pct_of_peak: bool
    table_name: str
    sub_table_name: str


class ExpressionRow(NamedTuple):
    metric_id: str
    value_name: str
    value: str


class SourceFrameCollector:
    """Creates one workload's source rows as its instructions reference them.

    A file is read and stored whole the first time a frame names it, so a
    sampled line can be read in its surrounding context.
    """

    def __init__(self, workload_path: Path, workload: orm.Workload) -> None:
        self._workload_path = workload_path
        self._workload = workload
        self._frames_by_comment: dict[str, list[SourceFrame]] = {}
        self._source_files: dict[str, orm.SourceFile] = {}
        self._source_lines: dict[SourceFrame, orm.SourceLine] = {}

    def add_instruction(
        self,
        instruction_line: orm.InstructionLine,
        source: Optional[str],
    ) -> None:
        """Link one instruction to its source frames, innermost first."""
        # The sampling path substitutes "N/A" for an absent comment, which
        # would otherwise become a source file named after itself.
        if not source or source == SOURCE_LINE_MISSING:
            return

        # Parse once per distinct comment; instructions repeat them heavily.
        if source not in self._frames_by_comment:
            self._frames_by_comment[source] = parse_source_frames(source)

        for frame_index, frame in enumerate(self._frames_by_comment[source]):
            Database.get_session().add(
                orm.InstructionSourceLine(
                    instruction_line=instruction_line,
                    source_line=self._get_or_create_source_line(frame),
                    frame_index=frame_index,
                )
            )

    def captured_source_paths(self) -> tuple[str, ...]:
        """Return the absolute paths whose snapshot copy this workload holds.

        A path the snapshot does not hold, and a frame naming a relative path,
        carries no digest and has no copy to export.
        """
        return tuple(
            absolute_path
            for absolute_path, source_file in self._source_files.items()
            if source_file.md5_checksum is not None
        )

    def _get_or_create_source_line(self, frame: SourceFrame) -> orm.SourceLine:
        """Return the row for one frame's line, creating it if absent.

        A frame naming a line the snapshot copy does not hold, including the
        null line of a ":?" frame, gets a row with no content.
        """
        absolute_path, line_number = frame
        # Resolve the file first: it seeds the cache with the lines it holds.
        source_file = self._get_or_create_source_file(absolute_path)
        if frame not in self._source_lines:
            self._source_lines[frame] = orm.SourceLine(
                source_file=source_file,
                line_number=line_number,
            )
            Database.get_session().add(self._source_lines[frame])
        return self._source_lines[frame]

    def _get_or_create_source_file(self, absolute_path: str) -> orm.SourceFile:
        """Return the row for one file, reading its snapshot copy if absent."""
        if absolute_path not in self._source_files:
            digest, line_contents = self._read_snapshot_file(absolute_path)
            source_file = orm.SourceFile(
                workload=self._workload,
                file_path=absolute_path,
                md5_checksum=digest,
            )
            Database.get_session().add(source_file)
            self._source_files[absolute_path] = source_file
            self._add_source_lines(source_file, absolute_path, line_contents)
        return self._source_files[absolute_path]

    def _read_snapshot_file(
        self, absolute_path: str
    ) -> tuple[Optional[str], dict[int, str]]:
        """Read one referenced file out of the workload's source snapshot."""
        if not Path(absolute_path).is_absolute():
            return None, {}

        return read_source_file_digest_and_lines(
            resolve_snapshot_path(self._workload_path, absolute_path)
        )

    def _add_source_lines(
        self,
        source_file: orm.SourceFile,
        absolute_path: str,
        line_contents: dict[int, str],
    ) -> None:
        """Create a row for every line of one file's snapshot copy."""
        source_lines = [
            orm.SourceLine(
                source_file=source_file,
                line_number=line_number,
                content=content,
            )
            for line_number, content in line_contents.items()
        ]
        Database.get_session().add_all(source_lines)
        self._source_lines.update({
            (absolute_path, source_line.line_number): source_line
            for source_line in source_lines
        })


class db_analysis(OmniAnalyze_Base):
    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()

        self._roofline_ceilings_per_workload = self.calc_roofline_ceilings()
        self._pc_sampling_tool_data_per_workload = (
            {path: load_pc_sampling_results(path) for path in self._runs}
            if self.pc_sampling_collected()
            else {}
        )
        self._pmc_df_per_workload = self.calc_pmc_df_data()
        self._pmc_df_per_workload = self.apply_pmc_filters()
        self._dispatch_data_per_workload = self.calc_dispatch_data(
            self._pc_sampling_tool_data_per_workload
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
        workload_source_snapshots: list[WorkloadSourceSnapshot] = []
        workload_objs: list[orm.Workload] = []
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
            workload_objs.append(workload_obj)

            # Add kernel
            kernel_objs: dict[KernelKey, orm.Kernel] = {}

            for dispatch in self._dispatch_data_per_workload.get(
                workload_path, pd.DataFrame()
            ).itertuples():
                kernel_key: KernelKey = dispatch.kernel_name
                # Add kernel object and map it, if not already added
                if kernel_key not in kernel_objs:
                    kernel_objs[kernel_key] = orm.Kernel(
                        kernel_name=dispatch.kernel_name,
                        workload=workload_obj,
                    )
                    Database.get_session().add(kernel_objs[kernel_key])

                # Add dispatch object and link with kernel object
                Database.get_session().add(
                    orm.Dispatch(
                        dispatch_id=dispatch.dispatch_id,
                        gpu_id=dispatch.gpu_id,
                        start_timestamp=dispatch.start_timestamp,
                        end_timestamp=dispatch.end_timestamp,
                        kernel=kernel_objs[kernel_key],
                    )
                )

            # Add kernel-level roofline data points
            for roofline_data in self._roofline_data_per_kernel.get(
                workload_path, pd.DataFrame()
            ).itertuples():
                kernel_name = getattr(roofline_data, "kernel_name", None)
                kernel_key: KernelKey = kernel_name
                if kernel_key not in kernel_objs:
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
                        kernel=kernel_objs[kernel_key],
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

            # Add pc sampling data, then the full code-object ISA
            source_frames = SourceFrameCollector(Path(workload_path), workload_obj)
            kernel_symbols: dict[KernelSymbolKey, orm.KernelSymbol] = {}
            code_object_stores = self.add_pc_sampling_data(
                workload_path,
                workload_obj,
                kernel_objs,
                kernel_symbols,
                source_frames,
            )
            self.add_code_object_isa(
                workload_path,
                workload_obj,
                kernel_objs,
                code_object_stores,
                kernel_symbols,
                source_frames,
            )
            workload_source_snapshots.append(
                WorkloadSourceSnapshot(
                    workload_path=Path(workload_path),
                    workload_name=workload_obj.name,
                    workload_sub_name=workload_obj.sub_name,
                    absolute_source_paths=source_frames.captured_source_paths(),
                )
            )

            # Add metrics and values - iterate on values, create metrics as needed
            self.run_analysis_metrics(workload_path, workload_obj, kernel_objs)

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
            csv_result_folder = Path(db_name).with_suffix("")
            # Before write_csv_dir, which closes the session these rows stream from.
            per_kernel_folder = export_per_kernel_isa_files(
                csv_result_directory=csv_result_folder,
                workload_isa_exports=self._build_workload_isa_exports(workload_objs),
            )
            Database.write_csv_dir(csv_result_folder)
            export_source_snapshot_files(
                workload_source_snapshots=workload_source_snapshots,
                export_directory=per_kernel_folder,
            )
        else:
            Database.create_views()
            Database.commit()
            Database.write()

    @staticmethod
    def _build_workload_isa_exports(
        workload_objs: list[orm.Workload],
    ) -> list[WorkloadIsaExport]:
        """Pair each workload's stall reasons with its instruction rows."""
        stall_reasons_by_workload = Database.get_stall_reasons_by_workload()
        workload_isa_exports = []
        for workload_obj in workload_objs:
            stall_reasons = stall_reasons_by_workload.get(workload_obj.workload_id, [])
            workload_isa_exports.append(
                WorkloadIsaExport(
                    workload_id=workload_obj.workload_id,
                    stall_reasons=stall_reasons,
                    isa_rows=Database.stream_per_kernel_isa_rows(
                        workload_obj.workload_id, stall_reasons
                    ),
                )
            )
        return workload_isa_exports

    def run_analysis_metrics(
        self,
        workload_path: str,
        workload_obj: orm.Workload,
        kernel_objs: dict[KernelKey, orm.Kernel],
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
            kernel_key: KernelKey = value.kernel_name
            # Check if kernel exists
            if kernel_key not in kernel_objs:
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
                    kernel=kernel_objs[kernel_key],
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

        for workload_path in self._runs.keys():
            if not (Path(workload_path) / "pmc_perf.csv").exists():
                continue

            pmc_df = utils_analysis.process_rocpd_csv(
                pd.read_csv(Path(workload_path) / "pmc_perf.csv")
            )

            utils_analysis.add_unit_counter(pmc_df)

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
            for dtype in SUPPORTED_DATATYPES[gpu_arch].keys():
                if OpsSupport.VALU in SUPPORTED_DATATYPES[gpu_arch][dtype]:
                    if dtype.startswith("F") or dtype.startswith("B"):
                        keys.append(f"{dtype}Flops")
                    elif dtype.startswith("I"):
                        keys.append(f"{dtype}Ops")
                if OpsSupport.MATRIX in SUPPORTED_DATATYPES[gpu_arch][dtype]:
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

    def add_pc_sampling_data(
        self,
        workload_path: str,
        workload_obj: orm.Workload,
        kernel_objs: dict[KernelKey, orm.Kernel],
        kernel_symbols: dict[KernelSymbolKey, orm.KernelSymbol],
        source_frames: SourceFrameCollector,
    ) -> dict[CodeObjectKey, orm.CodeObjectStore]:
        """Insert the normalized PC-sampling rows for one workload.

        A code object's store row is created by the first line that survives
        the kernel filter, so one whose kernels were all filtered out leaves no
        row with nothing under it.
        """
        code_object_stores: dict[CodeObjectKey, orm.CodeObjectStore] = {}
        tool_data_records = self._pc_sampling_tool_data_per_workload.get(
            workload_path, []
        )

        for tool_data in tool_data_records:
            pid: int = tool_data["metadata"]["pid"]

            for code_object in load_aggregated_pc_sampling(tool_data):
                for line in code_object.instruction_lines:
                    kernel = kernel_objs.get(line.kernel_name)
                    if kernel is None:
                        # Drop lines whose kernel was filtered out or never mapped.
                        continue

                    self._add_instruction_line(
                        line,
                        self._get_or_create_code_object_store(
                            code_object_stores,
                            (pid, code_object.code_object_id),
                            code_object.load_base,
                            workload_obj,
                        ),
                        kernel,
                        kernel_symbols,
                        source_frames,
                    )

        return code_object_stores

    @staticmethod
    def _get_or_create_code_object_store(
        code_object_stores: dict[CodeObjectKey, orm.CodeObjectStore],
        code_object_key: CodeObjectKey,
        load_base: Optional[int],
        workload_obj: orm.Workload,
    ) -> orm.CodeObjectStore:
        """Return the store for a (pid, code object) pair, creating it once."""
        if code_object_key not in code_object_stores:
            pid, code_object_id = code_object_key
            code_object_stores[code_object_key] = orm.CodeObjectStore(
                code_object_id=code_object_id,
                pid=pid,
                load_base=load_base,
                workload=workload_obj,
            )
            Database.get_session().add(code_object_stores[code_object_key])
        return code_object_stores[code_object_key]

    @staticmethod
    def _get_or_create_kernel_symbol(
        code_object_store: orm.CodeObjectStore,
        kernel: orm.Kernel,
        kernel_symbols: dict[KernelSymbolKey, orm.KernelSymbol],
    ) -> orm.KernelSymbol:
        """Return the symbol for a (code object, kernel) pair, creating it once."""
        key: KernelSymbolKey = (
            code_object_store.pid,
            code_object_store.code_object_id,
            kernel.kernel_name,
        )
        if key not in kernel_symbols:
            kernel_symbols[key] = orm.KernelSymbol(
                code_object_store=code_object_store,
                kernel=kernel,
            )
            Database.get_session().add(kernel_symbols[key])
        return kernel_symbols[key]

    @staticmethod
    def _add_instruction_line(
        line: InstructionLineRecord,
        code_object_store: orm.CodeObjectStore,
        kernel: orm.Kernel,
        kernel_symbols: dict[KernelSymbolKey, orm.KernelSymbol],
        source_frames: SourceFrameCollector,
    ) -> None:
        """Insert one instruction line, its sample state, and child counts."""
        instruction_line = orm.InstructionLine(
            code_object_offset=line.code_object_offset,
            instruction=line.instruction,
            kernel_symbol=db_analysis._get_or_create_kernel_symbol(
                code_object_store, kernel, kernel_symbols
            ),
        )
        Database.get_session().add(instruction_line)
        source_frames.add_instruction(instruction_line, line.source)

        sample_state = orm.PCSampleState(
            total_count=line.total_count,
            issue_count=line.issue_count,
            stall_count=line.stall_count,
            instruction_line=instruction_line,
        )
        Database.get_session().add(sample_state)

        for text, count in line.stall_reasons.items():
            Database.get_session().add(
                orm.PCSampleStallReason(
                    pc_sample_state=sample_state,
                    stall_reason_lookup=Database.get_or_create_type(
                        orm.PCSampleStallReasonLookup, text
                    ),
                    count=count,
                )
            )
        for text, count in line.inst_types.items():
            Database.get_session().add(
                orm.InstructionSample(
                    pc_sample_state=sample_state,
                    instruction_sample_lookup=Database.get_or_create_type(
                        orm.InstructionSampleLookup, text
                    ),
                    count=count,
                )
            )

    def add_code_object_isa(
        self,
        workload_path: str,
        workload_obj: orm.Workload,
        kernel_objs: dict[KernelKey, orm.Kernel],
        code_object_stores: dict[CodeObjectKey, orm.CodeObjectStore],
        kernel_symbols: dict[KernelSymbolKey, orm.KernelSymbol],
        source_frames: SourceFrameCollector,
    ) -> None:
        """Add dispatched kernels' disassembly as instruction lines,
        skipping any offset already present."""
        tool_data_records = self._pc_sampling_tool_data_per_workload.get(
            workload_path, []
        )
        tool_data_by_pid = {
            tool_data["metadata"]["pid"]: tool_data for tool_data in tool_data_records
        }

        for pid, disassemblies in load_code_object_disassemblies(workload_path).items():
            tool_data = tool_data_by_pid.get(pid)
            if tool_data is None:
                continue

            load_base_by_id = {
                code_object["code_object_id"]: code_object.get("load_base")
                for code_object in tool_data.get("code_objects", [])
            }
            kernel_by_symbol = self._kernel_by_symbol(tool_data, kernel_objs)
            invoked_code_object_ids = {
                code_object_id for code_object_id, _ in kernel_by_symbol
            }

            for disassembly in disassemblies:
                if disassembly.code_object_id not in invoked_code_object_ids:
                    continue

                code_object_store = self._get_or_create_code_object_store(
                    code_object_stores,
                    (pid, disassembly.code_object_id),
                    load_base_by_id.get(disassembly.code_object_id),
                    workload_obj,
                )

                # The disassembly's own offset is into the ELF file, not the
                # offset the PC-sampling rows use (measured from the code
                # object's load address). Without that load address we can't
                # derive it, so skip this ISA.
                if code_object_store.load_base is None:
                    console_debug(
                        "Code object info: skipped adding ISA for code object "
                        f"{disassembly.code_object_id} with no load_base"
                    )
                    continue

                for symbol in disassembly.symbols:
                    kernel = kernel_by_symbol.get((
                        disassembly.code_object_id,
                        symbol.name,
                    ))
                    if kernel is None:
                        continue
                    kernel_symbol = self._get_or_create_kernel_symbol(
                        code_object_store, kernel, kernel_symbols
                    )
                    # Offsets are relative to the runtime load_base, never the
                    # file offsets the code object info artifact carries.
                    kernel_symbol.code_object_offset = (
                        symbol.virtual_address - code_object_store.load_base
                    )
                    self._add_symbol_isa(kernel_symbol, symbol, source_frames)

    @staticmethod
    def _add_symbol_isa(
        kernel_symbol: orm.KernelSymbol,
        symbol: CodeObjectSymbol,
        source_frames: SourceFrameCollector,
    ) -> None:
        """Add a symbol's disassembly, skipping offsets it already holds."""
        existing_offsets = {
            line.code_object_offset for line in kernel_symbol.instruction_lines
        }
        load_base = kernel_symbol.code_object_store.load_base
        for instruction in symbol.instructions:
            code_object_offset = instruction.virtual_address - load_base
            if code_object_offset in existing_offsets:
                continue
            existing_offsets.add(code_object_offset)
            instruction_line = orm.InstructionLine(
                code_object_offset=code_object_offset,
                instruction=instruction.instruction,
                kernel_symbol=kernel_symbol,
            )
            Database.get_session().add(instruction_line)
            source_frames.add_instruction(instruction_line, instruction.source)

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
            original_value = value
            value = re.sub(
                r"\$([0-9A-Za-z_]+)",
                lambda m: f'sys_info["{m.group(1)}"]',
                value,
            )
            ast_node = ast.parse(value)
            if not transform_expression(ast_node, original_value):
                return None
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
                    {"__builtins__": EVAL_BUILTINS},
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
        return pd.Series(
            [
                db_analysis.evaluate(
                    f"{row.metric_id} - {row.value_name}",
                    row.value,
                    pmc_df,
                    sys_info,
                    emit_variance_warnings=emit_variance_warnings,
                )
                for row in expression_df.itertuples(index=False)
            ],
            index=expression_df.index,
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

        non_expression_columns = {
            "Metric",
            "Channel",
            "Unit",
            "Description",
            "Type",
            "Xfer",
            "Coherency",
            "Transaction",
            "Percent of Peak",
        }

        for workload_path in self._pmc_df_per_workload.keys():
            gfx_arch = self._runs[workload_path].sys_info.iloc[0]["gpu_arch"]
            arch_config = self._arch_configs[gfx_arch]

            # Build table_id -> title map
            # (e.g. 700 -> "Wavefront", 701 -> "Wavefront Launch Stats").
            table_names_map: dict[int, str] = {}
            for panel_config in arch_config.panel_configs.values():
                table_names_map[panel_config["id"]] = panel_config["title"]
                for source in panel_config["data source"]:
                    for table in source.values():
                        table_names_map[table["id"]] = table["title"]

            # Collect metric tables with table-level fields (table_name,
            # sub_table_name, value_columns) and rows computed once per table.
            metric_tables = [
                (
                    table_names_map[table_id // 100 * 100],
                    table_names_map[table_id],
                    [c for c in metric_df.columns if c not in non_expression_columns],
                    list(metric_df.iterrows()),
                )
                for table_id, metric_df in arch_config.dfs.items()
                if table_id != 402  # roofline points handled in calc_roofline_data
                if set(metric_df.columns).intersection({"Metric", "Channel"})
            ]

            metric_info_rows = [
                MetricInfoRow(
                    name=row.get("Metric") or row["Channel"].strip(),
                    metric_id=metric_id,
                    description=row.get("Description"),
                    unit=row.get("Unit"),
                    pct_of_peak=row.get("Percent of Peak") is True,
                    table_name=table_name,
                    sub_table_name=sub_table_name,
                )
                for table_name, sub_table_name, _value_columns, rows in metric_tables
                for metric_id, row in rows
            ]
            expression_rows = [
                ExpressionRow(
                    metric_id=metric_id,
                    value_name=value_name,
                    value=row[value_name].strip(),
                )
                for _table_name, _sub_table_name, value_columns, rows in metric_tables
                for metric_id, row in rows
                for value_name in value_columns
            ]

            metrics_info_df = pd.DataFrame(
                metric_info_rows, columns=MetricInfoRow._fields
            )
            expression_df = pd.DataFrame(expression_rows, columns=ExpressionRow._fields)

            metrics_info_data_per_workload[workload_path] = metrics_info_df
            metric_expression_data_per_workload[workload_path] = expression_df

        if metrics_info_data_per_workload or metric_expression_data_per_workload:
            console_debug("Collected metrics data")

        return metrics_info_data_per_workload, metric_expression_data_per_workload

    def calc_dispatch_data(
        self,
        tool_data_per_workload: dict[str, list[dict[str, Any]]],
    ) -> dict[str, pd.DataFrame]:
        dispatch_data_per_workload: dict[str, pd.DataFrame] = {}

        for workload_path in self._runs.keys():
            if self.pc_sampling_only():
                dispatch_data_per_workload[workload_path] = (
                    self._build_pc_sampling_dispatch_data(
                        tool_data_per_workload.get(workload_path, []),
                        self._runs[workload_path],
                    )
                )
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

    @staticmethod
    def _kernel_by_symbol(
        tool_data: dict[str, Any],
        kernel_objs: dict[KernelKey, orm.Kernel],
    ) -> dict[tuple[Any, str], orm.Kernel]:
        """Map process-local ELF symbols to their dispatched kernel objects.

        The key stays process-local because ``code_object_id`` is, but the
        kernel it resolves to is shared: a kernel is identified by name across
        the whole workload.
        """
        dispatched_kernel_ids = {
            dispatch_record["dispatch_info"]["kernel_id"]
            for dispatch_record in tool_data["buffer_records"]["kernel_dispatch"]
        }
        return {
            (
                symbol["code_object_id"],
                symbol["kernel_name"].removesuffix(".kd"),
            ): kernel
            for symbol in tool_data.get("kernel_symbols", [])
            if symbol["kernel_id"] in dispatched_kernel_ids
            if (kernel := kernel_objs.get(symbol["formatted_kernel_name"])) is not None
        }

    @staticmethod
    def _build_pc_sampling_dispatch_data(
        tool_data_records: list[dict[str, Any]],
        workload: schema.Workload,
    ) -> pd.DataFrame:
        """Convert combined sampling traces to the database dispatch schema.

        The filters run here because a sampling-only workload has no counter
        frame for them to reach the database through.
        """
        columns = [
            "dispatch_id",
            "kernel_name",
            "gpu_id",
            "start_timestamp",
            "end_timestamp",
        ]
        trace_df = process_pc_sampling_kernel_traces(tool_data_records)
        if trace_df.empty:
            return pd.DataFrame(columns=columns)

        # The trace names the column Dispatch_Id, the filters read Dispatch_ID.
        trace_df = filter_dispatch_frame(
            trace_df.rename(columns={"Dispatch_Id": "Dispatch_ID"}),
            workload.filter_gpu_ids,
            workload.filter_kernel_ids,
            workload.filter_dispatch_ids,
        )
        dispatch_df = pd.DataFrame({
            "dispatch_id": trace_df["Dispatch_ID"],
            "kernel_name": trace_df["Kernel_Name"],
            "gpu_id": trace_df["GPU_ID"],
            "start_timestamp": trace_df["Start_Timestamp"],
            "end_timestamp": trace_df["End_Timestamp"],
        })
        return dispatch_df.reset_index(drop=True)

    def apply_pmc_filters(self) -> dict[str, pd.DataFrame]:
        pmc_df_per_workload = self._pmc_df_per_workload.copy()

        for workload_path, pmc_df in pmc_df_per_workload.items():
            pmc_df_per_workload[workload_path] = filter_dispatch_frame(
                pmc_df,
                self._runs[workload_path].filter_gpu_ids,
                self._runs[workload_path].filter_kernel_ids,
                self._runs[workload_path].filter_dispatch_ids,
            )

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
