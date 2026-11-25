# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT


import argparse
import os
import shutil
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

import config
from rocprof_compute_soc.metric_library import loader
from rocprof_compute_soc.metric_library.loader import Collectables
from roofline.run_benchmark import BENCHMARKING_SUPPORTED, run_roofline_benchmark
from utils import amdsmi_interface, rocprofv3_avail_interface
from utils.counters.allocator import (
    allocate_single_pass_metrics,
    metric_counter_sets,
)
from utils.counters.counter_file import CounterFile
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs
from utils.utils_common import (
    INVALID_BLOCK_HINT,
    MEMBW_ANALYSIS_PANEL_ID,
    METRIC_ID_RE,
    METRIC_TABLE,
    add_counter_extra_config_input_yaml,
    convert_metric_id_to_panel_info,
    create_temp_rocprofiler_metrics_path,
    get_arch_alias_to_panel_id,
    is_only_pc_sampling,
    is_tcc_channel_counter,
    iter_view_rows,
    load_views,
    numbered_view_tables,
    parse_sets_yaml,
    row_metric_ids,
    selects_roofline,
    validate_roofline_csv,
    view_table_render,
)
from utils.utils_counter_defs import (
    metric_expressions,
)
from vendored import yaml


class PanelConfigMetric(NamedTuple):
    """One metric of a panel config metric_table."""

    panel_id: int  # e.g. 200 for System Speed-of-Light
    table_id: int
    metric_idx: int  # zero-based position of the metric in its table
    metric_name: str
    expressions: List[str]  # metric_expressions of the metric
    table: Dict[str, Any]
    metric: Dict[str, Any]


class ProfileMetric(NamedTuple):
    """One metric library metric that profile collects counters for."""

    metric_id: str
    expressions: List[str]  # MetricLibrary.counter_expressions of the metric


class OmniSoC_Base:
    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        # new info field will contain rocminfo or sysinfo to populate properties
        console_debug("[omnisoc init]")
        self.__args = args
        self.__arch: Optional[str] = None
        # The arch's views and metric library, read on first use
        self.__views: Optional[Dict[int, Dict[str, Any]]] = None
        self.__library: Optional[loader.MetricLibrary] = None
        self._mspec = mspec
        # Per IP block, max number of simultaneous counters. GFX IP Blocks.
        self.__perfmon_config: Dict[str, int] = {}
        self.populate_mspec()

    def __hash__(self) -> int:
        return hash(self.__arch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.__arch == other.get_soc()

    def set_perfmon_config(self, config: Dict[str, int]) -> None:
        self.__perfmon_config = config

    def set_arch(self, arch: str) -> None:
        self.__arch = arch
        self.__panel_configs = None

    def get_arch(self) -> Optional[str]:
        return self.__arch

    def get_args(self) -> argparse.Namespace:
        return self.__args

    def populate_mspec(self) -> None:
        from utils.specs import search_pattern, total_sqc

        if (
            not hasattr(self._mspec, "rocminfo_lines")
            or self._mspec.rocminfo_lines is None
        ):
            return

        # load stats from rocminfo
        self._mspec.gpu_l1 = ""
        self._mspec.gpu_l2 = ""

        for linetext in self._mspec.rocminfo_lines:
            key = search_pattern(r"^\s*L1:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.gpu_l1 = key
                continue

            key = search_pattern(r"^\s*L2:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.gpu_l2 = key
                continue

            key = search_pattern(r"^\s*Max Clock Freq\. \(MHz\):\s+([0-9]+)", linetext)
            if key is not None:
                self._mspec.max_sclk = key
                continue

            key = search_pattern(r"^\s*Compute Unit:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.cu_per_gpu = key
                continue

            key = search_pattern(r"^\s*SIMDs per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.simd_per_cu = key
                continue

            key = search_pattern(r"^\s*Shader Engines:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.se_per_gpu = key
                continue

            key = search_pattern(
                r"^\s*Shader Arrs. per Eng.:\s+ ([a-zA-Z0-9]+)\s*", linetext
            )
            if key is not None:
                self._mspec.sa_per_se = key
                continue

            key = search_pattern(r"^\s*Wavefront Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.wave_size = key
                continue

            key = search_pattern(
                r"^\s*Workgroup Max Size:\s+ ([a-zA-Z0-9]+)\s*", linetext
            )
            if key is not None:
                self._mspec.workgroup_max_size = key
                continue

            key = search_pattern(
                r"^\s*Max Waves Per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext
            )
            if key is not None:
                self._mspec.max_waves_per_cu = key
                break

        if self._mspec.gpu_arch and self._mspec.cu_per_gpu and self._mspec.se_per_gpu:
            self._mspec.sqc_per_gpu = str(
                total_sqc(
                    self._mspec.gpu_arch, self._mspec.cu_per_gpu, self._mspec.se_per_gpu
                )
            )

        with amdsmi_interface.amdsmi_ctx():
            self._mspec.max_mclk = str(amdsmi_interface.get_mem_max_clock())

        # These are just max values now, because the parsing was broken and this was
        # inconsistent with how we use the clocks elsewhere (all max, all the time)
        self._mspec.cur_sclk = self._mspec.max_sclk
        self._mspec.cur_mclk = self._mspec.max_mclk

        if self._mspec.gpu_arch:
            self._mspec.gpu_series = mi_gpu_specs.get_gpu_series(self._mspec.gpu_arch)
        # specify gpu model name for gfx942 hardware
        self._mspec.gpu_model = mi_gpu_specs.get_gpu_model(
            self._mspec.gpu_arch, self._mspec.gpu_chip_id
        )

        if not self._mspec.gpu_model:
            self._mspec.gpu_model = self.detect_gpu_model(self._mspec.gpu_arch)

    @demarcate
    def detect_gpu_model(self, gpu_arch: str) -> Optional[str]:
        """
        Detects the GPU model using various identifiers from 'amd-smi static'.
        Falls back through multiple methods if the primary method fails.
        """
        with amdsmi_interface.amdsmi_ctx():
            gpu_model = "N/A"
            for model in mi_gpu_specs.get_all_gpu_models():
                for amdsmi_gpu_model in amdsmi_interface.get_gpu_model():
                    if model.lower() in amdsmi_gpu_model.lower():
                        gpu_model = model
                        break

        gpu_model = self._adjust_mi300_model(gpu_model.lower(), gpu_arch.lower())

        if gpu_model.lower() not in mi_gpu_specs.get_num_xcds_dict().keys():
            console_warning(f'Unknown GPU model detected: "{gpu_model}".')
            return

        return gpu_model.upper()

    def _adjust_mi300_model(self, gpu_model: str, gpu_arch: str) -> str:
        """
        Applies specific adjustments for MI300 series GPU models based on architecture.
        """

        if gpu_model in ["mi300a", "mi300x"] and gpu_arch == "gfx942":
            gpu_model += "_a1"

        return gpu_model

    def _filter_token_metric_ids(self, raw_token: str) -> List[str]:
        """Return the metric ids of the view rows that a -b block, table or row id,
        or a block alias, selects."""
        block_id = raw_token
        if METRIC_ID_RE.match(block_id):
            pass
        else:
            alias = block_id
            panel_alias_dict = get_arch_alias_to_panel_id(self._mspec.gpu_arch)
            if alias not in panel_alias_dict:
                console_error(f"Invalid --block value {alias!r}.{INVALID_BLOCK_HINT}")
            block_id = str(panel_alias_dict[alias])
            console_log(f"alias: {alias}, block id: {block_id}")

        file_id, panel_id, metric_id = convert_metric_id_to_panel_info(block_id)
        views = self._profile_views()
        if int(file_id) not in views:
            console_warning(f"Skipping {block_id}: panel {file_id} not found")
            return []
        if panel_id is None:
            return [
                selected
                for _, row in iter_view_rows({int(file_id): views[int(file_id)]})
                for selected in row_metric_ids(row)
            ]

        tables = {
            table_id: table
            for table_id, table in numbered_view_tables(
                int(file_id), views[int(file_id)]
            )
            if view_table_render(table) == METRIC_TABLE
        }
        if panel_id not in tables:
            console_warning(
                f"Skipping {block_id}: metric table {panel_id} not found in "
                f"panel {file_id}"
            )
            return []
        rows = tables[panel_id]["metrics"]
        if metric_id is None:
            return [selected for row in rows for selected in row_metric_ids(row)]
        if metric_id >= len(rows):
            console_warning(
                f"Skipping {block_id}: metric id {metric_id} not found in "
                f"panel id {panel_id}"
            )
            return []
        return row_metric_ids(rows[metric_id])

    @demarcate
    def detect_counters(self) -> Tuple[Set[str], List[str]]:
        """Return the counters of the metrics that the selected blocks show, from
        the metric library, and the selected blocks."""
        args = self.get_args()

        filter_blocks = args.filter_blocks
        if args.set_selected and self.__arch:
            sets_info = parse_sets_yaml(self.__arch)
            if args.set_selected not in sets_info:
                console_error(
                    f'argument --set: invalid choice: "{args.set_selected}" '
                    f"(choose from {', '.join(sets_info)})"
                )
            filter_blocks = [
                next(iter(metric.keys()))
                for metric in sets_info[args.set_selected]["metric"]
            ]

        if filter_blocks:
            metric_ids = [
                metric_id
                for block_id in filter_blocks
                for metric_id in self._filter_token_metric_ids(block_id)
            ]
        else:
            # Every metric by default; block 30 only with --membw-analysis
            metric_ids = [metric.metric_id for metric in self._iter_profile_metrics()]

        counters = self._profile_library().get_counters_for_metrics(
            list(dict.fromkeys(metric_ids))
        )
        counters = self._expand_tcc_template_counters(counters)

        return counters, filter_blocks

    def _expand_tcc_template_counters(self, counters: Set[str]) -> Set[str]:
        """
        Expand TCC channel templates (name ending with '[') into indexed channel
        counters, matching perfmon allocation.
        """
        out = set(counters)
        # num_xcd is absent on single-die gfx115x; default to 1.
        num_xcd = int(getattr(self._mspec, "num_xcd", 1) or 1)
        l2_banks = int(self._mspec.l2_banks)
        for counter_name in counters.copy():
            if counter_name.startswith("TCC") and counter_name.endswith("["):
                out.discard(counter_name)
                base = counter_name.split("[")[0]
                out.update(f"{base}[{i}]" for i in range(num_xcd * l2_banks))
        return out

    @demarcate
    def perfmon_filter(self) -> List[str]:
        """Filter default performance counter set based on user arguments"""
        counters, filter_blocks = self.detect_counters()

        if is_only_pc_sampling(filter_blocks):
            console_log(
                "profiling",
                "PC sampling only mode -- skipping counter collection setup",
            )
            return filter_blocks

        # Coalesce and writeback workload specific perfmon
        self.perfmon_coalesce(counters)

        return filter_blocks

    def _allocate_perfmon_counter_files(self, counters: Set[str]) -> List[CounterFile]:
        """Pack counters into perfmon passes, so each metric whose counters fit
        one pass reads them all from one pass."""
        gpu_arch = self._mspec.gpu_arch
        collectables = Collectables.of_arch(gpu_arch)
        return allocate_single_pass_metrics(
            counters,
            metric_counter_sets(
                [
                    *(metric.expressions for metric in self._iter_profile_metrics()),
                    # Each collectable gets a pass that holds all its counters
                    *([formula] for formula in collectables.values()),
                ],
                collectables,
                counters,
            ),
            self.__perfmon_config,
            gpu_arch,
        )

    def _profile_views(self) -> Dict[int, Dict[str, Any]]:
        """Return the arch's analysis views, whose rows name the metrics profile
        collects counters for."""
        if self.__views is None:
            self.__views = load_views(self.__arch)
        return self.__views

    def _profile_library(self) -> loader.MetricLibrary:
        """Return the arch's metric library, which gives each metric's counters."""
        if self.__library is None:
            self.__library = loader.MetricLibrary.load(
                loader.default_metric_library_dir(), self.__arch
            )
        return self.__library

    def _iter_profile_metrics(self) -> Iterator[ProfileMetric]:
        """Iterate the metrics of the arch's views that profile collects by default,
        once each, in view order.

        Block 30 is skipped unless --membw-analysis is set.
        """
        views = self._profile_views()
        if not self.get_args().membw_analysis:
            views = {
                panel_id: view
                for panel_id, view in views.items()
                if panel_id != MEMBW_ANALYSIS_PANEL_ID
            }
        metric_ids = [
            metric_id
            for _, row in iter_view_rows(views)
            for metric_id in row_metric_ids(row)
        ]
        for metric_id in dict.fromkeys(metric_ids):
            yield ProfileMetric(
                metric_id, self._profile_library().counter_expressions(metric_id)
            )

    def get_rocprof_supported_counters(self) -> Set[str]:
        args = self.get_args()
        rocprof_counters: Set[str] = set()

        # Point to counter definition
        old_rocprofiler_metrics_path = os.environ.get("ROCPROFILER_METRICS_PATH")
        with open(
            config.rocprof_compute_home
            / "rocprof_compute_soc"
            / "profile_configs"
            / "sdk_config.yaml",
            encoding="utf-8",
        ) as filename:
            sdk_config = yaml.safe_load(filename)
        os.environ["ROCPROFILER_METRICS_PATH"] = create_temp_rocprofiler_metrics_path(
            sdk_config
        )

        counters = rocprofv3_avail_interface.get_counters(
            args.rocprofiler_sdk_tool_path
        )
        rocprof_counters = {
            counter.name
            for counter in counters[list(counters.keys())[0]]
            if hasattr(counter, "block") or hasattr(counter, "expression")
        }
        # Delete counter definition temporary directory
        if os.environ.get("ROCPROFILER_METRICS_PATH"):
            shutil.rmtree(os.environ["ROCPROFILER_METRICS_PATH"], ignore_errors=True)
        # Reset env. var.
        if old_rocprofiler_metrics_path is None:
            del os.environ["ROCPROFILER_METRICS_PATH"]
        else:
            os.environ["ROCPROFILER_METRICS_PATH"] = old_rocprofiler_metrics_path

        return rocprof_counters

    @demarcate
    def perfmon_coalesce(self, counters: Set[str]) -> None:
        """Pack counters into perfmon passes and write one pmc_perf_<N>.yaml per
        pass."""
        workload_perfmon_dir = Path(self.get_args().output_directory) / "perfmon"
        workload_perfmon_dir.mkdir(parents=True, exist_ok=True)

        rocprof_counters = self.get_rocprof_supported_counters()
        # rocprof does not support TCC channel counters in the avail output,
        # so remove channel suffix for comparison
        not_supported_counters = {
            counter.split("[")[0] if is_tcc_channel_counter(counter) else counter
            for counter in counters
        } - rocprof_counters

        if not_supported_counters:
            console_warning(
                "Following counters might not be supported by rocprof: "
                f"{', '.join(not_supported_counters)}"
            )

        # We might be providing definitions of unsupported counters, so still try to
        # collect them
        if not counters:
            console_error(
                "profiling",
                "No performance counters to collect, "
                "please check the provided profiling filters",
            )

        console_debug(f"Collecting following counters: {', '.join(counters)} ")

        output_files = self._allocate_perfmon_counter_files(counters)

        console_debug("profiling", f"perfmon_coalesce file_count {len(output_files)}")

        # Output to files
        for f in output_files:
            pmc_filename = workload_perfmon_dir / f"pmc_perf_{f.name}.yaml"
            counter_def_filename = workload_perfmon_dir / f"counter_def_{f.name}.yaml"

            pmc = []
            counter_def: Dict[str, Any] = {}

            for ctr in [
                ctr for block_name in f.blocks for ctr in f.blocks[block_name].elements
            ]:
                pmc.append(ctr)
                # Add TCC channel counters definitions
                if is_tcc_channel_counter(ctr):
                    counter_name = ctr.split("[")[0]
                    idx = int(ctr.split("[")[1].split("]")[0])
                    xcd_idx = idx // int(self._mspec.l2_banks)
                    channel_idx = idx % int(self._mspec.l2_banks)
                    expression = (
                        f"select({counter_name},"
                        f"[DIMENSION_XCC=[{xcd_idx}], "
                        f"DIMENSION_INSTANCE=[{channel_idx}]])"
                    )
                    description = (
                        f"{counter_name} on {xcd_idx}th XCC and {channel_idx}th channel"
                    )
                    counter_def = add_counter_extra_config_input_yaml(
                        counter_def,
                        ctr,
                        description,
                        expression,
                        [self.__arch],
                    )

            # Write counters to file
            with open(pmc_filename, "w", encoding="utf-8") as fd:
                fd.write(yaml.dump({"jobs": [{"pmc": pmc}]}, sort_keys=False))

            # Write counter definitions to file
            if counter_def:
                with open(counter_def_filename, "w", encoding="utf-8") as fp:
                    fp.write(yaml.dump(counter_def, sort_keys=False))

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def profiling_setup(self) -> Optional[List[str]]:
        """Perform any SoC-specific setup prior to profiling."""
        console_debug("profiling", f"perform SoC profiling setup for {self.__arch}")

    @abstractmethod
    def post_profiling(self) -> None:
        """Perform any SoC-specific post profiling activities."""
        console_debug("profiling", f"perform SoC post processing for {self.__arch}")
        # Roofline can be skipped via the deprecated --no-roof
        # Roofline not supported on MI 100
        # If -b or --set selected blocks, roofline (block 4) must be one of them
        if (
            self.get_args().no_roof
            or self.__arch not in BENCHMARKING_SUPPORTED
            or not selects_roofline(self.get_args().filter_blocks)
        ):
            console_log("roofline", "Skipping roofline")
        else:
            roofline_csv = Path(self.get_args().output_directory) / "roofline.csv"
            console_log(
                "roofline",
                f"Checking for roofline.csv in {self.get_args().output_directory}",
            )
            if not roofline_csv.is_file():
                try:
                    run_roofline_benchmark(
                        self.get_args().device,
                        roofline_csv,
                        self._mspec.cache_sizes,
                        self._mspec.gpu_arch,
                    )
                except Exception as e:
                    console_error(
                        "roofline",
                        f"Benchmark execution failed: {e}. Skipping roofline.",
                        exit=False,
                    )
                    return

            is_valid, error_msg = validate_roofline_csv(
                self.get_args().output_directory
            )
            if not is_valid:
                console_error(
                    "roofline",
                    f"Invalid roofline.csv: {error_msg}",
                    exit=False,
                )
                return

            console_log(
                "roofline",
                "Roofline data saved to "
                f"{self.get_args().output_directory}/roofline.csv\n"
                "  Run 'rocprof-compute analyze -p "
                f"{self.get_args().output_directory}' "
                "for charts",
            )


def _table_expressions(table: Dict[str, Any]) -> List[str]:
    """Expressions of every metric in one metric_table."""
    return [
        expression
        for metric in (table["metric"] or {}).values()
        for expression in metric_expressions(table, metric)
    ]


def iter_panel_config_metrics(
    panel_configs: "OrderedDict[int, Dict[str, Any]]", exclude_panel_ids: Set[int]
) -> Iterator[PanelConfigMetric]:
    """Iterate the metrics of panel_configs, skipping the panels in
    exclude_panel_ids."""
    for panel_id, panel_config in panel_configs.items():
        if panel_id in exclude_panel_ids:
            continue
        for section in panel_config["data source"]:
            table = section.get("metric_table")
            if table is None:
                continue
            metrics = table["metric"] or {}
            for idx, (metric_name, metric) in enumerate(metrics.items()):
                yield PanelConfigMetric(
                    panel_id,
                    table["id"],
                    idx,
                    metric_name,
                    metric_expressions(table, metric),
                    table,
                    metric,
                )
