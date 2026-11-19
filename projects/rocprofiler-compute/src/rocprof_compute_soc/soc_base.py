# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT


import argparse
import os
import shutil
from abc import abstractmethod
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
from roofline.run_benchmark import BENCHMARKING_SUPPORTED, run_roofline_benchmark
from utils import amdsmi_interface, rocprofv3_avail_interface
from utils.analysis_views import get_arch_alias_to_panel_id
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
    add_counter_extra_config_input_yaml,
    canonical_config_arch,
    convert_metric_id_to_panel_info,
    create_temp_rocprofiler_metrics_path,
    is_only_pc_sampling,
    is_tcc_channel_counter,
    parse_sets_yaml,
    selects_roofline,
    validate_roofline_csv,
)
from utils.utils_counter_defs import (
    Collectables,
    extract_counters_and_variables,
    metric_expressions,
)
from vendored import yaml


class AnalysisConfigMetric(NamedTuple):
    """One metric of an analysis config metric_table."""

    stem_id: str  # file id, e.g. "0200" from "0200_system_speed_of_light.yaml"
    table_id: int
    metric_idx: int  # zero-based position of the metric in its table
    metric_name: str
    expressions: List[str]  # metric_expressions of the metric
    table: Dict[str, Any]
    metric: Dict[str, Any]


class OmniSoC_Base:
    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        # new info field will contain rocminfo or sysinfo to populate properties
        console_debug("[omnisoc init]")
        self.__args = args
        self.__arch: Optional[str] = None
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

    def _append_expressions_for_filter_token(
        self,
        raw_token: str,
        config_filename_dict: Dict[str, str],
        config_root_dir: str,
        expressions: List[str],
    ) -> None:
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

        if file_id not in config_filename_dict:
            console_warning(
                f"Skipping {block_id}: file id {file_id} not found in {config_root_dir}"
            )
            return

        with open(config_filename_dict[file_id], encoding="utf-8") as stream:
            file_config = yaml.safe_load(stream)
        if panel_id is None:
            for section in file_config["Panel Config"]["data source"]:
                if "metric_table" in section:
                    expressions.extend(_table_expressions(section["metric_table"]))
            return

        panel_dict = {
            section["metric_table"]["id"]: section["metric_table"]
            for section in file_config["Panel Config"]["data source"]
            if "metric_table" in section
        }
        if panel_id not in panel_dict:
            console_warning(
                f"Skipping {block_id}: metric table {panel_id} not found in "
                f"{config_filename_dict[file_id]}"
            )
            return
        if metric_id is None:
            expressions.extend(_table_expressions(panel_dict[panel_id]))
            return

        metric_dict = {
            idx: panel_dict[panel_id]["metric"][metric]
            for idx, metric in enumerate(panel_dict[panel_id]["metric"].keys())
        }
        if metric_id not in metric_dict:
            console_warning(
                f"Skipping {block_id}: metric id {metric_id} not found in "
                f"panel id {panel_id}"
            )
            return
        expressions.extend(
            metric_expressions(panel_dict[panel_id], metric_dict[metric_id])
        )

    @demarcate
    def detect_counters(self) -> Tuple[Set[str], List[str]]:
        """
        Create a set of counters required for the selected report sections,
        from the metric expressions of those sections.
        """
        args = self.get_args()

        # File id dict
        config_arch = canonical_config_arch(self.__arch) or self.__arch
        config_root_dir = str(config.analysis_configs_dir / config_arch)
        config_filename_dict = {
            filename.name.split("_")[0]: str(filename)
            for filename in Path(config_root_dir).glob("*.yaml")
        }

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

        expressions: List[str] = []
        if not filter_blocks:
            # Select all sections by default; block 30 only with --membw-analysis
            for metric in self._iter_arch_analysis_yaml_metrics():
                expressions.extend(metric.expressions)

        for block_id in filter_blocks:
            self._append_expressions_for_filter_token(
                block_id, config_filename_dict, config_root_dir, expressions
            )

        counters, _ = extract_counters_and_variables(
            "\n".join(expressions), Collectables.of_arch(self._mspec.gpu_arch)
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
                    *(
                        metric.expressions
                        for metric in self._iter_arch_analysis_yaml_metrics()
                    ),
                    # Each collectable gets a pass that holds all its counters
                    *([formula] for formula in collectables.values()),
                ],
                collectables,
                counters,
            ),
            self.__perfmon_config,
            gpu_arch,
        )

    def _iter_arch_analysis_yaml_metrics(self) -> Iterator[AnalysisConfigMetric]:
        """Iterate the metrics of this arch's analysis configs.

        Block 30 is skipped unless --membw-analysis is set. See
        iter_analysis_config_metrics for what each item holds.
        """
        args = self.get_args()
        arch = self.__arch
        if not arch:
            return
        exclude: Set[str] = set()
        if not args.membw_analysis:
            exclude.add(str(MEMBW_ANALYSIS_PANEL_ID))
        yield from iter_analysis_config_metrics(
            config.analysis_configs_dir / (canonical_config_arch(arch) or arch), exclude
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


def iter_analysis_config_metrics(
    config_root: Path, exclude_file_ids: Set[str]
) -> Iterator[AnalysisConfigMetric]:
    """Iterate the metrics of the analysis config files in config_root, skipping files
    whose id is in exclude_file_ids."""
    for config_path in sorted(config_root.glob("*.yaml")):
        stem_id = config_path.name.split("_")[0]
        if stem_id in exclude_file_ids:
            continue
        with open(config_path, encoding="utf-8") as stream:
            panel_config = yaml.safe_load(stream)["Panel Config"]
        for section in panel_config["data source"]:
            table = section.get("metric_table")
            if table is None:
                continue
            metrics = table["metric"] or {}
            for idx, (metric_name, metric) in enumerate(metrics.items()):
                yield AnalysisConfigMetric(
                    stem_id,
                    table["id"],
                    idx,
                    metric_name,
                    metric_expressions(table, metric),
                    table,
                    metric,
                )
