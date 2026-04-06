# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import config
from utils.amdsmi_interface import amdsmi_ctx, get_gpu_model, get_mem_max_clock
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
    BUILD_IN_VARS,
    METRIC_ID_RE,
    SUPPORTED_DENOM,
    add_counter_extra_config_input_yaml,
    convert_metric_id_to_panel_info,
    create_temp_rocprofiler_metrics_path,
    get_panel_alias,
    is_only_pc_sampling,
    is_tcc_channel_counter,
    parse_sets_yaml,
    resolve_rocm_library_path,
    validate_roofline_csv,
)
from vendored import yaml


class OmniSoC_Base:
    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        # new info field will contain rocminfo or sysinfo to populate properties
        console_debug("[omnisoc init]")
        self.__args = args
        self.__arch: Optional[str] = None
        self._mspec = mspec
        # Per IP block, max number of simultaneous counters. GFX IP Blocks.
        self.__perfmon_config: dict[str, int] = {}
        self.__compatible_profilers: list[str] = []  # Store SoC compatible profilers
        self.populate_mspec()

    def __hash__(self) -> int:
        return hash(self.__arch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.__arch == other.get_soc()

    def set_perfmon_config(self, config: dict[str, int]) -> None:
        self.__perfmon_config = config

    def set_arch(self, arch: str) -> None:
        self.__arch = arch

    def set_compatible_profilers(self, profiler_names: list[str]) -> None:
        self.__compatible_profilers = profiler_names

    def get_arch(self) -> Optional[str]:
        return self.__arch

    def get_args(self) -> argparse.Namespace:
        return self.__args

    def get_compatible_profilers(self) -> list[str]:
        return self.__compatible_profilers

    def populate_mspec(self) -> None:
        from utils.specs import search, total_sqc

        if (
            not hasattr(self._mspec, "rocminfo_lines")
            or self._mspec.rocminfo_lines is None
        ):
            return

        # load stats from rocminfo
        self._mspec.gpu_l1 = ""
        self._mspec.gpu_l2 = ""

        for linetext in self._mspec.rocminfo_lines:
            key = search(r"^\s*L1:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.gpu_l1 = key
                continue

            key = search(r"^\s*L2:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.gpu_l2 = key
                continue

            key = search(r"^\s*Max Clock Freq\. \(MHz\):\s+([0-9]+)", linetext)
            if key is not None:
                self._mspec.max_sclk = key
                continue

            key = search(r"^\s*Compute Unit:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.cu_per_gpu = key
                continue

            key = search(r"^\s*SIMDs per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.simd_per_cu = key
                continue

            key = search(r"^\s*Shader Engines:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.se_per_gpu = key
                continue

            key = search(r"^\s*Wavefront Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.wave_size = key
                continue

            key = search(r"^\s*Workgroup Max Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.workgroup_max_size = key
                continue

            key = search(r"^\s*Max Waves Per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.max_waves_per_cu = key
                break

        if self._mspec.gpu_arch and self._mspec.cu_per_gpu and self._mspec.se_per_gpu:
            self._mspec.sqc_per_gpu = str(
                total_sqc(
                    self._mspec.gpu_arch, self._mspec.cu_per_gpu, self._mspec.se_per_gpu
                )
            )

        with amdsmi_ctx():
            self._mspec.max_mclk = str(get_mem_max_clock())

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

        self._mspec.num_xcd = str(
            mi_gpu_specs.get_num_xcds(
                self._mspec.gpu_arch,
                self._mspec.gpu_model,
                self._mspec.compute_partition,
            )
        )

    @demarcate
    def detect_gpu_model(self, gpu_arch: str) -> Optional[str]:
        """
        Detects the GPU model using various identifiers from 'amd-smi static'.
        Falls back through multiple methods if the primary method fails.
        """
        with amdsmi_ctx():
            gpu_model = "N/A"
            for model in mi_gpu_specs.get_all_gpu_models():
                for amdsmi_gpu_model in get_gpu_model():
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

        if gpu_model in ["mi300a", "mi300x"]:
            if gpu_arch in ["gfx940", "gfx941"]:
                gpu_model += "_a0"
            elif gpu_arch == "gfx942":
                gpu_model += "_a1"

        return gpu_model

    @demarcate
    def detect_counters(self) -> tuple[set[str], list[str]]:
        """
        Create a set of counters required for the selected report sections.
        Parse analysis report configuration files based on the selected report
        sections to be filtered.
        """
        args = self.get_args()

        # File id dict
        config_root_dir = f"{args.config_dir}/{self.__arch}"
        config_filename_dict = {
            filename.name.split("_")[0]: str(filename)
            for filename in Path(config_root_dir).glob("*.yaml")
        }

        filter_blocks = args.filter_blocks
        if args.set_selected and self.__arch:
            sets_info = parse_sets_yaml(self.__arch)
            if args.set_selected not in set(sets_info.keys()):
                console_error(
                    f'argument --set: invalid choice: "{args.set_selected}" '
                    f"(choose from {sets_info.keys()})"
                )
            filter_blocks = [
                next(iter(metric.keys()))
                for metric in sets_info[args.set_selected]["metric"]
            ]
        elif args.roof_only:
            filter_blocks = ["4"]

        texts: list[str] = []
        if not filter_blocks:
            # Do not profile block 30 unless explicitly requested
            exclude_file_ids: set[str] = set()
            if not args.membw_analysis:
                exclude_file_ids.add("3000")

            # Select all sections by default
            for file_id, filename in config_filename_dict.items():
                if file_id in exclude_file_ids:
                    continue

                with open(filename) as stream:
                    texts.append(stream.read())

        for block_id in filter_blocks:
            if METRIC_ID_RE.match(block_id):
                block_id = block_id
            else:
                alias = block_id
                panel_alias_dict = get_panel_alias()
                if alias not in panel_alias_dict:
                    raise KeyError(f"Unknown panel alias: {alias!r}")
                block_id = panel_alias_dict[alias]  # int
                print(f"alias: {alias}, block id: {block_id}")

            file_id, panel_id, metric_id = convert_metric_id_to_panel_info(block_id)

            # File id filtering
            if file_id not in config_filename_dict:
                console_warning(
                    f"Skipping {block_id}: file id {file_id} not found in "
                    f"{config_root_dir}"
                )
                continue

            with open(config_filename_dict[file_id]) as stream:
                file_config = yaml.safe_load(stream)
            if panel_id is None:
                # If no panel id level filtering, then read the whole file
                texts.append(yaml.dump(file_config, sort_keys=False))
                continue

            # Panel id filtering
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
                continue
            if metric_id is None:
                # If no metric id level filtering, then read the whole panel
                texts.append(yaml.dump(panel_dict[panel_id], sort_keys=False))
                continue

            # Metric id filtering
            metric_dict = {
                id: panel_dict[panel_id]["metric"][metric]
                for id, metric in enumerate(panel_dict[panel_id]["metric"].keys())
            }
            if metric_id not in metric_dict:
                console_warning(
                    f"Skipping {block_id}: metric id {metric_id} not found in "
                    f"panel id {panel_id}"
                )
                continue
            texts.append(yaml.dump(metric_dict[metric_id], sort_keys=False))

        counters = self.parse_counters("\n".join(texts))

        # Handle TCC channel counters: if hw_counter_matches has elems ending with '['
        # Expand and interleve the TCC channel counters
        # e.g.  TCC_HIT[0] TCC_ATOMIC[0] ... TCC_HIT[1] TCC_ATOMIC[1] ...
        num_xcd_for_pmc_file = int(self._mspec.num_xcd)
        for counter_name in counters.copy():
            if counter_name.startswith("TCC") and counter_name.endswith("["):
                counters.remove(counter_name)
                counter_name = counter_name.split("[")[0]
                counters = counters.union({
                    f"{counter_name}[{i}]"
                    for i in range(num_xcd_for_pmc_file * int(self._mspec.l2_banks))
                })

        return counters, filter_blocks

    @demarcate
    def perfmon_filter(self) -> list[str]:
        """Filter default performance counter set based on user arguments"""
        counters, filter_blocks = self.detect_counters()

        if is_only_pc_sampling(filter_blocks):
            console_log(
                "profiling",
                "PC sampling only mode -- skipping counter collection setup",
            )
            return filter_blocks

        # SQ_ACCUM_PREV_HIRES will be injected for level counters later on
        counters = counters - {"SQ_ACCUM_PREV_HIRES"}

        # Coalesce and writeback workload specific perfmon
        self.perfmon_coalesce(counters)

        return filter_blocks

    @demarcate
    def parse_counters(self, config_text: str) -> set[str]:
        """
        Create a set of all hardware counters mentioned in the given config file
        content string.
        """
        hw_counter_matches, variable_matches = self.parse_counters_text(config_text)

        # get hw counters and variables for all supported denominators
        for formula in SUPPORTED_DENOM.values():
            hw_counter_matches_denom, variable_matches_denom = self.parse_counters_text(
                formula
            )
            hw_counter_matches.update(hw_counter_matches_denom)
            variable_matches.update(variable_matches_denom)

        # get hw counters corresponding to variables recursively
        while variable_matches:
            subvariable_matches: set[str] = set()
            for var in variable_matches:
                if var in BUILD_IN_VARS:
                    (
                        hw_counter_matches_vars,
                        variable_matches_vars,
                    ) = self.parse_counters_text(BUILD_IN_VARS[var])
                    hw_counter_matches.update(hw_counter_matches_vars)
                    subvariable_matches.update(variable_matches_vars)
            # process new found variables
            variable_matches = subvariable_matches - variable_matches

        return hw_counter_matches

    def parse_counters_text(self, text: str) -> tuple[set[str], set[str]]:
        """Parse out hardware counters and variables from given text"""
        # hw counter name should start with ip block name
        # hw counter name should have all capital letters or digits
        # and should not end with underscore
        # he counter name can either optionally end with '[' or '_sum'
        hw_counter_regex = (
            r"(?:SQ|SQC|TA|TD|TCP|TCC|CPC|CPF|SPI|GRBM)_[0-9A-Z_]*[0-9A-Z](?:\[|_sum)*"
        )
        # only capture the variable name after $ using capturing group
        variable_regex = r"\$([0-9A-Za-z_]*[0-9A-Za-z])"
        hw_counter_matches = set(re.findall(hw_counter_regex, text))
        variable_matches = set(re.findall(variable_regex, text))
        # variable matches cannot be counters
        hw_counter_matches = hw_counter_matches - variable_matches
        return hw_counter_matches, variable_matches

    def get_rocprof_supported_counters(self) -> set[str]:
        args = self.get_args()
        rocprof_counters: set[str] = set()

        # Point to counter definition
        old_rocprofiler_metrics_path = os.environ.get("ROCPROFILER_METRICS_PATH")
        with open(
            config.rocprof_compute_home
            / "rocprof_compute_soc"
            / "profile_configs"
            / "sdk_config.yaml",
        ) as filename:
            sdk_config = yaml.safe_load(filename)
        os.environ["ROCPROFILER_METRICS_PATH"] = create_temp_rocprofiler_metrics_path(
            sdk_config
        )

        # Backward compatibility support for sdk avail module moved from
        # <rocm_path>/bin/rocprofv3_avail_module/avail.py to
        # <rocm_path>/lib/python3/site-packages/rocprofv3/avail.py
        new_path = str(
            Path(args.rocprofiler_sdk_tool_path).parents[1] / "python3/site-packages"
        )
        old_path = str(Path(args.rocprofiler_sdk_tool_path).parents[2] / "bin")
        try:
            sys.path.append(new_path)
            from rocprofv3 import avail
        except ImportError:
            console_debug(
                f"Could not import rocprofiler-sdk avail module from {new_path}, "
                f"trying {old_path}"
            )
            try:
                sys.path.remove(new_path)
                sys.path.append(old_path)
                from rocprofv3_avail_module import avail
            except ImportError:
                console_error("Failed to import rocprofiler-sdk avail module.")

        # librocprofv3-list-avail.so location varies by ROCm version:
        #   ROCm >= 7.1: <rocm_path>/lib/rocprofiler-sdk/
        #   ROCm 7.0.x:  <rocm_path>/libexec/rocprofiler-sdk/
        avail_lib_name = "librocprofv3-list-avail.so"
        avail_lib_path = resolve_rocm_library_path(
            str(Path(args.rocprofiler_sdk_tool_path).parent / avail_lib_name)
        )
        if not Path(avail_lib_path).exists():
            avail_lib_path = resolve_rocm_library_path(
                str(
                    Path(args.rocprofiler_sdk_tool_path).parents[2]
                    / "libexec"
                    / "rocprofiler-sdk"
                    / avail_lib_name
                )
            )
        avail.loadLibrary.libname = avail_lib_path
        counters = avail.get_counters()
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
    def _allocate_perfmon_counter_files(
        self, counters: set[str]
    ) -> tuple[list[CounterFile], int, int]:
        """Bin-pack counters into perfmon buckets (first-fit)."""
        output_files: list[CounterFile] = []
        accu_file_count = 0

        # Create separate perfmon file for LEVEL counters without _sum suffix
        # TCC LEVEL counters are handled channel wise, so ignore them
        # Convert set to sorted list for determinism in pmc txt files
        work = sorted(list(counters))
        for counter in work.copy():
            if (
                "LEVEL" in counter
                and not counter.endswith("_sum")
                and not is_tcc_channel_counter(counter)
            ):
                work.remove(counter)
                output_files.append(CounterFile(counter, self.__perfmon_config))
                output_files[-1].add(counter)
                output_files[-1].add(f"{counter}_ACCUM")
                accu_file_count += 1

        file_count = 0
        # Store all channels for a TCC channel counter in the same file
        tcc_channel_counter_file_map: dict[str, CounterFile] = {}

        for ctr in work:
            # Store all channels for a TCC channel counter in the same file
            if is_tcc_channel_counter(ctr):
                output_file = tcc_channel_counter_file_map.get(ctr.split("[")[0])
                if output_file:
                    output_file.add(ctr)
                    continue

            # Add counter to first file that has room
            added = False
            for output_file in output_files:
                if output_file.add(ctr):
                    added = True
                    # Store all channels for a TCC channel counter in the same file
                    if is_tcc_channel_counter(ctr):
                        tcc_channel_counter_file_map[ctr.split("[")[0]] = output_file
                    break

            # All files are full, create a new file
            if not added:
                output_files.append(CounterFile(str(file_count), self.__perfmon_config))
                file_count += 1
                output_files[-1].add(ctr)

        return output_files, file_count, accu_file_count

    @demarcate
    def perfmon_coalesce(self, counters: set[str]) -> None:
        """
        Sort and bucket all related performance counters to minimize required
        application passes
        """
        workload_perfmon_dir = Path(self.get_args().path) / "perfmon"
        workload_perfmon_dir.mkdir(parents=True, exist_ok=True)

        # Dry-run has no rocprofiler-sdk avail list; rely on YAML counter names only.
        if not getattr(self.get_args(), "profile_dry_run", False):
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

        output_files, file_count, accu_file_count = (
            self._allocate_perfmon_counter_files(counters)
        )

        console_debug("profiling", f"perfmon_coalesce file_count {file_count}")

        # TODO: rewrite the above logic for spatial_multiplexing later
        if self.get_args().spatial_multiplexing:
            # TODO: more error checking
            if len(self.get_args().spatial_multiplexing) != 3:
                console_error(
                    "profiling",
                    "multiplexing need provide node_idx node_count and gpu_count",
                )

            node_idx, node_count, gpu_count = map(
                int, self.get_args().spatial_multiplexing
            )

            old_group_num = file_count + accu_file_count
            new_bucket_count = node_count * gpu_count
            groups_per_bucket = math.ceil(
                old_group_num / new_bucket_count
            )  # It equals to file num per node
            max_groups_per_node = groups_per_bucket * gpu_count

            group_start = node_idx * max_groups_per_node
            group_end = min((node_idx + 1) * max_groups_per_node, old_group_num)

            console_debug(
                "profiling",
                f"spatial_multiplexing node_idx {node_idx}, node_count {node_count}, "
                f"gpu_count: {gpu_count},\n"
                f"old_group_num {old_group_num}, new_bucket_count {new_bucket_count}, "
                f"groups_per_bucket {groups_per_bucket},\n"
                f"max_groups_per_node {max_groups_per_node}, "
                f"group_start {group_start}, group_end {group_end}",
            )

            for f_idx in range(groups_per_bucket):
                file_name = (
                    Path(workload_perfmon_dir)
                    / f"pmc_perf_node_{node_idx}_{f_idx}.yaml"
                )

                pmc = []
                for g_idx in range(
                    group_start + f_idx * gpu_count,
                    min(group_end, group_start + (f_idx + 1) * gpu_count),
                ):
                    gpu_idx = g_idx % gpu_count
                    for block_name in output_files[g_idx].blocks.keys():
                        for ctr in output_files[g_idx].blocks[block_name].elements:
                            pmc.append(f"{ctr}:device={gpu_idx}")

                # Write counters to file
                with open(file_name, "w") as fd:
                    fd.write(yaml.dump({"jobs": [{"pmc": pmc}]}, sort_keys=False))
        else:
            # Output to files
            for f in output_files:
                pmc_filename = workload_perfmon_dir / f.pmc_filename
                counter_def_filename = workload_perfmon_dir / f.counter_def_filename

                pmc = []
                counter_def: dict[str, Any] = {}

                for ctr in [
                    ctr
                    for block_name in f.blocks
                    for ctr in f.blocks[block_name].elements
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
                            f"{counter_name} on {xcd_idx}th XCC and "
                            f"{channel_idx}th channel"
                        )
                        counter_def = add_counter_extra_config_input_yaml(
                            counter_def,
                            ctr,
                            description,
                            expression,
                            [self.__arch],
                        )

                # Write counters to file
                with open(pmc_filename, "w") as fd:
                    fd.write(yaml.dump({"jobs": [{"pmc": pmc}]}, sort_keys=False))

                # Write counter definitions to file
                if counter_def:
                    with open(counter_def_filename, "w") as fp:
                        fp.write(yaml.dump(counter_def, sort_keys=False))

    def _expand_tcc_template_counters(self, counters: set[str]) -> set[str]:
        """
        Expand TCC channel templates (name ending with '[') into indexed channel
        counters, matching perfmon allocation.
        """
        out = set(counters)
        num_xcd = int(self._mspec.num_xcd)
        l2_banks = int(self._mspec.l2_banks)
        for counter_name in counters.copy():
            if counter_name.startswith("TCC") and counter_name.endswith("["):
                out.discard(counter_name)
                base = counter_name.split("[")[0]
                out.update(f"{base}[{i}]" for i in range(num_xcd * l2_banks))
        return out

    def _iter_arch_analysis_yaml_metrics(
        self,
    ) -> Iterator[tuple[str, Any, int, str, str]]:
        """Iterate analysis_configs/<arch> YAML metric_table rows."""
        args = self.get_args()
        arch = self.__arch
        if not arch:
            return
        config_root = Path(args.config_dir) / arch
        if not config_root.is_dir():
            return
        exclude: set[str] = set()
        if not args.membw_analysis:
            exclude.add("3000")

        for ypath in sorted(config_root.glob("*.yaml")):
            stem_id = ypath.name.split("_")[0]
            if stem_id in exclude:
                continue
            try:
                with open(ypath, encoding="utf-8") as stream:
                    doc = yaml.safe_load(stream)
            except (OSError, UnicodeError, yaml.YAMLError):
                continue
            if not isinstance(doc, dict):
                continue
            panel_cfg = doc.get("Panel Config")
            if not isinstance(panel_cfg, dict):
                continue
            sources = panel_cfg.get("data source")
            if not isinstance(sources, list):
                continue
            for section in sources:
                if not isinstance(section, dict) or "metric_table" not in section:
                    continue
                mt = section["metric_table"]
                if not isinstance(mt, dict):
                    continue
                metrics = mt.get("metric")
                if not isinstance(metrics, dict):
                    continue
                panel_id = mt.get("id")
                for idx, (metric_name, metric_body) in enumerate(metrics.items()):
                    try:
                        metric_text = yaml.dump(
                            metric_body, sort_keys=False, allow_unicode=True
                        )
                    except (TypeError, yaml.YAMLError):
                        continue
                    yield stem_id, panel_id, idx, metric_name, metric_text

    def _log_perfmon_bucket_plan(self, output_files: list[CounterFile]) -> None:
        """Stdout markdown tables: one Bucket section per perfmon pass file."""
        arch = self.get_arch() or "unknown"
        print(f"Perfmon dry-run plan (architecture: {arch})")
        global_columns, col_widths = _dry_run_global_ip_column_widths(output_files)
        total_assignments = 0
        for counter_file in output_files:
            bucket_label = counter_file.file_name_txt.replace(".txt", "")
            flat = _flat_counters_in_perfmon_file(counter_file)
            total_assignments += len(flat)
            print(f"\nBucket: {bucket_label}")
            if not flat:
                print("(no PMC counters)")
                continue
            if not global_columns:
                continue
            print(_format_dry_run_bucket_markdown(flat, global_columns, col_widths))
        print(
            f"\nSummary: {len(output_files)} bucket(s), "
            f"{total_assignments} counter assignment(s).",
            flush=True,
        )

    def _print_dry_run_multi_bucket_metrics(
        self, output_files: list[CounterFile]
    ) -> None:
        """
        After bucket allocation, report metrics by how in-collection PMCs map to
        buckets: (1) 2+ bucket labels, (2) exactly one label. Same YAML scan scope.
        """
        args = self.get_args()
        arch = self.__arch or "unknown"
        scan_root = Path(args.config_dir) / arch
        counter_to_bucket = _dry_run_counter_to_bucket_map(output_files)

        if not scan_root.is_dir():
            print(
                "\nMetrics spanning multiple perfmon buckets "
                "(skipped: config directory missing)\n",
                flush=True,
            )
            return

        multi_rows: list[tuple[str, str, int, str, int, str]] = []
        single_rows: list[tuple[str, str, int, str, str]] = []
        total_metrics = 0
        for (
            file_id,
            panel_id,
            metric_idx,
            metric_name,
            metric_yaml,
        ) in self._iter_arch_analysis_yaml_metrics():
            total_metrics += 1
            hw = self.parse_counters(metric_yaml)
            hw = self._expand_tcc_template_counters(hw)
            buckets: set[str] = set()
            # Only PMCs present in this dry-run's collection have a bucket mapping.
            for c in hw:
                b = counter_to_bucket.get(c)
                if b is not None:
                    buckets.add(b)
            panel_s = str(panel_id) if panel_id is not None else "-"
            n_b = len(buckets)
            if n_b > 1:
                multi_rows.append((
                    file_id,
                    panel_s,
                    metric_idx,
                    metric_name,
                    n_b,
                    ", ".join(sorted(buckets)),
                ))
            elif n_b == 1:
                single_rows.append((
                    file_id,
                    panel_s,
                    metric_idx,
                    metric_name,
                    next(iter(buckets)),
                ))

        multi_count = len(multi_rows)
        pct = (100.0 * multi_count / total_metrics) if total_metrics else 0.0
        pct_str = f"{pct:.1f}%" if total_metrics else "n/a"
        print(
            f"\nMetrics with PMC counters assigned to more than one perfmon bucket "
            f"({multi_count} of {total_metrics} metrics, {pct_str})"
        )
        print(
            "All *.yaml under the arch are scanned. Listed rows are metrics where at "
            "least one formula counter is in this dry-run's collection and those "
            "counters map to 2+ buckets in the layout above."
        )
        print(f"Config tree: {scan_root}")

        if total_metrics == 0:
            print("(no metrics found in YAML scan)\n", flush=True)
            return

        if not multi_rows:
            print(
                "(none listed -- in-collection counters stay in one bucket"
                " per metric)\n"
            )
        else:
            _dry_run_print_markdown_metric_table(
                ["File", "Panel", "Idx", "Metric name", "#Bkts", "Buckets"],
                [[r[0], r[1], str(r[2]), r[3], str(r[4]), r[5]] for r in multi_rows],
            )

        single_count = len(single_rows)
        spct = (100.0 * single_count / total_metrics) if total_metrics else 0.0
        spct_str = f"{spct:.1f}%" if total_metrics else "n/a"
        print(
            f"\nMetrics with PMC counters assigned to one perfmon bucket "
            f"({single_count} of {total_metrics} metrics, {spct_str})"
        )
        print(
            "Listed rows are metrics where at least one formula counter is in this "
            "dry-run's collection and every such counter maps to the same single "
            "bucket above (metrics with no in-collection PMCs are omitted)."
        )
        if not single_rows:
            print(
                "(none listed -- no metric has in-collection PMCs confined to one "
                "bucket)\n"
            )
        else:
            _dry_run_print_markdown_metric_table(
                ["File", "Panel", "Idx", "Metric name", "Bucket"],
                [[r[0], r[1], str(r[2]), r[3], r[4]] for r in single_rows],
            )
        print(flush=True)

    @demarcate
    def perfmon_filter_dry_run(self) -> list[str]:
        """Counter resolution like profiling_setup; stdout plan + multi-bucket scan."""
        counters, filter_blocks = self.detect_counters()
        if is_only_pc_sampling(filter_blocks):
            console_log(
                "profiling",
                "Dry-run: PC sampling only -- no PMC perfmon buckets.",
            )
            return filter_blocks

        counters = counters - {"SQ_ACCUM_PREV_HIRES"}
        if not counters:
            console_error(
                "profiling",
                "No performance counters to collect for dry-run, "
                "please check the provided profiling filters",
            )

        console_debug(f"Collecting following counters: {', '.join(counters)} ")
        output_files, file_count, _accu = self._allocate_perfmon_counter_files(counters)
        console_debug("profiling", f"perfmon dry-run file_count {file_count}")
        self._log_perfmon_bucket_plan(output_files)
        self._print_dry_run_multi_bucket_metrics(output_files)
        if self.get_args().spatial_multiplexing:
            console_warning(
                "profiling",
                "Dry-run lists logical PMC buckets only; --spatial-multiplexing "
                "rewrites perfmon layout for multi-node runs.",
            )
        return filter_blocks

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def profiling_setup(self) -> Optional[list[str]]:
        """Perform any SoC-specific setup prior to profiling."""
        console_debug("profiling", f"perform SoC profiling setup for {self.__arch}")

    @abstractmethod
    def post_profiling(self) -> None:
        """Perform any SoC-specific post profiling activities."""
        console_debug("profiling", f"perform SoC post processing for {self.__arch}")
        # Roofline can be skipped via --no-roof
        # Roofline not supported on MI 100
        # If --filter-blocks is provided, roofline block (block 4) should be mentioned
        if (
            self.get_args().no_roof
            or self.__arch == "gfx908"
            or (
                self.get_args().filter_blocks
                and "4" not in self.get_args().filter_blocks
                and "roof" not in self.get_args().filter_blocks
            )
        ):
            console_log("roofline", "Skipping roofline")
        else:
            # Dynamic import to isolate hip dependency during profile time only
            from roofline.run_benchmark import load_bench

            console_log(
                "roofline", f"Checking for roofline.csv in {self.get_args().path}"
            )
            if not (Path(self.get_args().path) / "roofline.csv").is_file():
                try:
                    bench = load_bench([self.get_args().device])
                    result = bench.run_on_devices([self.get_args().device])
                    bench.dump_csv(result, f"{self.get_args().path}/roofline.csv")
                except Exception as e:
                    console_error(
                        "roofline",
                        f"Benchmark execution failed: {e}. Skipping roofline.",
                        exit=False,
                    )
                    return

            is_valid, error_msg = validate_roofline_csv(self.get_args().path)
            if not is_valid:
                console_error(
                    "roofline",
                    f"Invalid roofline.csv: {error_msg}",
                    exit=False,
                )
                return

            console_log(
                "roofline",
                f"Roofline data saved to {self.get_args().path}/roofline.csv\n"
                f"  Run 'rocprof-compute analyze -p {self.get_args().path}' "
                f"for charts",
            )


# Set with limited size
class LimitedSet:
    def __init__(self, maxsize: int) -> None:
        self.avail: int = maxsize
        self.elements: list[str] = []

    def add(self, element: str) -> bool:
        if element in self.elements:
            return True
        # Store all channels for a TCC channel counter in the same file
        if element.split("[")[0] in {elem.split("[")[0] for elem in self.elements}:
            self.elements.append(element)
            return True
        if self.avail > 0:
            self.avail -= 1
            self.elements.append(element)
            return True
        return False


# Represents a file that lists PMC counters. Number of counters for each
# block limited according to perfmon config.
class CounterFile:
    def __init__(self, name: str, perfmon_config: dict[str, int]) -> None:
        self.file_name_txt: str = f"pmc_perf_{name}.txt"
        self.pmc_filename: str = f"pmc_perf_{name}.yaml"
        self.counter_def_filename: str = f"counter_def_{name}.yaml"
        self.blocks: dict[str, LimitedSet] = {
            block: LimitedSet(capacity) for block, capacity in perfmon_config.items()
        }

    def add(self, counter: str) -> bool:
        block = counter.split("_")[0]

        # SQ, SQC, and SP belong to the same IP block
        if block == "SQC":
            block = "SQ"
        if block == "SP":
            block = "SQ"

        return self.blocks[block].add(counter)


def _flat_counters_in_perfmon_file(counter_file: CounterFile) -> list[str]:
    """Ordered list of PMC counter names assigned to one perfmon bucket file."""
    return [
        ctr
        for block_name in counter_file.blocks
        for ctr in counter_file.blocks[block_name].elements
    ]


def _counter_display_ip_prefix(counter: str) -> str:
    """First IP-style token of a PMC name (column key for dry-run tables)."""
    if is_tcc_channel_counter(counter):
        base = counter.split("[")[0]
        return base.split("_", 1)[0] if "_" in base else base
    if "_" not in counter:
        return counter
    return counter.split("_", 1)[0]


def _counters_grouped_by_ip_sorted(counters: list[str]) -> dict[str, list[str]]:
    by_ip: dict[str, list[str]] = {}
    for ctr in counters:
        ip = _counter_display_ip_prefix(ctr)
        by_ip.setdefault(ip, []).append(ctr)
    for names in by_ip.values():
        names.sort()
    return by_ip


def _dry_run_global_ip_column_widths(
    output_files: list[CounterFile],
) -> tuple[list[str], dict[str, int]]:
    """
    Column order and cell widths from the longest header or counter name
    per IP block across every bucket (aligned pipes between tables).
    """
    max_cell: dict[str, int] = {}
    for counter_file in output_files:
        flat = _flat_counters_in_perfmon_file(counter_file)
        by_ip = _counters_grouped_by_ip_sorted(flat)
        for ip, names in by_ip.items():
            longest = max((len(n) for n in names), default=0)
            max_cell[ip] = max(max_cell.get(ip, 0), longest, len(ip))
    columns = sorted(max_cell.keys())
    widths = {ip: max_cell[ip] for ip in columns}
    return columns, widths


def _format_dry_run_bucket_markdown(
    counters: list[str],
    global_columns: list[str],
    widths: dict[str, int],
) -> str:
    """GitHub-style pipe table with fixed column widths (spaces, not tabs)."""
    by_ip = _counters_grouped_by_ip_sorted(counters)
    height = max((len(by_ip.get(c, [])) for c in global_columns), default=0)

    def pipe_row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines = [
        pipe_row([ip.ljust(widths[ip]) for ip in global_columns]),
        pipe_row(["-" * widths[ip] for ip in global_columns]),
    ]
    for row_idx in range(height):
        lines.append(
            pipe_row([
                (by_ip[c][row_idx] if row_idx < len(by_ip.get(c, [])) else "").ljust(
                    widths[c]
                )
                for c in global_columns
            ])
        )
    return "\n".join(lines)


def _dry_run_counter_to_bucket_map(
    output_files: list[CounterFile],
) -> dict[str, str]:
    """Map each PMC counter string to its perfmon bucket label (e.g. pmc_perf_0)."""
    result: dict[str, str] = {}
    for counter_file in output_files:
        label = counter_file.file_name_txt.replace(".txt", "")
        for ctr in _flat_counters_in_perfmon_file(counter_file):
            result[ctr] = label
    return result


def _dry_run_print_markdown_metric_table(
    headers: list[str],
    str_rows: list[list[str]],
) -> None:
    """Print a left-padded pipe table for dry-run metric listings."""
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in str_rows))
        for i in range(len(headers))
    ]

    def pipe_line(parts: list[str]) -> str:
        return (
            "| "
            + " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts)))
            + " |"
        )

    print(pipe_line(headers))
    print(pipe_line(["-" * widths[i] for i in range(len(widths))]))
    for row in str_rows:
        print(pipe_line(row))
