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
import math
import os
import re
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

import yaml

import config
from roofline import Roofline
from utils.amdsmi_interface import amdsmi_ctx, get_gpu_model, get_mem_max_clock
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.parser import BUILD_IN_VARS, SUPPORTED_DENOM
from utils.specs import MachineSpecs
from utils.utils import (
    METRIC_ID_RE,
    add_counter_extra_config_input_yaml,
    convert_metric_id_to_panel_info,
    detect_rocprof,
    get_panel_alias,
    get_submodules,
    is_tcc_channel_counter,
    mibench,
    parse_sets_yaml,
)


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
        # Create roofline object if mode is provided; skip for --specs
        if hasattr(self.__args, "mode") and self.__args.mode:
            self.roofline_obj = Roofline(args, self._mspec)

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
            # Select all sections by default
            for filename in config_filename_dict.values():
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

        # TCP_TCP_LATENCY_sum not supported for MI300 (gfx940, gfx941, gfx942)
        if self.__arch in ("gfx940", "gfx941", "gfx942"):
            counters = counters - {"TCP_TCP_LATENCY_sum"}

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
        rocprof_cmd = detect_rocprof(args)

        if rocprof_cmd != "rocprofiler-sdk":
            console_warning(
                "rocprofv3 interface is deprecated and will be removed "
                "in a future release."
            )

        rocprof_counters: set[str] = set()

        if not (
            str(rocprof_cmd).endswith("rocprofv3")
            or str(rocprof_cmd) == "rocprofiler-sdk"
        ):
            console_error(
                f"Incompatible profiler: {rocprof_cmd}. "
                "Supported profilers include: "
                f"{get_submodules('rocprof_compute_profile')}"
            )

            # Point to counter definition
            old_rocprofiler_metrics_path = os.environ.get("ROCPROFILER_METRICS_PATH")
            os.environ["ROCPROFILER_METRICS_PATH"] = str(
                config.rocprof_compute_home / "rocprof_compute_soc" / "profile_configs"
            )
            sys.path.append(
                str(
                    Path(self.get_args().rocprofiler_sdk_library_path).parent
                    / "python3/site-packages"
                )
            )
            from rocprofv3 import avail

            avail.loadLibrary.libname = str(
                Path(args.rocprofiler_sdk_library_path).parent
                / "rocprofiler-sdk"
                / "librocprofv3-list-avail.so"
            )
            counters = avail.get_counters()
            rocprof_counters = {
                counter.name
                for counter in counters[list(counters.keys())[0]]
                if hasattr(counter, "block") or hasattr(counter, "expression")
            }
            # Reset env. var.
            if old_rocprofiler_metrics_path is None:
                del os.environ["ROCPROFILER_METRICS_PATH"]
            else:
                os.environ["ROCPROFILER_METRICS_PATH"] = old_rocprofiler_metrics_path

        return rocprof_counters

    @demarcate
    def perfmon_coalesce(self, counters: set[str]) -> None:
        """
        Sort and bucket all related performance counters to minimize required
        application passes
        """
        workload_perfmon_dir = Path(self.get_args().path) / "perfmon"
        workload_perfmon_dir.mkdir(parents=True, exist_ok=True)

        # Sanity check whether counters are supported by underlying rocprof tool
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

        output_files: list[CounterFile] = []
        accu_file_count = 0

        # Create separate perfmon file for LEVEL counters without _sum suffix
        # TCC LEVEL counters are handled channel wise, so ignore them
        # Convert set to sorted list for determinism in pmc txt files
        counters = sorted(list(counters))
        for counter in counters.copy():
            if (
                "LEVEL" in counter
                and not counter.endswith("_sum")
                and not is_tcc_channel_counter(counter)
            ):
                counters.remove(counter)
                output_files.append(
                    CounterFile(counter + ".txt", self.__perfmon_config)
                )
                output_files[-1].add(counter)
                output_files[-1].add(f"{counter}_ACCUM")
                accu_file_count += 1

        file_count = 0
        # Store all channels for a TCC channel counter in the same file
        tcc_channel_counter_file_map: dict[str, CounterFile] = {}

        for ctr in counters:
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
                output_files.append(
                    CounterFile(f"pmc_perf_{file_count}.txt", self.__perfmon_config)
                )
                file_count += 1
                output_files[-1].add(ctr)

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
                    Path(workload_perfmon_dir) / f"pmc_perf_node_{node_idx}_{f_idx}.txt"
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
                    fd.write(f"pmc: {' '.join(pmc)}\n\n")
        else:
            # Output to files
            for f in output_files:
                file_name_txt = workload_perfmon_dir / f.file_name_txt
                file_name_yaml = workload_perfmon_dir / f.file_name_yaml

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
                with open(file_name_txt, "w") as fd:
                    fd.write(f"pmc: {' '.join(pmc)}\n\n")
                    fd.write("gpu:\n")
                    fd.write("range:\n")
                    fd.write("kernel:\n")

                # Write counter definitions to file
                if counter_def:
                    with open(file_name_yaml, "w") as fp:
                        fp.write(yaml.dump(counter_def, sort_keys=False))

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
            pmc_path = Path(self.get_args().path) / "pmc_perf.csv"
            if not pmc_path.is_file():
                console_warning(
                    "Incomplete or missing profiling data. Skipping roofline."
                )
                return
            console_log(
                "roofline", f"Checking for roofline.csv in {self.get_args().path}"
            )
            if not (Path(self.get_args().path) / "roofline.csv").is_file():
                mibench(self.get_args(), self._mspec)
            self.roofline_obj.post_processing()

    @abstractmethod
    def analysis_setup(self, roofline_parameters: Optional[dict[str, Any]]) -> None:
        """Perform any SoC-specific setup prior to analysis."""
        console_debug("analysis", f"perform SoC analysis setup for {self.__arch}")
        if roofline_parameters:
            self.roofline_obj = Roofline(
                self.get_args(), self._mspec, roofline_parameters
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
        name_no_extension = name.split(".")[0]
        self.file_name_txt: str = name_no_extension + ".txt"
        self.file_name_yaml: str = name_no_extension + ".yaml"
        self.blocks: dict[str, LimitedSet] = {
            block: LimitedSet(capacity) for block, capacity in perfmon_config.items()
        }

    def add(self, counter: str) -> bool:
        block = counter.split("_")[0]

        # SQ and SQC belong to the same IP block
        if block == "SQC":
            block = "SQ"

        return self.blocks[block].add(counter)
