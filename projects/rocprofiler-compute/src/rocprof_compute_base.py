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
import importlib
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import config
from argparser import omniarg_parser
from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils import file_io, parser, schema
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
    setup_console_handler,
    setup_file_handler,
    setup_logging_priority,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs, generate_machine_specs
from utils.utils import (
    detect_rocprof,
    get_panel_alias,
    get_submodules,
    get_version,
    get_version_display,
    parse_sets_yaml,
    set_locale_encoding,
)


class RocProfCompute:
    def __init__(self) -> None:
        self.__args: Optional[argparse.Namespace] = None
        self.__profiler_mode = None
        self.__analyze_mode = None
        self.__soc: dict[str, OmniSoC_Base] = {}
        self.__version: dict[str, Optional[str]] = {"ver": None, "ver_pretty": None}
        self.__supported_archs = mi_gpu_specs.get_gpu_series_dict()
        self.__mspec: MachineSpecs  # to be initialized in load_soc_specs()

        setup_console_handler()
        self.set_version()
        self.parse_args()
        assert self.__args is not None
        self.__mode = self.__args.mode

        gui_value = getattr(self.__args, "gui", None)
        self.__loglevel = setup_logging_priority(
            self.__args.verbose, self.__args.quiet, self.__mode, gui_value
        )
        setattr(self.__args, "loglevel", self.__loglevel)
        set_locale_encoding()

        self.sanitize()

        if self.__mode == "profile":
            self.detect_profiler()
        elif self.__mode == "analyze":
            self.detect_analyze()

        console_debug(f"Execution mode = {self.__mode}")

    def print_graphic(self) -> None:
        print(
            r"""
                                 __                                       _
 _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
| '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
| | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
|_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
               |_|                                           |_|
"""
        )

    def get_mode(self) -> Optional[str]:
        return self.__mode

    def set_version(self) -> None:
        vData = get_version(config.rocprof_compute_home)
        self.__version["ver"] = vData["version"]
        self.__version["ver_pretty"] = get_version_display(
            vData["version"], vData["sha"], vData["mode"]
        )

    def detect_profiler(self) -> None:
        profiler_mode = detect_rocprof(self.__args)
        if str(profiler_mode).endswith("rocprofv3"):
            self.__profiler_mode = "rocprofv3"
        elif str(profiler_mode) == "rocprofiler-sdk":
            self.__profiler_mode = "rocprofiler-sdk"
        else:
            console_error(
                f"Incompatible profiler: {profiler_mode}. Supported profilers "
                f"include: {get_submodules('rocprof_compute_profile')}"
            )

    def detect_analyze(self) -> None:
        if self.__args.gui:
            self.__analyze_mode = "web_ui"
        elif self.__args.tui:
            self.__analyze_mode = "tui"
        elif self.__args.output_format == "db":
            self.__analyze_mode = "db"
        else:
            self.__analyze_mode = "cli"

    def sanitize(self) -> None:
        block = False
        if (hasattr(self.__args, "filter_metrics") and self.__args.filter_metrics) or (
            hasattr(self.__args, "filter_blocks") and self.__args.filter_blocks
        ):
            block = True

        if self.__args.list_metrics is not None and block:
            console_error("Cannot use --list-metrics with --blocks")
        if self.__args.list_blocks is not None and block:
            console_error("Cannot use --list-blocks with --blocks")
        if (
            hasattr(self.__args, "list_available_metrics")
            and self.__args.list_available_metrics
        ) and block:
            console_error("Cannot use --list-available-metrics with --blocks")

    @demarcate
    def load_soc_specs(self, sysinfo: Optional[dict] = None) -> None:
        """Load OmniSoC instance for RocProfCompute run"""
        self.__mspec = generate_machine_specs(self.__args, sysinfo)
        if self.__args and self.__args.specs:
            print(self.__mspec)
            sys.exit(0)

        arch = self.__mspec.gpu_arch
        soc_module = importlib.import_module(f"rocprof_compute_soc.soc_{arch}")
        soc_class = getattr(soc_module, f"{arch}_soc")
        self.__soc[arch] = soc_class(self.__args, self.__mspec)

    def parse_args(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Command line interface for AMD's GPU profiler, ROCm Compute Profiler"
            ),
            prog="tool",
            formatter_class=lambda prog: argparse.RawTextHelpFormatter(
                prog, max_help_position=30
            ),
            usage="rocprof-compute [mode] [options]",
        )
        omniarg_parser(
            parser, config.rocprof_compute_home, self.__supported_archs, self.__version
        )
        self.__args = parser.parse_args()

        if (
            hasattr(self.__args, "format_rocprof_output")
            and self.__args.format_rocprof_output != "rocpd"
        ):
            console_warning(
                f"The option --format-rocprof-output currently set to "
                f"{self.__args.format_rocprof_output} will default to rocpd "
                "in a future release."
            )

        if self.__args.mode is None:
            if self.__args.specs:
                print(generate_machine_specs(self.__args))
                sys.exit(0)
            elif self.__args.list_metrics is not None:
                self.list_metrics()
                sys.exit(0)
            elif self.__args.list_blocks is not None:
                self.list_blocks()
                sys.exit(0)
            elif self.__args.config_dir:
                parser.print_help(sys.stderr)
                console_error(
                    "rocprof-compute requires you to pass --list-metrics "
                    "with --config-dir."
                )
            parser.print_help(sys.stderr)
            console_error(
                "rocprof-compute requires you to pass a valid mode. Detected None."
            )
        elif self.__args.mode == "profile":
            self.handle_profile_args()
        elif self.__args.mode == "analyze":
            self.handle_analyze_args()

    def handle_profile_args(self) -> None:
        # Handle list operations first - these are independent and exit immediately
        if getattr(self.__args, "list_sets", False):
            return
        if getattr(self.__args, "list_available_metrics", False):
            return

    def handle_analyze_args(self) -> None:
        """Handle analyze-specific argument processing"""
        # Block all filters during spatial-multiplexing
        if self.__args.spatial_multiplexing:
            self.__args.gpu_id = None
            self.__args.gpu_kernel = None
            self.__args.gpu_dispatch_id = None
            self.__args.nodes = None

    @demarcate
    def list_metrics(self) -> None:
        for_current_arch = getattr(self.__args, "list_available_metrics", False)

        arch = (
            self.__mspec.gpu_arch
            if (for_current_arch or self.__args.list_metrics is None)
            else self.__args.list_metrics
        )
        if arch in self.__supported_archs.keys():
            ac = schema.ArchConfig()
            ac.panel_configs = file_io.load_panel_configs([
                str(Path(self.__args.config_dir) / arch)
            ])
            sys_info = (
                self.__mspec.get_class_members().iloc[0] if for_current_arch else None
            )
            parser.build_dfs(arch_configs=ac, filter_metrics=[], sys_info=sys_info)
            for key, value in ac.metric_list.items():
                prefix = "\t" * min(key.count("."), 2)
                print(f"{prefix}{key} -> {value}")
            sys.exit(0)
        else:
            console_error("Unsupported arch")

    @demarcate
    def list_blocks(self) -> None:
        for_current_arch = getattr(self.__args, "list_available_metrics", False)

        arch = (
            self.__mspec.gpu_arch
            if (for_current_arch or self.__args.list_blocks is None)
            else self.__args.list_blocks
        )
        if arch in self.__supported_archs.keys():
            ac = schema.ArchConfig()
            ac.panel_configs = file_io.load_panel_configs([
                str(Path(self.__args.config_dir) / arch)
            ])
            sys_info = (
                self.__mspec.get_class_members().iloc[0] if for_current_arch else None
            )
            parser.build_dfs(arch_configs=ac, filter_metrics=[], sys_info=sys_info)

            print(f"{'INDEX':<8} {'BLOCK ALIAS':<16} {'BLOCK NAME'}")
            for key, value in ac.metric_list.items():
                if key.count(".") > 0:
                    continue
                print(f"{key:<8} {get_panel_alias()[value]:<16} {value}")
            sys.exit(0)
        else:
            console_error("Unsupported arch")

    @demarcate
    def list_sets(self) -> None:
        sets_info = parse_sets_yaml(self.__mspec.gpu_arch)

        if not sets_info:
            console_error("No sets configuration found.")

        print("\nAvailable Sets:")
        print("=" * 115)

        # Print header
        print(
            f"{'Set Option':<35} {'Set Title':<35}"
            f" {'Metric Name':<30} {'Metric ID':<10}"
        )
        print("-" * 115)

        # Print data grouped by set
        for set_option, set_data in sets_info.items():
            title = set_data.get("title", set_option)
            metrics = set_data.get("metric", [])

            first_row = True
            for metric in metrics:
                if isinstance(metric, dict) and metric:
                    metric_id = next(iter(metric.keys()))
                    metric_name = next(iter(metric.values()))

                    # Only show set info on first row of each set
                    set_display = set_option if first_row else ""
                    title_display = title if first_row else ""

                    print(
                        f"{set_display:<35} {title_display:<35}"
                        f" {metric_name:<30} {metric_id:<10}"
                    )
                    first_row = False
            # Empty line between sets
            print()

        print("Usage Examples:")
        if sets_info:
            first_set = next(iter(sets_info.keys()))
            print(f"  rocprof-compute profile --set {first_set}  # Profile this set")
        print("  rocprof-compute profile --list-sets        # Show this help")
        print()

        sys.exit(0)

        profiler_classes = {
            "rocprofv3": (
                "rocprof_compute_profile.profiler_rocprof_v3",
                "rocprof_v3_profiler",
            ),
            "rocprofiler-sdk": (
                "rocprof_compute_profile.profiler_rocprofiler_sdk",
                "rocprofiler_sdk_profiler",
            ),
        }

        if self.__profiler_mode not in profiler_classes:
            console_error("Unsupported profiler")

        module_name, class_name = profiler_classes[self.__profiler_mode]
        module = importlib.import_module(module_name)
        profiler_class = getattr(module, class_name)

        return profiler_class(
            self.__args,
            self.__profiler_mode,
            self.__soc[self.__mspec.gpu_arch],
        )

    def create_profiler(self) -> object:
        profiler_classes = {
            "rocprofv3": (
                "rocprof_compute_profile.profiler_rocprof_v3",
                "rocprof_v3_profiler",
            ),
            "rocprofiler-sdk": (
                "rocprof_compute_profile.profiler_rocprofiler_sdk",
                "rocprofiler_sdk_profiler",
            ),
        }

        if self.__profiler_mode not in profiler_classes:
            console_error("Unsupported profiler")

        module_name, class_name = profiler_classes[self.__profiler_mode]
        module = importlib.import_module(module_name)
        profiler_class = getattr(module, class_name)

        return profiler_class(
            self.__args,
            self.__profiler_mode,
            self.__soc[self.__mspec.gpu_arch],
        )

    @demarcate
    def run_profiler(self) -> None:
        self.print_graphic()
        self.load_soc_specs()

        if self.__args.list_metrics is not None or self.__args.list_available_metrics:
            self.list_metrics()
        elif self.__args.list_sets:
            self.list_sets()
        elif self.__args.name is None:
            sys.exit("Either --list-name or --name is required")

        if "/" in self.__args.name:
            console_error('"/" is not permitted in profile name')

        # instantiate desired profiler
        profiler = self.create_profiler()
        profiler.sanitize()

        # Add --name to workload path if --path is not given
        if self.__args.path == str(Path.cwd() / "workloads"):
            if not hasattr(self.__args, "name") or not self.__args.name:
                console_error("-n/--name is required")
            self.__args.path = str(Path(self.__args.path) / self.__args.name)
            # Add node name to workload path
            if self.__args.subpath == "node_name":
                self.__args.path = str(Path(self.__args.path) / socket.gethostname())
            # OR, Add gpu model name to workload path
            elif self.__args.subpath == "gpu_model":
                self.__args.path = str(Path(self.__args.path) / self.__mspec.gpu_model)

        # Create workload directory if it does not exist
        p = Path(self.__args.path)
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                console_error("Directory already exists.")

        # enable file-based logging
        setup_file_handler(self.__args.loglevel, self.__args.path)

        profiler.pre_processing()
        console_debug('starting "run_profiling" and about to start rocprof\'s workload')

        time_start_prof = time.time()
        profiler.run_profiling(self.__version["ver"], config.PROJECT_NAME)
        time_end_prof = time.time()

        prof_duration = time_end_prof - time_start_prof
        console_debug(
            f'finished "run_profiling" and finished rocprof\'s workload, '
            f"time taken was {int(prof_duration / 60)} m {prof_duration % 60} sec"
        )

        profiler.post_processing()
        time_end_post = time.time()

        post_duration = int(time_end_post - time_end_prof)
        console_debug(f'time taken for "post_processing" was {post_duration} seconds')
        self.__soc[self.__mspec.gpu_arch].post_profiling()

    @demarcate
    def run_analysis(self) -> None:
        self.print_graphic()
        console_log(f"Analysis mode = {self.__analyze_mode}")

        if self.__analyze_mode == "cli":
            from rocprof_compute_analyze.analysis_cli import cli_analysis

            analyzer = cli_analysis(self.__args, self.__supported_archs)
        elif self.__analyze_mode == "web_ui":
            from rocprof_compute_analyze.analysis_webui import webui_analysis

            analyzer = webui_analysis(self.__args, self.__supported_archs)
        elif self.__analyze_mode == "tui":
            from rocprof_compute_tui.tui_app import run_tui

            run_tui(self.__args, self.__supported_archs)
            return
        elif self.__analyze_mode == "db":
            from rocprof_compute_analyze.analysis_db import db_analysis

            analyzer = db_analysis(self.__args, self.__supported_archs)
        else:
            console_error(f"Unsupported analysis mode -> {self.__analyze_mode}")

        # -----------------------
        # run analysis workflow
        # -----------------------
        analyzer.sanitize()

        # Load required SoC(s) from input
        for path_list in analyzer.get_args().path:
            base_path = path_list[0] if isinstance(path_list, list) else path_list

            # Determine sysinfo path
            if (
                analyzer.get_args().nodes is None
                and not analyzer.get_args().spatial_multiplexing
            ):
                sysinfo_path = base_path
            else:
                sysinfo_path = file_io.find_1st_sub_dir(base_path)

            sys_info = file_io.load_sys_info(f"{sysinfo_path}/sysinfo.csv")
            sys_info_dict = {
                key: value[0] for key, value in sys_info.to_dict("list").items()
            }
            self.load_soc_specs(sys_info_dict)

        analyzer.set_soc(self.__soc)
        analyzer.pre_processing()
        analyzer.run_analysis()
