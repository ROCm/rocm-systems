##############################################################################bl
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

import argparse
import importlib
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import yaml

import config
from argparser import omniarg_parser
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
    get_submodules,
    get_version,
    get_version_display,
    set_locale_encoding,
)


class RocProfCompute:
    def __init__(self):
        self.__args = None
        self.__profiler_mode = None
        self.__analyze_mode = None
        self.__soc_name = (
            set()
        )  # gpu name, or in case of analyze mode, all loaded gpu name(s)
        self.__soc = dict()  # set of key, value pairs. Where arch->OmniSoc() obj
        self.__version = {
            "ver": None,
            "ver_pretty": None,
        }
        self.__options = {}
        self.__supported_archs = mi_gpu_specs.get_gpu_series_dict()
        self.__mspec: MachineSpecs = None  # to be initalized in load_soc_specs()
        setup_console_handler()
        self.set_version()
        self.parse_args()
        self.__mode = self.__args.mode
        gui_value = getattr(self.__args, "gui", None)
        self.__loglevel = setup_logging_priority(
            self.__args.verbose, self.__args.quiet, self.__mode, gui_value
        )
        setattr(self.__args, "loglevel", self.__loglevel)
        set_locale_encoding()

        if self.__mode == "profile":
            self.detect_profiler()
        elif self.__mode == "analyze":
            self.detect_analyze()

        console_debug("Execution mode = %s" % self.__mode)

    def print_graphic(self):
        """Log program name as ascii art to terminal."""
        ascii_art = r"""
                                 __                                       _
 _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
| '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
| | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
|_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
               |_|                                           |_|
"""
        print(ascii_art)

    def get_mode(self):
        return self.__mode

    def set_version(self):
        vData = get_version(config.rocprof_compute_home)
        self.__version["ver"] = vData["version"]
        self.__version["ver_pretty"] = get_version_display(
            vData["version"], vData["sha"], vData["mode"]
        )
        return

    def detect_profiler(self):
        profiler_mode = detect_rocprof(self.__args)
        if str(profiler_mode).endswith("rocprof"):
            self.__profiler_mode = "rocprofv1"
        elif str(profiler_mode).endswith("rocprofv2"):
            self.__profiler_mode = "rocprofv2"
        elif str(profiler_mode).endswith("rocprofv3"):
            self.__profiler_mode = "rocprofv3"
        elif str(profiler_mode) == "rocprofiler-sdk":
            self.__profiler_mode = "rocprofiler-sdk"
        else:
            console_error(
                "Incompatible profiler: %s. Supported profilers include: %s"
                % (profiler_mode, get_submodules("rocprof_compute_profile"))
            )
        return

    def detect_analyze(self):
        if self.__args.gui:
            self.__analyze_mode = "web_ui"
        elif self.__args.tui:
            self.__analyze_mode = "tui"
        else:
            self.__analyze_mode = "cli"
        return

    @demarcate
    def load_soc_specs(self, sysinfo: dict = None):
        """Load OmniSoC instance for RocProfCompute run"""
        self.__mspec = generate_machine_specs(self.__args, sysinfo)
        if self.__args.specs:
            print(self.__mspec)
            sys.exit(0)

        arch = self.__mspec.gpu_arch

        soc_module = importlib.import_module("rocprof_compute_soc.soc_" + arch)
        soc_class = getattr(soc_module, arch + "_soc")
        self.__soc[arch] = soc_class(self.__args, self.__mspec)
        return

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Command line interface for AMD's GPU profiler, ROCm Compute Profiler",
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

        if self.__args.mode == None:
            if self.__args.specs:
                print(generate_machine_specs(self.__args))
                sys.exit(0)
            parser.print_help(sys.stderr)
            console_error(
                "rocprof-compute requires you to pass a valid mode. Detected None."
            )
        elif self.__args.mode == "profile":

            # FIXME:
            #     Might want to get host name from detected spec
            if self.__args.subpath == "node_name":
                self.__args.path = str(
                    Path(self.__args.path).joinpath(socket.gethostname())
                )
            elif self.__args.subpath == "gpu_model":
                self.__args.path = str(
                    Path(self.__args.path).joinpath(self.__mspec.gpu_model)
                )

            p = Path(self.__args.path)
            if not p.exists():
                try:
                    p.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    console_error("Directory already exists.")

        elif self.__args.mode == "analyze":
            # block all filters during spatial-multiplexing
            if self.__args.spatial_multiplexing:
                self.__args.gpu_id = None
                self.__args.gpu_kernel = None
                self.__args.gpu_dispatch_id = None
                self.__args.nodes = None

        return

    @demarcate
    def list_metrics(self):
        if not self.__args.list_metrics:
            arch = self.__mspec.gpu_arch
        else:
            arch = self.__args.list_metrics
        if arch in self.__supported_archs.keys():
            ac = schema.ArchConfig()
            ac.panel_configs = file_io.load_panel_configs(
                self.__args.config_dir.joinpath(arch)
            )
            sys_info = self.__mspec.get_class_members().iloc[0]
            parser.build_dfs(archConfigs=ac, filter_metrics=[], sys_info=sys_info)
            for key, value in ac.metric_list.items():
                prefix = ""
                if "." not in str(key):
                    prefix = ""
                elif str(key).count(".") == 1:
                    prefix = "\t"
                else:
                    prefix = "\t\t"
                print(prefix + key, "->", value)
            sys.exit(0)
        else:
            console_error("Unsupported arch")

    @demarcate
    def run_profiler(self):
        self.print_graphic()
        self.load_soc_specs()

        if self.__args.list_metrics is not None:
            self.list_metrics()
        elif self.__args.name is None:
            sys.exit("Either --list-name or --name is required")

        if self.__args.name.find("/") != -1:
            console_error("'/' not permitted in profile name")

        # Deprecation warning for hardware blocks
        if [
            name
            for name, type in self.__args.filter_blocks.items()
            if type == "hardware_block"
        ]:
            console_warning("Hardware block based filtering will be deprecated soon")

        # FIXME:
        #     Changing default path should be done at the end of arg parsing stage,
        #     unless there is a specific reason to do here.

        # Update default path
        if self.__args.path == str(Path(os.getcwd()).joinpath("workloads")):
            self.__args.path = str(
                Path(self.__args.path).joinpath(self.__args.name, self.__mspec.gpu_model)
            )

        # instantiate desired profiler
        if self.__profiler_mode == "rocprofv1":
            from rocprof_compute_profile.profiler_rocprof_v1 import rocprof_v1_profiler

            profiler = rocprof_v1_profiler(
                self.__args,
                self.__profiler_mode,
                self.__soc[self.__mspec.gpu_arch],
                self.__supported_archs,
            )
        elif self.__profiler_mode == "rocprofv2":
            from rocprof_compute_profile.profiler_rocprof_v2 import rocprof_v2_profiler

            profiler = rocprof_v2_profiler(
                self.__args,
                self.__profiler_mode,
                self.__soc[self.__mspec.gpu_arch],
                self.__supported_archs,
            )
        elif self.__profiler_mode == "rocprofv3":
            from rocprof_compute_profile.profiler_rocprof_v3 import rocprof_v3_profiler

            profiler = rocprof_v3_profiler(
                self.__args,
                self.__profiler_mode,
                self.__soc[self.__mspec.gpu_arch],
                self.__supported_archs,
            )
        elif self.__profiler_mode == "rocprofiler-sdk":
            from rocprof_compute_profile.profiler_rocprofiler_sdk import (
                rocprofiler_sdk_profiler,
            )

            profiler = rocprofiler_sdk_profiler(
                self.__args,
                self.__profiler_mode,
                self.__soc[self.__mspec.gpu_arch],
                self.__supported_archs,
            )
        else:
            console_error("Unsupported profiler")

        # -----------------------
        # run profiling workflow
        # -----------------------

        self.__soc[self.__mspec.gpu_arch].profiling_setup()
        # Write profiling configuration as yaml file
        with open(Path(self.__args.path).joinpath("profiling_config.yaml"), "w") as f:
            args_dict = vars(self.__args)
            args_dict["config_dir"] = str(args_dict["config_dir"])
            yaml.dump(args_dict, f)
        # enable file-based logging
        setup_file_handler(self.__args.loglevel, self.__args.path)

        profiler.pre_processing()
        console_debug('starting "run_profiling" and about to start rocprof\'s workload')
        time_start_prof = time.time()
        profiler.run_profiling(self.__version["ver"], config.prog)
        time_end_prof = time.time()
        console_debug(
            'finished "run_profiling" and finished rocprof\'s workload, time taken was {} m {} sec'.format(
                int((time_end_prof - time_start_prof) / 60),
                str((time_end_prof - time_start_prof) % 60),
            )
        )
        profiler.post_processing()
        time_end_post = time.time()
        console_debug(
            'time taken for "post_processing" was {} seconds'.format(
                int((time_end_post - time_end_prof) / 60),
                str((time_end_post - time_end_prof) % 60),
            )
        )
        self.__soc[self.__mspec.gpu_arch].post_profiling()

        return

    @demarcate
    def update_db(self):
        self.print_graphic()

        console_warning(
            "Database update mode is deprecated and will be removed in a future release "
            "and no fixes will be made for this mode."
        )

        from utils.db_connector import DatabaseConnector

        db_connection = DatabaseConnector(self.__args)

        # -----------------------
        # run database workflow
        # -----------------------
        db_connection.pre_processing()
        if self.__args.upload:
            db_connection.db_import()
        else:
            db_connection.db_remove()

        return

    @demarcate
    def run_analysis(self):
        self.print_graphic()

        console_log("Analysis mode = %s" % self.__analyze_mode)

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
        else:
            console_error("Unsupported analysis mode -> %s" % self.__analyze_mode)

        # -----------------------
        # run analysis workflow
        # -----------------------
        analyzer.sanitize()

        # Load required SoC(s) from input
        for d in analyzer.get_args().path:
            # FIXME
            # sys_info = pd.read_csv(Path(d[0], "sysinfo.csv"))
            sysinfo_path = (
                Path(d[0])
                if analyzer.get_args().nodes is None
                and analyzer.get_args().spatial_multiplexing is not True
                else file_io.find_1st_sub_dir(d[0])
            )
            sys_info = file_io.load_sys_info(sysinfo_path.joinpath("sysinfo.csv"))

            sys_info = sys_info.to_dict("list")
            sys_info = {key: value[0] for key, value in sys_info.items()}
            self.load_soc_specs(sys_info)

        analyzer.set_soc(self.__soc)
        analyzer.pre_processing()
        analyzer.run_analysis()

        return
