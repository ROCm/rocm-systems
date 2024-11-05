##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
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
import logging
import sys
import os
from pathlib import Path
import shutil
from utils.specs import MachineSpecs, generate_machine_specs
from utils.utils import (
    demarcate,
    get_version,
    get_version_display,
    detect_rocprof,
    get_submodules,
    console_debug,
    console_log,
    console_error,
    set_locale_encoding,
)
from utils.logger import (
    setup_console_handler,
    setup_logging_priority,
    setup_file_handler,
)
from argparser import omniarg_parser
import config
import pandas as pd
import importlib

SUPPORTED_ARCHS = {
    "gfx906": {"mi50": ["MI50", "MI60"]},
    "gfx908": {"mi100": ["MI100"]},
    "gfx90a": {"mi200": ["MI210", "MI250", "MI250X"]},
    "gfx940": {"mi300": ["MI300A_A0"]},
    "gfx941": {"mi300": ["MI300X_A0"]},
    "gfx942": {"mi300": ["MI300A_A1", "MI300X_A1"]},
}

MI300_CHIP_IDS = {
    "29856": "MI300A_A1",
    "29857": "MI300X_A1",
    "29858": "MI308X",
}


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
        self.__supported_archs = SUPPORTED_ARCHS
        self.__mspec: MachineSpecs = None  # to be initalized in load_soc_specs()
        setup_console_handler()
        self.set_version()
        self.parse_args()
        self.__mode = self.__args.mode
        self.__loglevel = setup_logging_priority(
            self.__args.verbose, self.__args.quiet, self.__mode
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
        if (
            self.__args.lucky == True
            or self.__args.summaries == True
            or self.__args.use_rocscope
        ):
            if not shutil.which("rocscope"):
                console_error("Rocscope must be in PATH")
            else:
                self.__profiler_mode = "rocscope"
        else:
            rocprof_cmd = detect_rocprof()
            if str(rocprof_cmd).endswith("rocprof"):
                self.__profiler_mode = "rocprofv1"
            elif str(rocprof_cmd).endswith("rocprofv2"):
                self.__profiler_mode = "rocprofv2"
            else:
                console_error(
                    "Incompatible profiler: %s. Supported profilers include: %s"
                    % (rocprof_cmd, get_submodules("rocprof_compute_profile"))
                )
        return

    def detect_analyze(self):
        if self.__args.gui:
            self.__analyze_mode = "web_ui"
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

        # NB: This checker is a bit redundent. We already check this in specs module
        if arch not in self.__supported_archs.keys():
            console_error("%s is an unsupported SoC" % arch)

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
                "rocprof-compute requires you pass a valid mode. Detected None."
            )
        return

    @demarcate
    def run_profiler(self):
        self.print_graphic()
        self.load_soc_specs()

        # Update default path
        if self.__args.path == os.path.join(os.getcwd(), "workloads"):
            self.__args.path = os.path.join(
                self.__args.path, self.__args.name, self.__mspec.gpu_model
            )

        # instantiate desired profiler
        if self.__profiler_mode == "rocprofv1":
            from rocprof_compute_profile.profiler_rocprof_v1 import rocprof_v1_profiler

            profiler = rocprof_v1_profiler(
                self.__args, self.__profiler_mode, self.__soc[self.__mspec.gpu_arch]
            )
        elif self.__profiler_mode == "rocprofv2":
            from rocprof_compute_profile.profiler_rocprof_v2 import rocprof_v2_profiler

            profiler = rocprof_v2_profiler(
                self.__args, self.__profiler_mode, self.__soc[self.__mspec.gpu_arch]
            )
        elif self.__profiler_mode == "rocscope":
            from rocprof_compute_profile.profiler_rocscope import rocscope_profiler

            profiler = rocscope_profiler(
                self.__args, self.__profiler_mode, self.__soc[self.__mspec.gpu_arch]
            )
        else:
            console_error("Unsupported profiler")

        # -----------------------
        # run profiling workflow
        # -----------------------

        self.__soc[self.__mspec.gpu_arch].profiling_setup()
        # enable file-based logging
        setup_file_handler(self.__args.loglevel, self.__args.path)

        profiler.pre_processing()
        profiler.run_profiling(self.__version["ver"], config.prog)
        profiler.post_processing()
        self.__soc[self.__mspec.gpu_arch].post_profiling()

        return

    @demarcate
    def update_db(self):
        self.print_graphic()
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
        else:
            console_error("Unsupported analysis mode -> %s" % self.__analyze_mode)

        # -----------------------
        # run analysis workflow
        # -----------------------
        analyzer.sanitize()

        # Load required SoC(s) from input
        for d in analyzer.get_args().path:
            sys_info = pd.read_csv(Path(d[0], "sysinfo.csv"))
            sys_info = sys_info.to_dict("list")
            sys_info = {key: value[0] for key, value in sys_info.items()}
            self.load_soc_specs(sys_info)

        analyzer.set_soc(self.__soc)
        analyzer.pre_processing()
        analyzer.run_analysis()

        return
