##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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
from utils.specs import get_machine_specs
from utils.utils import demarcate, trace_logger, get_version, get_version_display, detect_rocprof, error
from argparser import omniarg_parser
import config
import pandas as pd
import importlib

class Omniperf:
    def __init__(self):
        self.__args = None
        self.__profiler_mode = None
        self.__analyze_mode = None
        self.__soc_name = set() #TODO: Should we make this a list? To accommodate analyze mode 
        self.__soc = dict() # set of key, value pairs. Where arch->OmniSoc() obj
        self.__version = {
            "ver": None,
            "ver_pretty": None,
        }
        self.__options = {}
        self.__supported_archs = {
            "gfx906": {"mi50": ["MI50", "MI60"]},
            "gfx908": {"mi100": ["MI100"]},
            "gfx90a": {"mi200": ["MI210", "MI250", "MI250X"]},
        }

        self.setup_logging()
        self.set_version()
        self.parse_args()

        self.__mode = self.__args.mode

        if self.__mode == "profile":
            self.detect_profiler()
        elif self.__mode == "analyze":
            self.detect_analyze()
        
        logging.info("Execution mode = %s" % self.__mode)
   
    def print_graphic(self):
        """Log program name as ascii art to terminal.
        """
        ascii_art = '''
  ___                  _                  __ 
 / _ \ _ __ ___  _ __ (_)_ __   ___ _ __ / _|
| | | | '_ ` _ \| '_ \| | '_ \ / _ \ '__| |_ 
| |_| | | | | | | | | | | |_) |  __/ |  |  _|
 \___/|_| |_| |_|_| |_|_| .__/ \___|_|  |_|  
                        |_|                  
'''
        logging.info(ascii_art)
    
    def setup_logging(self):
        # register a trace level logger
        logging.TRACE = logging.DEBUG - 5
        logging.addLevelName(logging.TRACE, "TRACE")
        setattr(logging, "TRACE", logging.TRACE)
        setattr(logging, "trace", trace_logger)

        # demonstrate override of default loglevel via env variable
        loglevel=logging.INFO
        if "OMNIPERF_LOGLEVEL" in os.environ.keys():
            loglevel = os.environ['OMNIPERF_LOGLEVEL']
            if loglevel in {"DEBUG","debug"}:
                loglevel = logging.DEBUG
            elif loglevel in {"TRACE","trace"}:
                loglevel = logging.TRACE
            elif loglevel in {"INFO","info"}:
                loglevel = logging.INFO
            elif loglevel in {"ERROR","error"}:
                loglevel = logging.ERROR
            else:
                print("Ignoring unsupported OMNIPERF_LOGLEVEL setting (%s)" % loglevel)
                sys.exit(1)

        logging.basicConfig(format="%(message)s", level=loglevel, stream=sys.stdout)

    def get_mode(self):
        return self.__mode
    
    def set_version(self):
        vData = get_version(config.omniperf_home)
        self.__version["ver"] = vData["version"]
        self.__version["ver_pretty"] = get_version_display(vData["version"], vData["sha"], vData["mode"])
        return

    def detect_profiler(self):
        #TODO:
        # Currently this will only be called in profile mode
        # could we also utilize this function to detect "profiler origin" in analyze mode,
        # or is there a better place to do this?
        if self.__args.lucky == True or self.__args.summaries == True or self.__args.use_rocscope:
            if not shutil.which("rocscope"):
                logging.error("Rocscope must be in PATH")
                sys.exit(1)
            else:
                self.__profiler_mode = "rocscope"
        else:
            #TODO: Add detection logic for rocprofv2
            rocprof_cmd = detect_rocprof()
            self.__profiler_mode = "rocprofv1"
        return
    def detect_analyze(self):
        if self.__args.gui:
            self.__analyze_mode = "web_ui"
        else:
            self.__analyze_mode = "cli"
        return

    @demarcate
    def detect_soc(self, arch=None):
        """Load OmniSoC instance for Omniperf run
        """
        # in case of analyze mode, we can explicitly specify an arch
        # rather than detect from rocminfo
        if not arch:
            mspec = get_machine_specs(0)
            arch = mspec.arch

        # instantiate underlying SoC support class
        # in case of analyze mode, __soc can accommodate multiple archs
        if arch not in self.__supported_archs.keys():
            logging.error("Unsupported SoC")
            sys.exit(1)
        else:
            target = list(self.__supported_archs[arch].keys())[0]
            self.__soc_name.add(target)
            if hasattr(self.__args, 'target'):
                self.__args.target = target

            soc_module = importlib.import_module('omniperf_soc.soc_'+arch)
            soc_class = getattr(soc_module, arch+'_soc')
            self.__soc[arch] = soc_class(self.__args)

        logging.info("SoC = %s" % self.__soc_name)
        return arch

    @demarcate
    def parse_args(self):
        parser = argparse.ArgumentParser(
                description="Command line interface for AMD's GPU profiler, Omniperf",
                prog="tool",
                formatter_class=lambda prog: argparse.RawTextHelpFormatter(
                prog, max_help_position=30
            ),
            usage="omniperf [mode] [options]",
        )
        omniarg_parser(parser, config.omniperf_home, self.__supported_archs ,self.__version)
        self.__args = parser.parse_args()

        if self.__args.mode == None:
            parser.print_help(sys.stderr)
            error("Omniperf requires a valid mode.")

        return

    @demarcate
    def run_profiler(self):
        self.print_graphic()
        targ_arch = self.detect_soc()

        # Update default path
        if self.__args.path == os.path.join(os.getcwd(), "workloads"):
            self.__args.path = os.path.join(self.__args.path, self.__args.name, self.__args.target)

        logging.info("Profiler choice = %s" % self.__profiler_mode)

        # instantiate desired profiler
        if self.__profiler_mode == "rocprofv1":
            from omniperf_profile.profiler_rocprof_v1 import rocprof_v1_profiler
            profiler = rocprof_v1_profiler(self.__args, self.__profiler_mode, self.__soc[targ_arch])
        elif self.__profiler_mode == "rocprofv2":
            from omniperf_profile.profiler_rocprof_v2 import rocprof_v2_profiler
            profiler = rocprof_v2_profiler(self.__args, self.__profiler_mode, self.__soc[targ_arch])        
        elif self.__profiler_mode == "rocscope":
            from omniperf_profile.profiler_rocscope import rocscope_profiler
            profiler = rocscope_profiler(self.__args, self.__profiler_mode, self.__soc[targ_arch])
        else:
            logging.error("Unsupported profiler")
            sys.exit(1)

        #-----------------------
        # run profiling workflow
        #-----------------------
        self.__soc[targ_arch].profiling_setup()
        profiler.pre_processing()
        profiler.run_profiling(self.__version["ver"], config.prog)
        profiler.post_processing()
        self.__soc[targ_arch].post_profiling()

        return

    @demarcate
    def update_DB(self):
        self.print_graphic()
        #TODO: Add a DB workflow
        return

    @demarcate
    def run_analysis(self):
        self.print_graphic()

        logging.info("Analysis mode = %s" % self.__analyze_mode)

        if self.__analyze_mode == "cli":
            from omniperf_analyze.analysis_cli import cli_analysis
            analyzer = cli_analysis(self.__args, self.__supported_archs)
        elif self.__analyze_mode == "web_ui":
            from omniperf_analyze.analysis_webui import webui_analysis
            analyzer = webui_analysis(self.__args, self.__supported_archs)
        else:
            error("Unsupported anlaysis mode -> %s" % self.__analyze_mode)

        #-----------------------
        # run analysis workflow
        #-----------------------

        analyzer.sanitize()
        # Load required SoC(s) from input
        for d in analyzer.get_args().path:
            sys_info = pd.read_csv(Path(d[0], "sysinfo.csv"))
            arch = sys_info.iloc[0]["gpu_soc"]
            # Create and load new SoC object
            self.detect_soc(arch)

        analyzer.set_soc(self.__soc)
        analyzer.pre_processing()
        analyzer.run_analysis()

        return

