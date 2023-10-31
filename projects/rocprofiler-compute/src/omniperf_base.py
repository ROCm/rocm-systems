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
from utils.utils import demarcate, trace_logger, get_version, get_version_display, detect_rocprof
from argparser import omniarg_parser
import config
import pyfiglet

class Omniperf:
    def __init__(self):
        self.__args = None
        self.__profiler_mode = None
        self.__soc_name = None
        self.__soc = None
        self.__version = {
            "ver": None,
            "ver_pretty": None,
        }
        self.__options = {}

        self.setup_logging()
        self.set_version()
        self.parse_args()

        self.__mode = self.__args.mode

        if self.__mode == "profile":
            self.detect_profiler()
        
        # self.__analyze_mode = "cli"
        # self.__analyze_mode = "webui"
        
        logging.info("Execution mode = %s" % self.__mode)
   
    def print_graphic(self):
        """Read program name and log ascii art to terminal.
        """
        ascii_art = pyfiglet.figlet_format(config.prog.capitalize())
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

    def error(self,message):
        logging.error("")
        logging.error("[ERROR]: " + message)
        logging.error("")
        sys.exit(1)

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
        

    @demarcate
    def detect_soc(self):
        mspec = get_machine_specs(0)

        target=""
        if mspec.GPU == "gfx900":
            target = "vega10"
        elif mspec.GPU == "gfx906":
            target = "mi50"
        elif mspec.GPU == "gfx908":
            target = "mi100"
        elif mspec.GPU == "gfx90a":
            target = "mi200"
        else:
            self.error("Unsupported SoC -> %s" % mspec.GPU)
        
        self.__soc_name = target
        self.__args.target = target
        
        # instantiate underlying SoC support class
        if self.__soc_name == "vega10":
            from omniperf_soc.soc_gfx900 import gfx900_soc
            self.__soc = gfx900_soc(self.__args)
        elif self.__soc_name == "mi50":
            from omniperf_soc.soc_gfx906 import gfx906_soc
            self.__soc = gfx906_soc(self.__args)
        elif self.__soc_name == "mi100":
            from omniperf_soc.soc_gfx908 import gfx908_soc
            self.__soc = gfx908_soc(self.__args)
        elif self.__soc_name == "mi200":
            from omniperf_soc.soc_gfx90a import gfx90a_soc
            self.__soc = gfx90a_soc(self.__args)
        else:
            self.error("Unsupported SoC")
            sys.exit(1)

        logging.info("SoC = %s" % self.__soc_name)
        return

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
        omniarg_parser(parser, config.omniperf_home, self.__version)
        self.__args = parser.parse_args()

        if self.__args.mode == None:
            parser.print_help(sys.stderr)
            self.error("Omniperf requires a valid mode.")

        return

    @demarcate
    def run_profiler(self):
        self.print_graphic()
        self.detect_soc()

        # Update default path
        if self.__args.path == os.path.join(os.getcwd(), "workloads"):
            self.__args.path = os.path.join(self.__args.path, self.__args.name, self.__args.target)

        logging.info("Profiler choice = %s" % self.__profiler_mode)

        # instantiate desired profiler
        if self.__profiler_mode == "rocprofv1":
            from omniperf_profile.profiler_rocprof_v1 import rocprof_v1_profiler
            profiler = rocprof_v1_profiler(self.__args, self.__profiler_mode, self.__soc)
        elif self.__profiler_mode == "rocprofv2":
            from omniperf_profile.profiler_rocprof_v2 import rocprof_v2_profiler
            profiler = rocprof_v2_profiler(self.__args, self.__profiler_mode, self.__soc)        
        elif self.__profiler_mode == "rocscope":
            from omniperf_profile.profiler_rocscope import rocscope_profiler
            profiler = rocscope_profiler(self.__args, self.__profiler_mode, self.__soc)
        else:
            logging.error("Unsupported profiler")
            sys.exit(1)

        #-----------------------
        # run profiling workflow
        #-----------------------

        self.__soc.profiling_setup()
        profiler.pre_processing()
        profiler.run_profiling(self.__version["ver"], config.prog)
        profiler.post_processing()
        self.__soc.post_profiling()

        return

    @demarcate
    def update_DB(self):
        self.print_graphic()
        #TODO: Add a DB workflow
        return

    @demarcate
    def run_analysis(self):
        self.print_graphic()
        self.detect_soc() #NB: See comment in detect_profiler() to explain why this is here

        logging.info("Analysis mode choie = %s" % self.__analyze_mode)

        if self.__analyze_mode == "cli":
            from analyze.analysis_cli import cli_analysis
            analyzer = cli_analysis(self.__args,self.__options)
        elif self.__analyze_mode == "webui":
            from analyze.analysis_webui import webui_analysis
            analyzer = webui_analysis(self.__args,self.__options)
        else:
            self.error("Unsupported anlaysis mode -> %s" % self.__analyze_mode)

        #-----------------------
        # run analysis workflow
        #-----------------------

        self.__soc.analysis_setup()
        analyzer.pre_processing()
        analyzer.run_analysis()

        return
