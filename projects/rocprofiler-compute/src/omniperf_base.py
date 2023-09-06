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
from utils.utils import demarcate, trace_logger


class Omniperf:
    def __init__(self):
        self.__args = None
        self.__profiler_mode = None
        self.__soc_name = None
        self.__soc = None
        self.__options = {}

        self.setup_logging()
        self.parseArgs()

        # hard-coding dummy examples (comment/uncomment below to instantiate different implementations)
        self.__profiler_mode = "rocprof_v1"
        self.__profiler_mode = "rocprof_v2"
        #self.__profiler_mode = "rocscope"

        self.__analyze_mode = "cli"
        self.__analyze_mode = "webui"

        # hard-code execution mode
        self.__mode = "profile"
        #self.__mode = "analyze"

        logging.info("Execution mode = %s" % self.__mode)
   

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

    def error(self,message):
        logging.error("")
        logging.error("[ERROR]: " + message)
        logging.error("")
        sys.exit(1)

    @demarcate
    def detectSoC(self):
        # hard-coded example
        self.__soc_name = "gfx906"
        self.__soc_name = "gfx908"
        self.__soc_name = "gfx90a"
        
        # instantiate underlying SoC support class
        if self.__soc_name == "gfx906":
            from soc_gfx906 import gfx906_soc
            self.__soc = gfx906_soc(self.__args)
        elif self.__soc_name == "gfx908":
            from soc_gfx908 import gfx908_soc
            self.__soc = gfx908_soc(self.__args)
        elif self.__soc_name == "gfx90a":
            from soc_gfx90a import gfx90a_soc
            self.__soc = gfx90a_soc(self.__args)
        else:
            logging.error("Unsupported SoC")
            sys.exit(1)

        logging.info("SoC = %s" % self.__soc_name)
        return

    def parseArgs(self):
        self.__args = "some arguments"
        return

    @demarcate
    def run_profiler(self):
        self.detectSoC()

        logging.info("Profiler choice = %s" % self.__profiler_mode)

        # instantiate desired profiler
        if self.__profiler_mode == "rocprof_v1":
            from profiler_rocprof import rocprof_v1_profiler
            profiler = rocprof_v1_profiler(self.__soc, self.__args)
        elif self.__profiler_mode == "rocprof_v2":
            from profiler_rocprof_v2 import rocprof_v2_profiler
            profiler = rocprof_v2_profiler(self.__soc, self.__args)        
        elif self.__profiler_mode == "rocscope":
            from profiler_rocscope import rocscope_profiler
            profiler = rocscope_profiler(self.__soc, self.__args)
        else:
            logging.error("Unsupported profiler")
            sys.exit(1)

        #-----------------------
        # run profiling workflow
        #-----------------------

        self.__soc.profiling_setup()
        profiler.pre_processing()
        profiler.run_profiling()
        profiler.post_processing()

        return

    @demarcate
    def update_DB(self):
        return

    @demarcate
    def run_analysis(self):

        self.detectSoC()

        logging.info("Analysis mode choie = %s" % self.__analyze_mode)

        if self.__analyze_mode == "cli":
            from analysis_cli import cli_analysis
            analyzer = cli_analysis(self.__args,self.__options)
        elif self.__analyze_mode == "webui":
            from analysis_webui import webui_analysis
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
