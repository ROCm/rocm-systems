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

from abc import ABC, abstractmethod
import logging
import glob
import sys
import os
from utils.utils import capture_subprocess_output, run_prof, gen_sysinfo, run_rocscope, error
import config

class OmniProfiler_Base():
    def __init__(self,args, profiler_mode,soc):
        self.__args = args
        self.__profiler = profiler_mode
        self.__soc = soc
        
        self.__perfmon_dir = os.path.join(str(config.omniperf_home), "omniperf_soc", "profile_configs")

    def get_args(self):
        return self.__args

    #----------------------------------------------------
    # Required methods to be implemented by child classes
    #----------------------------------------------------
    @abstractmethod
    def pre_processing(self):
        """Perform any pre-processing steps prior to profiling.
        """
        logging.debug("[profiling] pre-processing using %s profiler" % self.__profiler)

        # verify not accessing parent directories
        if ".." in str(self.__args.path):
            error("Access denied. Cannot access parent directories in path (i.e. ../)")
        
        # verify correct formatting for application binary
        self.__args.remaining = self.__args.remaining[1:]
        if self.__args.remaining:
            if not os.path.isfile(self.__args.remaining[0]):
                error("Your command %s doesn't point to a executable. Please verify." % self.__args.remaining[0])
            self.__args.remaining = " ".join(self.__args.remaining)
        else:
            error("Profiling command required. Pass application executable after -- at the end of options.\n\t\ti.e. omniperf profile -n vcopy -- ./vcopy 1048576 256")
        
        # verify name meets MongoDB length requirements and no illegal chars
        if len(self.__args.name) > 35:
            error("-n/--name exceeds 35 character limit. Try again.")
        if self.__args.name.find(".") != -1 or self.__args.name.find("-") != -1:
            error("'-' and '.' are not permitted in -n/--name")

    @abstractmethod
    def run_profiling(self, version:str, prog:str):
        """Run profiling.
        """
        logging.debug("[profiling] performing profiling using %s profiler" % self.__profiler)
        
        # log basic info
        logging.info(str(prog) + " ver: " + str(version))
        logging.info("Path: " + str(os.path.abspath(self.__args.path)))
        logging.info("Target: " + str(self.__args.target))
        logging.info("Command: " + str(self.__args.remaining))
        logging.info("Kernel Selection: " + str(self.__args.kernel))
        logging.info("Dispatch Selection: " + str(self.__args.dispatch))
        if self.__args.ipblocks == None:
            logging.info("IP Blocks: All")
        else:
            logging.info("IP Blocks: "+ str(self.__args.ipblocks))
        if self.__args.kernel_verbose > 5:
            logging.info("KernelName verbose: DISABLED")
        else:
            logging.info("KernelName verbose: " + str(self.__args.kernel_verbose))

        # Run profiling on each input file
        for fname in glob.glob(self.get_args().path + "/perfmon/*.txt"):
            # Kernel filtering (in-place replacement)
            if not self.__args.kernel == None:
                success, output = capture_subprocess_output(
                    [
                        "sed",
                        "-i",
                        "-r",
                        "s%^(kernel:).*%" + "kernel: " + ",".join(self.__args.kernel) + "%g",
                        fname,
                    ]
                )
                # log output from profile filtering
                if not success:
                    error(output)
                else:
                    logging.debug(output)

            # Dispatch filtering (inplace replacement)
            if not self.__args.dispatch == None:
                success, output = capture_subprocess_output(
                    [
                        "sed",
                        "-i",
                        "-r",
                        "s%^(range:).*%" + "range: " + " ".join(self.__args.dispatch) + "%g",
                        fname,
                    ]
                )
                # log output from profile filtering
                if not success:
                    error(output)
                else:
                    logging.debug(output)
            logging.info("\nCurrent input file: %s" % fname)
            if self.__profiler == "rocprofv1":
                #TODO: Look back at run_prof() definition. We may want to separate this based on SoC
                run_prof(fname, self.get_args().path, self.__perfmon_dir, self.__args.remaining, self.__args.target, self.__args.verbose)

            elif self.__profiler == "rocscope":
                run_rocscope(self.__args, fname)
            else:
                #TODO: Finish logic
                error("profiler not supported")

    @abstractmethod
    def post_processing(self):
        """Perform any post-processing steps prior to profiling.
        """
        logging.debug("[profiling] performing post-processing using %s profiler" % self.__profiler)
        gen_sysinfo(self.__args.name, self.get_args().path, self.__args.ipblocks, self.__args.remaining, self.__args.no_roof)