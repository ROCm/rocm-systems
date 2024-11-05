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

from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from utils.utils import demarcate, console_log


class rocscope_profiler(RocProfCompute_Base):
    def __init__(self, profiling_args, profiler_mode, soc):
        super().__init__(profiling_args, profiler_mode, soc)

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self):
        """Perform any pre-processing steps prior to profiling."""
        self.__profiler = "rocscope"
        console_log("profiling", "pre-processing using %s profiler" % self.__profiler)
        # TODO: Finish implementation

    @demarcate
    def run_profiling(self, version, prog):
        """Run profiling."""
        console_log(
            "profiling", "performing profiling using %s profiler" % self.__profiler
        )
        # TODO: Finish implementation

    @demarcate
    def post_processing(self):
        """Perform any post-processing steps prior to profiling."""
        console_log(
            "profiling",
            "performing post-processing using %s profiler" % self.__profiler,
        )
        # TODO: Finish implementation
