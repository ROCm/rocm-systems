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

from omniperf_analyze.analysis_base import OmniAnalyze_Base
from utils.utils import demarcate, error
from utils import file_io, parser, tty
from utils.csv_processor import kernel_name_shortener

class cli_analysis(OmniAnalyze_Base):

    #-----------------------
    # Required child methods
    #-----------------------
    @demarcate
    def pre_processing(self, omni_soc):
        """Perform any pre-processing steps prior to analysis.
        """
        super().pre_processing(omni_soc)
        if self.get_args().random_port:
            error("--gui flag is required to enable --random-port")
        for d in self.get_args().path:
            # demangle and overwrite original 'KernelName'
            kernel_name_shortener(d[0], self.get_args().kernel_verbose)

            file_io.create_df_kernel_top_stats(
                d[0],
                self._runs[d[0]].filter_gpu_ids,
                self._runs[d[0]].filter_dispatch_ids,
                self.get_args().time_unit,
                self.get_args().max_kernel_num
            )
            # create 'mega dataframe'
            self._runs[d[0]].raw_pmc = file_io.create_df_pmc(
                d[0], self.get_args().verbose
            )
            is_gui = False
            # create the loaded table
            parser.load_table_data(
                self._runs[d[0]], d[0], is_gui, self.get_args().g, self.get_args().verbose
            )


    @demarcate
    def run_analysis(self):
        """Run CLI analysis.
        """
        if self.get_args().list_kernels:
            tty.show_kernels(
                self.get_args(),
                self._runs,
                self._arch_configs[self._runs[self.get_args().path[0][0]].sys_info.iloc[0]["gpu_soc"]],
                self._output
            )
        else:
            tty.show_all(
                self.get_args(),
                self._runs,
                self._arch_configs[self._runs[self.get_args().path[0][0]].sys_info.iloc[0]["gpu_soc"]],
                self._output
            )
