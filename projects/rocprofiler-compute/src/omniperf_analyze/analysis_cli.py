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

from omniperf_analyze.analysis_base import OmniAnalyze_Base
from utils.utils import demarcate, console_error
from utils import file_io, parser, tty
from utils.kernel_name_shortener import kernel_name_shortener


class cli_analysis(OmniAnalyze_Base):
    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self):
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()
        if self.get_args().random_port:
            console_error("--gui flag is required to enable --random-port")
        for i, d in enumerate(self.get_args().path):
            file_io.create_df_kernel_top_stats(
                raw_data_dir=d[0],
                filter_gpu_ids=self._runs[i].filter_gpu_ids,
                filter_dispatch_ids=self._runs[i].filter_dispatch_ids,
                time_unit=self.get_args().time_unit,
                max_stat_num=self.get_args().max_stat_num,
                kernel_verbose=self.get_args().kernel_verbose,
            )
            # create 'mega dataframe'
            self._runs[i].raw_pmc = file_io.create_df_pmc(
                d[0], self.get_args().kernel_verbose, self.get_args().verbose
            )
            # demangle and overwrite original 'Kernel_Name'
            kernel_name_shortener(self._runs[i].raw_pmc, self.get_args().kernel_verbose)

            # create the loaded table
            parser.load_table_data(
                workload=self._runs[i],
                dir=d[0],
                is_gui=False,
                debug=self.get_args().debug,
                verbose=self.get_args().verbose,
            )

    @demarcate
    def run_analysis(self):
        """Run CLI analysis."""
        super().run_analysis()
        if self.get_args().list_stats:
            tty.show_kernel_stats(
                self.get_args(),
                self._runs,
                self._arch_configs[self._runs[0].sys_info.iloc[0]["gpu_arch"]],
                self._output,
            )
        else:
            tty.show_all(
                self.get_args(),
                self._runs,
                self._arch_configs[self._runs[0].sys_info.iloc[0]["gpu_arch"]],
                self._output,
            )
