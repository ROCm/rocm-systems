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


from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import file_io, parser, tty
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_error, demarcate


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
        for d in self.get_args().path:

            # create 'mega dataframe'
            self._runs[d[0]].raw_pmc = file_io.create_df_pmc(
                d[0],
                self.get_args().nodes,
                self.get_args().spatial_multiplexing,
                self.get_args().kernel_verbose,
                self.get_args().verbose,
                self._profiling_config,
            )

            if self.get_args().spatial_multiplexing:
                self._runs[d[0]].raw_pmc = self.spatial_multiplex_merge_counters(
                    self._runs[d[0]].raw_pmc
                )

            file_io.create_df_kernel_top_stats(
                df_in=self._runs[d[0]].raw_pmc,
                raw_data_dir=d[0],
                filter_gpu_ids=self._runs[d[0]].filter_gpu_ids,
                filter_dispatch_ids=self._runs[d[0]].filter_dispatch_ids,
                filter_nodes=self._runs[d[0]].filter_nodes,
                time_unit=self.get_args().time_unit,
                max_stat_num=self.get_args().max_stat_num,
                kernel_verbose=self.get_args().kernel_verbose,
            )

            # demangle and overwrite original 'Kernel_Name'
            kernel_name_shortener(
                self._runs[d[0]].raw_pmc, self.get_args().kernel_verbose
            )

            # create the loaded table
            parser.load_table_data(
                workload=self._runs[d[0]],
                dir=d[0],
                is_gui=False,
                args=self.get_args(),
                config=self._profiling_config,
            )

    @demarcate
    def run_analysis(self):
        """Run CLI analysis."""
        super().run_analysis()

        if self.get_args().list_stats:
            tty.show_kernel_stats(
                self.get_args(),
                self._runs,
                self._arch_configs[
                    self._runs[self.get_args().path[0][0]].sys_info.iloc[0]["gpu_arch"]
                ],
                self._output,
            )
        else:
            roof_plot = None
            # 1. check if not baseline && compatible soc:
            if (len(self.get_args().path)) == 1 and self._runs[
                self.get_args().path[0][0]
            ].sys_info.iloc[0]["gpu_arch"] in [
                "gfx90a",
                "gfx940",
                "gfx941",
                "gfx942",
                "gfx950",
            ]:
                # add roofline plot to cli output
                roof_obj = self.get_socs()[
                    self._runs[self.get_args().path[0][0]].sys_info.iloc[0]["gpu_arch"]
                ].roofline_obj

                if roof_obj:
                    # NOTE: using default data type
                    roof_plot = roof_obj.cli_generate_plot(roof_obj.get_dtype()[0])

            tty.show_all(
                self.get_args(),
                self._runs,
                self._arch_configs[
                    self._runs[self.get_args().path[0][0]].sys_info.iloc[0]["gpu_arch"]
                ],
                self._output,
                self._profiling_config,
                roof_plot=roof_plot,
            )
