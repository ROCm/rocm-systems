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
            workload = self._runs[d[0]]
            # create 'mega dataframe'
            workload.raw_pmc = file_io.create_df_pmc(
                d[0],
                self.get_args().nodes,
                self.get_args().spatial_multiplexing,
                self.get_args().kernel_verbose,
                self.get_args().verbose,
                self._profiling_config,
            )

            if self.get_args().spatial_multiplexing:
                workload.raw_pmc = self.spatial_multiplex_merge_counters(
                    workload.raw_pmc
                )

            file_io.create_df_kernel_top_stats(
                df_in=workload.raw_pmc,
                raw_data_dir=d[0],
                filter_gpu_ids=workload.filter_gpu_ids,
                filter_dispatch_ids=workload.filter_dispatch_ids,
                filter_nodes=workload.filter_nodes,
                time_unit=self.get_args().time_unit,
                max_stat_num=self.get_args().max_stat_num,
                kernel_verbose=self.get_args().kernel_verbose,
            )

            # demangle and overwrite original 'Kernel_Name'
            kernel_name_shortener(workload.raw_pmc, self.get_args().kernel_verbose)

            # create the loaded table
            parser.load_table_data(
                workload=workload,
                dir=d[0],
                is_gui=False,
                args=self.get_args(),
                config=self._profiling_config,
            )

    @demarcate
    def run_analysis(self):
        """Run CLI analysis."""
        super().run_analysis()

        workload_path = self.get_args().path[0][0]
        workload = self._runs[workload_path]
        gpu_arch = workload.sys_info.iloc[0]["gpu_arch"]
        arch_config = self._arch_configs[gpu_arch]

        if self.get_args().list_stats:
            tty.show_kernel_stats(
                self.get_args(),
                self._runs,
                arch_config,
                self._output,
            )
        else:
            roof_plot = None
            # 1. check if not baseline && compatible soc:
            if (len(self.get_args().path)) == 1:
                if gpu_arch in ["gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"]:
                    roof_obj = self.get_socs()[gpu_arch].roofline_obj

                    if roof_obj:
                        # store path in workload for calc_ai_analyze
                        workload.path = workload_path

                        # NOTE: using default data type
                        roof_plot = roof_obj.cli_generate_plot(
                            dtype=roof_obj.get_dtype()[0],
                            workload=workload,
                            config=self._profiling_config,
                            arch_config=arch_config,
                        )

            tty.show_all(
                self.get_args(),
                self._runs,
                arch_config,
                self._output,
                self._profiling_config,
                roof_plot=roof_plot,
            )
