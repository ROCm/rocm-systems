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
from utils import file_io, parser, schema, tty
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_error, demarcate
from utils.roofline_calc import calc_ai_analyze


class cli_analysis(OmniAnalyze_Base):
    SUPPORTED_ROOFLINE_ARCHS = ["gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"]

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()
        args = self.get_args()

        if args.random_port:
            console_error("--gui flag is required to enable --random-port")

        for path_info in args.path:
            workload = self._runs[path_info[0]]

            # create 'mega dataframe'
            workload.raw_pmc = file_io.create_df_pmc(
                path_info[0],
                args.nodes,
                args.spatial_multiplexing,
                args.kernel_verbose,
                args.verbose,
                self._profiling_config,
            )

            if args.spatial_multiplexing:
                workload.raw_pmc = self.spatial_multiplex_merge_counters(
                    workload.raw_pmc
                )

            if self._profiling_config.get("iteration_multiplexing") is not None:
                workload.raw_pmc = self.iteration_multiplex_merge_counters(
                    workload.raw_pmc,
                    policy=self._profiling_config["iteration_multiplexing"],
                )

            file_io.create_df_kernel_top_stats(
                df_in=workload.raw_pmc,
                raw_data_dir=path_info[0],
                filter_gpu_ids=workload.filter_gpu_ids,
                filter_dispatch_ids=workload.filter_dispatch_ids,
                filter_nodes=workload.filter_nodes,
                time_unit=args.time_unit,
                kernel_verbose=args.kernel_verbose,
            )

            # demangle and overwrite original 'Kernel_Name'
            kernel_name_shortener(workload.raw_pmc, args.kernel_verbose)

            # create the loaded table
            parser.load_table_data(
                workload=workload,
                dir_path=path_info[0],
                is_gui=False,
                args=args,
                config=self._profiling_config,
            )

    def _populate_roofline_metrics(
        self, workload_path: str, workload: schema.Workload
    ) -> None:
        """Populate per-run roofline metrics for CLI comparisons."""
        socs = self.get_socs()
        if not socs or workload.roofline_peaks.empty:
            return

        gpu_arch = workload.sys_info.iloc[0]["gpu_arch"]
        if gpu_arch not in self.SUPPORTED_ROOFLINE_ARCHS:
            return
        if gpu_arch not in socs or gpu_arch not in self._arch_configs:
            return

        workload.path = workload_path
        calc_ai_analyze(
            workload=workload,
            mspec=socs[gpu_arch]._mspec,
            sort_type=str(getattr(self.get_args(), "sort", "kernels")),
            config=self._profiling_config,
            arch_config=self._arch_configs[gpu_arch],
        )

    @demarcate
    def run_analysis(self) -> None:
        """Run CLI analysis."""
        super().run_analysis()

        args = self.get_args()

        workload_path = args.path[0][0]
        workload = self._runs[workload_path]
        workload.path = workload_path
        gpu_arch = workload.sys_info.iloc[0]["gpu_arch"]
        arch_config = self._arch_configs[gpu_arch]

        if args.list_stats:
            tty.show_kernel_stats(
                args,
                self._runs,
                arch_config,
                self._output,
            )
        else:
            roof_plot = None

            # Generate roofline plot for single-path, compatible architectures
            if len(args.path) == 1 and gpu_arch in self.SUPPORTED_ROOFLINE_ARCHS:
                soc = self.get_socs()
                if soc and gpu_arch in soc:
                    roof_obj = soc[gpu_arch].roofline_obj

                    if roof_obj:
                        # NOTE: using default data type
                        roof_plot = roof_obj.cli_generate_plot(
                            dtype=roof_obj.get_dtype()[0],
                            workload=workload,
                            config=self._profiling_config,
                            arch_config=arch_config,
                        )
            else:
                # Multi-run comparisons skip CLI plotting but still need per-run metrics.
                for run_path, run_workload in self._runs.items():
                    self._populate_roofline_metrics(run_path, run_workload)

            tty.show_all(
                args,
                self._runs,
                arch_config,
                self._output,
                self._profiling_config,
                roof_plot=roof_plot,
            )
