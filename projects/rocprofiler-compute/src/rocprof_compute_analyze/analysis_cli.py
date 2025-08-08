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
from utils.logger import console_error, console_log, console_warning, demarcate


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
            kernel_name_shortener(
                workload.raw_pmc, self.get_args().kernel_verbose
            )
            
            if len(self.get_args().path) == 1:
                self.calculate_per_kernel_roofline_tables(d[0])

            # create the loaded table
            parser.load_table_data(
                workload=workload,
                dir=d[0],
                is_gui=False,
                args=self.get_args(),
                config=self._profiling_config,
            )


    def calculate_per_kernel_roofline_tables(self, run_dir):
        """
        Calculate per-kernel roofline metrics using the eval_metric engine.
        """
        workload = self._runs[run_dir]
        arch = workload.sys_info.iloc[0]["gpu_arch"]
        
        # Check for architecture and panel compatibility
        if arch not in self._arch_configs or 400 not in self._arch_configs[arch].panel_configs:
            return
            
        arch_config = self._arch_configs[arch]
        roofline_panel = arch_config.panel_configs[400]

        # Find the two roofline tables from the config
        roofline_tables = {}
        for data_source in roofline_panel.get("data source", []):
            if "metric_table" in data_source:
                table_config = data_source["metric_table"]
                if table_config.get("cli_style") == "Roofline":
                    roofline_tables[table_config["id"]] = table_config
        
        if not roofline_tables:
            return

        console_log("Calculating per-kernel roofline metrics...")
        
        workload.per_kernel_roofline = {'performance_rates': [], 'calculation_data': []}
        
        # We need a function to get the top kernels list. Assuming one exists in file_io.
        # Let's define a helper for this.
        kernel_top_df = file_io.create_df_from_file(run_dir, "pmc_kernel_top.csv")
        if kernel_top_df.empty:
            console_warning("pmc_kernel_top.csv not found for per-kernel roofline. Skipping.")
            return

        unique_kernels = kernel_top_df.head(self.get_args().max_stat_num)

        # MAIN LOOP: Iterate through each kernel
        for kernel_idx, kernel_row in unique_kernels.iterrows():
            kernel_name = kernel_row['Kernel_Name']
            
            single_kernel_pmc_df = workload.raw_pmc[workload.raw_pmc['pmc_perf']['Kernel_Name'] == kernel_name]
            if single_kernel_pmc_df.empty:
                continue

            perf_results, calc_results = {}, {}
            for table_id, table_config in roofline_tables.items():
                temp_df = workload.dfs[table_id].copy()
                
                parser.eval_metric(
                    {table_id: temp_df}, {table_id: "metric_table"},
                    workload.sys_info.iloc[0],
                    workload.roofline_peaks, # <-- This comes from the base class now!
                    single_kernel_pmc_df,
                    self.get_args().debug, self._profiling_config,
                )

                if table_id == 401: perf_results = temp_df.set_index('Metric').to_dict('index')
                elif table_id == 402: calc_results = temp_df.set_index('Metric').to_dict('index')
            
            # Structure results for TTY
            perf_metrics_for_tty = [
                {'name': name, 'value': data['Value'], 'unit': data['Unit'], 'peak': data.get('Peak (Empirical)')}
                for name, data in perf_results.items()
            ]
            calc_metrics_for_tty = [
                {'name': name, 'value': data['Value'], 'unit': data['Unit']}
                for name, data in calc_results.items()
            ]
            
            workload.per_kernel_roofline['performance_rates'].append({
                'kernel_idx': kernel_idx, 'kernel_name': kernel_name, 'metrics': perf_metrics_for_tty
            })
            workload.per_kernel_roofline['calculation_data'].append({
                'kernel_idx': kernel_idx, 'kernel_name': kernel_name, 'metrics': calc_metrics_for_tty
            })

        console_log(f"Finished calculating per-kernel roofline for {len(unique_kernels)} kernels.")

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
            if len(self.get_args().path) == 1:
                workload = self._runs[self.get_args().path[0][0]]
                arch = workload.sys_info.iloc[0]["gpu_arch"]
                
                if arch in ["gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"]:
                    roof_obj = self.get_socs()[arch].roofline_obj
                    
                    if roof_obj:
                        roof_obj._workload = workload
                        roof_obj._arch_config = self._arch_configs[arch]
                        roof_obj._profiling_config = self._profiling_config
                        
                        roof_plot = roof_obj.cli_generate_plot(roof_obj.get_dtype()[0])
            
            tty.show_all(
                self.get_args(),
                self._runs,
                self._arch_configs[workload.sys_info.iloc[0]["gpu_arch"]],
                self._output,
                self._profiling_config,
                roof_plot=roof_plot,
            )
