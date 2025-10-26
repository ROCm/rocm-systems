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

"""
Jupyter Notebook interface for rocprofiler-compute analysis.

This module provides a simplified API for analyzing rocprofiler-compute
performance data within Jupyter notebooks.

Example usage:
    import rocprof_compute_jupyter as rc
    
    # Load performance data
    rc.open('/path/to/perf_data')
    
    # Run analysis and display results
    rc.analysis()
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Try to import IPython display - required for Jupyter
try:
    from IPython.display import display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    # Fallback display function
    def display(obj: Any) -> None:
        """Fallback display function when IPython is not available."""
        print(obj)

from config import HIDDEN_COLUMNS
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils import file_io, parser, schema
from utils.gui import build_bar_chart
from utils.logger import console_error, console_log, console_warning


class JupyterAnalysis(OmniAnalyze_Base):
    """Jupyter-compatible analysis class for rocprofiler-compute."""

    def __init__(
        self,
        perf_data_dir: str,
        supported_archs: dict[str, str],
        **kwargs: Any,
    ) -> None:
        """
        Initialize Jupyter analysis.

        Args:
            perf_data_dir: Path to performance data directory
            supported_archs: Dictionary of supported architectures
            **kwargs: Additional analysis options (kernel_filter, gpu_filter, etc.)
        """
        # Create minimal args namespace for base class
        args = argparse.Namespace(
            path=[[perf_data_dir]],
            config_dir=str(Path(__file__).parent / "rocprof_compute_soc" / "analysis_configs"),
            list_stats=False,
            list_metrics=None,
            list_blocks=None,
            filter_metrics=kwargs.get("filter_metrics"),
            nodes=kwargs.get("nodes"),
            list_nodes=False,
            spatial_multiplexing=kwargs.get("spatial_multiplexing", False),
            kernel_verbose=kwargs.get("kernel_verbose", 0),
            verbose=kwargs.get("verbose", 0),
            debug=kwargs.get("debug", False),
            time_unit=kwargs.get("time_unit", "ns"),
            normal_unit=kwargs.get("normal_unit", "per_wave"),
            max_stat_num=kwargs.get("max_stat_num", 10),
            decimal=kwargs.get("decimal", 2),
            output_format="stdout",
            output_name=None,
            no_roof=kwargs.get("no_roof", False),
            specs_correction=kwargs.get("specs_correction"),
            roofline_data_type=kwargs.get("roofline_data_type", "empirical"),
            tui=False,
            gui=False,
            random_port=False,
            gpu_kernel=kwargs.get("kernel_filter"),
            gpu_id=kwargs.get("gpu_filter"),
            gpu_dispatch_id=kwargs.get("dispatch_filter"),
            pc_sampling_sorting_type=kwargs.get("pc_sampling_sorting_type", "samples"),
        )

        super().__init__(args, supported_archs)
        self.dest_dir = str(Path(perf_data_dir).absolute().resolve())
        self.arch: Optional[str] = None
        self._hidden_columns = HIDDEN_COLUMNS
        self._comparable_columns: list[str] = []
        self._initialized = False

        # Define chart types
        self._barchart_elements: dict[str, list[int]] = {
            "instr_mix": [1001, 1002],
            "multi_bar": [1604, 1705],
            "sol": [1101, 1201, 1301, 1401, 1601, 1701],
        }

    def pre_processing(self) -> None:
        """Perform initialization prior to analysis."""
        super().pre_processing()

        args = self.get_args()

        # Create mega dataframe
        self._runs[self.dest_dir].raw_pmc = file_io.create_df_pmc(
            self.dest_dir,
            args.nodes,
            args.spatial_multiplexing,
            args.kernel_verbose,
            args.verbose,
            self._profiling_config,
        )

        if args.spatial_multiplexing:
            self._runs[self.dest_dir].raw_pmc = self.spatial_multiplex_merge_counters(
                self._runs[self.dest_dir].raw_pmc
            )

        file_io.create_df_kernel_top_stats(
            df_in=self._runs[self.dest_dir].raw_pmc,
            raw_data_dir=self.dest_dir,
            filter_gpu_ids=self._runs[self.dest_dir].filter_gpu_ids,
            filter_dispatch_ids=self._runs[self.dest_dir].filter_dispatch_ids,
            filter_nodes=self._runs[self.dest_dir].filter_nodes,
            time_unit=args.time_unit,
            kernel_verbose=args.kernel_verbose,
        )

        # Load non-metrics table
        parser.load_non_mertrics_table(self._runs[self.dest_dir], self.dest_dir, args)

        # Set architecture
        self.arch = self._runs[self.dest_dir].sys_info.iloc[0]["gpu_arch"]

        # Build comparable columns
        self._comparable_columns = parser.build_comparable_columns(args.time_unit)

        self._initialized = True

    def run_analysis(self) -> None:
        """Run analysis - required by base class but not used in Jupyter mode."""
        pass

    def _display_dataframe(
        self,
        df: pd.DataFrame,
        table_config: dict[str, Any],
        title: Optional[str] = None,
    ) -> None:
        """Display a dataframe with optional title."""
        if title:
            display(f"### {title}")

        if df.empty:
            console_warning("analysis", f"Table {table_config.get('id', 'unknown')} is empty")
            return

        # Filter hidden columns
        display_columns = [
            col for col in df.columns.values.tolist() if col not in self._hidden_columns
        ]
        display_df = df[display_columns]

        # Check if this should be a bar chart
        table_id = table_config.get("id", 0)
        if table_id in [x for i in self._barchart_elements.values() for x in i]:
            # Display as bar chart
            charts = build_bar_chart(display_df, table_config, self._barchart_elements)
            for chart in charts:
                display(chart)
        else:
            # Display as table
            display(display_df)

    def display_results(
        self,
        filter_kernel: Optional[list[str]] = None,
        filter_gpu: Optional[list[int]] = None,
        filter_dispatch: Optional[list[int]] = None,
        show_basic_only: bool = False,
    ) -> None:
        """
        Display analysis results in Jupyter notebook.

        Args:
            filter_kernel: List of kernel IDs to filter
            filter_gpu: List of GPU IDs to filter
            filter_dispatch: List of dispatch IDs to filter
            show_basic_only: If True, show only basic metrics
        """
        if not self._initialized:
            console_error("analysis", "Must call pre_processing() first")

        args = self.get_args()
        base_run = self.dest_dir
        base_data = self._runs[base_run]

        # Apply filters
        if filter_kernel:
            base_data.filter_kernel_ids = [str(k) for k in filter_kernel]
        if filter_gpu:
            base_data.filter_gpu_ids = filter_gpu
        if filter_dispatch:
            base_data.filter_dispatch_ids = filter_dispatch

        # Get architecture config
        if not self.arch or self.arch not in self._arch_configs:
            console_error("analysis", f"Architecture {self.arch} not supported")

        arch_config = self._arch_configs[self.arch]
        panel_configs = copy.deepcopy(arch_config.panel_configs)

        # Filter to basic panels if requested
        if show_basic_only or not (filter_kernel or filter_gpu or filter_dispatch):
            basic_panels_keep = [0, 100, 200, 300, 400]
            panel_configs = {
                key: panel_configs[key]
                for key in panel_configs
                if key in basic_panels_keep
            }

        # Load table data with filters
        parser.load_table_data(
            workload=base_data,
            dir_path=self.dest_dir,
            is_gui=False,
            args=args,
            config=self._profiling_config,
        )

        # Display system information
        display("# ROCProfiler-Compute Analysis Results")
        display(f"**Data Directory:** {self.dest_dir}")
        display(f"**Architecture:** {self.arch}")
        display("")

        # Display roofline if available
        has_roofline = (Path(self.dest_dir) / "roofline.csv").is_file()
        soc = self.get_socs()
        if soc and self.arch in soc and has_roofline:
            if hasattr(soc[self.arch], "roofline_obj"):
                soc[self.arch].analysis_setup(
                    roofline_parameters={
                        "workload_dir": self.dest_dir,
                        "device_id": 0,
                        "sort_type": "kernels",
                        "mem_level": "ALL",
                        "include_kernel_names": True,
                        "is_standalone": False,
                        "roofline_data_type": args.roofline_data_type,
                        "kernel_filter": False,
                    }
                )
                roof_obj = soc[self.arch].roofline_obj
                roofline_fig = roof_obj.empirical_roofline(
                    ret_df=parser.apply_filters(
                        workload=base_data,
                        dir_path=self.dest_dir,
                        is_gui=False,
                        debug=args.debug,
                    )
                )
                display("## Roofline Analysis")
                display(roofline_fig)

        # Display each panel
        for panel_id, panel in panel_configs.items():
            title = f"{panel_id // 100}. {panel['title']}"
            display(f"## {title}")

            # Display each table in the panel
            for data_source in panel["data source"]:
                for t_type, table_config in data_source.items():
                    if table_config["id"] not in base_data.dfs:
                        continue

                    df = base_data.dfs[table_config["id"]]
                    subtitle = table_config.get("title")
                    if subtitle:
                        subtitle = f"{table_config['id'] // 100}.{table_config['id'] % 100} {subtitle}"

                    self._display_dataframe(df, table_config, subtitle)

        # Show message if no filters applied
        if not (filter_kernel or filter_gpu or filter_dispatch):
            display("")
            display("**Note:** To see detailed metrics, apply filters using filter_kernel, filter_gpu, or filter_dispatch parameters.")


# Global state for the module-level API
_current_analysis: Optional[JupyterAnalysis] = None
_supported_archs: dict[str, str] = {}


def _initialize_supported_archs() -> dict[str, str]:
    """Initialize supported architectures."""
    global _supported_archs
    if not _supported_archs:
        # Import here to avoid circular dependencies
        from rocprof_compute_soc import soc_gfx908, soc_gfx90a, soc_gfx940, soc_gfx941, soc_gfx942, soc_gfx950

        _supported_archs = {
            "gfx908": "MI100",
            "gfx90a": "MI200",
            "gfx940": "MI300A",
            "gfx941": "MI300A",
            "gfx942": "MI300X",
            "gfx950": "MI350X",
        }
    return _supported_archs


def open(perf_data_dir: str, **kwargs: Any) -> None:
    """
    Load performance data from a directory.

    Args:
        perf_data_dir: Path to the performance data directory
        **kwargs: Additional options:
            - kernel_filter: List of kernel IDs to filter
            - gpu_filter: List of GPU IDs to filter
            - dispatch_filter: List of dispatch IDs to filter
            - time_unit: Time unit for display (default: 'ns')
            - normal_unit: Normalization unit (default: 'per_wave')
            - max_stat_num: Maximum number of statistics to show (default: 10)
            - decimal: Number of decimal places (default: 2)

    Example:
        >>> import rocprof_compute_jupyter as rc
        >>> rc.open('/path/to/perf_data')
    """
    global _current_analysis

    perf_data_path = Path(perf_data_dir).absolute().resolve()
    if not perf_data_path.is_dir():
        console_error("analysis", f"Invalid directory: {perf_data_dir}")

    console_log("analysis", f"Loading performance data from {perf_data_path}")

    # Initialize supported architectures
    archs = _initialize_supported_archs()

    # Create analysis instance
    _current_analysis = JupyterAnalysis(str(perf_data_path), archs, **kwargs)

    # Initialize SOCs
    from rocprof_compute_soc import soc_gfx908, soc_gfx90a, soc_gfx940, soc_gfx941, soc_gfx942, soc_gfx950
    from utils.specs import generate_machine_specs

    # Load system info from the workload directory
    sys_info = file_io.load_sys_info(f"{perf_data_path}/sysinfo.csv")
    sys_info_dict = {key: value[0] for key, value in sys_info.to_dict("list").items()}

    # Generate machine specs from the loaded sysinfo
    mspec = generate_machine_specs(_current_analysis.get_args(), sys_info_dict)

    omni_socs: dict[str, OmniSoC_Base] = {
        "gfx908": soc_gfx908.gfx908_soc(_current_analysis.get_args(), mspec),
        "gfx90a": soc_gfx90a.gfx90a_soc(_current_analysis.get_args(), mspec),
        "gfx940": soc_gfx940.gfx940_soc(_current_analysis.get_args(), mspec),
        "gfx941": soc_gfx941.gfx941_soc(_current_analysis.get_args(), mspec),
        "gfx942": soc_gfx942.gfx942_soc(_current_analysis.get_args(), mspec),
        "gfx950": soc_gfx950.gfx950_soc(_current_analysis.get_args(), mspec),
    }
    _current_analysis.set_soc(omni_socs)

    # Run preprocessing
    _current_analysis.sanitize()
    _current_analysis.pre_processing()

    console_log("analysis", "Performance data loaded successfully")


def analysis(
    filter_kernel: Optional[list[str]] = None,
    filter_gpu: Optional[list[int]] = None,
    filter_dispatch: Optional[list[int]] = None,
    show_basic_only: bool = False,
) -> None:
    """
    Display analysis results in the Jupyter notebook.

    Args:
        filter_kernel: List of kernel IDs to filter (optional)
        filter_gpu: List of GPU IDs to filter (optional)
        filter_dispatch: List of dispatch IDs to filter (optional)
        show_basic_only: If True, show only basic metrics (default: False)

    Example:
        >>> import rocprof_compute_jupyter as rc
        >>> rc.open('/path/to/perf_data')
        >>> rc.analysis()  # Show all metrics
        >>> rc.analysis(filter_kernel=['0', '1'])  # Filter specific kernels
    """
    global _current_analysis

    if _current_analysis is None:
        console_error(
            "analysis",
            "No performance data loaded. Call open() first.",
        )

    _current_analysis.display_results(
        filter_kernel=filter_kernel,
        filter_gpu=filter_gpu,
        filter_dispatch=filter_dispatch,
        show_basic_only=show_basic_only,
    )


def get_dataframe(table_id: int) -> Optional[pd.DataFrame]:
    """
    Get a specific dataframe by table ID.

    Args:
        table_id: The table ID to retrieve

    Returns:
        DataFrame if found, None otherwise

    Example:
        >>> import rocprof_compute_jupyter as rc
        >>> rc.open('/path/to/perf_data')
        >>> df = rc.get_dataframe(1)  # Get kernel top stats
    """
    global _current_analysis

    if _current_analysis is None:
        console_error(
            "analysis",
            "No performance data loaded. Call open() first.",
        )

    base_run = _current_analysis.dest_dir
    if base_run not in _current_analysis._runs:
        return None

    base_data = _current_analysis._runs[base_run]
    return base_data.dfs.get(table_id)


def list_tables() -> None:
    """
    List all available table IDs and their descriptions.

    Example:
        >>> import rocprof_compute_jupyter as rc
        >>> rc.open('/path/to/perf_data')
        >>> rc.list_tables()
    """
    global _current_analysis

    if _current_analysis is None:
        console_error(
            "analysis",
            "No performance data loaded. Call open() first.",
        )

    if not _current_analysis.arch or _current_analysis.arch not in _current_analysis._arch_configs:
        console_error("analysis", "Architecture configuration not found")

    arch_config = _current_analysis._arch_configs[_current_analysis.arch]
    panel_configs = arch_config.panel_configs

    print("Available Tables:")
    print(f"{'ID':<8} {'Panel':<40} {'Table'}")
    print("-" * 80)

    for panel_id, panel in panel_configs.items():
        panel_title = f"{panel_id // 100}. {panel['title']}"
        for data_source in panel["data source"]:
            for t_type, table_config in data_source.items():
                table_id = table_config["id"]
                table_title = table_config.get("title", "")
                print(f"{table_id:<8} {panel_title:<40} {table_title}")
