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

import argparse
import copy
import random
from pathlib import Path
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State

from config import HIDDEN_COLUMNS, PROJECT_NAME
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import file_io, parser, schema
from utils.gui import build_bar_chart, build_table_chart
from utils.gui_components.memchart import get_memchart
from utils.logger import console_debug, console_error, console_warning, demarcate


class webui_analysis(OmniAnalyze_Base):
    def __init__(
        self, args: argparse.Namespace, supported_archs: dict[str, str]
    ) -> None:
        super().__init__(args, supported_archs)
        self.app = dash.Dash(
            __name__, title=PROJECT_NAME, external_stylesheets=[dbc.themes.CYBORG]
        )
        self.dest_dir = str(Path(args.path[0][0]).absolute().resolve())
        self.arch: Optional[str] = None

        self.__hidden_sections = ["Memory Chart"]
        self.__hidden_columns = HIDDEN_COLUMNS

        # define different types of bar charts
        self.__barchart_elements: dict[str, list[int]] = {
            "instr_mix": [1001, 1002],
            # 1604: L1D - L2 Transactions
            # 1705: L2 - Fabric Interface Stalls
            "multi_bar": [1604, 1705],
            "sol": [1101, 1201, 1301, 1401, 1601, 1701],
            # "l2_cache_per_chan": [1802, 1803]
        }

        # define any elements which will have full width
        self.__full_width_elements: set[int] = {1801}
        self.__roofline_data_type = args.roofline_data_type

    @demarcate
    def build_layout(
        self, input_filters: dict[str, Any], arch_configs: schema.ArchConfig
    ) -> None:
        """
        Build gui layout
        """
        args = self.get_args()

        comparable_columns = parser.build_comparable_columns(args.time_unit)
        base_run, base_data = next(iter(self._runs.items()))

        self.app.layout = html.Div(style={"backgroundColor": "rgb(50, 50, 50)"})

        # get filtered kernel names from kernel ids
        filt_kernel_names: list[str] = []
        kernel_top_df = base_data.dfs[1]
        for kernel_id in base_data.filter_kernel_ids:
            filt_kernel_names.append(str(kernel_top_df.loc[kernel_id, "Kernel_Name"]))

        # setup app layout
        from utils.gui_components.header import get_header

        self.app.layout.children = html.Div(
            children=[
                dbc.Spinner(
                    children=[
                        get_header(base_data.raw_pmc, input_filters, filt_kernel_names),
                        html.Div(id="container", children=[]),
                    ],
                    fullscreen=True,
                    color="primary",
                    spinner_style={"width": "6rem", "height": "6rem"},
                )
            ]
        )

        @self.app.callback(
            Output("container", "children"),
            [Input("disp-filt", "value")],
            [Input("kernel-filt", "value")],
            [Input("gcd-filt", "value")],
            [Input("norm-filt", "value")],
            [Input("top-n-filt", "value")],
            [State("container", "children")],
        )
        def generate_from_filter(
            disp_filt: list[str],
            kernel_filter: list[str],
            gcd_filter: list[str],
            norm_filt: str,
            top_n_filt: int,
            div_children: list[html.Section],
        ) -> list[html.Section]:
            console_debug("analysis", f"gui normalization is {norm_filt}")

            # Re-initalizes everything
            base_data = self.initalize_runs(normalization_filter=norm_filt)
            panel_configs = copy.deepcopy(arch_configs.panel_configs)

            # Generate original raw df
            base_data[base_run].raw_pmc = file_io.create_df_pmc(
                self.dest_dir,
                args.nodes,
                args.spatial_multiplexing,
                args.kernel_verbose,
                args.verbose,
                self._profiling_config,
            )

            if args.spatial_multiplexing:
                base_data[base_run].raw_pmc = self.spatial_multiplex_merge_counters(
                    base_data[base_run].raw_pmc
                )

            # Apply filters to workload data
            console_debug("analysis", f"gui dispatch filter is {disp_filt}")
            console_debug("analysis", f"gui kernel filter is {kernel_filter}")
            console_debug("analysis", f"gui gpu filter is {gcd_filter}")
            console_debug("analysis", f"gui top-n filter is {top_n_filt}")

            base_data[base_run].filter_kernel_ids = (
                [str(k) for k in kernel_filter] if kernel_filter else []
            )
            base_data[base_run].filter_gpu_ids = (
                [int(g) for g in gcd_filter] if gcd_filter else []
            )
            base_data[base_run].filter_dispatch_ids = (
                [int(d) for d in disp_filt] if disp_filt else []
            )
            base_data[base_run].filter_top_n = top_n_filt

            # Reload the pmc_kernel_top.csv for Top Stats panel
            file_io.create_df_kernel_top_stats(
                df_in=base_data[base_run].raw_pmc,
                raw_data_dir=str(self.dest_dir),
                filter_gpu_ids=base_data[base_run].filter_gpu_ids,
                filter_dispatch_ids=base_data[base_run].filter_dispatch_ids,
                filter_nodes=self._runs[self.dest_dir].filter_nodes,
                time_unit=args.time_unit,
                kernel_verbose=args.kernel_verbose,
            )

            # Only display basic metrics if no filters are applied
            if not (disp_filt or kernel_filter or gcd_filter):
                basic_dfs_keep = [1, 2, 101, 201, 301, 401, 402]
                basic_panels_keep = [0, 100, 200, 300, 400]

                # Filter dataframes
                filtered_dfs = {
                    key: base_data[base_run].dfs[key]
                    for key in base_data[base_run].dfs
                    if key in basic_dfs_keep
                }
                base_data[base_run].dfs = filtered_dfs

                panel_configs = {
                    key: panel_configs[key]
                    for key in panel_configs
                    if key in basic_panels_keep
                }

            # All filtering will occur here
            parser.load_table_data(
                workload=base_data[base_run],
                dir_path=self.dest_dir,
                is_gui=True,
                args=args,
                config=self._profiling_config,
            )

            # ~~~~~~~~~~~~~~~~~~~~~~~
            # Generate GUI content
            # ~~~~~~~~~~~~~~~~~~~~~~~
            div_children = [
                get_memchart(panel_configs[300]["data source"], base_data[base_run])
            ]

            has_roofline = (Path(self.dest_dir) / "roofline.csv").is_file()
            soc = self.get_socs()
            if soc and self.arch in soc:
                if has_roofline and hasattr(soc[self.arch], "roofline_obj"):
                    # update roofline for visualization in GUI
                    soc[self.arch].analysis_setup(
                        roofline_parameters={
                            "workload_dir": self.dest_dir,
                            "device_id": 0,
                            "sort_type": "kernels",
                            "mem_level": "ALL",
                            "include_kernel_names": True,
                            "is_standalone": False,
                            "roofline_data_type": self.__roofline_data_type,
                            "kernel_filter": False,
                        }
                    )
                    roof_obj = soc[self.arch].roofline_obj
                    div_children.append(
                        roof_obj.empirical_roofline(
                            ret_df=parser.apply_filters(
                                workload=base_data[base_run],
                                dir_path=self.dest_dir,
                                is_gui=True,
                                debug=args.debug,
                            )
                        )
                    )

            # Iterate over each section as defined in panel configs
            for panel_id, panel in panel_configs.items():
                if panel["title"] in self.__hidden_sections:
                    continue
                title = f"{panel_id // 100}. {panel['title']}"
                section_title = (
                    panel["title"]
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "")
                    .replace(" ", "_")
                    .lower()
                )

                # Build content for a single panel
                html_section = []
                # Iterate over each table per section
                for data_source in panel["data source"]:
                    for t_type, table_config in data_source.items():
                        original_df = base_data[base_run].dfs[table_config["id"]]

                        # The sys info table need to add index back
                        if t_type == "raw_csv_table" and "Info" in original_df.keys():
                            original_df.reset_index(inplace=True)

                        content = determine_chart_type(
                            original_df=original_df,
                            table_config=table_config,
                            hidden_columns=self.__hidden_columns,
                            barchart_elements=self.__barchart_elements,
                            comparable_columns=comparable_columns,
                            decimal=args.decimal,
                        )

                        # Update content for this section
                        div_style = (
                            {"width": "100%"}
                            if table_config["id"] in self.__full_width_elements
                            else {}
                        )
                        html_section.append(
                            html.Div(
                                className="float-child",
                                children=content,
                                style=div_style,
                            )
                        )

                # Append the new section with all of it's contents
                div_children.append(
                    html.Section(
                        id=section_title,
                        children=[
                            html.H3(
                                children=title,
                                style={"color": "white"},
                            ),
                            html.Div(
                                className="float-container", children=html_section
                            ),
                        ],
                    )
                )

            # Display pop-up message if no filters are applied
            if not (disp_filt or kernel_filter or gcd_filter):
                div_children.append(
                    html.Section(
                        id="popup",
                        children=[
                            html.Div(
                                children=(
                                    "To dive deeper, use the top drop down menus to "
                                    "isolate particular kernel(s) or dispatch(s). "
                                    "You will then see the web page update with "
                                    "additional low-level metrics specific to the "
                                    "filter you've applied."
                                ),
                            ),
                        ],
                    )
                )

            return div_children

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()

        if len(self._runs) != 1:
            console_error(
                "analysis",
                "Multiple runs not yet supported in GUI. Retry without --gui flag.",
            )

        args = self.get_args()

        # create 'mega dataframe'
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
        # create the loaded kernel stats
        parser.load_non_mertrics_table(self._runs[self.dest_dir], self.dest_dir, args)
        # set architecture
        self.arch = self._runs[self.dest_dir].sys_info.iloc[0]["gpu_arch"]

    @demarcate
    def run_analysis(self) -> None:
        """Run webui analysis."""
        super().run_analysis()

        args = self.get_args()

        input_filters = {
            "kernel": self._runs[self.dest_dir].filter_kernel_ids,
            "gpu": self._runs[self.dest_dir].filter_gpu_ids,
            "dispatch": self._runs[self.dest_dir].filter_dispatch_ids,
            "normalization": args.normal_unit,
            "top_n": args.max_stat_num,
        }

        if self.arch and self.arch in self._arch_configs:
            self.build_layout(
                input_filters,
                self._arch_configs[self.arch],
            )

        port = random.randint(1024, 49151) if args.random_port else args.gui
        self.app.run(debug=False, host="0.0.0.0", port=port)


@demarcate
def determine_chart_type(
    original_df: pd.DataFrame,
    table_config: dict[str, Any],
    hidden_columns: list[str],
    barchart_elements: dict[str, list[int]],
    comparable_columns: list[str],
    decimal: int,
) -> list[html.Div]:
    content = []

    if original_df.empty:
        console_warning(
            "analysis",
            f"The dataframe with id={table_config['id']} is empty! Not displaying it.",
        )
        return content

    display_columns = [
        col for col in original_df.columns.values.tolist() if col not in hidden_columns
    ]
    display_df = original_df[display_columns]

    # Determine chart type:
    # a) Barchart
    if table_config["id"] in [x for i in barchart_elements.values() for x in i]:
        d_figs = build_bar_chart(display_df, table_config, barchart_elements)
        # Smaller formatting if barchart yeilds several graphs
        if len(d_figs) > 2:
            temp_obj = [
                html.Div(
                    className="float-child",
                    children=[dcc.Graph(figure=fig, style={"margin": "2%"})],
                )
                for fig in d_figs
            ]
            content.append(html.Div(className="float-container", children=temp_obj))
        # Normal formatting if < 2 graphs
        else:
            content.extend([
                dcc.Graph(figure=fig, style={"margin": "2%"}) for fig in d_figs
            ])

    # b) Tablechart
    else:
        d_figs = build_table_chart(
            display_df,
            table_config,
            original_df,
            display_columns,
            comparable_columns,
            decimal,
        )
        content.extend([html.Div([fig], style={"margin": "2%"}) for fig in d_figs])

    # subtitle for each table in a panel if existing
    if table_config.get("title"):
        subtitle = (
            f"{table_config['id'] // 100}.{table_config['id'] % 100} "
            f"{table_config['title']}\n"
        )

        content.insert(
            0,
            html.H4(
                children=subtitle,
                style={"color": "white"},
            ),
        )
    return content
