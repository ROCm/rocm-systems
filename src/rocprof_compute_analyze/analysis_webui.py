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


import copy
import random
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

from config import HIDDEN_COLUMNS, PROJECT_NAME
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import file_io, parser
from utils.gui import build_bar_chart, build_table_chart
from utils.logger import console_debug, console_error, console_warning, demarcate


class webui_analysis(OmniAnalyze_Base):
    def __init__(self, args, supported_archs):
        super().__init__(args, supported_archs)
        self.app = dash.Dash(
            __name__, title=PROJECT_NAME, external_stylesheets=[dbc.themes.CYBORG]
        )
        self.dest_dir = str(Path(args.path[0][0]).absolute().resolve())
        self.arch = None

        self.__hidden_sections = ["Memory Chart", "Roofline"]
        self.__hidden_columns = HIDDEN_COLUMNS
        # define different types of bar charts
        self.__barchart_elements = {
            "instr_mix": [1001, 1002],
            # 1604: L1D - L2 Transactions
            # 1705: L2 - Fabric Interface Stalls
            "multi_bar": [1604, 1705],
            "sol": [1101, 1201, 1301, 1401, 1601, 1701],
            # "l2_cache_per_chan": [1802, 1803]
        }
        # define any elements which will have full width
        self.__full_width_elements = {1801}

        self.__roofline_data_type = args.roofline_data_type

    @demarcate
    def build_layout(self, input_filters, arch_configs):
        """
        Build gui layout
        """
        from utils.gui_components.header import get_header
        from utils.gui_components.memchart import get_memchart

        comparable_columns = parser.build_comparable_columns(self.get_args().time_unit)
        base_run, base_data = next(iter(self._runs.items()))
        self.app.layout = html.Div(style={"backgroundColor": "rgb(50, 50, 50)"})

        filt_kernel_names = []
        kernel_top_df = base_data.dfs[1]
        for kernel_id in base_data.filter_kernel_ids:
            filt_kernel_names.append(kernel_top_df.loc[kernel_id, "Kernel_Name"])

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
            disp_filt, kernel_filter, gcd_filter, norm_filt, top_n_filt, div_children
        ):
            console_debug("analysis", "gui normalization is %s" % norm_filt)

            # Re-initalizes everything
            base_data = self.initalize_runs(normalization_filter=norm_filt)

            panel_configs = copy.deepcopy(arch_configs.panel_configs)
            # Generate original raw df
            base_data[base_run].raw_pmc = file_io.create_df_pmc(
                self.dest_dir,
                self.get_args().nodes,
                self.get_args().spatial_multiplexing,
                self.get_args().kernel_verbose,
                self.get_args().verbose,
                self._profiling_config,
            )

            if self.get_args().spatial_multiplexing:
                base_data[base_run].raw_pmc = self.spatial_multiplex_merge_counters(
                    base_data[base_run].raw_pmc
                )

            console_debug("analysis", "gui dispatch filter is %s" % disp_filt)
            console_debug("analysis", "gui kernel filter is %s" % kernel_filter)
            console_debug("analysis", "gui gpu filter is %s" % gcd_filter)
            console_debug("analysis", "gui top-n filter is %s" % top_n_filt)
            base_data[base_run].filter_kernel_ids = kernel_filter
            base_data[base_run].filter_gpu_ids = gcd_filter
            base_data[base_run].filter_dispatch_ids = disp_filt
            base_data[base_run].filter_top_n = top_n_filt

            # Reload the pmc_kernel_top.csv for Top Stats panel
            file_io.create_df_kernel_top_stats(
                df_in=base_data[base_run].raw_pmc,
                raw_data_dir=str(self.dest_dir),
                filter_gpu_ids=base_data[base_run].filter_gpu_ids,
                filter_dispatch_ids=base_data[base_run].filter_dispatch_ids,
                filter_nodes=self._runs[self.dest_dir].filter_nodes,
                time_unit=self.get_args().time_unit,
                max_stat_num=base_data[base_run].filter_top_n,
                kernel_verbose=self.get_args().kernel_verbose,
            )
            # Only display basic metrics if no filters are applied
            if not (disp_filt or kernel_filter or gcd_filter):
                temp = {}
                keep = [1, 2, 101, 201, 301, 401]
                for key in base_data[base_run].dfs:
                    if keep.count(key) != 0:
                        temp[key] = base_data[base_run].dfs[key]

                base_data[base_run].dfs = temp
                temp = {}
                keep = [0, 100, 200, 300, 400]
                for key in panel_configs:
                    if keep.count(key) != 0:
                        temp[key] = panel_configs[key]
                panel_configs = temp
            # All filtering will occur here
            parser.load_table_data(
                workload=base_data[base_run],
                dir=self.dest_dir,
                is_gui=True,
                args=self.get_args(),
                config=self._profiling_config,
            )

            # ~~~~~~~~~~~~~~~~~~~~~~~
            # Generate GUI content
            # ~~~~~~~~~~~~~~~~~~~~~~~
            div_children = []

            # Append memory chart and roofline
            div_children.append(
                get_memchart(panel_configs[300]["data source"], base_data[base_run])
            )
            has_roofline = Path(self.dest_dir).joinpath("roofline.csv").is_file()
            if has_roofline and hasattr(self.get_socs()[self.arch], "roofline_obj"):
                # update roofline for visualization in GUI
                self.get_socs()[self.arch].analysis_setup(
                    roofline_parameters={
                        "workload_dir": self.dest_dir,
                        "device_id": 0,
                        "sort_type": "kernels",
                        "mem_level": "ALL",
                        "include_kernel_names": False,
                        "is_standalone": False,
                        "roofline_data_type": self.__roofline_data_type,
                    }
                )
                roof_obj = self.get_socs()[self.arch].roofline_obj
                div_children.append(
                    roof_obj.empirical_roofline(
                        ret_df=parser.apply_filters(
                            workload=base_data[base_run],
                            dir=self.dest_dir,
                            is_gui=True,
                            debug=self.get_args().debug,
                        )
                    )
                )

            # Iterate over each section as defined in panel configs
            for panel_id, panel in panel_configs.items():
                title = str(panel_id // 100) + ". " + panel["title"]
                section_title = (
                    panel["title"]
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "")
                    .replace(" ", "_")
                    .lower()
                )
                html_section = []

                if panel["title"] not in self.__hidden_sections:
                    # Iterate over each table per section
                    for data_source in panel["data source"]:
                        for t_type, table_config in data_source.items():
                            original_df = base_data[base_run].dfs[table_config["id"]]
                            # The sys info table need to add index back
                            if (
                                t_type == "raw_csv_table"
                                and "Info" in original_df.keys()
                            ):
                                original_df.reset_index(inplace=True)

                            content = determine_chart_type(
                                original_df=original_df,
                                table_config=table_config,
                                hidden_columns=self.__hidden_columns,
                                barchart_elements=self.__barchart_elements,
                                norm_filt=norm_filt,
                                comparable_columns=comparable_columns,
                                decimal=self.get_args().decimal,
                            )

                            # Update content for this section
                            if table_config["id"] in self.__full_width_elements:
                                # Optionally override default (50%) width
                                html_section.append(
                                    html.Div(
                                        className="float-child",
                                        children=content,
                                        style={"width": "100%"},
                                    )
                                )
                            else:
                                html_section.append(
                                    html.Div(className="float-child", children=content)
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
                                children="To dive deeper, use the top drop down menus to isolate particular kernel(s) or dispatch(s). You will then see the web page update with additional low-level metrics specific to the filter you've applied.",
                            ),
                        ],
                    )
                )

            return div_children

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self):
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()
        if len(self._runs) == 1:
            args = self.get_args()

            # create 'mega dataframe'
            self._runs[self.dest_dir].raw_pmc = file_io.create_df_pmc(
                self.dest_dir,
                self.get_args().nodes,
                self.get_args().spatial_multiplexing,
                self.get_args().kernel_verbose,
                args.verbose,
                self._profiling_config,
            )

            if self.get_args().spatial_multiplexing:
                self._runs[self.dest_dir].raw_pmc = (
                    self.spatial_multiplex_merge_counters(
                        self._runs[self.dest_dir].raw_pmc
                    )
                )

            file_io.create_df_kernel_top_stats(
                df_in=self._runs[self.dest_dir].raw_pmc,
                raw_data_dir=self.dest_dir,
                filter_gpu_ids=self._runs[self.dest_dir].filter_gpu_ids,
                filter_dispatch_ids=self._runs[self.dest_dir].filter_dispatch_ids,
                filter_nodes=self._runs[self.dest_dir].filter_nodes,
                time_unit=args.time_unit,
                max_stat_num=args.max_stat_num,
                kernel_verbose=self.get_args().kernel_verbose,
            )
            # create the loaded kernel stats
            parser.load_kernel_top(
                self._runs[self.dest_dir], self.dest_dir, self.get_args()
            )
            # set architecture
            self.arch = self._runs[self.dest_dir].sys_info.iloc[0]["gpu_arch"]

        else:
            console_error(
                "Multiple runs not yet supported in GUI. Retry without --gui flag."
            )

    @demarcate
    def run_analysis(self):
        """Run CLI analysis."""
        super().run_analysis()
        args = self.get_args()
        input_filters = {
            "kernel": self._runs[self.dest_dir].filter_kernel_ids,
            "gpu": self._runs[self.dest_dir].filter_gpu_ids,
            "dispatch": self._runs[self.dest_dir].filter_dispatch_ids,
            "normalization": args.normal_unit,
            "top_n": args.max_stat_num,
        }

        self.build_layout(
            input_filters,
            self._arch_configs[self.arch],
        )
        if args.random_port:
            self.app.run(debug=False, host="0.0.0.0", port=random.randint(1024, 49151))
        else:
            self.app.run(debug=False, host="0.0.0.0", port=args.gui)


@demarcate
def determine_chart_type(
    original_df,
    table_config,
    hidden_columns,
    barchart_elements,
    norm_filt,
    comparable_columns,
    decimal,
):
    content = []

    display_columns = original_df.columns.values.tolist().copy()
    # Remove hidden columns. Better way to do it?
    for col in hidden_columns:
        if col in display_columns:
            display_columns.remove(col)
    display_df = original_df[display_columns]

    # Determine chart type:
    # a) Barchart
    if original_df.empty:
        console_warning(
            f"The dataframe with id={table_config['id']} is empty! Not displaying it."
        )
    elif table_config["id"] in [x for i in barchart_elements.values() for x in i]:
        d_figs = build_bar_chart(display_df, table_config, barchart_elements, norm_filt)
        # Smaller formatting if barchart yeilds several graphs
        if (
            len(d_figs)
            > 2
            # and not table_config["id"]
            # in barchart_elements["l2_cache_per_chan"]
        ):
            temp_obj = []
            for fig in d_figs:
                temp_obj.append(
                    html.Div(
                        className="float-child",
                        children=[dcc.Graph(figure=fig, style={"margin": "2%"})],
                    )
                )
            content.append(html.Div(className="float-container", children=temp_obj))
        # Normal formatting if < 2 graphs
        else:
            for fig in d_figs:
                content.append(dcc.Graph(figure=fig, style={"margin": "2%"}))
    # B) Tablechart
    else:
        d_figs = build_table_chart(
            display_df,
            table_config,
            original_df,
            display_columns,
            comparable_columns,
            decimal,
        )
        for fig in d_figs:
            content.append(html.Div([fig], style={"margin": "2%"}))

    # subtitle for each table in a panel if existing
    if "title" in table_config and table_config["title"]:
        subtitle = (
            str(table_config["id"] // 100)
            + "."
            + str(table_config["id"] % 100)
            + " "
            + table_config["title"]
            + "\n"
        )

        content.insert(
            0,
            html.H4(
                children=subtitle,
                style={"color": "white"},
            ),
        )
    return content
