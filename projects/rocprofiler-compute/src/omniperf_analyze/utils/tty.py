################################################################################
# Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import pandas as pd
from tabulate import tabulate

from omniperf_analyze.utils import schema, parser

hidden_columns = ["Tips", "coll_level"]
hidden_sections = [1900, 2000]


def string_multiple_lines(source, width, max_rows):
    """
    Adjust string with multiple lines by inserting '\n'
    """
    idx = 0
    lines = []
    while idx < len(source) and len(lines) < max_rows:
        lines.append(source[idx : idx + width])
        idx += width

    if idx < len(source):
        last = lines[-1]
        lines[-1] = last[0:-3] + "..."
    return "\n".join(lines)


def show_all(runs, archConfigs, output, decimal, time_unit, selected_cols, verbose):
    """
    Show all panels with their data in plain text mode.
    """
    comparable_columns = parser.build_comparable_columns(time_unit)

    for panel_id, panel in archConfigs.panel_configs.items():
        # Skip panels that don't support baseline comparison
        if panel_id in hidden_sections:
            continue
        ss = ""  # store content of all data_source from one pannel

        for data_source in panel["data source"]:
            for type, table_config in data_source.items():
                # take the 1st run as baseline
                base_run, base_data = next(iter(runs.items()))
                base_df = base_data.dfs[table_config["id"]]

                df = pd.DataFrame(index=base_df.index)

                for header in list(base_df.keys()):
                    if (
                        (not selected_cols)
                        or (
                            selected_cols
                            and base_df.columns.get_loc(header) in selected_cols
                        )
                        or (type == "raw_csv_table")
                    ):

                        if header in hidden_columns:
                            pass
                        elif header not in comparable_columns:

                            if (
                                type == "raw_csv_table"
                                and table_config["source"] == "pmc_kernel_top.csv"
                                and header == "KernelName"
                            ):
                                # NB: the width of kernel name might depend on the header of the table.
                                adjusted_name = base_df["KernelName"].apply(
                                    lambda x: string_multiple_lines(x, 40, 3)
                                )
                                df = pd.concat([df, adjusted_name], axis=1)
                            elif type == "raw_csv_table" and header == "Info":
                                for run, data in runs.items():
                                    cur_df = data.dfs[table_config["id"]]
                                    df = pd.concat([df, cur_df[header]], axis=1)
                            else:
                                df = pd.concat([df, base_df[header]], axis=1)
                        else:
                            for run, data in runs.items():
                                cur_df = data.dfs[table_config["id"]]
                                if (type == "raw_csv_table") or (
                                    type == "metric_table"
                                    and (not header in hidden_columns)
                                ):
                                    if run != base_run:
                                        # calc percentage over the baseline
                                        base_df[header] = [
                                            float(x) if x != "" else float(0)
                                            for x in base_df[header]
                                        ]
                                        cur_df[header] = [
                                            float(x) if x != "" else float(0)
                                            for x in cur_df[header]
                                        ]
                                        t_df = (
                                            pd.concat(
                                                [
                                                    base_df[header],
                                                    cur_df[header],
                                                ],
                                                axis=1,
                                            )
                                            .pct_change(axis="columns")
                                            .iloc[:, 1]
                                        )
                                        if verbose >= 2:
                                            print("---------", header, t_df)

                                        # show value + percentage
                                        # TODO: better alignment
                                        t_df = (
                                            cur_df[header]
                                            .astype(float)
                                            .round(decimal)
                                            .map(str)
                                            + " ("
                                            + t_df.astype(float)
                                            .mul(100)
                                            .round(decimal)
                                            .map(str)
                                            + "%)"
                                        )

                                        df = pd.concat([df, t_df], axis=1)
                                    else:
                                        df = pd.concat([df, cur_df[header]], axis=1)

                if not df.empty:
                    # subtitle for each table in a panel if existing
                    if "title" in table_config and table_config["title"]:
                        ss += (
                            str(table_config["id"] // 100)
                            + "."
                            + str(table_config["id"] % 100)
                            + " "
                            + table_config["title"]
                            + "\n"
                        )

                    # NB:
                    # "columnwise: True" is a special attr of a table/df
                    # For raw_csv_table, such as system_info, we transpose the
                    # df when load it, because we need those items in column.
                    # For metric_table, we only need to show the data in column
                    # fash for now.
                    ss += (
                        tabulate(
                            df.transpose()
                            if type != "raw_csv_table"
                            and "columnwise" in table_config
                            and table_config["columnwise"] == True
                            else df,
                            headers="keys",
                            tablefmt="fancy_grid",
                            floatfmt="." + str(decimal) + "f",
                        )
                        + "\n"
                    )

        if ss:
            print("\n" + "-" * 80, file=output)
            print(str(panel_id // 100) + ". " + panel["title"], file=output)
            print(ss, file=output)


def show_kernels(runs, archConfigs, output, decimal):
    """
    Show the kernels from top stats.
    """
    print("\n" + "-" * 80, file=output)
    print("Detected Kernels", file=output)

    df = pd.DataFrame()
    for panel_id, panel in archConfigs.panel_configs.items():
        for data_source in panel["data source"]:
            for type, table_config in data_source.items():
                for run, data in runs.items():
                    single_df = data.dfs[table_config["id"]]
                    # NB:
                    #   For pmc_kernel_top.csv, have to sort here if not
                    #   sorted when load_table_data.
                    df = pd.concat([df, single_df["KernelName"]], axis=1)

    print(
        tabulate(
            df, headers="keys", tablefmt="fancy_grid", floatfmt="." + str(decimal) + "f"
        ),
        file=output,
    )
