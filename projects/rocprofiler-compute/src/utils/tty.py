##############################################################################bl
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

import copy
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from utils import mem_chart, parser
from utils.logger import console_log, console_warning
from utils.utils import convert_metric_id_to_panel_idx

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


def get_table_string(df, transpose=False, decimal=2):
    return tabulate(
        df.transpose() if transpose else df,
        headers="keys",
        tablefmt="fancy_grid",
        floatfmt="." + str(decimal) + "f",
    )


def show_all(args, runs, archConfigs, output, profiling_config):
    """
    Show all panels with their data in plain text mode.
    """
    comparable_columns = parser.build_comparable_columns(args.time_unit)
    filter_panel_ids = [
        convert_metric_id_to_panel_idx(section)
        for section in [
            name
            for name, type in profiling_config.get("filter_blocks", {}).items()
            if type == "metric_id"
        ]
    ]
    comparable_columns = parser.build_comparable_columns(args.time_unit)

    for panel_id, panel in archConfigs.panel_configs.items():
        # Skip panels that don't support baseline comparison
        if panel_id in hidden_sections:
            continue
        ss = ""  # store content of all data_source from one pannel

        for data_source in panel["data source"]:
            for type, table_config in data_source.items():
                # If block filtering was used during analysis, then dont use profiling config
                # If block filtering was used in profiling config, only show those panels
                # If block filtering not used in profiling config, show all panels
                # Skip this table if table id or panel id is not present in block filters
                # However, always show panel id <= 100
                if (
                    not args.filter_metrics
                    and filter_panel_ids
                    and table_config["id"] not in filter_panel_ids
                    and panel_id not in filter_panel_ids
                    and panel_id > 100
                ):
                    table_id_str = (
                        str(table_config["id"] // 100)
                        + "."
                        + str(table_config["id"] % 100)
                    )
                    console_log(
                        f"Not showing table not selected during profiling: {table_id_str} {table_config['title']}"
                    )
                    continue
                # take the 1st run as baseline
                base_run, base_data = next(iter(runs.items()))
                base_df = base_data.dfs[table_config["id"]]

                df = pd.DataFrame(index=base_df.index)

                for header in list(base_df.keys()):
                    if (
                        (not args.cols)
                        or (args.cols and base_df.columns.get_loc(header) in args.cols)
                        or (type == "raw_csv_table")
                    ):
                        if header in hidden_columns:
                            pass
                        elif header not in comparable_columns:
                            if (
                                type == "raw_csv_table"
                                and (
                                    table_config["source"] == "pmc_kernel_top.csv"
                                    or table_config["source"] == "pmc_dispatch_info.csv"
                                )
                                and header == "Kernel_Name"
                            ):
                                # NB: the width of kernel name might depend on the header of the table.
                                if table_config["source"] == "pmc_kernel_top.csv":
                                    adjusted_name = base_df["Kernel_Name"].apply(
                                        lambda x: string_multiple_lines(x, 40, 3)
                                    )
                                else:
                                    adjusted_name = base_df["Kernel_Name"].apply(
                                        lambda x: string_multiple_lines(x, 80, 4)
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
                                        t_df = pd.concat(
                                            [
                                                base_df[header],
                                                cur_df[header],
                                            ],
                                            axis=1,
                                        )
                                        absolute_diff = (
                                            t_df.iloc[:, 1] - t_df.iloc[:, 0]
                                        ).round(args.decimal)
                                        t_df = absolute_diff / t_df.iloc[:, 0].replace(
                                            0, 1
                                        )
                                        if args.verbose >= 2:
                                            console_log("---------", header, t_df)

                                        t_df_pretty = (
                                            t_df.astype(float)
                                            .mul(100)
                                            .round(args.decimal)
                                        )
                                        # show value + percentage
                                        # TODO: better alignment
                                        t_df = (
                                            cur_df[header]
                                            .astype(float)
                                            .round(args.decimal)
                                            .map(str)
                                            .astype(str)
                                            + " ("
                                            + t_df_pretty.map(str)
                                            + "%)"
                                        )
                                        df = pd.concat([df, t_df], axis=1)

                                        # DEBUG: When in a CI setting and flag is set,
                                        #       then verify metrics meet threshold requirement
                                        if (
                                            header in ["Value", "Count", "Avg"]
                                            and t_df_pretty.abs()
                                            .gt(args.report_diff)
                                            .any()
                                        ):
                                            df["Abs Diff"] = absolute_diff
                                            if args.report_diff:
                                                violation_idx = t_df_pretty.index[
                                                    t_df_pretty.abs() > args.report_diff
                                                ]
                                                console_warning(
                                                    "Dataframe diff exceeds %s threshold requirement\nSee metric %s"
                                                    % (
                                                        str(args.report_diff) + "%",
                                                        violation_idx.to_numpy(),
                                                    )
                                                )
                                                console_warning(df)

                                    else:
                                        cur_df_copy = copy.deepcopy(cur_df)
                                        cur_df_copy[header] = [
                                            (
                                                round(float(x), args.decimal)
                                                if x != ""
                                                else x
                                            )
                                            for x in base_df[header]
                                        ]
                                        df = pd.concat([df, cur_df_copy[header]], axis=1)

                if not df.empty:
                    # subtitle for each table in a panel if existing
                    table_id_str = (
                        str(table_config["id"] // 100)
                        + "."
                        + str(table_config["id"] % 100)
                    )

                    # Check if any column in df is empty
                    is_empty_columns_exist = any(
                        [
                            df.columns[col_idx]
                            for col_idx in range(len(df.columns))
                            if df.replace("", None).iloc[:, col_idx].isnull().all()
                        ]
                    )
                    # Do not print the table if any column is empty
                    if is_empty_columns_exist:
                        if "title" in table_config:
                            console_log(
                                f"Not showing table with empty column(s): {table_id_str} {table_config['title']}"
                            )
                        else:
                            console_log(
                                f"Not showing table with empty column(s): {table_id_str}"
                            )
                    if (
                        "title" in table_config
                        and table_config["title"]
                        and not is_empty_columns_exist
                    ):
                        ss += table_id_str + " " + table_config["title"] + "\n"

                    if args.df_file_dir:
                        p = Path(args.df_file_dir)
                        if not p.exists():
                            p.mkdir()
                        if p.is_dir():
                            if "title" in table_config and table_config["title"]:
                                table_id_str += "_" + table_config["title"]
                            df.to_csv(
                                p.joinpath(table_id_str.replace(" ", "_") + ".csv"),
                                index=False,
                            )
                    # Only show top N kernels (as specified in --max-kernel-num) in "Top Stats" section
                    if type == "raw_csv_table" and (
                        table_config["source"] == "pmc_kernel_top.csv"
                        or table_config["source"] == "pmc_dispatch_info.csv"
                    ):
                        df = df.head(args.max_stat_num)
                    # NB:
                    # "columnwise: True" is a special attr of a table/df
                    # For raw_csv_table, such as system_info, we transpose the
                    # df when load it, because we need those items in column.
                    # For metric_table, we only need to show the data in column
                    # fash for now.
                    transpose = (
                        type != "raw_csv_table"
                        and "columnwise" in table_config
                        and table_config["columnwise"] == True
                    )
                    if not is_empty_columns_exist:

                        # enable mem_chart only with single run
                        if (
                            "cli_style" in table_config
                            and table_config["cli_style"] == "mem_chart"
                            and len(runs) == 1
                        ):
                            # NB: to avoid broken test with arbitrary number with "--cols" option
                            if "Metric" in df.columns and "Value" in df.columns:
                                ss += mem_chart.plot_mem_chart(
                                    "",
                                    args.normal_unit,
                                    pd.DataFrame([df["Metric"], df["Value"]])
                                    .transpose()
                                    .set_index("Metric")
                                    .to_dict()["Value"],
                                )
                        else:
                            ss += (
                                get_table_string(
                                    df, transpose=transpose, decimal=args.decimal
                                )
                                + "\n"
                            )

        if ss:
            print("\n" + "-" * 80, file=output)
            print(str(panel_id // 100) + ". " + panel["title"], file=output)
            print(ss, file=output)


def show_kernel_stats(args, runs, archConfigs, output):
    """
    Show the kernels and dispatches from "Top Stats" section.
    """

    df = pd.DataFrame()
    for panel_id, panel in archConfigs.panel_configs.items():
        for data_source in panel["data source"]:
            for type, table_config in data_source.items():
                for run, data in runs.items():
                    df = pd.DataFrame()
                    single_df = data.dfs[table_config["id"]]
                    # NB:
                    #   For pmc_kernel_top.csv, have to sort here if not
                    #   sorted when load_table_data.
                    if table_config["id"] == 1:
                        print("\n" + "-" * 80, file=output)
                        print(
                            "Detected Kernels (sorted descending by duration)",
                            file=output,
                        )
                        df = pd.concat([df, single_df["Kernel_Name"]], axis=1)

                    if table_config["id"] == 2:
                        print("\n" + "-" * 80, file=output)
                        print("Dispatch list", file=output)
                        df = single_df

                    print(
                        get_table_string(df, transpose=False, decimal=args.decimal),
                        file=output,
                    )
