#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import pandas as pd
import argparse
import logging


def read_csv(file_path):
    df = pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.info(f"Error reading {file_path}: {e}")
        raise
    return df


def get_counter_collection_files(root_path):
    file_paths = []
    for root, _, files in os.walk(root_path):
        if "pmc_" in root:
            for file in files:
                if file.endswith("counter_collection.csv"):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
    return file_paths


def get_combined_df(args):
    files_list = []
    for input in args.input:
        if os.path.isfile(input):
            files_list.append(input)
        elif os.path.isdir(input):
            files_list.extend(get_counter_collection_files(input))
    if not files_list:
        raise ValueError("Valid Input files not found")
    logging.info(f"Processing files: {files_list}")
    combined_df = pd.DataFrame()
    for file in files_list:
        combined_df = pd.concat([combined_df, read_csv(file)], ignore_index=True)
    return combined_df


def write_to_file(df, args):
    logging.info(f"Saving output file to : {args.output}")
    directory, file_path = os.path.split(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if not args.retain_agent_prefix:
        df["Agent_Id"] = df["Agent_Id"].apply(lambda x: x.split(" ")[-1])

    df.to_csv(args.output, index=False)


def main(args):
    logging.basicConfig(level=args.loglevel)
    input_df = get_combined_df(args)
    # Validate
    columns = [
        "Correlation_Id",
        "Dispatch_Id",
        "Agent_Id",
        "Queue_Id",
        "Process_Id",
        "Thread_Id",
        "Grid_Size",
        "Kernel_Id",
        "Kernel_Name",
        "Workgroup_Size",
        "LDS_Block_Size",
        "Scratch_Size",
        "VGPR_Count",
        "SGPR_Count",
        "Counter_Name",
        "Counter_Value",
        "Start_Timestamp",
        "End_Timestamp",
    ]
    for col in input_df.columns:
        if col not in columns:
            logging.debug(f"Unexpected column {col} found in rocprofv3 input file")

    # Convert
    indexes = [
        "Dispatch_Id",
        "Agent_Id",
        "Grid_Size",
        "Kernel_Name",
        "LDS_Block_Size",
        "Queue_Id",
        "SGPR_Count",
        "Scratch_Size",
        "VGPR_Count",
        "Workgroup_Size",
    ]

    # Drop duplicate counters in multiple PMC lines
    input_df.drop_duplicates(
        subset=indexes + ["Counter_Name"], keep="first", inplace=True
    )

    pivoted_data = input_df.pivot_table(
        index=indexes, columns="Counter_Name", values="Counter_Value", aggfunc="sum"
    ).reset_index()

    # Save
    write_to_file(pivoted_data, args)


def strtobool(val):
    """Convert a string representation of truth to true or false.
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if isinstance(val, (list, tuple)):
        if len(val) > 1:
            val_type = type(val).__name__
            raise ValueError(f"invalid truth value {val} (type={val_type})")
        else:
            val = val[0]

    if isinstance(val, bool):
        return val
    elif isinstance(val, str) and val.lower() in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif isinstance(val, str) and val.lower() in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        val_type = type(val).__name__
        raise ValueError(f"invalid truth value {val} (type={val_type})")


class booleanArgAction(argparse.Action):
    def __call__(self, parser, args, value, option_string=None):
        setattr(args, self.dest, strtobool(value))


def add_parser_bool_argument(gparser, *args, **kwargs):
    gparser.add_argument(
        *args,
        **kwargs,
        action=booleanArgAction,
        nargs="?",
        const=True,
        type=str,
        required=False,
        metavar="BOOL",
        default=False,
    )


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Rocprofv3 Counter Collection input files and/or directories containing `*counter_collection.csv` files",
        nargs="+",
        default=[],
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Rocprofv1 formatted output file name",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Debug Logs",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose Logs",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    advanced_options = parser.add_argument_group("Advanced options")
    add_parser_bool_argument(
        advanced_options,
        "--retain-agent-prefix",
        help="retains the agent prefix",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
