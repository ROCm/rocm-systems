#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

from __future__ import absolute_import

__author__ = "Advanced Micro Devices, Inc."
__copyright__ = "Copyright 2025, Advanced Micro Devices, Inc."
__license__ = "MIT"


def main(argv=None, config=None):
    """Main entry point for the rocpd command line tool.

    Args:
        argv (list, optional): List of command line arguments. Defaults to None.

    """
    import argparse
    from . import time_window
    from . import output_config
    from . import pftrace
    from . import csv
    from . import otf2
    from .importer import RocpdImportData

    convert_examples = """

Example usage:

    Convert 1 database, output perfetto trace
    $ python3 -m rocpd convert -i db1.db --output-format pftrace

    Convert 2 databases, output perfetto trace to path and filename, reduce time window to omit the first 30%
    $ python3 -m rocpd convert -i db1.db db2.db --output-format pftrace -d "./output/" -o "twoFileTraces" --start 30% --end 100%

    Convert 6 databases, output CSV and perfetto trace formats
    $ python3 -m rocpd convert -i db{0..5}.db --output-format csv pftrace -d "~/output_folder/" -o "sixFileTraces"

    Convert 2 databases, output CSV, OTF2, and perfetto trace formats
    $ python3 -m rocpd convert -i db{3,4}.db --output-format csv otf2 pftrace

"""

    parser = argparse.ArgumentParser(
        prog="rocpd",
        description="Aggregate and/or analyze ROCm Profiling Data (rocpd)",
        allow_abbrev=False,
    )

    subparsers = parser.add_subparsers(dest="command")
    converter = subparsers.add_parser(
        "convert",
        description="Convert rocPD data into another data format",
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=convert_examples,
    )

    def get_output_type(val):
        return val.lower().replace("perfetto", "pftrace")

    required_params = converter.add_argument_group("Required arguments")

    required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s), separated by spaces",
    )
    required_params.add_argument(
        "-f",
        "--output-format",
        help="For adding output format (supported formats: csv, pftrace, otf2)",
        nargs="+",
        default=None,
        choices=("csv", "pftrace", "otf2"),
        type=get_output_type,
        required=True,
    )

    # add args from any sub-modules
    valid_out_config_args = output_config.add_args(converter)
    valid_generic_args = output_config.add_generic_args(converter)
    valid_pftrace_args = pftrace.add_args(converter)
    valid_csv_args = csv.add_args(converter)
    valid_otf2_args = otf2.add_args(converter)
    valid_time_window_args = time_window.add_args(converter)

    # parse the command line arguments
    args = parser.parse_args(argv)

    # process the args
    out_cfg_args = output_config.process_args(args, valid_out_config_args)
    generic_out_cfg_args = output_config.process_generic_args(args, valid_generic_args)
    pftrace_args = pftrace.process_args(args, valid_pftrace_args)
    csv_args = csv.process_args(args, valid_csv_args)
    otf2_args = otf2.process_args(args, valid_otf2_args)
    window_args = time_window.process_args(args, valid_time_window_args)

    # now start processing the data.  Import the data and merge the views
    importData = RocpdImportData(args.input)

    # adjust the time window view of the data
    if window_args is not None:
        time_window.apply_time_window(importData, **window_args)

    all_args = {
        **out_cfg_args,
        **generic_out_cfg_args,
        **pftrace_args,
        **csv_args,
        **otf2_args,
    }
    # setup the config args
    config = (
        output_config.output_config(**all_args)
        if config is None
        else config.update(**all_args)
    )

    # process each requested output format
    format_handlers = {
        "pftrace": pftrace.write_pftrace,
        "csv": csv.write_csv,
        "otf2": otf2.write_otf2,
    }

    for out_format in args.output_format:
        if out_format in format_handlers:
            print(f"Converting database(s) to {out_format} format:")
            format_handlers[out_format](importData, config)
        else:
            print(f"Warning: Unsupported output format '{out_format}'")

    print("Done. Exiting...")


if __name__ == "__main__":
    main()
