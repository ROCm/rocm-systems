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
    from . import csv
    from . import merge
    from . import otf2
    from . import output_config
    from . import pftrace
    from . import query
    from . import summary
    from . import time_window
    from . import version_info
    from .importer import RocpdImportData

    convert_examples = """

Example usage:

    Convert 1 database, output perfetto trace
    $ rocpd convert -i db1.db --output-format pftrace

    Convert 2 databases, output perfetto trace to path and filename, reduce time window to omit the first 30%
    $ rocpd convert -i db1.db db2.db --output-format pftrace -d "./output/" -o "twoFileTraces" --start 30% --end 100%

    Convert 6 databases, output CSV and perfetto trace formats
    $ rocpd convert -i db{0..5}.db --output-format csv pftrace -d "~/output_folder/" -o "sixFileTraces"

    Convert 2 databases, output CSV, OTF2, and perfetto trace formats
    $ rocpd convert -i db{3,4}.db --output-format csv otf2 pftrace

"""

    merge_examples = """

Example usage:

    TODO: Add examples for merge command
"""

    query_examples = """

Example usage:

    Query the first 5 rows of the 'rocpd_info_agents' view and output to console
    $ rocpd query -i db0.db --query "SELECT * FROM rocpd_info_agents LIMIT 5"

    Combine 4 databases and query the first 10 rows of the 'top_kernels' view and output to CSV file
    $ rocpd query -i db{0..3}.db --query "SELECT * FROM top_kernels LIMIT 10" --format csv
"""

    summary_examples = """

Example usage:

    Output all summaries to console and include domain summary for 1 database
    $ rocpd summary -i db1.db --domain-summary

    Aggregate 3 databases and output all summary files and include summary by rank/process ID, to csv file output
    $ rocpd summary -i db{1..3}.db --summary-by-rank --format csv

    Output all summaries to console and exlude all regions to save processing time
    $ rocpd summary -i db0.db --region-categories NONE

    Aggregate 2 databases and output all summary files to HTML, only include HIP and MARKER regions, include domain summary
    $ rocpd summary -i db{0,1}.db --region-categories HIP MARKERS --domain-summary --format html

"""

    # Add the subparsers
    parser = argparse.ArgumentParser(
        prog="rocpd",
        description="Aggregate and/or analyze ROCm Profiling Data (rocpd)",
        allow_abbrev=False,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Print the version information and exit",
    )

    subparsers = parser.add_subparsers(dest="command")
    converter = subparsers.add_parser(
        "convert",
        description="Convert rocPD data into another data format",
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=convert_examples,
    )

    merger = subparsers.add_parser(
        "merge",
        description="Generate merged database from rocPD databases",
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=merge_examples,
    )

    query_reporter = subparsers.add_parser(
        "query",
        description="Generate output on a query",
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=query_examples,
    )

    generate_summary = subparsers.add_parser(
        "summary",
        description="Generate summary views from rocPD data",
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=summary_examples,
    )

    def get_output_type(val):
        return val.lower().replace("perfetto", "pftrace")

    # add required options for each subparser
    converter_required_params = converter.add_argument_group("Required options")
    converter_required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s)",
    )
    converter_required_params.add_argument(
        "-f",
        "--output-format",
        help="For adding output format (supported formats: csv, pftrace, otf2)",
        nargs="+",
        default=None,
        choices=("csv", "pftrace", "otf2"),
        type=get_output_type,
        required=True,
    )

    merger_required_params = merger.add_argument_group("Required options")
    merger_required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s)",
    )

    query_required_params = query_reporter.add_argument_group("Required options")
    query_required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s)",
    )

    summary_required_params = generate_summary.add_argument_group("Required options")
    summary_required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s)",
    )

    # converter: add args from any sub-modules
    valid_out_config_args = output_config.add_args(converter)
    valid_generic_args = output_config.add_generic_args(converter)
    valid_pftrace_args = pftrace.add_args(converter)
    valid_csv_args = csv.add_args(converter)
    valid_otf2_args = otf2.add_args(converter)
    valid_time_window_args = time_window.add_args(converter)

    # merge: subparser args
    valid_merge_args = merge.add_args(merger)

    # query: subparser args
    valid_out_config_args = output_config.add_args(query_reporter)
    valid_query_args = query.add_args(query_reporter)
    valid_time_window_args = time_window.add_args(query_reporter)

    # summary: subparser args
    valid_io_args = summary.add_io_args(generate_summary)
    valid_summary_args = summary.add_args(generate_summary)
    valid_time_window_args = time_window.add_args(generate_summary)

    # parse the command line arguments
    args = parser.parse_args(argv)

    if args.version:
        for key, itr in version_info.items():
            if key in ["major", "minor", "patch"]:
                continue
            print(f"    {key:>16}: {itr}")
        return 0

    # error check the command line arguments, if no subparser command is given, print the help message
    if args.command is None:
        parser.print_help()
        return

    # if the user requested converter, process the conversion
    if args.command == "convert":
        # process the args
        out_cfg_args = output_config.process_args(args, valid_out_config_args)
        generic_out_cfg_args = output_config.process_generic_args(
            args, valid_generic_args
        )
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

    # if the user requested merge module, execute the merge
    elif args.command == "merge":
        # merge subparser args
        merge_args = merge.process_args(args, valid_merge_args)
        merge.execute(args.input, **merge_args)

    # if the user requested query module, execute the query
    elif args.command == "query":
        # query subparser args
        query_args = query.process_args(args, valid_query_args)
        out_cfg_args = output_config.process_args(args, valid_out_config_args)
        window_args = time_window.process_args(args, valid_time_window_args)

        all_args = {**query_args, **out_cfg_args}

        query.execute(
            args.input,
            args,
            window_args=window_args,
            **all_args,
        )

    # if the user requested a summary, generate the views
    elif args.command == "summary":
        # summary subparser args
        summary_args = summary.process_args(args, valid_summary_args)
        io_args = output_config.process_args(args, valid_io_args)
        window_args = time_window.process_args(args, valid_time_window_args)

        # now start processing the data.  Import the data and merge the views
        importData = RocpdImportData(args.input)

        # adjust the time window view of the data
        if window_args is not None:
            time_window.apply_time_window(importData, **window_args)

        all_args = {**summary_args, **io_args}
        summary.generate_all_summaries(importData, **all_args)

    print("Done. Exiting...")


if __name__ == "__main__":
    main()
