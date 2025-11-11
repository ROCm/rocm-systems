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
    from . import package
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

    Merge the three databases and output to a folder called merged3DBs
    $ rocpd merge -i db0.db db1.db db2.db -d merged3DBs

    Merge all the databases from the node0 folder and output to the node0_output folder, with filename called largeMerged.db
    $ rocpd merge -i node0/*.db -d node0_output -o largeMerged
"""

    package_examples = """

Example usage:

    Index the three databases into a metadata file (index.yaml) in the current folder, just reference the databases where they are on the filesystem
    $ rocpd package -i node0/db0.db node1/db1.db node2/db2.db

    Package and copy/consolidate all the databases into a my_MPI_run_1.rpdb folder so it can be managed easier
    $ rocpd package -i node0/db0.db node1/db1.db node2/db2.db -d my_MPI_run_1 --consolidate --copy

    Package and copy/consolidate all the databases from my_MPI_run_1.rpdb folder append node5/db5.db and make new folder
    $ rocpd package -i my_MPI_run_1.rpdb node5/db5.db -d my_MPI_run_1_append_5 --consolidate --copy

    Use my_MPI_run_1.rpdb folder and move/consolidate node7/db7.db and re-use same .rpdb folder
    $ rocpd package -i my_MPI_run_1.rpdb node7/db7.db -d my_MPI_run_1 --consolidate
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
    input_help_string = "Input path and filename to one or more database(s). Wildcards accepted, as well as .rpdb folders"

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

    def add_required_args(_parser):
        _required_params = _parser.add_argument_group("Required options")
        _required_params.add_argument(
            "-i",
            "--input",
            required=True,
            type=output_config.check_file_exists,
            nargs="+",
            help=input_help_string,
        )
        return _required_params

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

    packager = subparsers.add_parser(
        "package",
        description="Package database files into .rpdb output",
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=package_examples,
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
    converter_required_params = add_required_args(converter)
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

    add_required_args(merger)
    add_required_args(packager)
    add_required_args(query_reporter)
    add_required_args(generate_summary)

    # converter: add args from any sub-modules
    process_converter_args = []
    process_converter_args.append(output_config.add_args(converter))
    process_converter_args.append(output_config.add_generic_args(converter))
    process_converter_args.append(pftrace.add_args(converter))
    process_converter_args.append(csv.add_args(converter))
    process_converter_args.append(otf2.add_args(converter))
    process_converter_args.append(time_window.add_args(converter))

    # merge: subparser args
    process_merger_args = []
    process_merger_args.append(merge.add_args(merger))

    # package: subparser args
    process_packager_args = []
    process_packager_args.append(package.add_args(packager))

    # query: subparser args
    process_query_reporter_args = []
    process_query_reporter_args.append(output_config.add_args(query_reporter))
    process_query_reporter_args.append(query.add_args(query_reporter))
    process_query_reporter_args.append(time_window.add_args(query_reporter))

    # summary: subparser args
    process_generate_summary_args = []
    process_generate_summary_args.append(output_config.add_args(generate_summary))
    process_generate_summary_args.append(summary.add_args(generate_summary))
    process_generate_summary_args.append(time_window.add_args(generate_summary))

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
        # construct the rocpd import data object
        input = RocpdImportData(
            args.input,
            automerge_limit=getattr(
                args, "automerge_limit", package.IDEAL_NUMBER_OF_DATABASE_FILES
            ),
        )

        all_args = {}
        for pitr in process_converter_args:
            all_args.update(pitr(input, args))

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
                format_handlers[out_format](input, config)
            else:
                print(f"Warning: Unsupported output format '{out_format}'")

    # if the user requested merge module, execute the merge
    elif args.command == "merge":
        # no construction of the import data object
        input = None

        # merge subparser args
        merge_args = {}
        for pitr in process_merger_args:
            merge_args.update(pitr(input, args))

        merge.execute(args.input, **merge_args)

    # if the user requested package module, package up the database
    elif args.command == "package":
        # construct the rocpd import data object
        input = None

        # package subparser args
        packager_args = {}
        for pitr in process_packager_args:
            packager_args.update(pitr(input, args))

        package.execute(args.input, **packager_args)

    # if the user requested query module, execute the query
    elif args.command == "query":
        # construct the rocpd import data object
        input = RocpdImportData(
            args.input,
            automerge_limit=getattr(
                args, "automerge_limit", package.IDEAL_NUMBER_OF_DATABASE_FILES
            ),
        )

        # query subparser args
        query_args = {}
        for pitr in process_query_reporter_args:
            query_args.update(pitr(input, args))

        query.execute(
            input,
            args,
            **query_args,
        )

    # if the user requested a summary, generate the views
    elif args.command == "summary":
        # construct the rocpd import data object
        input = RocpdImportData(
            args.input,
            automerge_limit=getattr(
                args, "automerge_limit", package.IDEAL_NUMBER_OF_DATABASE_FILES
            ),
        )

        # summary subparser args
        summary_args = {}
        for pitr in process_generate_summary_args:
            summary_args.update(pitr(input, args))

        summary.generate_all_summaries(input, **summary_args)

    print("Done. Exiting...")


if __name__ == "__main__":
    main()
