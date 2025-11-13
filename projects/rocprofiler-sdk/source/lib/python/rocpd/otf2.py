#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

from .importer import RocpdImportData
from . import output_config
from . import libpyrocpd


def write_otf2(importData, config):
    return libpyrocpd.write_otf2(importData, config)


def execute(input, config=None, **kwargs):

    importData = RocpdImportData(input)

    config = (
        output_config.output_config(**kwargs)
        if config is None
        else config.update(**kwargs)
    )

    write_otf2(importData, config)


def add_args(parser):
    """Add otf2 arguments."""

    # Currently, no otf2 specific args

    # otf2_options = parser.add_argument_group("OTF2 options")

    # otf2_options.add_argument(
    #     "--kernel-rename",
    #     help="Use kernel names from debugging symbols if available",
    #     action="store_true",
    #     default=False,
    # )

    def process_args(input, args):
        valid_args = []
        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is not None:
                    ret[itr] = val
        return ret

    return process_args


def main(argv=None):
    import argparse
    from . import time_window
    from . import output_config

    parser = argparse.ArgumentParser(
        description="Convert rocPD to OTF2 format", allow_abbrev=False
    )

    required_params = parser.add_argument_group("Required arguments")

    required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s), separated by spaces",
    )

    process_out_config_args = output_config.add_args(parser)
    process_otf2_args = add_args(parser)
    process_generic_args = output_config.add_generic_args(parser)
    process_time_window_args = time_window.add_args(parser)

    args = parser.parse_args(argv)

    input = RocpdImportData(args.input)

    out_cfg_args = process_out_config_args(input, args)
    generic_out_cfg_args = process_generic_args(input, args)
    otf2_args = process_otf2_args(input, args)
    process_time_window_args(input, args)

    all_args = {**out_cfg_args, **otf2_args, **generic_out_cfg_args}

    execute(input, **all_args)


if __name__ == "__main__":
    main()
