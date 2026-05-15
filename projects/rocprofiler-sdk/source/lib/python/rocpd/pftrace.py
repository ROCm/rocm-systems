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


def write_pftrace(importData, config):
    return libpyrocpd.write_perfetto(importData, config)


def execute(input, config=None, **kwargs):

    config = (
        output_config.output_config(**kwargs)
        if config is None
        else config.update(**kwargs)
    )

    write_pftrace(input, config)


def add_args(parser):
    """Add arguments for pftrace."""

    pftrace_options = parser.add_argument_group("Perfetto trace (pftrace) options")

    pftrace_options.add_argument(
        "--perfetto-backend",
        help="Perfetto data collection backend. 'system' mode requires starting traced and perfetto daemons (default: inprocess)",
        default="inprocess",
        choices=["inprocess", "system"],
    )

    pftrace_options.add_argument(
        "--perfetto-buffer-fill-policy",
        help="Policy for handling new records when perfetto has reached the buffer limit (default: discard)",
        default="discard",
        choices=["discard", "ring_buffer"],
    )

    pftrace_options.add_argument(
        "--perfetto-buffer-size",
        help="Size of buffer for perfetto output in KB (default: 1 GB)",
        default=None,
        type=int,
        metavar="KB",
    )

    pftrace_options.add_argument(
        "--perfetto-shmem-size-hint",
        help="Perfetto shared memory size hint in KB (default: 64 KB)",
        default=None,
        type=int,
        metavar="KB",
    )

    pftrace_options.add_argument(
        "--group-by-queue",
        help="For displaying the HIP streams that kernels and memory copy operations are submitted to rather than HSA queues",
        action="store_true",
        default=False,
    )

    def process_args(input, args):
        valid_args = [
            "perfetto_backend",
            "perfetto_buffer_fill_policy",
            "perfetto_buffer_size",
            "perfetto_shmem_size_hint",
            "group_by_queue",
        ]
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
    from .time_window import add_args as add_args_time_window
    from .output_config import add_args as add_args_output_config
    from .output_config import add_generic_args

    parser = argparse.ArgumentParser(
        description="Convert rocPD to Perfetto file", allow_abbrev=False
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

    process_out_config_args = add_args_output_config(parser)
    process_pftrace_args = add_args(parser)
    process_generic_args = add_generic_args(parser)
    process_time_window_args = add_args_time_window(parser)

    args = parser.parse_args(argv)
    input = RocpdImportData(
        args.input, automerge_limit=getattr(args, "automerge_limit", None)
    )

    out_cfg_args = process_out_config_args(input, args)
    pftrace_args = process_pftrace_args(input, args)
    generic_out_cfg_args = process_generic_args(input, args)
    process_time_window_args(input, args)

    all_args = {
        **pftrace_args,
        **out_cfg_args,
        **generic_out_cfg_args,
    }

    execute(
        input,
        **all_args,
    )


if __name__ == "__main__":
    main()
