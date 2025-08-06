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

import sys
import os
import argparse

try:
    import ctypes

    sqlite3lib = ctypes.CDLL("libsqlite3.so")
except Exception:
    pass

from . import libpyrocpd


__all__ = ["format_path", "output_config", "add_args", "process_args"]


def _generate_attribute_docs(data):
    properties = []
    for key, itr in data.items():
        if not isinstance(key, str):
            pass
        if key.startswith("__") and key.endswith("__"):
            pass
        else:
            properties += [key]
    return "\n\t- ".join(properties)


class output_config(libpyrocpd.output_config):
    __doc__ = f"""Output configuration

    Read/Write properties:\n\t- {_generate_attribute_docs(libpyrocpd.output_config.__dict__)}

    Example:
        # folder for output data
        output_dir = os.path.join(os.getcwd(), "rocpd-output")

        # create output config instance
        cfg = output_config(output_path=output_dir, output_file="out")

        # using read/write properties
        if cfg.output_path != output_dir:
            cfg.output_path = output_dir
    """

    def __init__(self, **kwargs):
        super(output_config, self).__init__()
        self.update(**kwargs)

    def update(self, **kwargs):
        _strict = kwargs.get("strict", True)
        # _verbose = kwargs.get("log-level", "config")
        for key, itr in kwargs.items():
            if hasattr(self, key):
                # if _verbose in ("info", "trace", "config"):
                #     print(f"  - output_config.{key} = {itr}")
                if key == "agent_index_value":
                    if itr == "absolute":
                        setattr(self, key, libpyrocpd.agent_indexing.node)
                    elif itr == "type-relative":
                        setattr(self, key, libpyrocpd.agent_indexing.logical_node_type)
                    else:
                        setattr(self, key, libpyrocpd.agent_indexing.logical_node)
                else:
                    setattr(self, key, itr)
            elif _strict:
                raise KeyError(f"output_config does not have {key} attribute")
        return self


def format_path(path, tag=os.path.basename(sys.executable)):
    return libpyrocpd.format_path(path, tag)


def check_file_exists(filename):
    if not os.path.exists(filename):
        raise argparse.ArgumentTypeError(f"File '{filename}' does not exist.")
    return filename


def add_args(parser):
    """Add output arguments to an existing parser."""

    io_options = parser.add_argument_group("I/O options")

    io_options.add_argument(
        "-o",
        "--output-file",
        help="Sets the base output file name (default base filename: `out`)",
        default=os.environ.get("ROCPD_OUTPUT_NAME", "out"),
        type=str,
        required=False,
    )
    io_options.add_argument(
        "-d",
        "--output-path",
        help="Sets the output path where the output files will be saved (default path: `./rocpd-output-data`)",
        default=os.environ.get("ROCPD_OUTPUT_PATH", "./rocpd-output-data"),
        type=str,
        required=False,
    )

    kernel_naming_options = parser.add_argument_group("Kernel naming options")

    kernel_naming_options.add_argument(
        "--kernel-rename",
        help="Use ROCTx marker names instead of kernel names",
        action="store_true",
        default=False,
    )

    return ["output_file", "output_path", "kernel_rename"]


def process_args(args, valid_args):

    ret = {}
    for itr in valid_args:
        if hasattr(args, itr):
            val = getattr(args, itr)
            if itr == "output_format":
                ret[itr] = val
            elif itr == "output_path" and val is not None:
                ret[itr] = format_path(val)
            elif val is not None:
                ret[itr] = val
    return ret


def add_generic_args(parser):
    """Add generic arguments that apply to multiple output formats."""

    generic_options = parser.add_argument_group("Generic options")

    generic_options.add_argument(
        "--agent-index-value",
        choices=("absolute", "relative", "type-relative"),
        help="""Device identification format in CSV/Perfetto/OTF2 output (default: relative):
        absolute: uses node_id (Agent-0, Agent-2, Agent-4) ignoring cgroups restrictions.
        relative: uses logical_node_id (Agent-0, Agent-1, Agent-2) considering cgroups restrictions.
        type-relative: uses logical_node_type_id (CPU-0, GPU-0, GPU-1) with numbering that resets for each device type.""",
        default="relative",
    )

    return [
        "agent_index_value",
    ]


def process_generic_args(args, valid_args):
    ret = {}
    for itr in valid_args:
        if hasattr(args, itr):
            val = getattr(args, itr)
            if val is not None:
                ret[itr] = val
    return ret
