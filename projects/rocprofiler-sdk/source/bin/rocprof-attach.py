#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import argparse
import ctypes
import os
import signal
import sys
import time

ROCPROF_ATTACH_DIR = os.path.dirname(os.path.realpath(__file__))
ROCM_DIR = os.path.dirname(ROCPROF_ATTACH_DIR)
ROCPROF_ATTACH_LIBRARY = f"{ROCM_DIR}/lib/librocprofiler-sdk-rocattach.so"


def parse_arguments(args=None):

    def format_help(formatter, w=120, h=40):
        """Return a wider HelpFormatter, if possible."""
        try:
            kwargs = {"width": w, "max_help_position": h}
            formatter(None, **kwargs)
            return lambda prog: formatter(prog, **kwargs)
        except TypeError:
            return formatter

    usage_examples = """

%(prog)s, e.g.

    $ rocprof-attach -p <pid> -t <tool library> [-a <attach tool library> -d <msec duration>]
    $ rocprof-attach -p 12345 -t path/to/your-tool-library.so -d 5000

"""
    parser = argparse.ArgumentParser(
        description="rocprofiler-sdk attachment profiler for custom tool libraries ",
        usage="%(prog)s [options] ",
        epilog=usage_examples,
        formatter_class=format_help(argparse.RawTextHelpFormatter),
    )

    parser.add_argument(
        "-p",
        "--pid",
        "--attach",
        help="""Attachment target's process identifier (PID).
  Can also be specified in environment variable ROCPROF_ATTACH_PID. This option overrides the environment variable if both are set.""",
        type=int,
        required=False,
        default=os.environ.get("ROCPROF_ATTACH_PID", None),
    )

    parser.add_argument(
        "-t",
        "--attach-tool-library",
        help="""Comma delimited list of tool libraries to use during attachment.
  Can also be specified in environment variable ROCPROF_ATTACH_TOOL_LIBRARY. This option overrides the environment variable if both are set.""",
        type=str,
        required=False,
        default=os.environ.get("ROCPROF_ATTACH_TOOL_LIBRARY", None),
    )

    parser.add_argument(
        "-d",
        "--attach-duration-msec",
        help="""Sets the amount of time in milliseconds the profiler will be attached before detaching. When unset, the profiler will wait until Enter is pressed or SIGINT (Ctrl+C) to detach.
  Can also be specified in environment variable ROCPROF_ATTACH_DURATION. This option overrides the environment variable if both are set.""",
        type=int,
        required=False,
        default=os.environ.get("ROCPROF_ATTACH_DURATION", None),
    )

    advanced_options = parser.add_argument_group("Advanced options")

    advanced_options.add_argument(
        "--attach-library",
        help="""Library used to attach and detach from the target process. Default will work for nearly all configurations.
  Defaults to rocprofiler-sdk-rocattach.so from this ROCm install, i.e. <ROCmdirectory>/lib/rocprofiler-sdk-rocattach.so
  Can also be specified in environment variable ROCPROF_ATTACH_LIBRARY. This option overrides the environment variable if both are set.""",
        type=str,
        required=False,
        default=os.environ.get("ROCPROF_ATTACH_LIBRARY", ROCPROF_ATTACH_LIBRARY),
    )
    return parser.parse_args(args)


def attach(
    pid,
    attach_tool_library,
    attach_duration_msec,
    attach_library=ROCPROF_ATTACH_LIBRARY,
):

    if pid is None:
        raise RuntimeError("rocprof-attach called with no PID specified")
    if attach_tool_library is None:
        raise RuntimeError("rocprof-attach called with no tool libraries specified")

    tool_libraries_tokens = attach_tool_library.split(":")
    for lib in tool_libraries_tokens:
        if not os.path.exists(lib):
            raise RuntimeError(f"rocprof-attach could not find tool library {lib}")

    # Program option overrides environment variable. This is consumed by rocprofiler-sdk on the target program side.
    os.environ["ROCPROF_ATTACH_TOOL_LIBRARY"] = attach_tool_library

    print(f"Attaching to PID {pid} using library {attach_library}")

    # Load the shared library into ctypes and attach
    try:
        c_lib = ctypes.CDLL(attach_library)
        c_lib.rocattach_attach.restype = ctypes.c_int
        c_lib.rocattach_attach.argtypes = [ctypes.c_int]
        c_lib.rocattach_detach.restype = ctypes.c_int
        c_lib.rocattach_detach.argtypes = [ctypes.c_int]
        attach_status = c_lib.rocattach_attach(pid)
    except Exception as e:
        raise RuntimeError(f"Exception during library load and attachment: {e}")

    if attach_status != 0:
        raise RuntimeError(
            f"Calling attach in {attach_library} returned non-zero status {attach_status}"
        )

    print(f"Attaching to PID {pid} using library {attach_library} :: success")

    def detach():
        print("Detaching. Please wait, this can take up to 1-2 minutes")
        sys.stdout.flush()
        try:
            detach_status = c_lib.rocattach_detach(int(pid))
        except Exception as e:
            print(f"Exception during detachment: {e}")

        if detach_status != 0:
            print(
                f"Calling detach in {attach_library} returned non-zero status {detach_status}"
            )
        else:
            print(f"Detaching from PID {pid} using library {attach_library} :: success")

    def signal_handler(sig, frame):
        print("\nCaught signal SIGINT")
        detach()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if attach_duration_msec is None:
        sys.stdout.write("Press Enter to detach...")
        sys.stdout.flush()  # Force the prompt to appear immediately
        input()  # Now wait for input
    else:
        print(f"Attaching for {attach_duration_msec} msec...\n")
        sys.stdout.flush()
        time.sleep(int(attach_duration_msec) / 1000)

    detach()


def main(cmd_args=None):
    args = parse_arguments(cmd_args)

    attach(
        pid=args.pid,
        attach_tool_library=args.attach_tool_library,
        attach_duration_msec=args.attach_duration_msec,
        attach_library=args.attach_library,
    )
    return 0


if __name__ == "__main__":
    ec = main(sys.argv[1:])
    sys.exit(ec)
