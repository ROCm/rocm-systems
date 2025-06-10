#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import sys
import argparse
import textwrap
import subprocess

# version info for rocprofiler-sdk / rocprofv3
CONST_VERSION_INFO = {
    "version": "@FULL_VERSION_STRING@",
    "git_revision": "@ROCPROFILER_SDK_GIT_REVISION@",
    "library_arch": "@CMAKE_LIBRARY_ARCHITECTURE@",
    "system_name": "@CMAKE_SYSTEM_NAME@",
    "system_processor": "@CMAKE_SYSTEM_PROCESSOR@",
    "system_version": "@CMAKE_SYSTEM_VERSION@",
    "compiler_id": "@CMAKE_CXX_COMPILER_ID@",
    "compiler_version": "@CMAKE_CXX_COMPILER_VERSION@",
    "rocm_version": "@rocm_version_FULL_VERSION@",
}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, d):
        super(dotdict, self).__init__(d)
        for k, v in self.items():
            if isinstance(v, dict):
                self.__setitem__(k, dotdict(v))
            elif isinstance(v, (list, tuple)):
                self.__setitem__(
                    k,
                    [dotdict(i) if isinstance(i, (dict)) else i for i in v],
                )


def patch_message(msg, *args):
    msg = textwrap.dedent(msg)

    if len(args) > 0:
        msg = msg.format(*args)

    return msg.strip("\n").strip()


def fatal_error(msg, *args, exit_code=1):
    msg = patch_message(msg, *args)
    sys.stderr.write(f"Fatal error: {msg}\n")
    sys.stderr.flush()
    sys.exit(exit_code)


def warning(msg, *args):
    msg = patch_message(msg, *args)
    sys.stderr.write(f"Warning: {msg}\n")
    sys.stderr.flush()


def format_help(formatter, w=120, h=40):
    """Return a wider HelpFormatter, if possible."""
    try:
        kwargs = {"width": w, "max_help_position": h}
        formatter(None, **kwargs)
        return lambda prog: formatter(prog, **kwargs)
    except TypeError:
        return formatter


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


def check_att_capability(args):

    ROCPROFV3_DIR = os.path.dirname(os.path.realpath(__file__))
    ROCM_DIR = os.path.dirname(ROCPROFV3_DIR)
    ld_library_paths = []

    if args.att_library_path:
        ld_library_paths.extend(args.att_library_path)
    else:
        for itr in os.environ.get("LD_LIBRARY_PATH", "").split(":") + [f"{ROCM_DIR}/lib"]:
            # don't add duplicates
            if itr not in ld_library_paths:
                ld_library_paths += [itr]

    lib_att_name = "librocprof-trace-decoder.so"

    for path in ld_library_paths:
        for root, dirs, files in os.walk(path, topdown=True):
            for itr in files:
                if lib_att_name in itr:
                    args.att_library_path = root
                    return True

    return False


class booleanArgAction(argparse.Action):
    def __call__(self, parser, args, value, option_string=None):
        setattr(args, self.dest, strtobool(value))


def parse_arguments(args=None):

    usage_examples = """

%(prog)s requires double-hyphen (--) before the application to be executed, e.g.

    $ rocprofv3 [<rocprofv3-option> ...] -- <application> [<application-arg> ...]
    $ rocprofv3 --hip-trace -- ./myapp -n 1

For MPI applications (or other job launchers such as SLURM), place rocprofv3 inside the job launcher:

    $ mpirun -n 4 rocprofv3 --hip-trace -- ./mympiapp

"""

    # Create the parser
    parser = argparse.ArgumentParser(
        description="ROCProfilerV3 Run Script",
        usage="%(prog)s [options] -- <application> [application options]",
        epilog=usage_examples,
        allow_abbrev=False,
        formatter_class=format_help(argparse.RawTextHelpFormatter),
    )

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
        )

    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Print the version information and exit",
    )

    io_options = parser.add_argument_group("I/O options")

    io_options.add_argument(
        "-i",
        "--input",
        help="Input file for run configuration (JSON or YAML) or counter collection (TXT)",
        required=False,
    )
    io_options.add_argument(
        "-o",
        "--output-file",
        help="For the output file name. If nothing specified default path is `%%hostname%%/%%pid%%`",
        default=os.environ.get("ROCPROF_OUTPUT_FILE_NAME", None),
        type=str,
        required=False,
    )
    io_options.add_argument(
        "-d",
        "--output-directory",
        help="For adding output path where the output files will be saved. If nothing specified default path is `%%hostname%%/%%pid%%`",
        default=os.environ.get("ROCPROF_OUTPUT_PATH", None),
        type=str,
        required=False,
    )
    io_options.add_argument(
        "-f",
        "--output-format",
        help="For adding output format (supported formats: csv, json, pftrace, otf2, rocpd)",
        nargs="+",
        default=None,
        choices=("csv", "json", "pftrace", "otf2", "rocpd"),
        type=str.lower,
    )
    add_parser_bool_argument(
        io_options,
        "--output-config",
        help="Generate a output file of the rocprofv3 configuration, e.g. out_config.json",
    )
    io_options.add_argument(
        "--log-level",
        help="Set the desired log level",
        default=None,
        choices=("fatal", "error", "warning", "info", "trace", "env", "config"),
        type=str.lower,
    )
    io_options.add_argument(
        "-E",
        "--extra-counters",
        help="Path to YAML file containing extra counter definitions",
        type=str,
        required=False,
    )

    aggregate_tracing_options = parser.add_argument_group("Aggregate tracing options")

    add_parser_bool_argument(
        aggregate_tracing_options,
        "-r",
        "--runtime-trace",
        help="Collect tracing data for HIP runtime API, Marker (ROCTx) API, RCCL API, rocDecode API, rocJPEG API, Memory operations (copies, scratch, and allocation), and Kernel dispatches. Similar to --sys-trace but without tracing HIP compiler API and the underlying HSA API.",
    )
    add_parser_bool_argument(
        aggregate_tracing_options,
        "-s",
        "--sys-trace",
        help="Collect tracing data for HIP API, HSA API, Marker (ROCTx) API, RCCL API, rocDecode API, rocJPEG API, Memory operations (copies, scratch, and allocations), and Kernel dispatches.",
    )

    basic_tracing_options = parser.add_argument_group("Basic tracing options")

    # Add the arguments
    add_parser_bool_argument(
        basic_tracing_options,
        "--hip-trace",
        help="Combination of --hip-runtime-trace and --hip-compiler-trace. This option only enables the tracing of the HIP API. Unlike previous iterations of rocprof, this does not enable kernel tracing, memory copy tracing, etc",
    )
    add_parser_bool_argument(
        basic_tracing_options,
        "--marker-trace",
        help="For collecting Marker (ROCTx) Traces. Similar to --roctx-trace option in earlier rocprof versions but with improved ROCTx library with more features",
    )
    add_parser_bool_argument(
        basic_tracing_options,
        "--kernel-trace",
        help="For collecting Kernel Dispatch Traces",
    )
    add_parser_bool_argument(
        basic_tracing_options,
        "--memory-copy-trace",
        help="For collecting Memory Copy Traces. This was part of HIP and HSA traces in previous rocprof versions but is now a separate option",
    )
    add_parser_bool_argument(
        basic_tracing_options,
        "--memory-allocation-trace",
        help="For collecting Memory Allocation Traces. Displays starting address, allocation size, and agent where allocation occurred.",
    )
    add_parser_bool_argument(
        basic_tracing_options,
        "--scratch-memory-trace",
        help="For collecting Scratch Memory operations Traces. Helps in determining scratch allocations and manage them efficiently",
    )
    add_parser_bool_argument(
        basic_tracing_options,
        "--hsa-trace",
        help="For collecting --hsa-core-trace, --hsa-amd-trace,--hsa-image-trace and --hsa-finalizer-trace. This option only enables the tracing of the HSA API. Unlike previous iterations of rocprof, this does not enable kernel tracing, memory copy tracing, etc",
    )
    add_parser_bool_argument(
        basic_tracing_options,
        "--rccl-trace",
        help="For collecting RCCL(ROCm Communication Collectives Library. Also pronounced as 'Rickle' ) Traces",
    )
    add_parser_bool_argument(
        basic_tracing_options,
        "--kokkos-trace",
        help="Enable built-in Kokkos Tools support (implies --marker-trace and --kernel-rename)",
    )
    add_parser_bool_argument(
        basic_tracing_options,
        "--rocdecode-trace",
        help="For collecting rocDecode Traces",
    )
    add_parser_bool_argument(
        basic_tracing_options,
        "--rocjpeg-trace",
        help="For collecting rocJPEG Traces",
    )

    extended_tracing_options = parser.add_argument_group("Granular tracing options")

    add_parser_bool_argument(
        extended_tracing_options,
        "--hip-runtime-trace",
        help="For collecting HIP Runtime API Traces, e.g. public HIP API functions starting with 'hip' (i.e. hipSetDevice).",
    )
    add_parser_bool_argument(
        extended_tracing_options,
        "--hip-compiler-trace",
        help="For collecting HIP Compiler generated code Traces, e.g. HIP API functions starting with '__hip' (i.e. __hipRegisterFatBinary).",
    )
    add_parser_bool_argument(
        extended_tracing_options,
        "--hsa-core-trace",
        help="For collecting HSA API Traces (core API), e.g. HSA functions prefixed with only 'hsa_' (i.e. hsa_init).",
    )
    add_parser_bool_argument(
        extended_tracing_options,
        "--hsa-amd-trace",
        help="For collecting HSA API Traces (AMD-extension API), e.g. HSA function prefixed with 'hsa_amd_' (i.e. hsa_amd_coherency_get_type).",
    )
    add_parser_bool_argument(
        extended_tracing_options,
        "--hsa-image-trace",
        help="For collecting HSA API Traces (Image-extension API), e.g. HSA functions prefixed with only 'hsa_ext_image_' (i.e. hsa_ext_image_get_capability).",
    )
    add_parser_bool_argument(
        extended_tracing_options,
        "--hsa-finalizer-trace",
        help="For collecting HSA API Traces (Finalizer-extension API), e.g. HSA functions prefixed with only 'hsa_ext_program_' (i.e. hsa_ext_program_create).",
    )

    counter_collection_options = parser.add_argument_group("Counter collection options")

    counter_collection_options.add_argument(
        "--pmc",
        help=(
            "Specify Performance Monitoring Counters to collect(comma OR space separated in case of more than 1 counters). "
            "Note: job will fail if entire set of counters cannot be collected in single pass"
        ),
        default=None,
        nargs="*",
    )

    pc_sampling_options = parser.add_argument_group("PC sampling options")

    add_parser_bool_argument(
        pc_sampling_options,
        "--pc-sampling-beta-enabled",
        help="enable pc sampling support; beta version",
    )

    pc_sampling_options.add_argument(
        "--pc-sampling-unit",
        help="",
        default=None,
        type=str.lower,
        choices=("instructions", "cycles", "time"),
    )

    pc_sampling_options.add_argument(
        "--pc-sampling-method",
        help="",
        default=None,
        type=str.lower,
        choices=("stochastic", "host_trap"),
    )

    pc_sampling_options.add_argument(
        "--pc-sampling-interval",
        help="",
        default=None,
        type=int,
    )

    post_processing_options = parser.add_argument_group("Post-processing tracing options")

    add_parser_bool_argument(
        post_processing_options,
        "--stats",
        help="For collecting statistics of enabled tracing types. Must be combined with one or more tracing options. No default kernel stats unlike previous rocprof versions",
    )
    add_parser_bool_argument(
        post_processing_options,
        "-S",
        "--summary",
        help="Output single summary of tracing data at the conclusion of the profiling session",
    )
    add_parser_bool_argument(
        post_processing_options,
        "-D",
        "--summary-per-domain",
        help="Output summary for each tracing domain at the conclusion of the profiling session",
    )
    post_processing_options.add_argument(
        "--summary-groups",
        help="Output a summary for each set of domains matching the regular expression, e.g. 'KERNEL_DISPATCH|MEMORY_COPY' will generate a summary from all the tracing data in the KERNEL_DISPATCH and MEMORY_COPY domains; '*._API' will generate a summary from all the tracing data in the HIP_API, HSA_API, and MARKER_API domains",
        nargs="+",
        default=None,
        type=str,
        metavar="REGULAR_EXPRESSION",
    )

    summary_options = parser.add_argument_group("Summary options")

    summary_options.add_argument(
        "--summary-output-file",
        help="Output summary to a file, stdout, or stderr (default: stderr)",
        default=None,
        type=str,
    )
    summary_options.add_argument(
        "-u",
        "--summary-units",
        help="Timing units for output summary",
        default=None,
        type=str,
        choices=("sec", "msec", "usec", "nsec"),
    )

    kernel_naming_options = parser.add_argument_group("Kernel naming options")

    add_parser_bool_argument(
        kernel_naming_options,
        "-M",
        "--mangled-kernels",
        help="Do not demangle the kernel names",
    )
    add_parser_bool_argument(
        kernel_naming_options,
        "-T",
        "--truncate-kernels",
        help="Truncate the demangled kernel names. In earlier rocprof versions, this was known as --basenames [on/off]",
    )
    add_parser_bool_argument(
        kernel_naming_options,
        "--kernel-rename",
        help="Use region names defined by roctxRangePush/roctxRangePop regions to rename the kernels. Known as --roctx-rename in earlier rocprof versions",
    )

    filter_options = parser.add_argument_group("Filtering options")

    filter_options.add_argument(
        "--kernel-include-regex",
        help="Include the kernels matching this filter from counter-collection and thread-trace data (non-matching kernels will be excluded)",
        default=None,
        type=str,
        metavar="REGULAR_EXPRESSION",
    )
    filter_options.add_argument(
        "--kernel-exclude-regex",
        help="Exclude the kernels matching this filter from counter-collection and thread-trace data (applied after --kernel-include-regex option)",
        default=None,
        type=str,
        metavar="REGULAR_EXPRESSION",
    )
    filter_options.add_argument(
        "--kernel-iteration-range",
        help="Iteration range",
        nargs="+",
        default=None,
        type=str,
    )
    filter_options.add_argument(
        "-P",
        "--collection-period",
        help="The times are specified in seconds by default, but the unit can be changed using the `--collection-period-unit` option. Start Delay Time is the time in seconds before the collection begins, Collection Time is the duration in seconds for which data is collected, and Rate is the number of times the cycle is repeated. A repeat of 0 indicates that the cycle will repeat indefinitely. Users can specify multiple configurations, each defined by a triplet in the format `start_delay:collection_time:repeat`",
        nargs="+",
        default=None,
        type=str,
        metavar=("(START_DELAY_TIME):(COLLECTION_TIME):(REPEAT)"),
    )
    filter_options.add_argument(
        "--collection-period-unit",
        help="To change the unit used in `--collection-period` or `-P`, you can specify the desired unit using the `--collection-period-unit` option. The available units are `hour` for hours, `min` for minutes, `sec` for seconds, `msec` for milliseconds, `usec` for microseconds, and `nsec` for nanoseconds",
        nargs=1,
        default=["sec"],
        type=str,
        choices=("hour", "min", "sec", "msec", "usec", "nsec"),
    )
    add_parser_bool_argument(
        filter_options,
        "--selected-regions",
        help="If set, rocprofv3 will only profile regions of code surrounded by roctxProfilerResume(0) and roctxProfilerPause(0)",
    )

    perfetto_options = parser.add_argument_group("Perfetto-specific options")

    perfetto_options.add_argument(
        "--perfetto-backend",
        help="Perfetto data collection backend. 'system' mode requires starting traced and perfetto daemons",
        default=None,
        type=str,
        nargs=1,
        choices=("inprocess", "system"),
    )
    perfetto_options.add_argument(
        "--perfetto-buffer-size",
        help="Size of buffer for perfetto output in KB. default: 1 GB",
        default=None,
        type=int,
        metavar="KB",
    )
    perfetto_options.add_argument(
        "--perfetto-buffer-fill-policy",
        help="Policy for handling new records when perfetto has reached the buffer limit",
        default=None,
        type=str,
        choices=("discard", "ring_buffer"),
    )
    perfetto_options.add_argument(
        "--perfetto-shmem-size-hint",
        help="Perfetto shared memory size hint in KB. default: 64 KB",
        default=None,
        type=int,
        metavar="KB",
    )

    display_options = parser.add_argument_group("Display options")

    add_parser_bool_argument(
        display_options,
        "-L",
        "--list-avail",
        help="List available PC sampling configurations and metrics for counter collection. Backed by a valid YAML file. In earlier rocprof versions, this was known as --list-basic, --list-derived and --list-counters",
    )

    add_parser_bool_argument(
        display_options,
        "--group-by-queue",
        help="For displaying the HIP streams that kernels and memory copy operations are submitted to rather than HSA queues.",
    )

    advanced_options = parser.add_argument_group("Advanced options")

    advanced_options.add_argument(
        "--preload",
        help="Libraries to prepend to LD_PRELOAD (useful for sanitizer libraries)",
        default=os.environ.get("ROCPROF_PRELOAD", "").split(":"),
        nargs="*",
    )

    advanced_options.add_argument(
        "--rocm-root",
        help="Use the given path as the root ROCm path instead of the relative path of this script",
        type=str,
        metavar="PATH",
        default=None,
    )
    add_parser_bool_argument(
        advanced_options,
        "--readlink",
        help=argparse.SUPPRESS,
    )
    add_parser_bool_argument(
        advanced_options,
        "--realpath",
        help=argparse.SUPPRESS,
    )
    advanced_options.add_argument(
        "--benchmark-mode",
        choices=(
            "disabled-sdk-contexts",
            "sdk-buffer-overhead",
            "sdk-callback-overhead",
            "tool-runtime-overhead",
            "execution-profile",
        ),
        help=argparse.SUPPRESS,
        default=None,
        type=str.lower,
    )
    advanced_options.add_argument(
        "-A",
        "--agent-index",
        choices=("absolute", "relative", "type-relative"),
        help="""absolute == node_id, e.g. Agent-0, Agent-2, Agent-4- absolute index of the agent regardless of cgroups masking.
        This is a monotonically increasing number that is incremented for every folder in /sys/class/kfd/kfd/topology/nodes.
    relative == logical_node_id,
        e.g. Agent-0, Agent-1, Agent-2- relative index of the agent accounting for cgroups masking.
        This is a monotonically increasing number which is incremented for every folder in /sys/class/kfd/kfd/topology/nodes/ whose properties file was non-empty.
    type-relative == logical_node_type_id,
        e.g. CPU-0, GPU-0, GPU-1- relative index of the agent accounting for cgroups masking where indexing starts at zero for each agent type.
        It is a monotonically increasing number for each agent type and is incremented for every type folder in /sys/class/kfd/kfd/topology/nodes/ whose properties file is non-empty.
        If agent-index is not provided then the default value for it is relative.""",
    )
    # below is available for CI because LD_PRELOADing a library linked to a sanitizer library
    # causes issues in apps where HIP is part of shared library.
    add_parser_bool_argument(
        advanced_options,
        "--suppress-marker-preload",
        help=argparse.SUPPRESS,
    )

    # just echo the command line
    add_parser_bool_argument(
        advanced_options,
        "--echo",
        help=argparse.SUPPRESS,
    )

    add_parser_bool_argument(
        advanced_options,
        "--disable-signal-handlers",
        help="""Enables the signal handlers in the rocprofv3 tool.
        When --disable-signal-handlers is set to true,
        and application has its signal handler on SIGSEGV or similar installed,
        then its signal handler will be used, not the rocprofv3 signal handler.
        Note: glog still installs signal handlers which provide backtraces""",
    )

    advanced_options.add_argument(
        "--minimum-output-data",
        help="""Output files are generated only if output data size > minimum output data".
        It can be used for controlling the generation of output files so that user don't recieve empty files.
        The input is in KB units.""",
        default=None,
        type=int,
        metavar="KB",
    )

    reserved_options = parser.add_argument_group("Reserved options")
    reserved_options.add_argument(
        "-p",
        "--pid",
        help=argparse.SUPPRESS,
        type=str,
        nargs="+",
        default=None,
    )

    if args is None:
        args = sys.argv[1:]

    rocp_args = args[:]

    app_args = []

    for idx, itr in enumerate(args):
        if itr == "--":
            rocp_args = args[0:idx]
            app_args = args[(idx + 1) :]
            break

    att_options = parser.add_argument_group("Advanced Thread Trace (ATT) options")

    add_parser_bool_argument(
        att_options,
        "--advanced-thread-trace",
        "--att",
        help="Enables thread trace",
    )

    att_options.add_argument(
        "--att-library-path",
        help="Search path to decoder library.",
        default=None,
        nargs="+",
    )

    att_options.add_argument(
        "--att-target-cu",
        help="Target compute unit ID (or WGP). Default 1",
        default=None,
    )

    att_options.add_argument(
        "--att-simd-select",
        help="Bitmask of SIMDs to enable (gfx9) or SIMD ID (gfx10+). Default 0xF",
        default=None,
        type=str,
    )

    att_options.add_argument(
        "--att-buffer-size",
        help="Thread trace buffer size. Default 96MB",
        default=None,
        type=str,
    )

    att_options.add_argument(
        "--att-shader-engine-mask",
        help="Bitmask of shader engines to enable. Default 0x1",
        default=None,
        type=str,
    )

    att_options.add_argument(
        "--att-perfcounters",
        help="(gfx9) List of performance counters, and optionally their SIMD mask.",
        default=None,
        type=str.upper,
    )

    att_options.add_argument(
        "--att-perfcounter-ctrl",
        help="(gfx9) Integer in [1,32] range specifying collection period. 0 = disabled.",
        default=None,
        type=int,
    )

    att_options.add_argument(
        "--att-activity",
        help="(gfx9) Collect HW activity counters. Integer in [1,16] range specifying collection period. Recommended: 8",
        default=None,
        type=int,
    )

    add_parser_bool_argument(
        att_options,
        "--att-serialize-all",
        default=False,
        help="Serialize all kernels, not just the traced ones.",
    )

    return (parser.parse_args(rocp_args), app_args)


def parse_yaml(yaml_file):
    try:
        import yaml
    except ImportError as e:
        fatal_error(
            f"{e}\n\nYAML package is not installed. Run '{sys.executable} -m pip install pyyaml' or use JSON or text format"
        )
    try:
        lst = []
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
        for itr in data["jobs"]:
            # TODO: support naming jobs
            # if isinstance(itr, str):
            #     itr = data["jobs"][itr]
            itr["sub_directory"] = "pass_"
            lst.append(itr)

        return [dotdict(itr) for itr in lst]

    except yaml.YAMLError as exc:
        fatal_error(f"{exc}")

    return None


def parse_json(json_file):
    import json

    try:
        lst = []
        with open(json_file, "r") as file:
            data = json.load(file)
        for itr in data["jobs"]:
            itr["sub_directory"] = "pass_"
            lst.append(itr)

        return [dotdict(itr) for itr in lst]

    except Exception as e:
        fatal_error(f"{e}")

    return None


def parse_text(text_file):
    def process_line(line):
        if "pmc:" not in line:
            return ""
        line = line.strip()
        pos = line.find("#")
        if pos >= 0:
            line = line[0:pos]

        def _dedup(_line, _sep):
            for itr in _sep:
                _line = " ".join(_line.split(itr))
            return _line.strip()

        # remove tabs and duplicate spaces
        return _dedup(line.replace("pmc:", ""), ["\n", "\t", " "]).split(" ")

    try:
        with open(text_file, "r") as file:
            return [
                litr
                for litr in [process_line(itr) for itr in file.readlines()]
                if len(litr) > 0
            ]
    except Exception as e:
        fatal_error(f"{e}")

    return None


def parse_input(input_file):

    _, extension = os.path.splitext(input_file)
    if extension == ".txt":
        text_input = parse_text(input_file)
        text_input_lst = [{"pmc": itr, "sub_directory": "pmc_"} for itr in text_input]
        return [dotdict(itr) for itr in text_input_lst]
    elif extension in (".yaml", ".yml"):
        return parse_yaml(input_file)
    elif extension == ".json":
        return parse_json(input_file)
    else:
        fatal_error(
            f"Input file '{input_file}' does not have a recognized extension (.txt, .json, .yaml, .yml)\n"
        )

    return None


def has_set_attr(obj, key):
    if obj and hasattr(obj, key) and getattr(obj, key) is not None:
        return True
    else:
        return False


def patch_args(data):
    """Used to handle certain fields which might be specified as a string instead of an array or vice-versa"""

    if hasattr(data, "kernel_iteration_range") and isinstance(
        data.kernel_iteration_range, str
    ):
        data.kernel_iteration_range = [data.kernel_iteration_range]
    return data


def get_args(cmd_args, inp_args):
    def ensure_type(name, var, type_id):
        if not isinstance(var, type_id):
            raise TypeError(
                f"{name} is of type {type(var).__name__}, expected {type(type_id).__name__}"
            )

    ensure_type("cmd_args", cmd_args, argparse.Namespace)
    ensure_type("inp_args", inp_args, dotdict)

    cmd_keys = list(cmd_args.__dict__.keys())
    inp_keys = list(inp_args.keys())
    data = {}

    def get_attr(key):
        if has_set_attr(cmd_args, key):
            return getattr(cmd_args, key)
        elif has_set_attr(inp_args, key):
            return getattr(inp_args, key)
        return None

    for itr in set(cmd_keys + inp_keys):
        if (
            has_set_attr(cmd_args, itr)
            and has_set_attr(inp_args, itr)
            and getattr(cmd_args, itr) != getattr(inp_args, itr)
        ):
            raise RuntimeError(
                f"conflicting value for {itr} : {getattr(cmd_args, itr)} vs {getattr(inp_args, itr)}"
            )
        else:
            data[itr] = get_attr(itr)

    return patch_args(dotdict(data))


def run(app_args, args, **kwargs):

    app_env = dict(os.environ)
    use_execv = kwargs.get("use_execv", True)
    app_pass = kwargs.get("pass_id", None)

    if args.pid is not None:
        fatal_error(
            """The -p shorthand option for --collection-period is now an upper-case -P
                    In the future, rocprofv3 plans to support debugger-like process attachment and -p
                    is de-facto standard shorthand option for this feature"""
        )

    def setattrifnone(obj, attr, value):
        if getattr(obj, f"{attr}") is None:
            setattr(obj, f"{attr}", value)

    def update_env(env_var, env_val, **kwargs):
        """Local function for updating application environment which supports
        various options for dealing with existing environment variables
        """
        _overwrite = kwargs.get("overwrite", True)
        _prepend = kwargs.get("prepend", False)
        _append = kwargs.get("append", False)
        _join_char = kwargs.get("join_char", ":")

        # only overwrite if env_val evaluates as true
        _overwrite_if_true = kwargs.get("overwrite_if_true", False)
        # only overwrite if env_val evaluates as false
        _overwrite_if_false = kwargs.get("overwrite_if_false", False)

        _formatter = kwargs.get(
            "formatter",
            lambda x: f"{x}" if not isinstance(x, bool) else "1" if x else "0",
        )

        for itr in kwargs.keys():
            if itr not in (
                "overwrite",
                "prepend",
                "append",
                "join_char",
                "overwrite_if_true",
                "overwrite_if_false",
                "formatter",
            ):
                fatal_error(
                    f"Internal error in update_env('{env_var}', {env_val}, {itr}={kwargs[itr]}). Invalid key: {itr}"
                )

        if env_val is None:
            return app_env.get(env_var, None)

        _val = _formatter(env_val)
        _curr_val = app_env.get(env_var, None)

        def _write_env_value():
            if _overwrite_if_true:
                if bool(env_val) is True:
                    app_env[env_var] = _val
            elif _overwrite_if_false:
                if bool(env_val) is False:
                    app_env[env_var] = _val
            else:
                app_env[env_var] = _val

        if _curr_val is not None:
            if not _overwrite:
                pass
            elif _prepend:
                app_env[env_var] = (
                    "{}{}{}".format(_val, _join_char, _curr_val) if _val else _curr_val
                ).strip(":")
            elif _append:
                app_env[env_var] = (
                    "{}{}{}".format(_curr_val, _join_char, _val) if _val else _curr_val
                ).strip(":")
            elif _overwrite:
                _write_env_value()
        else:
            _write_env_value()

        return app_env.get(env_var, None)

    update_env("ROCPROFILER_LIBRARY_CTOR", True)

    ROCPROFV3_DIR = os.path.dirname(os.path.realpath(__file__))
    ROCM_DIR = os.path.dirname(ROCPROFV3_DIR)
    if args.rocm_root is not None:
        ROCM_DIR = os.path.abspath(args.rocm_root)
    ROCPROF_TOOL_LIBRARY = f"{ROCM_DIR}/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
    ROCPROF_SDK_LIBRARY = f"{ROCM_DIR}/lib/librocprofiler-sdk.so"
    ROCPROF_ROCTX_LIBRARY = f"{ROCM_DIR}/lib/librocprofiler-sdk-roctx.so"
    ROCPROF_KOKKOSP_LIBRARY = (
        f"{ROCM_DIR}/lib/rocprofiler-sdk/librocprofiler-sdk-tool-kokkosp.so"
    )
    ROCPROF_LIST_AVAIL_TOOL_LIBRARY = (
        f"{ROCM_DIR}/libexec/rocprofiler-sdk/librocprofv3-list-avail.so"
    )

    def resolve_path(val):
        if not os.path.exists(val):
            fatal_error(f"{val} does not exist")
        if os.path.islink(val):
            if args.readlink:
                val = os.path.abspath(os.readlink(val))
            if args.realpath:
                val = os.path.realpath(val)
        return val

    ROCPROF_TOOL_LIBRARY = resolve_path(ROCPROF_TOOL_LIBRARY)
    ROCPROF_SDK_LIBRARY = resolve_path(ROCPROF_SDK_LIBRARY)
    ROCPROF_ROCTX_LIBRARY = resolve_path(ROCPROF_ROCTX_LIBRARY)
    ROCPROF_KOKKOSP_LIBRARY = resolve_path(ROCPROF_KOKKOSP_LIBRARY)
    ROCPROF_LIST_AVAIL_TOOL_LIBRARY = resolve_path(ROCPROF_LIST_AVAIL_TOOL_LIBRARY)

    prepend_preload = [itr for itr in args.preload if itr]
    append_preload = [
        ROCPROF_TOOL_LIBRARY,
        ROCPROF_SDK_LIBRARY,
    ]

    update_env("LD_PRELOAD", ":".join(prepend_preload), prepend=True)
    update_env("LD_PRELOAD", ":".join(append_preload), append=True)

    update_env(
        "ROCP_TOOL_LIBRARIES",
        f"{ROCPROF_TOOL_LIBRARY}",
        append=True,
    )
    update_env(
        "LD_LIBRARY_PATH",
        f"{ROCM_DIR}/lib",
        append=True,
    )

    _output_file = args.output_file
    _output_path = (
        args.output_directory if args.output_directory is not None else os.getcwd()
    )

    update_env("ROCPROF_OUTPUT_FILE_NAME", _output_file)
    update_env("ROCPROF_OUTPUT_PATH", _output_path)
    update_env("ROCPROF_OUTPUT_CONFIG_FILE", args.output_config, overwrite_if_true=True)
    if app_pass is not None and args.sub_directory is not None:
        app_env["ROCPROF_OUTPUT_PATH"] = os.path.join(
            f"{_output_path}", f"{args.sub_directory}{app_pass}"
        )

    if args.list_avail and (
        args.output_file is not None or args.output_directory is not None
    ):
        update_env("ROCPROF_OUTPUT_LIST_AVAIL_FILE", True)

    if not args.output_format:
        args.output_format = ["rocpd"]

    update_env(
        "ROCPROF_OUTPUT_FORMAT", ",".join(args.output_format), append=True, join_char=","
    )

    if args.kokkos_trace:
        update_env("KOKKOS_TOOLS_LIBS", ROCPROF_KOKKOSP_LIBRARY, append=True)
        for itr in (
            "marker_trace",
            "kernel_rename",
        ):
            setattrifnone(args, itr, True)

    if args.sys_trace:
        for itr in (
            "hip_trace",
            "hsa_trace",
            "marker_trace",
            "kernel_trace",
            "memory_copy_trace",
            "memory_allocation_trace",
            "scratch_memory_trace",
            "rccl_trace",
            "rocdecode_trace",
            "rocjpeg_trace",
        ):
            setattrifnone(args, itr, True)

    if args.runtime_trace:
        for itr in (
            "hip_runtime_trace",
            "marker_trace",
            "kernel_trace",
            "memory_copy_trace",
            "memory_allocation_trace",
            "scratch_memory_trace",
            "rccl_trace",
            "rocdecode_trace",
            "rocjpeg_trace",
        ):
            setattrifnone(args, itr, True)

    if args.hip_trace:
        for itr in ("compiler", "runtime"):
            setattrifnone(args, f"hip_{itr}_trace", True)

    if args.hsa_trace:
        for itr in ("core", "amd", "image", "finalizer"):
            setattrifnone(args, f"hsa_{itr}_trace", True)

    trace_count = 0
    trace_opts = ["--hip-trace", "--hsa-trace"]
    for opt, env_val in dict(
        [
            ["hip_compiler_trace", "HIP_COMPILER_API_TRACE"],
            ["hip_runtime_trace", "HIP_RUNTIME_API_TRACE"],
            ["hsa_core_trace", "HSA_CORE_API_TRACE"],
            ["hsa_amd_trace", "HSA_AMD_EXT_API_TRACE"],
            ["hsa_image_trace", "HSA_IMAGE_EXT_API_TRACE"],
            ["hsa_finalizer_trace", "HSA_FINALIZER_EXT_API_TRACE"],
            ["marker_trace", "MARKER_API_TRACE"],
            ["rccl_trace", "RCCL_API_TRACE"],
            ["rocdecode_trace", "ROCDECODE_API_TRACE"],
            ["rocjpeg_trace", "ROCJPEG_API_TRACE"],
            ["kernel_trace", "KERNEL_TRACE"],
            ["memory_copy_trace", "MEMORY_COPY_TRACE"],
            ["memory_allocation_trace", "MEMORY_ALLOCATION_TRACE"],
            ["scratch_memory_trace", "SCRATCH_MEMORY_TRACE"],
            ["group_by_queue", "GROUP_BY_QUEUE"],
        ]
    ).items():
        val = getattr(args, f"{opt}")
        update_env(f"ROCPROF_{env_val}", val, overwrite_if_true=True)
        trace_count += 1 if val else 0
        trace_opts += ["--{}".format(opt.replace("_", "-"))]

    for opt in ["advanced_thread_trace", "pmc", "pc_sampling_beta_enabled"]:
        if not hasattr(args, f"{opt}"):
            continue
        val = getattr(args, f"{opt}")
        trace_count += 1 if val else 0
        trace_opts += ["--{}".format(opt.replace("_", "-"))]

    if args.pmc_groups:
        trace_count += 1

    # if marker tracing was requested, LD_PRELOAD the rocprofiler-sdk-roctx library
    # to override the roctx symbols of an app linked to the old roctracer roctx
    if args.marker_trace and not args.suppress_marker_preload:
        update_env("LD_PRELOAD", ROCPROF_ROCTX_LIBRARY, append=True)

    if trace_count == 0:
        warning("No tracing options were enabled.")

        # if no tracing was enabled but the options below were enabled, raise an error
        for oitr in [
            "stats",
            "summary",
            "summary-per-domain",
            "summary-groups",
            "summary-output-file",
            "summary-units",
        ]:
            _attr = oitr.replace("-", "_")
            if not hasattr(args, _attr):
                fatal_error(
                    f"Internal error. parser does not support --{oitr} argument (i.e. args.{_attr})"
                )
            elif getattr(args, _attr):
                _len = max([len(f"{key}") for key in args.keys()])
                _args = "\n    ".join(
                    sorted([f"{key:<{_len}} = {val}" for key, val in args.items()])
                )
                fatal_error(
                    """
                    No tracing options were enabled for --{} option.

                    Configuration argument values:
                        {}

                    Tracing options:
                        {}
                    """,
                    oitr,
                    f"{_args}",
                    "\n    ".join(trace_opts),
                )

        rocprofiler_ci_env = os.environ.get("ROCPROFILER_CI", "0")
        if strtobool(rocprofiler_ci_env):
            fatal_error(
                f"rocprofv3 tracing options are required when ROCPROFILER_CI={rocprofiler_ci_env}"
            )

    _summary_groups = "##@@##".join(args.summary_groups) if args.summary_groups else None
    _summary_output_fname = args.summary_output_file
    if args.summary and _summary_output_fname is None:
        _summary_output_fname = "stderr"
    elif args.summary and _summary_output_fname.lower() in ("stdout", "stderr"):
        _summary_output_fname = _summary_output_fname.lower()

    update_env("ROCPROF_STATS", args.stats, overwrite_if_true=True)
    update_env("ROCPROF_STATS_SUMMARY", args.summary, overwrite_if_true=True)
    update_env("ROCPROF_STATS_SUMMARY_UNITS", args.summary_units, overwrite=True)
    update_env("ROCPROF_STATS_SUMMARY_OUTPUT", _summary_output_fname, overwrite=True)
    update_env("ROCPROF_STATS_SUMMARY_GROUPS", _summary_groups, overwrite=True)
    update_env(
        "ROCPROF_STATS_SUMMARY_PER_DOMAIN",
        args.summary_per_domain,
        overwrite_if_true=True,
    )
    update_env(
        "ROCPROF_DEMANGLE_KERNELS",
        not args.mangled_kernels,
        overwrite_if_false=True,
    )
    update_env(
        "ROCPROF_TRUNCATE_KERNELS",
        args.truncate_kernels,
        overwrite_if_true=True,
    )
    update_env(
        "ROCPROF_SELECTED_REGIONS",
        args.selected_regions,
        overwrite_if_true=True,
    )

    if args.list_avail:
        update_env(
            "ROCPROF_LIST_AVAIL_TOOL_LIBRARY",
            ROCPROF_LIST_AVAIL_TOOL_LIBRARY,
            overwrite_if_true=True,
        )

    if args.collection_period:
        factors = {
            "hour": 60 * 60 * 1e9,
            "min": 60 * 1e9,
            "sec": 1e9,
            "msec": 1e6,
            "usec": 1e3,
            "nsec": 1,
        }

        def to_nanosec(val):
            return int(float(val) * factors[args.collection_period_unit[0]])

        def convert_triplet(delay, duration, repeat):
            return ":".join(
                [
                    f"{itr}"
                    for itr in [to_nanosec(delay), to_nanosec(duration), int(repeat)]
                ]
            )

        periods = [convert_triplet(*itr.split(":")) for itr in args.collection_period]
        update_env(
            "ROCPROF_COLLECTION_PERIOD",
            ";".join(periods),
            overwrite_if_true=True,
        )

    if args.log_level and args.log_level not in ("env", "config"):
        for itr in ("ROCPROF", "ROCPROFILER", "ROCTX"):
            update_env(
                f"{itr}_LOG_LEVEL",
                args.log_level,
            )

    if args.benchmark_mode:
        if args.benchmark_mode == "execution-profile":
            if args.group_by_queue is None:
                update_env(
                    "ROCPROF_GROUP_BY_QUEUE",
                    False,
                    overwrite=True,
                )
            elif args.group_by_queue:
                fatal_error(
                    "rocprofv3 requires --group-by-queue=false for --benchmark-mode=execution-profile"
                )

        update_env(
            "ROCPROF_BENCHMARK_MODE",
            args.benchmark_mode,
        )

    for opt, env_val in dict(
        [
            ["kernel_rename", "KERNEL_RENAME"],
        ]
    ).items():
        val = getattr(args, f"{opt}")
        if val is not None:
            update_env(f"ROCPROF_{env_val}", val, overwrite_if_true=True)

    for opt, env_val in dict(
        [
            ["perfetto_buffer_size", "PERFETTO_BUFFER_SIZE_KB"],
            ["perfetto_shmem_size_hint", "PERFETTO_SHMEM_SIZE_HINT_KB"],
            ["perfetto_fill_policy", "PERFETTO_BUFFER_FILL_POLICY"],
            ["perfetto_backend", "PERFETTO_BACKEND"],
        ]
    ).items():
        val = getattr(args, f"{opt}")
        if val is not None:
            if isinstance(val, (list, tuple, set)):
                val = ", ".join(val)
            update_env(f"ROCPROF_{env_val}", val, overwrite=True)

    def log_config(_env):

        cfg_init_message = "\n- rocprofv3 configuration{}:\n".format(
            "" if app_pass is None else f" (pass {app_pass})"
        )
        if args.log_level in ("info", "trace", "config"):
            _len = max([len(f"{key}") for key in args.keys()])
            _args = sorted([f"{key:<{_len}} = {val}" for key, val in args.items()])
            sys.stderr.write(cfg_init_message)
            cfg_init_message = None
            for itr in _args:
                sys.stderr.write(f"    - {itr}\n")

        env_init_message = "\n- rocprofv3 environment{}:\n".format(
            "" if app_pass is None else f" (pass {app_pass})"
        )
        if args.log_level in ("info", "trace", "env"):
            existing_env = dict(os.environ)
            for key, itr in _env.items():
                if key not in existing_env.keys():
                    if env_init_message:
                        sys.stderr.write(env_init_message)
                        env_init_message = None
                    sys.stderr.write(f"    - {key}={itr}\n")

        if cfg_init_message is None or env_init_message is None:
            sys.stderr.write("\n")
        sys.stderr.flush()

    if args.list_avail:
        update_env("ROCPROFILER_PC_SAMPLING_BETA_ENABLED", "on")
        path = os.path.join(f"{ROCM_DIR}", "bin/rocprofv3-avail")
        if app_args:
            exit_code = subprocess.check_call(
                [sys.executable, path, "info", "--pmc"],
                env=app_env,
            )
            if exit_code != 0:
                fatal_error("rocprofv3-avail exit with error")

            exit_code = subprocess.check_call(
                [sys.executable, path, "info", "--pc-sampling"],
                env=app_env,
            )
        else:
            app_args = [sys.executable, path, "info", "--pmc"]
            exit_code = subprocess.check_call(
                [sys.executable, path, "info", "--pc-sampling"],
                env=app_env,
            )

    elif not app_args and not args.echo:
        log_config(app_env)
        fatal_error("No application provided")

    if args.kernel_include_regex:
        update_env(
            "ROCPROF_KERNEL_FILTER_INCLUDE_REGEX",
            args.kernel_include_regex,
        )

    if args.kernel_exclude_regex:
        update_env(
            "ROCPROF_KERNEL_FILTER_EXCLUDE_REGEX",
            args.kernel_exclude_regex,
        )

    if args.kernel_iteration_range:
        update_env("ROCPROF_KERNEL_FILTER_RANGE", ", ".join(args.kernel_iteration_range))

    if args.agent_index:
        update_env("ROCPROF_AGENT_INDEX", args.agent_index)

    if args.extra_counters is not None:
        with open(args.extra_counters, "r") as e_file:
            e_file_contents = e_file.read()
            update_env("ROCPROF_EXTRA_COUNTERS_CONTENTS", e_file_contents, overwrite=True)

    if args.pmc and args.pmc_groups:
        fatal_error("Cannot specify both --pmc and (input file) pmc_groups")

    if args.pmc:
        update_env("ROCPROF_COUNTER_COLLECTION", True, overwrite=True)
        update_env(
            "ROCPROF_COUNTERS", "pmc: {}".format(" ".join(args.pmc)), overwrite=True
        )

    if args.pmc_groups:
        group_env = ""
        update_env("ROCPROF_COUNTER_COLLECTION", True, overwrite=True)

        for row in map(" ".join, args.pmc_groups):
            # pmc: added to allow for the same parser to be shared between the two
            #      counter collection modes. Output will be new line delimited between
            #      groups.
            group_env += f"pmc: {row}\n"
        group_env = group_env.rstrip()

        update_env("ROCPROF_COUNTER_GROUPS", group_env, overwrite=True)

        if args.pmc_group_interval:
            update_env(
                "ROCPROF_COUNTER_GROUPS_INTERVAL",
                f"{str(args.pmc_group_interval)}",
                overwrite=True,
            )

    if args.pc_sampling_unit or args.pc_sampling_method or args.pc_sampling_interval:

        if (
            not args.pc_sampling_beta_enabled
            and os.environ.get("ROCPROFILER_PC_SAMPLING_BETA_ENABLED", None) is None
        ):
            fatal_error(
                "PC sampling unavailable. The feature is implicitly disabled. To enable it, use --pc-sampling-beta-enable option or set ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON in the environment"
            )

        update_env(
            "ROCPROFILER_PC_SAMPLING_BETA_ENABLED",
            args.pc_sampling_beta_enabled,
            overwrite_if_true=True,
        )

        if not (
            args.pc_sampling_unit
            and args.pc_sampling_method
            and args.pc_sampling_interval
        ):
            fatal_error("All three PC sampling configurations need to be set")

        if args.pc_sampling_interval <= 0:
            fatal_error("PC sampling interval must be a positive number.")

        update_env("ROCPROF_PC_SAMPLING_UNIT", args.pc_sampling_unit)
        update_env("ROCPROF_PC_SAMPLING_METHOD", args.pc_sampling_method)
        update_env("ROCPROF_PC_SAMPLING_INTERVAL", args.pc_sampling_interval)

    if args.disable_signal_handlers is not None:
        update_env("ROCPROF_SIGNAL_HANDLERS", not args.disable_signal_handlers)

    if args.minimum_output_data:
        update_env("ROCPROF_MINIMUM_OUTPUT_BYTES", args.minimum_output_data * 1024)

    if args.advanced_thread_trace:

        def int_auto(num_str):
            if isinstance(num_str, str):
                if "0x" in num_str:
                    return int(num_str, 16)
                else:
                    return int(num_str, 10)
            elif isinstance(num_str, int):
                return num_str
            else:
                raise ValueError(
                    f"{type(num_str)} is not supported. {num_str} should be of type integer or string."
                )

        update_env("ROCPROF_ADVANCED_THREAD_TRACE", True, overwrite=True)

        if args.att_target_cu is not None:
            update_env(
                "ROCPROF_ATT_PARAM_TARGET_CU",
                int_auto(args.att_target_cu),
                overwrite=True,
            )

        if args.att_shader_engine_mask:
            update_env(
                "ROCPROF_ATT_PARAM_SHADER_ENGINE_MASK",
                int_auto(args.att_shader_engine_mask),
                overwrite=True,
            )
        if args.att_buffer_size:
            update_env(
                "ROCPROF_ATT_PARAM_BUFFER_SIZE",
                int_auto(args.att_buffer_size),
                overwrite=True,
            )
        if args.att_simd_select:
            update_env(
                "ROCPROF_ATT_PARAM_SIMD_SELECT",
                int_auto(args.att_simd_select),
                overwrite=True,
            )
        if args.att_serialize_all:
            update_env(
                "ROCPROF_ATT_PARAM_SERIALIZE_ALL",
                args.att_serialize_all,
                overwrite=True,
            )
        if check_att_capability(args):
            update_env(
                "ROCPROF_ATT_LIBRARY_PATH",
                args.att_library_path,
                overwrite=True,
            )
        else:
            fatal_error(
                "rocprof-trace-decoder library path not found in", args.att_library_path
            )

        if args.att_perfcounters:
            if args.pmc:
                fatal_error("ATT perfcounters cannot be enabled with PMC")
            else:
                update_env(
                    "ROCPROF_ATT_PARAM_PERFCOUNTERS",
                    args.att_perfcounters,
                    overwrite=True,
                )
        if args.att_perfcounter_ctrl:
            if args.pmc:
                fatal_error("ATT perfcounters cannot be enabled with PMC")
            else:
                update_env(
                    "ROCPROF_ATT_PARAM_PERFCOUNTER_CTRL",
                    args.att_perfcounter_ctrl,
                    overwrite=True,
                )
        if args.att_activity:
            if args.pmc:
                fatal_error("ATT activity cannot be enabled with PMC")
            elif args.att_perfcounters or args.att_perfcounter_ctrl:
                fatal_error(
                    "ATT activity cannot be enabled with att-perfcounters or att-perfcounter-ctrl."
                )
            else:
                update_env(
                    "ROCPROF_ATT_PARAM_PERFCOUNTER_CTRL",
                    args.att_activity,
                    overwrite=True,
                )
                update_env(
                    "ROCPROF_ATT_PARAM_PERFCOUNTERS",
                    "SQ_BUSY_CU_CYCLES SQ_VALU_MFMA_BUSY_CYCLES SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_LDS SQ_ACTIVE_INST_VMEM SQ_ACTIVE_INST_FLAT SQ_ACTIVE_INST_SCA SQ_ACTIVE_INST_MISC",
                    overwrite=True,
                )

    if args.log_level in ("info", "trace", "env", "config"):
        log_config(app_env)

    if use_execv:
        if args.echo:
            sys.stderr.flush()
            print(f"command: {app_args}")
            sys.stdout.flush()
        else:
            # does not return
            os.execvpe(app_args[0], app_args, env=app_env)
    else:
        if args.echo:
            sys.stderr.flush()
            print(f"command: {app_args}")
            sys.stdout.flush()
            return 0
        try:
            exit_code = subprocess.check_call(app_args, env=app_env)
            if exit_code != 0:
                fatal_error("Application exited with non-zero exit code", exit_code)
        except Exception as e:
            fatal_error(f"{e}\n")
        return exit_code


def main(argv=None):

    cmd_args, app_args = parse_arguments(argv)

    if cmd_args.version:
        for key, itr in CONST_VERSION_INFO.items():
            print(f"    {key:>16}: {itr}")
        return 0

    inp_args = (
        parse_input(cmd_args.input) if getattr(cmd_args, "input") else [dotdict({})]
    )

    if len(inp_args) == 1:
        args = get_args(cmd_args, inp_args[0])
        pass_idx = None
        if has_set_attr(args, "pmc") and len(args.pmc) > 0:
            pass_idx = 1
        ec = run(app_args, args, pass_id=pass_idx)
    else:
        ec = 0
        for idx, itr in enumerate(inp_args):
            args = get_args(cmd_args, itr)
            _ec = run(
                app_args,
                args,
                pass_id=(idx + 1),
                use_execv=False,
            )
            if _ec is not None and _ec != 0:
                ec = _ec

    # return error code
    return ec if not None else 0


if __name__ == "__main__":
    ec = main(sys.argv[1:])
    sys.exit(ec)
