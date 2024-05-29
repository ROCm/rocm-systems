#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess


def fatal_error(msg, exit_code=1):
    sys.stderr.write(f"Fatal error: {msg}\n")
    sys.stderr.flush()
    sys.exit(exit_code)


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
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Add the arguments
    parser.add_argument(
        "--hip-trace",
        action="store_true",
        help="For Collecting HIP Traces (runtime + compiler)",
        required=False,
    )
    parser.add_argument(
        "--hip-runtime-trace",
        action="store_true",
        help="For Collecting HIP Runtime API Traces",
        required=False,
    )
    parser.add_argument(
        "--hip-compiler-trace",
        action="store_true",
        help="For Collecting HIP Compiler generated code Traces",
        required=False,
    )
    parser.add_argument(
        "--marker-trace",
        action="store_true",
        help="For Collecting Marker (ROCTx) Traces",
        required=False,
    )
    parser.add_argument(
        "--kernel-trace",
        action="store_true",
        help="For Collecting Kernel Dispatch Traces",
        required=False,
    )
    parser.add_argument(
        "--memory-copy-trace",
        action="store_true",
        help="For Collecting Memory Copy Traces",
        required=False,
    )
    parser.add_argument(
        "--scratch-memory-trace",
        action="store_true",
        help="For Collecting Scratch Memory operations Traces",
        required=False,
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="For Collecting statistics of enabled tracing types",
        required=False,
    )
    parser.add_argument(
        "--hsa-trace",
        action="store_true",
        help="For Collecting HSA Traces (core + amd + image + finalizer)",
        required=False,
    )
    parser.add_argument(
        "--hsa-core-trace",
        action="store_true",
        help="For Collecting HSA API Traces (core API)",
        required=False,
    )
    parser.add_argument(
        "--hsa-amd-trace",
        action="store_true",
        help="For Collecting HSA API Traces (AMD-extension API)",
        required=False,
    )
    parser.add_argument(
        "--hsa-image-trace",
        action="store_true",
        help="For Collecting HSA API Traces (Image-extenson API)",
        required=False,
    )
    parser.add_argument(
        "--hsa-finalizer-trace",
        action="store_true",
        help="For Collecting HSA API Traces (Finalizer-extension API)",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--sys-trace",
        action="store_true",
        help="For Collecting HIP, HSA, Marker (ROCTx), Memory copy, Scratch memory, and Kernel dispatch traces",
        required=False,
    )
    parser.add_argument(
        "-M",
        "--mangled-kernels",
        action="store_true",
        help="Do not demangle the kernel names",
        required=False,
    )
    parser.add_argument(
        "-T",
        "--truncate-kernels",
        action="store_true",
        help="Truncate the demangled kernel names",
        required=False,
    )
    parser.add_argument(
        "-L",
        "--list-metrics",
        action="store_true",
        help="List metrics for counter collection",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input file for counter collection",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="For the output file name",
        default=os.environ.get("ROCPROF_OUTPUT_FILE_NAME", None),
        type=str,
        required=False,
    )
    parser.add_argument(
        "-d",
        "--output-directory",
        help="For adding output path where the output files will be saved",
        default=os.environ.get("ROCPROF_OUTPUT_PATH", None),
        type=str,
        required=False,
    )
    parser.add_argument(
        "--output-format",
        help="For adding output format (supported formats: csv, json, pftrace)",
        nargs="+",
        default=["csv"],
        choices=("csv", "json", "pftrace"),
        type=str.lower,
    )
    parser.add_argument(
        "--log-level",
        help="Set the log level",
        default=None,
        choices=("fatal", "error", "warning", "info", "trace"),
        type=str.lower,
    )
    parser.add_argument(
        "--kernel-names",
        help="Filter kernel names",
        default=None,
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--preload",
        help="Libraries to prepend to LD_PRELOAD (usually for sanitizers)",
        default=os.environ.get("ROCPROF_PRELOAD", "").split(":"),
        nargs="*",
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

    return (parser.parse_args(rocp_args), app_args)


def parse_yaml(yaml_file):
    try:
        import yaml
    except ImportError as e:
        fatal_error(
            f"{e}\n\nYAML package is not installed. Run '{sys.executable} -m pip install pyyaml' or use JSON or text format"
        )

    try:
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
        return [" ".join(itr["pmc"]) for itr in data["metrics"]]
    except yaml.YAMLError as exc:
        fatal_error(f"{exc}")

    return None


def parse_json(json_file):
    import json

    try:
        with open(json_file, "r") as file:
            data = json.load(file)
        return [" ".join(itr["pmc"]) for itr in data["metrics"]]
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
            return _line

        # remove tabs and duplicate spaces
        return _dedup(line.replace("pmc:", ""), ["\t", " "]).strip()

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
    pmc_lines = []
    _, extension = os.path.splitext(input_file)
    if extension == ".txt":
        pmc_lines = parse_text(input_file)
    elif extension in (".yaml", ".yml"):
        pmc_lines = parse_yaml(input_file)
    elif extension == ".json":
        pmc_lines = parse_json(input_file)
    else:
        fatal_error(
            f"Input file '{input_file}' does not have a recognized extension (.txt, .json, .yaml, .yml)\n"
        )

    return pmc_lines


def main(argv=None):

    app_env = dict(os.environ)

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
                app_env[env_var] = "{}{}{}".format(_val, _join_char, _curr_val)
            elif _append:
                app_env[env_var] = "{}{}{}".format(_curr_val, _join_char, _val)
            elif _overwrite:
                _write_env_value()
        else:
            _write_env_value()

        return app_env.get(env_var, None)

    update_env("ROCPROFILER_LIBRARY_CTOR", True)

    ROCPROFV3_DIR = os.path.dirname(os.path.realpath(__file__))
    ROCM_DIR = os.path.dirname(ROCPROFV3_DIR)
    ROCPROF_TOOL_LIBRARY = f"{ROCM_DIR}/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
    ROCPROF_SDK_LIBRARY = f"{ROCM_DIR}/lib/librocprofiler-sdk.so"

    args, app_args = parse_arguments(argv)

    _preload = ":".join(args.preload) if args.preload else None

    update_env("LD_PRELOAD", _preload, prepend=True)
    update_env("LD_PRELOAD", f"{ROCPROF_TOOL_LIBRARY}:{ROCPROF_SDK_LIBRARY}", append=True)
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

    if args.output_file is not None or args.output_directory is not None:
        update_env("ROCPROF_OUTPUT_LIST_METRICS_FILE", True)

    update_env(
        "ROCPROF_OUTPUT_FORMAT", ",".join(args.output_format), append=True, join_char=","
    )

    _kernel_names = ",".join(args.kernel_names) if args.kernel_names else None
    update_env("ROCPROF_KERNEL_NAMES", _kernel_names, append=True, join_char=",")

    if args.sys_trace:
        for itr in (
            "hip_trace",
            "hsa_trace",
            "marker_trace",
            "kernel_trace",
            "memory_copy_trace",
            "scratch_memory_trace",
        ):
            setattr(args, itr, True)

    if args.hip_trace:
        for itr in ("compiler", "runtime"):
            setattr(args, f"hip_{itr}_trace", True)

    if args.hsa_trace:
        for itr in ("core", "amd", "image", "finalizer"):
            setattr(args, f"hsa_{itr}_trace", True)

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
            ["kernel_trace", "KERNEL_TRACE"],
            ["memory_copy_trace", "MEMORY_COPY_TRACE"],
            ["scratch_memory_trace", "SCRATCH_MEMORY_TRACE"],
        ]
    ).items():
        val = getattr(args, f"{opt}")
        update_env(f"ROCPROF_{env_val}", val, overwrite_if_true=True)
        trace_count += 1 if val else 0
        trace_opts += ["--{}".format(opt.replace("_", "-"))]

    if trace_count == 0 and args.stats:
        fatal_error(
            "No tracing options were enabled for --stats option. Tracing options:\n\t{}".format(
                "\n\t".join(trace_opts)
            )
        )

    update_env("ROCPROF_STATS", args.stats, overwrite_if_true=True)
    update_env(
        "ROCPROF_DEMANGLE_KERNELS", not args.mangled_kernels, overwrite_if_false=True
    )
    update_env(
        "ROCPROF_TRUNCATE_KERNELS",
        args.truncate_kernels,
        overwrite_if_true=True,
    )
    update_env(
        "ROCPROF_LIST_METRICS",
        args.list_metrics,
        overwrite_if_true=True,
    )

    for itr in ("ROCPROF", "ROCPROFILER", "ROCTX"):
        update_env(
            f"{itr}_LOG_LEVEL",
            args.log_level,
        )

    def log_config(_env):
        existing_env = dict(os.environ)
        init_message = "- rocprofv3 configuration:\n"
        for key, itr in _env.items():
            if key not in existing_env.keys():
                if init_message:
                    sys.stderr.write(init_message)
                    init_message = None
                sys.stderr.write(f"\t- {key}={itr}\n")
        sys.stderr.flush()

    if args.list_metrics:
        app_args = [f"{ROCM_DIR}/lib/rocprofiler-sdk/rocprofv3-trigger-list-metrics"]
    elif not app_args:
        log_config(app_env)
        fatal_error("No application provided")

    pmc_lines = []
    if args.input:
        pmc_lines = parse_input(args.input)

    if pmc_lines:
        exit_code = 0
        update_env("ROCPROF_COUNTER_COLLECTION", True, overwrite_if_true=True)

        for idx, pmc_line in enumerate(pmc_lines):
            COUNTER = idx + 1
            pmc_env = dict(app_env)
            pmc_env["ROCPROF_COUNTERS"] = f"pmc: {pmc_line}"
            pmc_env["ROCPROF_OUTPUT_PATH"] = os.path.join(
                f"{_output_path}", f"pmc_{COUNTER}"
            )

            if args.log_level in ("info", "trace"):
                log_config(pmc_env)

            try:
                exit_code = subprocess.check_call(app_args, env=pmc_env)
                if exit_code != 0:
                    fatal_error("Application exited with non-zero exit code", exit_code)
            except Exception as e:
                fatal_error(f"{e}\n")

        return exit_code
    else:
        if args.log_level in ("info", "trace"):
            log_config(app_env)
        # does not return
        os.execvpe(app_args[0], app_args, env=app_env)


if __name__ == "__main__":
    ec = main(sys.argv[1:])
    sys.exit(ec)
