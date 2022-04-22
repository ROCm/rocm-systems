#!/usr/bin/env python@_VERSION@
# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

from __future__ import absolute_import

__author__ = "AMD Research"
__copyright__ = "Copyright 2022, Advanced Micro Devices, Inc."
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "AMD Research"
__status__ = "Development"

""" @file __main__.py
Command line execution for profiler
"""

import os
import sys
import argparse
import traceback

PY3 = sys.version_info[0] == 3


# Python 3.x compatibility utils: execfile
try:
    execfile
except NameError:
    # Python 3.x doesn't have 'execfile' builtin
    import builtins

    exec_ = getattr(builtins, "exec")

    def execfile(filename, globals=None, locals=None):
        with open(filename, "rb") as f:
            exec_(compile(f.read(), filename, "exec"), globals, locals)


def find_script(script_name):
    """Find the script.

    If the input is not a file, then $PATH will be searched.
    """
    if os.path.isfile(script_name):
        return script_name
    path = os.getenv("PATH", os.defpath).split(os.pathsep)
    for dir in path:
        if dir == "":
            continue
        fn = os.path.join(dir, script_name)
        if os.path.isfile(fn):
            return fn

    sys.stderr.write("Could not find script %s\n" % script_name)
    raise SystemExit(1)


def parse_args(args=None):
    """Parse the arguments"""

    if args is None:
        args = sys.argv[1:]

    from .libpyomnitrace.profiler import config as _profiler_config

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(
        "omnitrace",
        add_help=True,
        epilog="usage: {} -m omnitrace <OMNITRACE_ARGS> -- <SCRIPT> <SCRIPT_ARGS>".format(
            os.path.basename(sys.executable)
        ),
    )
    parser.add_argument(
        "-b",
        "--builtin",
        action="store_true",
        help=(
            "Put 'profile' in the builtins. Use '@profile' to decorate a single function, "
            "or 'with profile:' to profile a single section of code."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        metavar="FILE",
        help="Omnitrace configuration file",
    )
    parser.add_argument(
        "-s",
        "--setup",
        default=None,
        metavar="FILE",
        help="Code to execute before the code to profile",
    )
    parser.add_argument(
        "--trace-c",
        type=str2bool,
        nargs="?",
        metavar="BOOL",
        const=True,
        default=_profiler_config.trace_c,
        help="Enable profiling C functions",
    )
    parser.add_argument(
        "-a",
        "--include-args",
        type=str2bool,
        nargs="?",
        metavar="BOOL",
        const=True,
        default=_profiler_config.include_args,
        help="Encode the argument values",
    )
    parser.add_argument(
        "-l",
        "--include-line",
        type=str2bool,
        nargs="?",
        metavar="BOOL",
        const=True,
        default=_profiler_config.include_line,
        help="Encode the function line number",
    )
    parser.add_argument(
        "-f",
        "--include-file",
        type=str2bool,
        nargs="?",
        metavar="BOOL",
        const=True,
        default=_profiler_config.include_filename,
        help="Encode the function filename",
    )
    parser.add_argument(
        "-F",
        "--full-filepath",
        type=str2bool,
        nargs="?",
        metavar="BOOL",
        const=True,
        default=_profiler_config.full_filepath,
        help="Encode the full function filename (instead of basename)",
    )
    parser.add_argument(
        "-I",
        "--function-include",
        type=str,
        nargs="+",
        metavar="FUNC",
        default=_profiler_config.include_functions,
        help="Include any entries with these function names",
    )
    parser.add_argument(
        "-E",
        "--function-exclude",
        type=str,
        nargs="+",
        metavar="FUNC",
        default=_profiler_config.exclude_functions,
        help="Filter out any entries with these function names",
    )
    parser.add_argument(
        "-R",
        "--function-restrict",
        type=str,
        nargs="+",
        metavar="FUNC",
        default=_profiler_config.restrict_functions,
        help="Select only entries with these function names",
    )
    parser.add_argument(
        "-MI",
        "--module-include",
        type=str,
        nargs="+",
        metavar="FILE",
        default=_profiler_config.include_modules,
        help="Include any entries from these files",
    )
    parser.add_argument(
        "-ME",
        "--module-exclude",
        type=str,
        nargs="+",
        metavar="FILE",
        default=_profiler_config.exclude_modules,
        help="Filter out any entries from these files",
    )
    parser.add_argument(
        "-MR",
        "--module-restrict",
        type=str,
        nargs="+",
        metavar="FILE",
        default=_profiler_config.restrict_modules,
        help="Select only entries from these files",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=_profiler_config.verbosity,
        help="Logging verbosity",
    )

    return parser.parse_args(args)


def get_value(env_var, default_value, dtype, arg=None):
    if arg is not None:
        return dtype(arg)
    else:
        val = os.environ.get(env_var)
        if val is None:
            os.environ[env_var] = "{}".format(default_value)
            return dtype(default_value)
        else:
            return dtype(val)


def run(prof, cmd):
    if len(cmd) == 0:
        return

    progname = cmd[0]
    sys.path.insert(0, os.path.dirname(progname))
    with open(progname, "rb") as fp:
        code = compile(fp.read(), progname, "exec")

    import __main__

    dict = __main__.__dict__
    print("code: {} {}".format(type(code).__name__, code))
    globs = {
        "__file__": progname,
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
        **dict,
    }

    prof.runctx(code, globs, None)


def main():
    """Main function"""

    opts = None
    argv = None
    if "--" in sys.argv:
        _idx = sys.argv.index("--")
        _argv = sys.argv[(_idx + 1) :]
        opts = parse_args(sys.argv[1:_idx])
        argv = _argv
    else:
        if "-h" in sys.argv or "--help" in sys.argv:
            opts = parse_args()
        else:
            argv = sys.argv[1:]
            opts = parse_args([])
            if len(argv) == 0 or not os.path.isfile(argv[0]):
                raise RuntimeError(
                    "Could not determine input script. Use '--' before "
                    "the script and its arguments to ensure correct parsing. \nE.g. "
                    "python -m omnitrace -- ./script.py"
                )

    if len(argv) > 1:
        if argv[0] == "-m":
            argv = argv[1:]
        elif argv[0] == "-c":
            argv[0] = os.path.basename(sys.executable)
        else:
            while len(argv) > 1 and argv[0].startswith("-"):
                argv = argv[1:]
                if os.path.exists(argv[0]):
                    break

    if argv:
        os.environ["OMNITRACE_COMMAND_LINE"] = " ".join(argv)

    if opts.config is not None:
        os.environ["OMNITRACE_CONFIG_FILE"] = ":".join(
            [os.environ.get("OMNITRACE_CONFIG_FILE", ""), opts.config]
        )

    from .libpyomnitrace import initialize

    if os.path.isfile(argv[0]):
        argv[0] = os.path.realpath(argv[0])

    initialize(argv)

    from .libpyomnitrace.profiler import config as _profiler_config

    _profiler_config.trace_c = opts.trace_c
    _profiler_config.include_args = opts.include_args
    _profiler_config.include_line = opts.include_line
    _profiler_config.include_filename = opts.include_file
    _profiler_config.full_filepath = opts.full_filepath
    _profiler_config.include_functions = opts.function_include
    _profiler_config.include_modules = opts.module_include
    _profiler_config.exclude_functions = opts.function_exclude
    _profiler_config.exclude_modules = opts.module_exclude
    _profiler_config.restrict_functions = opts.function_restrict
    _profiler_config.restrict_modules = opts.module_restrict
    _profiler_config.verbosity = opts.verbosity

    print("[omnitrace]> profiling: {}".format(argv))

    sys.argv[:] = argv
    if opts.setup is not None:
        # Run some setup code outside of the profiler. This is good for large
        # imports.
        setup_file = find_script(opts.setup)
        __file__ = setup_file
        __name__ = "__main__"
        # Make sure the script's directory is on sys.path
        sys.path.insert(0, os.path.dirname(setup_file))
        ns = locals()
        execfile(setup_file, ns, ns)

    from . import Profiler, FakeProfiler

    script_file = find_script(sys.argv[0])
    __file__ = script_file
    __name__ = "__main__"
    # Make sure the script's directory is on sys.path
    sys.path.insert(0, os.path.dirname(script_file))

    prof = Profiler()
    fake = FakeProfiler()

    if PY3:
        import builtins
    else:
        import __builtin__ as builtins

    builtins.__dict__["profile"] = prof
    builtins.__dict__["noprofile"] = fake
    builtins.__dict__["trace"] = prof
    builtins.__dict__["notrace"] = fake

    try:
        try:
            if not opts.builtin:
                prof.start()
            execfile_ = execfile
            ns = locals()
            if opts.builtin:
                execfile(script_file, ns, ns)
            else:
                prof.runctx("execfile_(%r, globals())" % (script_file,), ns, ns)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            if not opts.builtin:
                prof.stop()
            del prof
            del fake
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10)
        print("Exception - {}".format(e))


if __name__ == "__main__":
    main()
    from .libpyomnitrace import finalize

    finalize()
