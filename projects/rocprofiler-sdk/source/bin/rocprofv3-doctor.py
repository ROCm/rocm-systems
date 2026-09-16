#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""rocprofv3-doctor: diagnose a ROCprofiler-SDK installation and system setup."""

import argparse
import os
import sys

# version info for rocprofiler-sdk / rocprofv3, substituted by configure_file
CONST_VERSION_INFO = {
    "version": "@FULL_VERSION_STRING@",
    "git_revision": "@ROCPROFILER_SDK_GIT_REVISION@",
    "rocm_version": "@rocm_version_FULL_VERSION@",
}


def _substituted(value):
    """False when configure_file has not replaced an @VAR@ placeholder."""
    return bool(value) and not (value.startswith("@") and value.endswith("@"))


def _script_prefix():
    """``$PREFIX`` for the installed layout ``$PREFIX/bin/rocprofv3-doctor``."""
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _bootstrap_package(rocm_root_override):
    """Import the rocprofv3 package, falling back to the installed site-packages.

    The package is looked for next to this script first, so the checks always
    match the tool version. Locating the ROCm root to *inspect* is the
    package's job (doctor_layout.detect_rocm_root), because that root is not
    necessarily where this script lives -- nor under /opt/rocm.
    """
    try:
        from rocprofv3 import doctor

        return doctor
    except ImportError:
        pass

    prefixes = [_script_prefix()]
    if rocm_root_override:
        prefixes.append(os.path.abspath(rocm_root_override))

    candidates = []
    for prefix in prefixes:
        candidates.append(os.path.join(prefix, "lib", "python3", "site-packages"))
        candidates.append(
            os.path.join(
                prefix, "lib", "python{}".format(sys.version_info[0]), "site-packages"
            )
        )
    # running straight out of a source checkout
    candidates.append(os.path.join(_script_prefix(), "lib", "python"))

    # Prepend rather than append, and drop any cached partial import: a
    # rocprofv3.py sitting next to this script (or in the cwd) would otherwise
    # shadow the rocprofv3 *package* and make the import fail permanently.
    for candidate in reversed(candidates):
        if os.path.isdir(candidate):
            if candidate in sys.path:
                sys.path.remove(candidate)
            sys.path.insert(0, candidate)

    for name in list(sys.modules):
        if name == "rocprofv3" or name.startswith("rocprofv3."):
            del sys.modules[name]

    from rocprofv3 import doctor

    return doctor


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        prog="rocprofv3-doctor",
        description=(
            "Inspect the ROCprofiler-SDK installation and system configuration, "
            "and report what would prevent rocprofv3 from working."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes:\n"
            "  0  no check failed (warnings are not failures)\n"
            "  1  at least one check failed\n"
            "  2  rocprofv3-doctor itself could not run, an --only pattern\n"
            "     matched no check, or a check broke (status error) and none failed\n"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="PATTERN",
        default=None,
        help="run only checks whose id or group matches this glob (repeatable)",
    )
    parser.add_argument(
        "--skip",
        action="append",
        metavar="PATTERN",
        default=None,
        help="skip checks whose id or group matches this glob (repeatable)",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        default=False,
        help="print the available check ids and groups, then exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="show details and structured data for passing checks too",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="show only failing and warning checks",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="disable ANSI color codes",
    )
    parser.add_argument(
        "--rocm-root",
        metavar="PATH",
        default=None,
        help=(
            "ROCm root directory to inspect (default: auto-detected from this "
            "script's location, ROCM_PATH/ROCM_HOME/ROCM_DIR, rocprofv3 on PATH, "
            "an installed TheRock Python package, then /opt/rocm and "
            "/opt/rocm/core-*)"
        ),
    )
    parser.add_argument(
        "--run-smoke-test",
        action="store_true",
        default=False,
        help=(
            "also run the opt-in smoke checks, which run rocprofv3 itself "
            "(may take 30+ seconds); naming them with --only also runs them"
        ),
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="write the report to this file in addition to stdout",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        default=False,
        help="print version information and exit",
    )
    # accepted so that `rocprofv3 --doctor` can forward its argv verbatim
    parser.add_argument(
        "--doctor",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _use_color(args):
    if args.no_color or args.format == "json":
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    return bool(getattr(sys.stdout, "isatty", None)) and sys.stdout.isatty()


def _header(doctor, accessor):
    tool_version = CONST_VERSION_INFO["version"]
    rocm_version = CONST_VERSION_INFO["rocm_version"]
    parts = ["rocprofv3-doctor"]
    if _substituted(tool_version):
        parts.append("v{}".format(tool_version))
    if _substituted(rocm_version):
        parts.append("(ROCm {})".format(rocm_version))
    return "{}\nChecking: {}".format(
        " ".join(parts), doctor.describe_installation(accessor)
    )


def main(argv=None):
    args = parse_arguments(argv)

    try:
        doctor = _bootstrap_package(args.rocm_root)
    except ImportError as exc:
        sys.stderr.write(
            "rocprofv3-doctor: cannot import the rocprofv3 Python package "
            "({}).\n"
            "Try: export PYTHONPATH={}/lib/python3/site-packages:$PYTHONPATH\n".format(
                exc, _script_prefix()
            )
        )
        return 2

    try:
        doctor.validate_registry()
    except ValueError as exc:
        sys.stderr.write("rocprofv3-doctor: invalid check registry ({})\n".format(exc))
        return doctor.EXIT_TOOL_ERROR

    if args.version:
        for key, value in CONST_VERSION_INFO.items():
            print("    {:>16}: {}".format(key, value))
        return 0

    if args.list_checks:
        sys.stdout.write(doctor.render_check_list())
        return 0

    rocm_dir, rocm_root_source = doctor.resolve_rocm_root(__file__, args.rocm_root)
    tool_version = CONST_VERSION_INFO["version"]
    accessor = doctor.build_accessor(
        rocm_root=rocm_dir,
        tool_version=tool_version if _substituted(tool_version) else None,
        rocm_root_source=rocm_root_source,
    )

    try:
        outcomes = doctor.run(
            accessor,
            only=args.only,
            skip=args.skip,
            include_default_disabled=args.run_smoke_test,
        )
    except doctor.SelectionError as exc:
        sys.stderr.write("rocprofv3-doctor: {}\n".format(exc))
        return doctor.EXIT_TOOL_ERROR
    except Exception as exc:  # noqa: BLE001 -- the runner itself failing is exit 2
        sys.stderr.write("rocprofv3-doctor: internal error ({})\n".format(exc))
        return doctor.EXIT_TOOL_ERROR

    if args.format == "json":
        report = doctor.render_json(
            outcomes,
            tool_version=(
                CONST_VERSION_INFO["version"] if _substituted(tool_version) else ""
            ),
            rocm_version=(
                CONST_VERSION_INFO["rocm_version"]
                if _substituted(CONST_VERSION_INFO["rocm_version"])
                else ""
            ),
            rocm_root=rocm_dir,
            rocm_root_source=rocm_root_source,
            install_kind=doctor.installation_kind(accessor),
        )
    else:
        report = doctor.render_text(
            outcomes,
            header=_header(doctor, accessor),
            use_color=_use_color(args),
            verbose=args.verbose,
            quiet=args.quiet,
        )

    sys.stdout.write(report)

    if args.output:
        try:
            with open(args.output, "w") as handle:
                if args.format == "json":
                    handle.write(report)
                else:
                    # the file copy is always uncolored so it stays readable
                    handle.write(
                        doctor.render_text(
                            outcomes,
                            header=_header(doctor, accessor),
                            use_color=False,
                            verbose=args.verbose,
                            quiet=args.quiet,
                        )
                    )
        except (IOError, OSError) as exc:
            sys.stderr.write(
                "rocprofv3-doctor: cannot write {} ({})\n".format(args.output, exc)
            )
            return doctor.EXIT_TOOL_ERROR

    return doctor.exit_code(outcomes)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
