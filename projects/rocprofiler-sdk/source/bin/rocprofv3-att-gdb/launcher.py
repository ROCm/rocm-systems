#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Launch a single application under ROCgdb with breakpoint-controlled ATT."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "share/rocprofiler-sdk"))
from rocprofv3_att_gdb_common import parse_duration, parse_skip


def executable(value, fallback):
    candidate = value or next((str(p) for p in fallback if Path(p).is_file()), None)
    found = shutil.which(candidate) if candidate else None
    if not found:
        raise ValueError(f"executable not found: {value or fallback[0]}")
    return os.path.abspath(found)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Single-process CPU-breakpoint ATT prototype. Remaining options go to rocprofv3.",
        usage="%(prog)s [--start LOCATION (--stop LOCATION | --timeout 10ms)] [ATT options] -- APPLICATION [args]",
        epilog="Without --start, use 'att arm --start LOCATION --stop LOCATION' inside ROCgdb, then 'run'. "
        "Choose a start after GPU initialization. Application ROCTx Pause/Resume shares prototype control state.",
        allow_abbrev=False,
    )
    parser.add_argument("--start", help="CPU function or file:line that starts capture")
    parser.add_argument(
        "--skip",
        type=parse_skip,
        default=0,
        help="ignore the first N start-breakpoint hits; --skip 100 starts on hit 101",
        metavar="N",
    )
    parser.add_argument("--stop", help="CPU function or file:line that stops capture")
    parser.add_argument(
        "--timeout",
        type=parse_duration,
        help="one-shot interval after debugger continuation, excluding setup, e.g. 10ms",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="unattended mode: exit after the application instead of keeping ROCgdb open; nonzero on incomplete capture",
    )
    parser.add_argument(
        "--rocgdb", help="ROCgdb executable (default: next to rocprofv3 or on PATH)"
    )
    parser.add_argument(
        "--rocprofv3", help="rocprofv3 executable (default: from the same installation)"
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="GDB command executed before run; may be repeated",
    )
    parser.add_argument(
        "--preload",
        nargs="*",
        default=os.environ.get("ROCPROF_PRELOAD", "").split(":"),
        help="additional preloads, passed before the prototype helper",
    )
    # Useful when testing a standalone prototype build against an installed SDK.
    parser.add_argument("--helper", help=argparse.SUPPRESS)
    parser.add_argument("--extension", help=argparse.SUPPRESS)
    split = argv.index("--") if "--" in argv else len(argv)
    args, profiler_args = parser.parse_known_args(argv[:split])
    application = argv[split + 1 :]
    if not application:
        parser.error("provide an application after --")
    if bool(args.start) != bool(args.stop or args.timeout):
        parser.error("--start requires --stop and/or --timeout")
    if args.batch and not args.start:
        parser.error("--batch requires --start and --stop or --timeout")
    if args.skip and not args.start:
        parser.error("--skip requires --start")
    incompatible = {
        "--attach",
        "--pid",
        "-p",
        "--att-no-intercept",
        "--att-consecutive-kernels",
        "--collection-period",
        "--selected-regions-ref-count",
        "--echo",
        "-i",
        "--input",
    }
    for option in profiler_args:
        if option.split("=", 1)[0] in incompatible or re.match(r"^-[pi].+", option):
            parser.error(
                f"{option} is incompatible with this single-process launch prototype"
            )

    prefix = Path(__file__).resolve().parent.parent
    try:
        profiler = executable(
            args.rocprofv3,
            [
                prefix / "bin/rocprofv3",
                shutil.which("rocprofv3") or "/opt/rocm/bin/rocprofv3",
            ],
        )
        debugger = executable(
            args.rocgdb,
            [
                Path(profiler).parent / "rocgdb",
                shutil.which("rocgdb") or "/opt/rocm/bin/rocgdb",
            ],
        )
        helper = (
            Path(args.helper)
            if args.helper
            else next(
                (
                    p
                    for d in ("lib", "lib64")
                    if (
                        p := prefix / d / "rocprofiler-sdk/librocprofv3-att-gdb-helper.so"
                    ).is_file()
                ),
                prefix / "lib/rocprofiler-sdk/librocprofv3-att-gdb-helper.so",
            )
        )
        extension = (
            Path(args.extension)
            if args.extension
            else prefix / "share/rocprofiler-sdk/rocprofv3_att_gdb.py"
        )
        if not helper.is_file() or not extension.is_file():
            raise ValueError(
                "prototype helper/extension missing; build and install rocprofv3-att-gdb first"
            )
        # Let rocprofv3 choose its matching SDK; --rocm-root may override its installation.
        rocm_root = Path(profiler).resolve().parent.parent
        for i, option in enumerate(profiler_args):
            if option == "--rocm-root" and i + 1 < len(profiler_args):
                rocm_root = Path(profiler_args[i + 1])
            elif option.startswith("--rocm-root="):
                rocm_root = Path(option.split("=", 1)[1])
        roctx = rocm_root / "lib/librocprofiler-sdk-roctx.so"
        if not roctx.is_file():
            raise ValueError(f"ROCTx library not found: {roctx}")
        app = executable(application[0], [application[0]])
    except ValueError as error:
        parser.error(str(error))

    # Keep AF_UNIX paths below sun_path's limit even when TMPDIR is a long path.
    with tempfile.TemporaryDirectory(
        prefix="rocprofv3-att-gdb-", dir="/tmp"
    ) as directory:
        config = {
            "socket": f"{directory}/control.sock",
            "batch": args.batch,
            "start": args.start,
            "skip": args.skip,
            "stop": args.stop,
            "timeout": args.timeout or 0,
            "wrapper": [
                sys.executable,
                profiler,
                *profiler_args,
                "--att",
                "--selected-regions",
                "--selected-regions-ref-count=false",
                "--preload",
                *[p for p in args.preload if p],
                str(helper.resolve()),
                str(roctx.resolve()),
                "--",
            ],
        }
        config_path = Path(directory) / "config.json"
        config_path.write_text(json.dumps(config))
        init_path = Path(directory) / "init.gdb"
        init_path.write_text(
            "python\nimport os, gdb, traceback\nos.environ['ROCPROFV3_GDB_CONFIG'] = "
            + repr(str(config_path))
            + "\n__file__ = "
            + repr(str(extension.resolve()))
            + "\ntry:\n    exec(compile(open(__file__).read(), __file__, 'exec'))\n"
            + "except Exception:\n    traceback.print_exc()\n    gdb.execute('set confirm off')\n    gdb.execute('quit 1')\nend\n"
        )
        command = [debugger, "-q", "-nx", "-x", str(init_path)]
        for item in args.command:
            command += ["-ex", item]
        if args.start:
            command += ["-ex", "att run"]
        command += ["--args", app, *application[1:]]
        # Keep the debugger's event loop alive in batch mode. GDB's own --batch
        # would exit after the first non-stop breakpoint before deferred work runs.
        with subprocess.Popen(
            command, stdin=subprocess.PIPE if args.batch else None
        ) as process:
            while True:
                try:
                    return process.wait()
                except KeyboardInterrupt:
                    if args.batch:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        return 130
                    # The terminal also delivers SIGINT to ROCgdb; keep its session open.
                    continue


if __name__ == "__main__":
    sys.exit(main())
