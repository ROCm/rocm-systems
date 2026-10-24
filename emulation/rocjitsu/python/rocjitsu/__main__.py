# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bootstrap Python, or forward native CLI flags before an explicit separator."""

import os
import sys

from ._launch import environment, launcher


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m rocjitsu <script.py | -m module | -c code> [args ...]")
        print(
            "       python -m rocjitsu [native options] -- [Python options] <script ...>"
        )
        print("       python -m rocjitsu --native <native CLI arguments ...>")
        print(
            "Without native launch options, call rocjitsu.enable(config) before GPU discovery."
        )
        print(
            "Example: python -m rocjitsu --config gpu.json --cpu-thread-budget 2 -- app.py"
        )
        return
    executable = launcher()
    if args[0] == "--native":
        command = args[1:]
    elif args[0].startswith("--") and args[0] != "--":
        if "--" not in args:
            # Diagnostic and server modes do not require a Python application.
            command = args
        else:
            index = args.index("--")
            command = [*args[:index], "--", sys.executable, *args[index + 1 :]]
    else:
        if args[0] == "--":
            args = args[1:]
        command = ["--preload-only", "--", sys.executable, *args]
    os.execve(executable, [executable, *command], environment())


if __name__ == "__main__":
    main()
