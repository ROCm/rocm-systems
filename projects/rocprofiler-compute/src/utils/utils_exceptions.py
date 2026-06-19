# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import shlex


class WorkloadCommandError(Exception):
    """Base for all workload command validation errors raised by sanitize()."""


class ExecutableNotFoundError(WorkloadCommandError, FileNotFoundError):
    """The command could not be resolved to an executable via PATH."""

    def __init__(self, command: str) -> None:
        self.command = command
        super().__init__(f"'{command}' doesn't point to an executable. Please verify.")


class PythonScriptNotFoundError(WorkloadCommandError, FileNotFoundError):
    """A Python script was specified but does not exist on disk."""

    def __init__(self, script: str) -> None:
        self.script = script
        super().__init__(f"Python script not found: {script}")


class NoScriptInCommandError(WorkloadCommandError, ValueError):
    """A Python interpreter was invoked without a script argument."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        super().__init__(
            f"No Python script found in the workload command: {shlex.join(argv)}"
        )
