# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess
import sys


def invoke(*args, status=0):
    result = subprocess.run(
        [sys.argv[1], *args], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == status, (args, result.returncode, result.stderr)
    return result.stdout if status == 0 else result.stderr


help_text = invoke("run", "--help")
assert help_text.index("Options:") < help_text.index("Run settings:")
assert "--max-items" in help_text
assert "Run settings:" in invoke("--help", "run", "--help")
assert "--limit=128" in invoke("run", "-l", "128", "input.bin")
assert "positional=--limit" in invoke("run", "--", "--limit")
assert "--label=" in invoke("run", "--label=")
assert "--limit=2" in invoke("run", "-l=1", "--limit=2")
assert "requires a value" in invoke("run", "--limit", status=1)
assert "unknown flag" in invoke("run", "--unknown", status=1)
assert "unknown command" in invoke("rnu", status=1)
assert "subcommand is required" in invoke(status=1)
prefix = ("--plugin", sys.argv[2], "plugin", "inspect")
help_text = invoke(*prefix, "--help")
assert "--format, -f" in help_text
assert "Output formatting:" in help_text
assert "Run settings:" not in help_text
output = invoke(*prefix, "-f=json", "input.bin", "--", "--format=bad")
assert "command=rj_cli_probe plugin inspect" in output
assert "--format=json" in output
assert "positional=--format=bad" in output
assert "expected text or json" in invoke(*prefix, "--format=bad", status=1)
assert "unknown flag" in invoke("run", "--plugin", sys.argv[2], status=1)
