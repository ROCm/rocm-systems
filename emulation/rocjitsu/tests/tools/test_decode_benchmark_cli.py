# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the actual benchmark executable without a GPU or assembler."""

from pathlib import Path
import subprocess
import sys
import tempfile

from cli_fixtures import write_code_object


def invoke(*args, status=0):
    result = subprocess.run(
        [sys.argv[1], *map(str, args)], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == status, (args, result.returncode, result.stderr)
    return result.stdout if status == 0 else result.stderr


with tempfile.TemporaryDirectory() as directory:
    source = Path(directory) / "input.co"
    write_code_object(source)
    existing = invoke(
        source, "--target", "gfx950", "--iterations", "2", "--invalid-limit", "1"
    )
    assert "2 valid instructions" in existing, existing
    assert "valid: 4 attempts" in existing, existing
    assert "invalid: 2 attempts" in existing, existing
    assert existing.count("0 unexpected") == 2, existing
    extended = invoke("-t=gfx950", "-n=2", "--invalid-limit=0", "--", source)
    assert "valid: 4 attempts" in extended, extended
    help_text = invoke(source, "--help")
    assert "--iterations, -n" in help_text
    assert "Decoder settings:" in help_text
    assert "exactly one" in invoke(status=1)
    assert "exactly one" in invoke("", status=1)
    assert "unsupported decoder target" in invoke(source, "--target=bad", status=1)
    assert "expected an integer" in invoke(source, "--iterations=0", status=1)
    assert "requires a value" in invoke(source, "--target", status=1)
    assert "valid: 4 attempts" in invoke(
        source, "-n=1", "--iterations=2", "--invalid-limit=0"
    )
    assert "exactly one" in invoke(source, source, status=1)
    assert "unknown flag" in invoke("--help", "--unknown", status=1)
    assert "does not match" in invoke(source, "--target=gfx942", status=2)
