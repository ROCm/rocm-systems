# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration checks for every native CLI consumer, without a GPU or SDK."""

from pathlib import Path
import subprocess
import sys
import tempfile

from cli_fixtures import write_code_object

translate, pretranslate, discovery, fuzz, launcher = (
    Path(arg).resolve() for arg in sys.argv[1:]
)


def invoke(tool, *args, status=0, cwd=None):
    result = subprocess.run(
        [str(tool), *map(str, args)], cwd=cwd, capture_output=True, timeout=20
    )
    assert result.returncode == status, (
        tool.name,
        args,
        result.returncode,
        result.stdout,
        result.stderr,
    )
    return result.stdout + result.stderr


for tool, failure in (
    (translate, 1),
    (pretranslate, 2),
    (discovery, 1),
    (fuzz, 2),
    (launcher, 1),
):
    assert b"Usage:" in invoke(tool, "--help")
    assert b"--help, -h" in invoke(tool, "-h")
    assert b"unknown flag" in invoke(tool, "--help", "--unknown", status=failure)

launcher_help = invoke(launcher, "--help")
assert b"launch overrides do not rewrite the source config" in launcher_help
assert b"off disables checks" in launcher_help
assert b"on enables ordinary checks" in launcher_help
assert b"all includes gfx1250 XCNT checks" in launcher_help
assert b"ROCJITSU_ENABLE_VFIO=ON" in launcher_help
assert b"memory-backend-memfd" in launcher_help
assert b"memory-backend=mem" in launcher_help

assert b"application arguments must follow --" in invoke(
    launcher, "bare-application", status=1
)
assert b"requires a value" in invoke(launcher, "--config", status=1)
assert b"nonempty path" in invoke(launcher, "--vfio-socket=", status=1)
assert b"requires a value" in invoke(pretranslate, "--store-root", status=2)
assert b"unsigned integer" in invoke(
    launcher, "--cpu-thread-budget=4294967296", status=1
)
assert b"unsigned integer" in invoke(launcher, "--vfio-ready-fd=2147483648", status=1)
assert b"expected a decimal or hexadecimal" in invoke(
    translate, "--code-object-index=0x", status=1
)
assert b"at most 65535" in invoke(
    translate, "--debug-conservative-liveness=65536", status=1
)
assert b"unsupported value" in invoke(translate, "--input-target=unknown", status=1)
assert b"unsupported value" in invoke(translate, "--output-mode=unknown", status=1)
assert b"requires a value" in invoke(translate, "--input-revision", status=1)
assert b"Usage:" in invoke(fuzz, "unexpected-positional", status=2)

with tempfile.TemporaryDirectory(prefix="rj-cli-tools-") as directory:
    root = Path(directory)
    source = root / "-input.co"
    write_code_object(source)
    assert b"no generation named" in invoke(
        discovery, "missing", root / "output.bin", status=1
    )
    invoke(discovery, "--", "gfx1250", "-discovery.bin", cwd=root)
    assert (root / "-discovery.bin").stat().st_size > 0
    # The pretranslator should distinguish expected target rejections from I/O failures.
    skipped = invoke(
        pretranslate, f"--store-root={root / 'store'}", "--portable", "--", source
    )
    assert b"skipped=1 failed=0" in skipped, skipped
    skipped = invoke(
        pretranslate,
        "--store-root",
        root / "store",
        "--fail-on-skipped",
        source,
        status=1,
    )
    assert b"skipped=1 failed=0" in skipped, skipped
    failed = invoke(
        pretranslate, "--store-root", root / "store", root / "absent.co", status=1
    )
    assert b"cannot read" in failed, failed
    # Repeated options exercise the full decoder path and prove the last value wins.
    sample = root / "sample.bin"
    sample.write_bytes(bytes.fromhex("000080bf") + bytes(12))
    replay = invoke(
        fuzz, f"--input={sample}", "--json", "--target=bogus", "--target=gfx950"
    )
    assert b'"mnemonic":"s_nop"' in replay.replace(b" ", b""), replay
    assert b"unsupported decoder target: bogus" in invoke(
        fuzz,
        f"--input={sample}",
        "--json",
        "--target=gfx950",
        "--target=bogus",
        status=2,
    )
    # A leading-dash input reaches domain validation. This minimal fixture
    # intentionally has no kernel descriptor; full translation is covered by the smoke suite.
    translated = invoke(
        translate,
        "--input-target=gfx950",
        "--output-target=gfx950",
        "--output-mode=code-object",
        "--",
        "-input.co",
        cwd=root,
        status=3,
    )
    assert b"kernel descriptors are required" in translated, translated
