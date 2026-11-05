#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the dynolog RDC shim (``dynolog/src/gpumon/amd``) against a built RDC.

Configures and builds the standalone target defined by dynolog's
``src/gpumon/amd/CMakeLists.txt`` -- the ``dynolog_rdc_lib`` shim plus the
``RdcWrapperExample`` binary -- against the RDC that was installed under
``--rocm-dir``. This compile+link is the primary CI signal: it fails when an
RDC change breaks the header/library surface the dynolog integration depends
on.

The standalone build gates its metric set on ``DYNOLOG_ROCM_VERSION``; if that
is left undefined every metric is preprocessed out and the example aborts at
runtime with "enabledMetrics is empty". The version code is derived from the
installed ROCm the same way dynolog's own CMake derives it -- from
``<rocm-dir>/.info/version`` as ``major*10000 + minor*100 + patch`` (e.g.
``70200`` for ROCm 7.2.0) -- and passed explicitly as ``-DROCM_VERSION``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

from ._common import configure_logging, gh_error, gh_warning, run

logger = logging.getLogger("dynolog.build")

# Relative path from a dynolog checkout to the standalone RDC shim CMake project.
_SHIM_CMAKE_SUBDIR = Path("dynolog/src/gpumon/amd")
# Where the example binary lands inside the shim build tree.
_EXAMPLE_SUBPATH = Path("examples/RdcWrapperExample")

# Fallback used only when the installed ROCm version cannot be read. 70000 keeps
# the full ROCm 7.0 metric set enabled so the example is still meaningful.
_DEFAULT_ROCM_VERSION_CODE = 70000

_VERSION_RE = re.compile(r"\d+")


def rocm_version_code(rocm_dir: Path) -> int:
    """Return ``major*10000 + minor*100 + patch`` for the ROCm at ``rocm_dir``.

    Reads ``<rocm_dir>/.info/version`` (e.g. ``7.2.0-...``) and mirrors the
    version-code math in dynolog's ``src/gpumon/amd/CMakeLists.txt``. Falls back
    to ``_DEFAULT_ROCM_VERSION_CODE`` with a warning if the file is missing or
    unparsable.
    """
    version_file = rocm_dir / ".info" / "version"
    try:
        raw = version_file.read_text().splitlines()[0]
    except (OSError, IndexError):
        gh_warning(
            f"Could not read {version_file}; defaulting ROCM_VERSION to "
            f"{_DEFAULT_ROCM_VERSION_CODE}. Pass --rocm-version to override."
        )
        return _DEFAULT_ROCM_VERSION_CODE

    parts = _VERSION_RE.findall(raw)
    if len(parts) < 3:
        gh_warning(
            f"Unexpected version string {raw!r} in {version_file}; defaulting "
            f"ROCM_VERSION to {_DEFAULT_ROCM_VERSION_CODE}."
        )
        return _DEFAULT_ROCM_VERSION_CODE

    major, minor, patch = (int(p) for p in parts[:3])
    code = major * 10000 + minor * 100 + patch
    logger.info("Detected ROCm %s.%s.%s -> ROCM_VERSION=%d", major, minor, patch, code)
    return code


def build(
    *,
    dynolog_dir: Path,
    rocm_dir: Path,
    rocm_version: int,
    build_dir: Path,
    build_type: str,
    jobs: int,
) -> Path:
    """Configure + build the shim/example, returning the example binary path."""
    shim_dir = dynolog_dir / _SHIM_CMAKE_SUBDIR
    if not (shim_dir / "CMakeLists.txt").is_file():
        gh_error(f"dynolog RDC shim not found at {shim_dir}")
        raise SystemExit(1)

    # The shim links the `rdc` target only (embedded mode), whose interface is
    # rdc_bootstrap/pthread/amd_smi/cap -- no gRPC -- so the ROCm prefix is the
    # only one find_package(rdc/amd_smi) needs.
    run(
        [
            "cmake",
            "-S",
            str(shim_dir),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            f"-DROCM_VERSION={rocm_version}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_PREFIX_PATH={rocm_dir}",
        ]
    )
    run(["cmake", "--build", str(build_dir), "--parallel", str(jobs)])

    example = build_dir / _EXAMPLE_SUBPATH
    if not example.is_file():
        gh_error(f"Build succeeded but example binary is missing: {example}")
        raise SystemExit(1)
    logger.info("Built dynolog RDC example: %s", example)
    return example


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynolog-dir", required=True, type=Path)
    parser.add_argument("--rocm-dir", default=Path("/opt/rocm"), type=Path)
    parser.add_argument(
        "--rocm-version",
        type=int,
        default=None,
        help="ROCM_VERSION code (major*10000+minor*100+patch). Auto-detected if omitted.",
    )
    parser.add_argument("--build-dir", default=None, type=Path)
    parser.add_argument("--build-type", default="RelWithDebInfo")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    build_dir = args.build_dir or (args.dynolog_dir / _SHIM_CMAKE_SUBDIR / "build")
    rocm_version = (
        args.rocm_version if args.rocm_version is not None else rocm_version_code(args.rocm_dir)
    )
    build(
        dynolog_dir=args.dynolog_dir,
        rocm_dir=args.rocm_dir,
        rocm_version=rocm_version,
        build_dir=build_dir,
        build_type=args.build_type,
        jobs=args.jobs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
