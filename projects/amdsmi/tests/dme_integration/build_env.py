#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lay out the GPU Agent build tree against an installed AMDSMI.

GPU Agent's Makefile hard-codes ``ABS_DIR`` to a fixed absolute path, and
expects AMDSMI headers/libraries under
``sw/nic/third-party/rocm/amd_smi_lib``. This helper:

* Creates a symlink from ``--gpu-agent-workdir`` -> ``--gpuagent-src``.
* Copies the installed AMDSMI headers and shared libraries into both the
  third-party staging dir and the gpu-agent runtime ``lib`` directory.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from ._common import configure_logging, run

logger = logging.getLogger("dme.build_env")


def _copy_libs(src_dir: Path, dst_dir: Path, glob: str) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    matches = sorted(src_dir.glob(glob))
    if not matches:
        raise FileNotFoundError(f"No files matching {glob} in {src_dir}")
    for src in matches:
        dst = dst_dir / src.name
        if src.is_symlink():
            link_target = src.readlink()
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(link_target)
        else:
            shutil.copy2(src, dst)
        logger.info("copied %s -> %s", src, dst)
    return len(matches)


def _copy_tree(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory missing: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for entry in src_dir.iterdir():
        target = dst_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True, symlinks=True)
        else:
            shutil.copy2(entry, target)


def prepare(*, gpuagent_src: Path, gpu_agent_workdir: Path, rocm_dir: Path) -> None:
    # Symlink to satisfy gpu-agent Makefile's hard-coded ABS_DIR.
    gpu_agent_workdir.parent.mkdir(parents=True, exist_ok=True)
    if gpu_agent_workdir.is_symlink() or gpu_agent_workdir.exists():
        run(["rm", "-rf", str(gpu_agent_workdir)])
    gpu_agent_workdir.symlink_to(gpuagent_src)
    logger.info("symlinked %s -> %s", gpu_agent_workdir, gpuagent_src)

    rocm_lib = rocm_dir / "lib"
    rocm_include = rocm_dir / "include/amd_smi"

    third_party = gpuagent_src / "sw/nic/third-party/rocm/amd_smi_lib"
    _copy_libs(rocm_lib, third_party / "x86_64/lib", "libamd_smi.so*")
    _copy_tree(rocm_include, third_party / "include")

    # gpu-agent's runtime lib dir for sim builds
    sim_lib = gpuagent_src / "sw/nic/build/x86_64/sim/lib"
    _copy_libs(rocm_lib, sim_lib, "libamd_smi.so*")

    logger.info("GPU Agent build environment ready")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpuagent-src", required=True, type=Path)
    parser.add_argument("--gpu-agent-workdir", required=True, type=Path)
    parser.add_argument("--rocm-dir", required=True, type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    prepare(
        gpuagent_src=args.gpuagent_src,
        gpu_agent_workdir=args.gpu_agent_workdir,
        rocm_dir=args.rocm_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
