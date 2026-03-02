#!/usr/bin/env python3
"""
Install WSL (libhsakmt) build from S3 artifacts.

Downloads and extracts the WSL artifact tarball(s) from a given CI run
(therock-wsl-artifacts-external bucket, path {run_id}-windows/artifacts/).
Similar in spirit to TheRock's install_rocm_from_artifacts.py.

Requires: AWS CLI on PATH. For public read, no credentials needed; otherwise
configure AWS (e.g. same role as upload).

Usage:
  python install_wsl_from_artifacts.py --run-id RUN_ID [--output-dir DIR] [--s3-bucket BUCKET]
  python install_wsl_from_artifacts.py --run-id 123456789 --output-dir ./wsl-libhsakmt

Example (after ROCm install from TheRock script):
  python install_rocm_from_artifacts.py --run-id R --amdgpu-family gfx94X-dcgpu --output-dir therock-build
  python install_wsl_from_artifacts.py --run-id W --output-dir wsl-libhsakmt
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def log(*args):
    print(*args)
    sys.stdout.flush()


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    log(f"++ $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract WSL (libhsakmt) artifacts from S3.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="GitHub workflow run ID that produced the WSL artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wsl-libhsakmt"),
        help="Install prefix: libs/headers/bin will go here (default: wsl-libhsakmt).",
    )
    parser.add_argument(
        "--s3-bucket",
        default="therock-wsl-artifacts-external",
        help="S3 bucket name (default: therock-wsl-artifacts-external).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be downloaded.",
    )
    args = parser.parse_args()

    if not shutil.which("aws"):
        log("ERROR: AWS CLI not found on PATH.")
        sys.exit(1)

    platform_suffix = "windows"
    s3_prefix = f"s3://{args.s3_bucket}/{args.run_id}-{platform_suffix}/artifacts/"
    # We upload only *.tar.xz; the WSL workflow produces wsl-libhsakmt-build.tar.xz
    artifact_name = "wsl-libhsakmt-build.tar.xz"
    s3_uri = f"{s3_prefix}{artifact_name}"

    log(f"Run ID: {args.run_id}")
    log(f"S3 prefix: {s3_prefix}")
    log(f"Output dir: {args.output_dir.resolve()}")

    if args.dry_run:
        log("Dry run: would download", s3_uri, "and extract to", args.output_dir)
        return

    # Download to a temp dir; archive has top-level "build/" from CI.
    tmp_dir = args.output_dir.parent / (args.output_dir.name + ".tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        log("Downloading", s3_uri)
        run_cmd(
            [
                "aws",
                "s3",
                "cp",
                s3_uri,
                str(tmp_dir / artifact_name),
                "--region",
                "us-east-2",
            ],
        )
        archive_path = tmp_dir / artifact_name
        if not archive_path.is_file():
            log("ERROR: Download did not produce", archive_path)
            sys.exit(1)

        log("Extracting (install prefix:", args.output_dir, ")")
        with tarfile.open(archive_path) as tf:
            tf.extractall(tmp_dir)
        archive_path.unlink(missing_ok=True)

        # Archive contains top-level "build/"; use its contents as the install prefix.
        inner_build = tmp_dir / "build"
        if not inner_build.is_dir():
            log("ERROR: Archive has no top-level 'build/' directory.")
            sys.exit(1)
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        shutil.move(str(inner_build), str(args.output_dir))
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    log("Done. WSL libhsakmt installed to", args.output_dir.resolve())


if __name__ == "__main__":
    main()
