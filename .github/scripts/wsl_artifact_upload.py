#!/usr/bin/env python3
"""
Standalone WSL artifact upload for the ROCR Runtime WSL workflow.

Uploads staged build artifacts and logs to S3 using the same path shape as
TheRock's post_build_upload.py. No dependency on TheRock; to be replaced by
TheRock's post_build_upload.py once those changes are merged.

Requires: --s3-bucket (when uploading), AWS CLI on PATH with credentials
(e.g. OIDC role therock-wsl-artifacts-external). Bucket used by this workflow:
therock-wsl-artifacts-external.

Usage:
  wsl_artifact_upload.py --run-id RUN_ID --artifact-group GROUP
    [--build-dir DIR] [--upload | --no-upload] [--job-status STATUS]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def log(*args):
    print(*args)
    sys.stdout.flush()


def run_command(cmd: list, cwd: Path | None = None):
    log(f"++ Exec $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or Path.cwd())


def create_ninja_log_archive(build_dir: Path):
    """Create logs/ninja_logs.tar.gz from any .ninja_log under build_dir."""
    log_dir = build_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    found = list(build_dir.glob("**/.ninja_log"))
    if not found:
        log("No .ninja_log files found. Skipping ninja log archive.")
        return
    archive_path = log_dir / "ninja_logs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in found:
            tar.add(p)
            log(f"[+] Add: {p}")
    log(f"[*] Created {archive_path}")


def upload_artifacts(build_dir: Path, bucket_uri: str, region: str = "us-east-2"):
    """Upload artifacts/*.tar.xz to bucket_uri."""
    artifacts_dir = build_dir / "artifacts"
    if not artifacts_dir.is_dir():
        log(f"[INFO] No artifacts dir at {artifacts_dir}. Skipping.")
        return
    run_command(
        [
            "aws",
            "s3",
            "cp",
            str(artifacts_dir),
            bucket_uri,
            "--recursive",
            "--no-follow-symlinks",
            "--exclude",
            "*",
            "--include",
            "*.tar.xz*",
            "--region",
            region,
        ],
    )


def upload_logs(
    artifact_group: str, build_dir: Path, bucket_uri: str, region: str = "us-east-2"
):
    """Upload logs/ to bucket_uri/logs/<artifact_group>/."""
    log_dir = build_dir / "logs"
    if not log_dir.is_dir():
        log(f"[INFO] No log dir at {log_dir}. Skipping.")
        return
    s3_logs = f"{bucket_uri}/logs/{artifact_group}"
    run_command(
        ["aws", "s3", "cp", str(log_dir), s3_logs, "--recursive", "--region", region],
    )


def write_gha_summary(bucket_url: str, artifact_group: str, job_status: str):
    """Append Build Logs and Artifacts links to GITHUB_STEP_SUMMARY."""
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_file:
        log("GITHUB_STEP_SUMMARY not set. Skipping job summary.")
        return
    # Logs: link to logs subfolder (index.html may not exist until TheRock is used)
    lines = [f"[Build Logs]({bucket_url}/logs/{artifact_group}/)"]
    if not job_status or job_status == "success":
        # Artifacts: link to run prefix (no index page in standalone mode)
        lines.append(f"[Artifacts]({bucket_url}/)")
    body = "\n".join(lines) + "\n\n"
    with open(summary_file, "a") as f:
        f.write(body)
    log("Wrote job summary links.")


def main():
    parser = argparse.ArgumentParser(
        description="Upload WSL build artifacts and logs to S3 (standalone for rocm-systems).",
    )
    parser.add_argument("--run-id", required=True, help="GitHub run ID")
    parser.add_argument(
        "--artifact-group",
        required=True,
        help="Artifact group name (e.g. wsl-libhsakmt)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.getenv("BUILD_DIR", "build")),
        help="Staging dir with artifacts/ and logs/ (default: BUILD_DIR or 'build')",
    )
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=str(os.getenv("CI", "false")).lower() == "true",
        help="Perform S3 upload (default: true if CI is set)",
    )
    parser.add_argument(
        "--job-status", default="", help="Job status: success, failure, etc."
    )
    parser.add_argument(
        "--s3-bucket",
        default="",
        help="S3 bucket name for uploads (required when --upload).",
    )
    args = parser.parse_args()

    if not args.build_dir.is_dir():
        log(f"Build dir not found: {args.build_dir}. Exiting.")
        sys.exit(1)

    log("Creating ninja log archive")
    log("--------------------------")
    create_ninja_log_archive(args.build_dir)

    if not args.upload:
        log("Upload disabled. Done.")
        return

    bucket = (args.s3_bucket or "").strip()
    if not bucket:
        log("--s3-bucket is required when uploading. Cannot upload.")
        sys.exit(1)
    if not shutil.which("aws"):
        log("AWS CLI not found on PATH.")
        sys.exit(1)

    # Same path shape as TheRock: no repo prefix (dedicated bucket), run_id-PLATFORM
    platform = "windows"
    path_suffix = f"{args.run_id}-{platform}"
    bucket_uri = f"s3://{bucket}/{path_suffix}"
    bucket_url = f"https://{bucket}.s3.amazonaws.com/{path_suffix}"
    region = "us-east-2"

    log("Upload build artifacts")
    log("----------------------")
    if not args.job_status or args.job_status == "success":
        upload_artifacts(args.build_dir, bucket_uri, region)
    else:
        log("Job did not succeed; skipping artifact upload.")

    log("Upload logs")
    log("----------")
    upload_logs(args.artifact_group, args.build_dir, bucket_uri, region)

    log("Write job summary")
    log("-----------------")
    write_gha_summary(bucket_url, args.artifact_group, args.job_status)


if __name__ == "__main__":
    main()
