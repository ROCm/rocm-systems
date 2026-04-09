#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Download and extract the latest GPUOpen Machine-Readable ISA archive."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
import zipfile
from pathlib import Path

LATEST_URL = "https://gpuopen.com/download/machine-readable-isa/latest/"
CURRENT_URL = "https://gpuopen.com/download/AMD_GPU_MR_ISA_XML_2025_09_05.zip"
DEFAULT_URL = CURRENT_URL
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the latest GPUOpen Machine-Readable ISA archive and "
            "extract it into a local directory."
        )
    )
    parser.add_argument(
        "-u",
        "--url",
        default=DEFAULT_URL,
        help=f"Download URL to fetch (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("third_party/machine-readable-isa"),
        help="Directory where the archive will be extracted",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Optional path for the downloaded archive file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing archive file and extracted output directory",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Network timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )
    return parser.parse_args()


def archive_name_from_response(response: urllib.response.addinfourl) -> str:
    content_disposition = response.headers.get("Content-Disposition", "")
    if "filename=" in content_disposition:
        _, _, raw_filename = content_disposition.partition("filename=")
        filename = raw_filename.strip().strip('"')
        if filename:
            return Path(filename).name

    redirected_url = response.geturl()
    parsed = urllib.parse.urlparse(redirected_url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    if name:
        return name
    return "machine-readable-isa.zip"


def download_archive(url: str, destination: Path, timeout: float) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "rocjitsu-mr-isa-downloader/1.0",
            "Accept": "application/zip, application/octet-stream, */*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            filename = archive_name_from_response(response)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file, length=CHUNK_SIZE)
            return filename
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc}") from exc


def ensure_extract_target(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise RuntimeError(
                f"output directory already exists: {path} (pass --force to replace it)"
            )
        if not path.is_dir():
            raise RuntimeError(f"output path exists and is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_archive_target(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise RuntimeError(
            f"archive file already exists: {path} (pass --force to replace it)"
        )


def validate_zip_member(member: zipfile.ZipInfo, destination: Path) -> Path:
    member_path = Path(member.filename)
    if member_path.is_absolute():
        raise RuntimeError(f"refusing to extract absolute path from archive: {member.filename}")

    resolved_destination = destination.resolve()
    resolved_member = (destination / member_path).resolve()
    if os.path.commonpath([str(resolved_destination), str(resolved_member)]) != str(
        resolved_destination
    ):
        raise RuntimeError(f"refusing to extract path outside target directory: {member.filename}")
    return resolved_member


def extract_archive(archive_path: Path, destination: Path) -> list[Path]:
    extracted_files: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target_path = validate_zip_member(member, destination)
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as output:
                shutil.copyfileobj(source, output, length=CHUNK_SIZE)
            extracted_files.append(target_path)
    return extracted_files


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if args.archive_path is not None:
        archive_path = args.archive_path.resolve()
    else:
        archive_path = output_dir.parent / "machine-readable-isa-latest.zip"

    ensure_archive_target(archive_path, args.force)
    ensure_extract_target(output_dir, args.force)

    try:
        downloaded_name = download_archive(args.url, archive_path, args.timeout)
        extracted_files = extract_archive(archive_path, output_dir)
    except Exception as exc:  # noqa: BLE001
        if archive_path.exists() and not args.keep_archive:
            archive_path.unlink()
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.keep_archive:
        archive_path.unlink()

    print(f"downloaded: {downloaded_name}")
    print(f"extracted_to: {output_dir}")
    print(f"files: {len(extracted_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())