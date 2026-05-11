#!/usr/bin/env python3

"""Upload and download rocprofiler-sdk CI handoff artifacts from S3.

This is intentionally small and independent of TheRock's full artifact layout:
rocprofiler-sdk is passing build-tree tarballs between build and test jobs, not
publishing component artifacts. Uploads use boto3 credentials from the runner;
downloads use the bucket's public HTTPS endpoint so GPU test jobs do not need
AWS credentials or boto3 installed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import mimetypes
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2
DEFAULT_CONTENT_TYPE = "application/octet-stream"


def log(message):
    print(message, flush=True)


def s3_key(prefix, filename):
    prefix = prefix.strip("/")
    return "{}/{}".format(prefix, filename) if prefix else filename


def s3_url(bucket, key):
    return "https://{}.s3.amazonaws.com/{}".format(bucket, quote(key, safe="/"))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_type(path):
    guessed, _ = mimetypes.guess_type(str(path))
    if path.name.endswith(".tar.gz"):
        return "application/gzip"
    if path.name.endswith(".tar.zst") or path.suffix == ".zst":
        return "application/zstd"
    return guessed or DEFAULT_CONTENT_TYPE


def retry(operation, location, func):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < MAX_RETRIES:
                delay = INITIAL_BACKOFF_SECONDS * (2**attempt)
                log(
                    "{} failed for {} (attempt {}/{}): {}. Retrying in {}s.".format(
                        operation, location, attempt + 1, MAX_RETRIES, exc, delay
                    )
                )
                time.sleep(delay)
    raise RuntimeError(
        "{} failed after {} attempts: {}".format(operation, MAX_RETRIES, location)
    ) from last_error


def boto3_client():
    try:
        import boto3
        from botocore.config import Config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "boto3 is required for S3 uploads. Install it before calling upload."
        ) from exc

    return boto3.client("s3", config=Config(max_pool_connections=10))


def upload_one(client, bucket, prefix, source):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError("Upload source not found: {}".format(source))

    key = s3_key(prefix, source.name)
    checksum = sha256_file(source)
    log("Uploading {} -> s3://{}/{}".format(source, bucket, key))
    retry(
        "upload",
        "s3://{}/{}".format(bucket, key),
        lambda: client.upload_file(
            str(source),
            bucket,
            key,
            ExtraArgs={"ContentType": content_type(source)},
        ),
    )

    checksum_key = key + ".sha256sum"
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=source.name + ".", suffix=".sha256sum", delete=False
    ) as f:
        checksum_path = Path(f.name)
        f.write("{}  {}\n".format(checksum, source.name))

    try:
        log("Uploading checksum -> s3://{}/{}".format(bucket, checksum_key))
        retry(
            "upload",
            "s3://{}/{}".format(bucket, checksum_key),
            lambda: client.upload_file(
                str(checksum_path),
                bucket,
                checksum_key,
                ExtraArgs={"ContentType": "text/plain"},
            ),
        )
    finally:
        checksum_path.unlink()

    return source.name, s3_url(bucket, key)


def download_url(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _download():
        with urlopen(url, timeout=120) as response:
            with dest.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

    try:
        retry("download", url, _download)
    except (HTTPError, URLError):
        raise


def read_expected_checksum(path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("Checksum file is empty: {}".format(path))
    return text.split()[0]


def download_one(bucket, prefix, dest):
    dest = Path(dest)
    key = s3_key(prefix, dest.name)
    checksum_dest = dest.with_name(dest.name + ".sha256sum")

    artifact_url = s3_url(bucket, key)
    checksum_url = s3_url(bucket, key + ".sha256sum")
    log("Downloading {} -> {}".format(artifact_url, dest))
    download_url(artifact_url, dest)
    log("Downloading {} -> {}".format(checksum_url, checksum_dest))
    download_url(checksum_url, checksum_dest)

    expected = read_expected_checksum(checksum_dest)
    actual = sha256_file(dest)
    if actual != expected:
        raise RuntimeError(
            "Checksum mismatch for {}: expected {}, got {}".format(
                dest, expected, actual
            )
        )
    log("Verified sha256 for {}".format(dest))
    return str(dest)


def run_parallel(operation, items, jobs, func):
    if jobs <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    workers = min(jobs, len(items))
    log("{} {} file(s) with {} worker(s).".format(operation, len(items), workers))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(func, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError("{} failed for {}".format(operation, item)) from exc
    return results


def append_step_summary(uploads):
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    with open(summary, "a", encoding="utf-8") as f:
        for name, url in uploads:
            f.write("- [{}]({})\n".format(name, url))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    upload = subparsers.add_parser("upload")
    upload.add_argument("--bucket", required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--file", action="append", required=True)
    upload.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("ROCPROFILER_SDK_S3_JOBS", "4")),
        help="Maximum parallel file transfers.",
    )

    download = subparsers.add_parser("download")
    download.add_argument("--bucket", required=True)
    download.add_argument("--prefix", required=True)
    download.add_argument("--file", action="append", required=True)
    download.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("ROCPROFILER_SDK_S3_JOBS", "4")),
        help="Maximum parallel file transfers.",
    )

    args = parser.parse_args(argv)
    if not args.command:
        parser.error("expected a command: upload or download")
    if args.jobs < 1:
        parser.error("--jobs must be greater than zero")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "upload":
            client = boto3_client()
            uploads = run_parallel(
                "Uploading",
                args.file,
                args.jobs,
                lambda file_path: upload_one(
                    client, args.bucket, args.prefix, file_path
                ),
            )
            append_step_summary(uploads)
        elif args.command == "download":
            run_parallel(
                "Downloading",
                args.file,
                args.jobs,
                lambda file_path: download_one(args.bucket, args.prefix, file_path),
            )
        else:
            raise AssertionError("unhandled command {}".format(args.command))
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
