#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the rocDecode conformance sweep against an installed ROCm payload.

Invoked by run_rocdecode.py after the CTest suite, and runnable on its own for
local debugging. Three things happen here:

  1. Fetch the conformance stream corpus. It is not redistributable, so it lives
     in a private S3 bucket as a single zip rather than alongside the public
     build artifacts. Repeat runs reuse the extracted copy.
  2. Build the videoDecode sample, which is what actually decodes the streams.
  3. Run testScripts/run_rocDecode_Conformance.py once per codec and collect the
     results. That script does the decoding and MD5 comparison; this one supplies
     it with a corpus, a binary and somewhere to write, and turns the per-codec
     exit codes into a single verdict.

A run with no AWS credentials skips rather than fails. GitHub does not expose
repository secrets to pull requests from forks, so that case is expected.

Environment variables:
  ROCM_PATH                     ROCm install prefix override.
  ROCDECODE_CONFORMANCE_DIR     Corpus location (also the fetch destination).
  ROCDECODE_CONFORMANCE_BUCKET  S3 bucket holding the corpus zip.
  ROCDECODE_CONFORMANCE_KEY     Object key of the corpus zip.
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
"""

import argparse
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")

PROJECT_NAME = "rocdecode"

DEFAULT_BUCKET = "knagesh-media-test"
DEFAULT_KEY = "rocDecodeConformance.zip"
DEFAULT_REGION = "us-east-2"

# Codec -> the folder name(s) it ships under, tried in order. These are the literal
# names in the corpus; the AV1 set has shipped as both Av1Conformance and
# Av1Conformance_v1, so accept either rather than renaming on extraction.
CODEC_DIRS = {
    "AVC": ["AvcConformance"],
    "AV1": ["Av1Conformance", "Av1Conformance_v1"],
    "HEVC": ["HevcConformance"],
    "VP9": ["Vp9Conformance"],
}

# Written beside the extracted corpus so a repeat run can skip the download.
ETAG_MARKER = ".rocdecode-conformance-etag"

SKIP_BANNER = "SKIPPING rocDecode conformance"

PASSED = "PASSED"
FAILED = "FAILED"
SKIPPED = "SKIPPED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the rocDecode conformance sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--conformance-dir",
        type=Path,
        default=None,
        help="Use the corpus already at this path instead of fetching one.",
    )
    parser.add_argument(
        "--codecs",
        default=",".join(CODEC_DIRS),
        help="Comma-separated codecs to run (default: %(default)s).",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help="GPU device index to decode on (default: %(default)s).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Where to write per-codec logs (default: ./conformance-results).",
    )
    parser.add_argument(
        "--videodecode-exe",
        type=Path,
        default=None,
        help="Use this videodecode binary instead of building the sample.",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("ROCDECODE_CONFORMANCE_BUCKET", DEFAULT_BUCKET),
        help="S3 bucket holding the corpus zip (default: %(default)s).",
    )
    parser.add_argument(
        "--key",
        default=os.getenv("ROCDECODE_CONFORMANCE_KEY", DEFAULT_KEY),
        help="Object key of the corpus zip (default: %(default)s).",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_DEFAULT_REGION", DEFAULT_REGION),
        help="Bucket region (default: %(default)s). Corrected automatically if "
        "S3 reports the bucket lives elsewhere.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# ROCm layout
# --------------------------------------------------------------------------


def derive_rocm_path(script_dir: Path) -> Path:
    """Recover the ROCm prefix from the installed test layout.

    Mirrors run_rocdecode.py: these scripts install to
    <prefix>/share/<project>/test, so the prefix is three levels up.
    """
    if script_dir.name == "test" and script_dir.parent.name == PROJECT_NAME:
        if script_dir.parent.parent.name == "share":
            return script_dir.parent.parent.parent
    raise RuntimeError(
        "Could not derive ROCM_PATH from an installed rocdecode test layout. "
        "Set ROCM_PATH explicitly."
    )


def get_rocm_lib_dir(rocm_path: Path) -> Path:
    for name in ("lib", "lib64"):
        candidate = rocm_path / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find ROCm library directory under {rocm_path}")


def setup_env(env: Dict[str, str], rocm_path: Path) -> None:
    env["ROCM_PATH"] = str(rocm_path)
    logging.info(f"++ rocdecode setting ROCM_PATH={rocm_path}")
    lib_dir = get_rocm_lib_dir(rocm_path)
    logging.info(f"++ rocdecode setting LD_LIBRARY_PATH={lib_dir}")
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env['LD_LIBRARY_PATH']}"
    else:
        env["LD_LIBRARY_PATH"] = str(lib_dir)


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


def default_corpus_dir() -> Path:
    configured = os.getenv("ROCDECODE_CONFORMANCE_DIR")
    if configured:
        return Path(configured)
    return Path.home() / "rocDecodeConformance"


def find_codec_dirs(root: Path) -> Dict[str, Path]:
    """Return {codec: folder} for the per-codec folders present under root.

    A folder counts only if it holds both Streams/ and MD5/. Requiring both is
    what distinguishes the real thing from a stray directory carrying the name.
    """
    found: Dict[str, Path] = {}
    if not root.is_dir():
        return found
    for codec, names in CODEC_DIRS.items():
        for name in names:
            candidate = root / name
            if (candidate / "Streams").is_dir() and (candidate / "MD5").is_dir():
                found[codec] = candidate
                break
    return found


def resolve_corpus_root(dest: Path) -> Optional[Path]:
    """Find the directory holding the per-codec folders.

    The zip may or may not carry its own top-level wrapper directory, so look in
    dest and then one level below rather than assuming either shape.
    """
    if find_codec_dirs(dest):
        return dest
    if dest.is_dir():
        for child in sorted(dest.iterdir()):
            if child.is_dir() and find_codec_dirs(child):
                return child
    return None


def have_credentials() -> bool:
    """Report whether boto3 can find any credentials at all.

    Checked up front so a job without them skips with a clear message instead of
    failing on an opaque NoCredentialsError part-way through.
    """
    import boto3

    return boto3.Session().get_credentials() is not None


def make_client(region: str, bucket: str):
    """Build an S3 client, retrying once against the bucket's actual region.

    A client pointed at the wrong region fails every call, and the region is not
    something a caller should have to know. S3 reports the right one in the
    x-amz-bucket-region response header, so read it back and rebuild.
    """
    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client("s3", region_name=region)
    try:
        client.head_bucket(Bucket=bucket)
        return client
    except ClientError as exc:
        headers = exc.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        actual = headers.get("x-amz-bucket-region")
        if actual and actual != region:
            logging.info(
                f"++ bucket {bucket} is in {actual}, not {region}; reconnecting"
            )
            return boto3.client("s3", region_name=actual)
        # head_bucket also fails when the caller may read the object but not the
        # bucket itself, which is a legitimate least-privilege policy. Carry on
        # and let the download report any real problem.
        code = exc.response.get("Error", {}).get("Code")
        logging.info(f"++ head_bucket on {bucket} failed ({code}); continuing")
        return client


def safe_members(archive: zipfile.ZipFile, dest: Path) -> List[zipfile.ZipInfo]:
    """Reject archive entries that would write outside dest.

    zipfile.extractall already sanitizes member paths, but the corpus is an
    opaque blob from outside the repo and a traversal entry is worth failing on
    rather than silently normalizing away.
    """
    members = []
    dest_resolved = dest.resolve()
    for member in archive.infolist():
        target = (dest_resolved / member.filename).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise ValueError(
                f"Refusing to extract {member.filename!r}: it resolves outside {dest}"
            )
        members.append(member)
    return members


def download_corpus(client, bucket: str, key: str, dest: Path, size: int) -> None:
    """Replace dest's contents with a freshly downloaded copy of the corpus.

    Downloaded and extracted under a scratch directory first, so an interrupted
    transfer cannot leave a half-extracted corpus that the next run would take
    for a complete one.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rocdecode-conformance-") as scratch:
        archive_path = Path(scratch) / "corpus.zip"
        logging.info(
            f"++ Downloading s3://{bucket}/{key} ({size / (1024 ** 2):.1f} MiB)"
        )
        with archive_path.open("wb") as handle:
            client.download_fileobj(bucket, key, handle)

        logging.info(f"++ Extracting into {dest}")
        staging = Path(scratch) / "extracted"
        staging.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(staging, members=safe_members(archive, staging))

        for existing in dest.iterdir():
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()
        for extracted in staging.iterdir():
            shutil.move(str(extracted), str(dest / extracted.name))


def fetch_corpus(bucket: str, key: str, dest: Path, region: str) -> Path:
    """Ensure the corpus is present locally and return the per-codec parent dir."""
    client = make_client(region, bucket)
    response = client.head_object(Bucket=bucket, Key=key)
    etag = response["ETag"].strip('"')
    size = int(response["ContentLength"])

    marker = dest / ETAG_MARKER
    cached = marker.read_text().strip() if marker.is_file() else None
    corpus_root = resolve_corpus_root(dest)
    if cached == etag and corpus_root is not None:
        logging.info(f"++ Corpus already up to date at {corpus_root}")
        return corpus_root
    if cached and cached != etag:
        logging.info(f"++ Cached corpus is stale (have {cached}, want {etag})")

    download_corpus(client, bucket, key, dest, size)
    marker.write_text(etag + "\n")

    corpus_root = resolve_corpus_root(dest)
    if corpus_root is None:
        expected = ", ".join(sorted(n for names in CODEC_DIRS.values() for n in names))
        raise RuntimeError(
            f"Extracted s3://{bucket}/{key} into {dest} but found none of "
            f"[{expected}] containing Streams/ and MD5/. The archive layout is "
            "not what this script expects."
        )
    found = find_codec_dirs(corpus_root)
    logging.info(
        f"++ Corpus ready at {corpus_root}: "
        + ", ".join(f"{c}={p.name}" for c, p in found.items())
    )
    return corpus_root


def resolve_corpus(args: argparse.Namespace) -> Optional[Path]:
    """Locate the corpus, fetching it if needed. None means 'skip the sweep'."""
    if args.conformance_dir is not None:
        corpus = args.conformance_dir.resolve()
        if not find_codec_dirs(corpus):
            raise FileNotFoundError(
                f"No conformance corpus at {corpus}: expected per-codec folders "
                "each containing Streams/ and MD5/."
            )
        return corpus

    if not have_credentials():
        logging.warning("")
        logging.warning("=" * 72)
        logging.warning(f"{SKIP_BANNER}: no AWS credentials available.")
        logging.warning("")
        logging.warning("The stream corpus lives in a private bucket. GitHub does not")
        logging.warning(
            "expose repository secrets to pull requests from forks, so this"
        )
        logging.warning("is expected on fork PRs and is not a failure.")
        logging.warning("=" * 72)
        logging.warning("")
        return None

    return fetch_corpus(args.bucket, args.key, default_corpus_dir(), args.region)


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------


def build_videodecode(env: Dict[str, str], rocm_path: Path, build_dir: Path) -> Path:
    """Build the videoDecode sample and return the resulting binary.

    The sample has no stable installed location: the CTest suite builds it per
    test via `ctest --build-and-test` into a throwaway directory. The sweep needs
    one binary it can point every codec at, so build it once here and pass the
    path explicitly. run_rocDecode_Conformance.py's default lookup
    (<rocDecode_directory>/samples/videoDecode/build/videodecode) matches nothing
    in an installed layout.
    """
    source_dir = rocm_path / "share" / PROJECT_NAME / "samples" / "videoDecode"
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"videoDecode sample sources not found at {source_dir}. "
            "Fetch the rocdecode test artifact (--rocdecode --tests)."
        )

    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["cmake"]
    if shutil.which("ninja"):
        cmd += ["-GNinja"]
    cmd += ["-S", str(source_dir), "-B", str(build_dir)]
    logging.info(f"++ Exec $ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    cmd = ["cmake", "--build", str(build_dir)]
    logging.info(f"++ Exec $ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    exe = build_dir / "videodecode"
    if not exe.is_file():
        raise FileNotFoundError(
            f"Built the videoDecode sample but found no binary at {exe}."
        )
    return exe


def run_codec(
    env: Dict[str, str],
    conformance_script: Path,
    codec_dir: Path,
    videodecode_exe: Path,
    results_dir: Path,
    gpu_device_id: int,
) -> bool:
    """Run one codec's sweep. Returns True on a clean pass.

    Each codec gets its own results directory. Without that they share one
    rocDecode_output.log, which run_rocDecode_Conformance.py appends to, and each
    codec's tally ends up taken partly over the previous codec's output.
    """
    cmd = [
        sys.executable,
        str(conformance_script),
        "--files_directory",
        str(codec_dir),
        "--videodecode_exe",
        str(videodecode_exe),
        "--results_directory",
        str(results_dir),
        "--gpu_device_id",
        str(gpu_device_id),
    ]
    logging.info(f"++ Exec $ {shlex.join(cmd)}")
    # run_rocDecode_Conformance.py reports the outcome through its exit code.
    return subprocess.run(cmd, env=env).returncode == 0


def report(results: Dict[str, str]) -> None:
    width = max((len(name) for name in results), default=0)
    rule = "=" * (width + 14)
    logging.info("")
    logging.info(rule)
    logging.info("rocDecode conformance")
    logging.info(rule)
    for name, status in results.items():
        logging.info(f"  {name.ljust(width)}   {status}")
    logging.info(rule)


def main() -> int:
    args = parse_args()

    if platform.system() != "Linux":
        logging.info("++ rocdecode conformance is only supported on Linux")
        return 0

    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = (
        Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    )

    conformance_script = (
        rocm_path
        / "share"
        / PROJECT_NAME
        / "test"
        / "testScripts"
        / "run_rocDecode_Conformance.py"
    )
    if not conformance_script.is_file():
        raise FileNotFoundError(
            f"Conformance script not found at {conformance_script}. "
            "Fetch the rocdecode test artifact (--rocdecode --tests)."
        )

    requested = [c.strip().upper() for c in args.codecs.split(",") if c.strip()]
    unknown = [c for c in requested if c not in CODEC_DIRS]
    if unknown:
        raise ValueError(
            f"Unknown codec(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(CODEC_DIRS)}."
        )

    corpus = resolve_corpus(args)
    if corpus is None:
        return 0
    available = find_codec_dirs(corpus)

    env = os.environ.copy()
    setup_env(env, rocm_path)

    results_dir = (args.results_dir or Path.cwd() / "conformance-results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.videodecode_exe is not None:
        videodecode_exe = args.videodecode_exe.resolve()
        if not videodecode_exe.is_file():
            raise FileNotFoundError(f"No videodecode binary at {videodecode_exe}")
    else:
        videodecode_exe = build_videodecode(
            env, rocm_path, results_dir / "videoDecode-build"
        )
    logging.info(f"++ Using videodecode binary {videodecode_exe}")

    results: Dict[str, str] = {}
    for codec in requested:
        codec_dir = available.get(codec)
        if codec_dir is None:
            wanted = " or ".join(CODEC_DIRS[codec])
            logging.warning(f"++ {codec}: no {wanted} under {corpus}, skipping")
            results[codec] = SKIPPED
            continue
        logging.info("")
        logging.info(f"=== Running conformance: {codec} ({codec_dir.name}) ===")
        ok = run_codec(
            env,
            conformance_script,
            codec_dir,
            videodecode_exe,
            results_dir / codec,
            args.gpu_device_id,
        )
        results[codec] = PASSED if ok else FAILED

    report(results)

    if all(status == SKIPPED for status in results.values()):
        logging.error("No codec was actually run; treating that as a failure.")
        return 1
    return 1 if FAILED in results.values() else 0


if __name__ == "__main__":
    sys.exit(main())
