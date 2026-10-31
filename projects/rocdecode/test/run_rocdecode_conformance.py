#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the rocDecode conformance sweep against an installed ROCm payload.

Invoked by run_rocdecode.py after the CTest suite, and runnable on its own for
local debugging. Three things happen here:

  1. Fetch the conformance stream corpus. It is not redistributable, so it lives
     in a private S3 bucket rather than alongside the public build artifacts.
     Repeat runs reuse the extracted copy.
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
  ROCDECODE_CONFORMANCE_BUCKET  S3 bucket holding the corpus zips.
  ROCDECODE_CONFORMANCE_KEYS    Comma-separated object keys of the corpus zips.
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
# The corpus is published as more than one object. rocDecodeConformance.zip is a
# container: it holds one zip per codec (AVC, AV1 and VP9) rather than the stream
# folders themselves. HEVC is published beside it as its own zip instead of being
# folded in, so it has to be fetched separately.
DEFAULT_KEYS = ["rocDecodeConformance.zip", "HevcConformance.zip"]
DEFAULT_REGION = "us-east-2"

# Codec -> the literal prefix of the folder it ships under. Matched as a prefix
# rather than in full because the AV1 set carries a version suffix in its folder
# name -- Av1Conformance_v1.0 at the time of writing -- and pinning the exact
# spelling is what breaks the next time that version moves.
CODEC_DIR_PREFIXES = {
    "AVC": "AvcConformance",
    "AV1": "Av1Conformance",
    "HEVC": "HevcConformance",
    "VP9": "Vp9Conformance",
}

# Written beside the extracted corpus so a repeat run can skip the download.
ETAG_MARKER = ".rocdecode-conformance-etag"

SKIP_BANNER = "SKIPPING rocDecode conformance"

PASSED = "PASSED"
FAILED = "FAILED"
# Every stream in the codec's set was skipped because this GPU cannot handle that
# codec. Nothing was decoded, so it is not a pass; nothing was wrong either, so it
# does not fail the sweep.
SKIPPED = "SKIPPED"
# A requested codec whose corpus is absent. Reported separately from FAILED so the
# summary distinguishes "decoded wrong" from "never had the streams", but counted
# as a failure either way.
MISSING = "MISSING"


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
        default=",".join(CODEC_DIR_PREFIXES),
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
        help="S3 bucket holding the corpus zips (default: %(default)s).",
    )
    parser.add_argument(
        "--keys",
        default=os.getenv("ROCDECODE_CONFORMANCE_KEYS", ",".join(DEFAULT_KEYS)),
        help="Comma-separated object keys of the corpus zips "
        "(default: %(default)s).",
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


def codec_of(dir_name: str) -> Optional[str]:
    """Return the codec whose folder this directory name is, or None."""
    for codec, prefix in CODEC_DIR_PREFIXES.items():
        if dir_name.startswith(prefix):
            return codec
    return None


def scan_codec_dirs(root: Path) -> Dict[str, Path]:
    """Return {codec: folder} for root's immediate children that are corpora.

    A folder counts only if it holds both Streams/ and MD5/. Requiring both is
    what distinguishes the real thing from a stray directory carrying the name.
    """
    found: Dict[str, Path] = {}
    if not root.is_dir():
        return found
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        codec = codec_of(child.name)
        if codec is None or codec in found:
            continue
        if (child / "Streams").is_dir() and (child / "MD5").is_dir():
            found[codec] = child
    return found


def find_codec_dirs(root: Path) -> Dict[str, Path]:
    """Return {codec: folder}, looking in root and one level below it.

    Both levels are searched, and merged, because the codecs do not all arrive
    the same way: the zips may or may not carry their own wrapper directory, and
    HEVC comes from a separate archive from the rest, so it can land at a
    different depth than the codecs it sits beside.
    """
    found = scan_codec_dirs(root)
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir():
                for codec, path in scan_codec_dirs(child).items():
                    found.setdefault(codec, path)
    return found


def fetch_feasibility() -> Optional[str]:
    """Return what stops the corpus being fetched, or None if nothing does.

    Both checks run before anything else so an environment that cannot reach the
    corpus skips with an explanation, rather than failing part-way through on a
    ModuleNotFoundError or an opaque NoCredentialsError.

    boto3 is not part of the installed test payload. CI gets it from
    requirements-test.txt, but someone running run_rocdecode.py from an installed
    ROCm tree may not have it, and the CTest suite they came for does not need
    it -- so a missing SDK skips the sweep rather than failing the whole run.
    """
    try:
        import boto3
    except ImportError:
        return (
            "boto3 is not installed, so the stream corpus cannot be downloaded.\n"
            "Install it (pip install boto3) to run the conformance sweep."
        )

    if boto3.Session().get_credentials() is None:
        return (
            "no AWS credentials available.\n"
            "\n"
            "The stream corpus lives in a private bucket. GitHub does not\n"
            "expose repository secrets to pull requests from forks, so this\n"
            "is expected on fork PRs and is not a failure."
        )
    return None


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


def extract_zip(archive_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(dest, members=safe_members(archive, dest))


def extract_nested_zips(root: Path) -> None:
    """Unpack the per-codec zips shipped inside the corpus archive.

    rocDecodeConformance.zip does not contain the stream folders directly, it
    contains one zip per codec. Each is extracted where it sits and then removed,
    so what remains is the folder layout the sweep expects and the corpus is not
    left holding a second, compressed copy of itself.

    Looping rather than making a single pass costs four lines and means a future
    archive that nests one level deeper does not quietly yield an empty corpus.
    """
    while True:
        archives = sorted(root.rglob("*.zip"))
        if not archives:
            return
        for archive_path in archives:
            logging.info(f"++ Extracting {archive_path.name}")
            extract_zip(archive_path, archive_path.parent)
            archive_path.unlink()


def download_corpus(
    client, bucket: str, heads: Dict[str, Tuple[str, int]], dest: Path
) -> None:
    """Replace dest's contents with freshly downloaded copies of the corpus zips.

    Everything is downloaded and extracted under a scratch directory first, so an
    interrupted transfer cannot leave a half-extracted corpus that the next run
    would take for a complete one.

    The scratch directory sits beside dest rather than in the system temp dir.
    The extracted corpus runs to several GiB, which /tmp inside a CI container
    frequently is not, and staging on the same filesystem makes the final move a
    rename instead of a copy of every stream.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rocdecode-conformance-", dir=dest.parent
    ) as scratch:
        staging = Path(scratch) / "extracted"
        staging.mkdir()
        for key, (_, size) in heads.items():
            archive_path = Path(scratch) / Path(key).name
            logging.info(
                f"++ Downloading s3://{bucket}/{key} ({size / (1024 ** 2):.1f} MiB)"
            )
            with archive_path.open("wb") as handle:
                client.download_fileobj(bucket, key, handle)
            logging.info(f"++ Extracting {archive_path.name}")
            extract_zip(archive_path, staging)
            archive_path.unlink()

        extract_nested_zips(staging)

        logging.info(f"++ Installing corpus into {dest}")
        for existing in dest.iterdir():
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()
        for extracted in staging.iterdir():
            shutil.move(str(extracted), str(dest / extracted.name))


def fetch_corpus(
    bucket: str, keys: List[str], dest: Path, region: str
) -> Dict[str, Path]:
    """Ensure the corpus is present locally and return its per-codec folders."""
    client = make_client(region, bucket)
    heads: Dict[str, Tuple[str, int]] = {}
    for key in keys:
        response = client.head_object(Bucket=bucket, Key=key)
        heads[key] = (response["ETag"].strip('"'), int(response["ContentLength"]))

    # One line per object, so changing any of them invalidates the cached copy.
    want = "".join(f"{key} {etag}\n" for key, (etag, _) in heads.items())
    marker = dest / ETAG_MARKER
    cached = marker.read_text() if marker.is_file() else None
    found = find_codec_dirs(dest)
    if cached == want and found:
        logging.info(f"++ Corpus already up to date at {dest}")
        return found
    if cached and cached != want:
        logging.info("++ Cached corpus is stale; fetching again")

    download_corpus(client, bucket, heads, dest)
    marker.write_text(want)

    found = find_codec_dirs(dest)
    if not found:
        expected = ", ".join(sorted(CODEC_DIR_PREFIXES.values()))
        raise RuntimeError(
            f"Extracted {len(keys)} archive(s) from s3://{bucket} into {dest} but "
            f"found no folder named for [{expected}] containing Streams/ and "
            "MD5/. The archive layout is not what this script expects."
        )
    logging.info(
        f"++ Corpus ready at {dest}: "
        + ", ".join(f"{c}={p.name}" for c, p in found.items())
    )
    return found


def resolve_corpus(args: argparse.Namespace) -> Optional[Dict[str, Path]]:
    """Locate the per-codec folders, fetching them if needed.

    None means 'skip the sweep', which is distinct from an empty mapping.
    """
    if args.conformance_dir is not None:
        corpus = args.conformance_dir.resolve()
        found = find_codec_dirs(corpus)
        if not found:
            raise FileNotFoundError(
                f"No conformance corpus at {corpus}: expected per-codec folders "
                "each containing Streams/ and MD5/."
            )
        return found

    blocker = fetch_feasibility()
    if blocker is not None:
        logging.warning("")
        logging.warning("=" * 72)
        for line in f"{SKIP_BANNER}: {blocker}".splitlines():
            logging.warning(line)
        logging.warning("=" * 72)
        logging.warning("")
        return None

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if not keys:
        raise ValueError("--keys selected no objects to download.")
    return fetch_corpus(args.bucket, keys, default_corpus_dir(), args.region)


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


# run_rocDecode_Conformance.py ends by printing one of three verdicts: PASSED,
# FAILED or SKIPPED. Only two of those reach the exit status -- a sweep whose
# streams the GPU cannot decode has nothing to report as failed, so it exits 0
# exactly like a pass. The verdict line is therefore the only place the skip is
# visible, which is why this is matched rather than read off the exit code.
CHILD_SKIP_VERDICT = "Conformance test SKIPPED"


def run_codec(
    env: Dict[str, str],
    conformance_script: Path,
    codec_dir: Path,
    videodecode_exe: Path,
    results_dir: Path,
    gpu_device_id: int,
) -> str:
    """Run one codec's sweep and return its status: PASSED, FAILED or SKIPPED.

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
    # stderr is folded into stdout so a traceback from the child is printed rather
    # than swallowed, and so the whole transcript is echoed in order.
    result = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        return FAILED
    return SKIPPED if CHILD_SKIP_VERDICT in result.stdout else PASSED


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
    unknown = [c for c in requested if c not in CODEC_DIR_PREFIXES]
    if unknown:
        raise ValueError(
            f"Unknown codec(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(CODEC_DIR_PREFIXES)}."
        )
    # Otherwise the sweep runs nothing and the all-PASSED check below is vacuously
    # true, which would report a clean pass for having done no work.
    if not requested:
        raise ValueError(
            f"--codecs selected no codecs. Choose from {', '.join(CODEC_DIR_PREFIXES)}."
        )

    available = resolve_corpus(args)
    if available is None:
        return 0

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
            # A requested codec with no corpus is a failure, not a skip. Silently
            # passing over it is how a partial download ends up reporting green
            # having covered a quarter of the streams. Run a deliberate subset
            # with --codecs instead.
            logging.error(
                f"++ {codec}: no {CODEC_DIR_PREFIXES[codec]}* folder containing "
                "Streams/ and MD5/ was found in the corpus"
            )
            results[codec] = MISSING
            continue
        logging.info("")
        logging.info(f"=== Running conformance: {codec} ({codec_dir.name}) ===")
        results[codec] = run_codec(
            env,
            conformance_script,
            codec_dir,
            videodecode_exe,
            results_dir / codec,
            args.gpu_device_id,
        )

    report(results)

    # SKIPPED is a GPU that cannot handle the codec, which is not this build's
    # fault. MISSING is a corpus that should have been there and was not.
    return 1 if any(s in (FAILED, MISSING) for s in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
