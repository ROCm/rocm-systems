# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve, download, verify, and extract a nightly ROCm tarball."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import NoReturn


from .command import die, log, run
from .constants import NIGHTLY_TARBALL_BASE, NIGHTLY_TARBALL_INDEX


def _require_https(url: str) -> None:
    """Refuse anything but HTTPS (defense-in-depth against downgraded fetches)."""
    if not url.lower().startswith("https://"):
        die(f"refusing to fetch a non-HTTPS URL: {url}")


def _http_get_text(url: str, timeout: int = 60) -> str:
    _require_https(url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


_INDEX_HTML: str | None = None


def fetch_tarball_index(timeout: int = 60) -> str:
    """Return the nightly index HTML, fetching it at most once per run.

    Both the preflight reachability check and the tarball resolution need this
    page, and it does not change mid-run, so the second caller reuses the first
    one's copy instead of making another request.
    """
    global _INDEX_HTML
    if _INDEX_HTML is None:
        _INDEX_HTML = _http_get_text(NIGHTLY_TARBALL_INDEX, timeout=timeout)
    return _INDEX_HTML


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_published_sha256(url: str) -> tuple[str | None, str | None]:
    """Best-effort fetch of a published checksum next to ``url``.

    Returns ``(hex_digest, source_url)`` or ``(None, None)`` if none is published.
    """
    for ext in (".sha256", ".sha256sum", ".SHA256"):
        try:
            text = _http_get_text(url + ext, timeout=30)
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"\b[0-9a-fA-F]{64}\b", text)
        if m:
            return m.group(0).lower(), url + ext
    return None, None


def verify_tarball(
    url: str, tarball: Path, expected: str | None, require: bool, facts: dict
) -> None:
    """Verify the downloaded archive's SHA-256 before it is extracted/executed.

    Priority: an explicit ``--sha256`` value, else a checksum published next to the
    tarball. If neither is available, warn (or abort when ``require`` is set) since
    the extracted binaries are subsequently executed.
    """
    origin = "cli (--sha256)"
    if not expected:
        expected, origin = fetch_published_sha256(url)

    if not expected:
        msg = (
            "no SHA-256 checksum available for the downloaded tarball "
            "(none published alongside it and none passed via --sha256)"
        )
        if require:
            die(msg + "; aborting because --require-checksum was set.")
        log(
            "WARNING: " + msg + ". Skipping integrity verification. Pass --sha256 "
            "or --require-checksum to enforce."
        )
        facts["sha256_verified"] = "no (unavailable)"
        return

    log(f"Verifying tarball SHA-256 (source: {origin})...")
    actual = _sha256_file(tarball)
    if actual.lower() != expected.lower():
        die(
            "SHA-256 MISMATCH - the download is corrupt or has been tampered with:\n"
            f"       expected: {expected}\n"
            f"       actual:   {actual}"
        )
    log(f"SHA-256 verified OK: {actual}")
    facts["sha256_verified"] = "yes"
    facts["sha256"] = actual


_DIST_TARBALL_RE = re.compile(
    r"therock-dist-linux-(?P<variant>.+)-(?P<version>\d+\.\d+\.\d+a\d{8})\.tar\.gz\Z"
)


def parse_dist_tarball(filename: str) -> tuple[str, str] | None:
    """Split a dist tarball filename into (variant, ROCm version), or None.

    Lets a run that reuses an already-extracted tree (--offline / --skip-download)
    report the variant and version it is *actually* testing, rather than echoing back
    what was requested on the command line.
    """
    m = _DIST_TARBALL_RE.match(filename)
    return (m.group("variant"), m.group("version")) if m else None


def index_dist_variants(html: str) -> list[str]:
    """Return the dist tarball variants present in the index, sorted.

    Only the ``therock-dist-linux-<variant>-<version>.tar.gz`` families are
    reported; the parallel ``<variant>-tests-`` tarballs (sample data, not
    redistributables) are skipped.
    """
    variants = set()
    for m in re.finditer(
        r"therock-dist-linux-(.+?)-\d+\.\d+\.\d+a\d{8}\.tar\.gz",
        html,
    ):
        variant = m.group(1)
        if not variant.endswith("-tests"):
            variants.add(variant)
    return sorted(variants)


def _variant_family_pattern(variant: str) -> re.Pattern | None:
    """Compile the GPU-arch family a tarball variant covers, or None if not a family.

    TheRock names per-family tarballs with an upper-case ``X`` standing in for the
    last digit of the arch, plus a market suffix: ``gfx94X-dcgpu`` covers gfx940 /
    gfx941 / gfx942, ``gfx103X-all`` covers gfx1030 / gfx1031 / ... The arch token
    is alphanumeric, so only the ``X`` needs substituting.
    """
    token = variant.split("-", 1)[0]
    if not token.startswith("gfx"):
        return None
    return re.compile(token.replace("X", "[0-9a-f]") + r"\Z")


def redirect_variant(
    requested: str, available: list[str], fallback: str | None = None
) -> str:
    """Resolve a requested variant to one the index actually publishes.

    A specific arch is accepted where only its family ships, e.g. ``gfx942`` ->
    ``gfx94X-dcgpu``, which is how users refer to their GPU (and what rocminfo
    reports) even though no ``gfx942`` tarball exists.
    """
    if requested in available:
        return requested
    matches = [
        v for v in available if (p := _variant_family_pattern(v)) and p.match(requested)
    ]
    if len(matches) == 1:
        log(f"Variant '{requested}' ships as '{matches[0]}'; using that tarball.")
        return matches[0]
    if len(matches) > 1:
        die(
            f"variant '{requested}' is ambiguous: it matches "
            f"{', '.join(matches)}.\n"
            "       Pass one of those to --variant explicitly."
        )
    if fallback and fallback in available:
        log(
            f"WARNING: no tarball covers '{requested}'; falling back to " f"'{fallback}'."
        )
        return fallback
    die(
        f"no nightly tarball variant matches '{requested}'.\n"
        f"       Available variants: {', '.join(available)}\n"
        f"       Browse {NIGHTLY_TARBALL_INDEX} for the full listing."
    )


def resolve_tarball(
    variant: str, version: str | None, fallback: str | None = None
) -> tuple[str, str, str]:
    """Return (filename, url, variant) of the nightly dist tarball to download.

    ``variant`` is e.g. ``multiarch``, ``gfx94X-dcgpu``, or a specific arch such
    as ``gfx942`` that is redirected to the family tarball that ships it. The
    returned variant is the one actually resolved against the index.

    The filename regex deliberately anchors a digit right after ``<variant>-`` so
    the separate ``<variant>-tests-`` tarballs are never matched.
    """
    log(f"Reading nightly tarball index: {NIGHTLY_TARBALL_INDEX}")
    try:
        html = fetch_tarball_index()
    except Exception as exc:  # noqa: BLE001
        die(f"could not fetch tarball index: {exc}")

    available = index_dist_variants(html)
    if not available:
        die(
            f"could not parse any dist tarball from {NIGHTLY_TARBALL_INDEX}; the "
            "index layout may have changed."
        )
    variant = redirect_variant(variant, available, fallback)

    pattern = re.compile(
        r"therock-dist-linux-"
        + re.escape(variant)
        + r"-((\d+\.\d+\.\d+)a(\d{8}))\.tar\.gz"
    )

    # candidates: list of (date_int, version_str, filename)
    candidates = {}
    for m in pattern.finditer(html):
        filename = m.group(0)
        full_version = m.group(1)  # e.g. 7.15.0a20260717
        date_int = int(m.group(3))  # e.g. 20260717
        candidates[filename] = (date_int, full_version)

    if not candidates:
        die(
            f"no nightly tarballs found for variant '{variant}'.\n"
            f"       Available variants: {', '.join(available)}\n"
            f"       Check {NIGHTLY_TARBALL_INDEX}"
        )

    if version:
        matches = [
            fn
            for fn, (date_int, ver) in candidates.items()
            if _rocm_version_matches(version, ver, date_int)
        ]
        if not matches:
            die(
                f"requested version '{version}' not found for variant '{variant}'.\n"
                f"       Browse {NIGHTLY_TARBALL_INDEX} for valid values."
            )
        filename = max(matches, key=lambda fn: candidates[fn][0])
    else:
        filename = max(candidates, key=lambda fn: candidates[fn][0])

    url = f"{NIGHTLY_TARBALL_BASE}/{filename}"
    return filename, url, variant


def _rocm_version_matches(requested: str, full_version: str, date_int: int) -> bool:
    """Return True when ``requested`` exactly selects ``full_version`` (X.Y.ZaYYYYMMDD).

    Accepts the full nightly string (``7.15.0a20260717``), the trailing build date
    (``20260717``), or the ROCm semver prefix without the date (``7.15.0``). Rejects
    loose substring matches such as ``0`` or ``10.1`` matching unrelated tarballs.
    """
    if requested == full_version:
        return True
    if requested.isdigit() and len(requested) == 8 and int(requested) == date_int:
        return True
    semver = re.fullmatch(r"(\d+\.\d+\.\d+)", requested)
    if semver and full_version.startswith(semver.group(1) + "a"):
        return True
    return False


def _looks_like_gzip(path: Path) -> bool:
    """Return True when ``path`` begins with the gzip magic header."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _cached_download_is_valid(
    path: Path,
    url: str,
    *,
    expected_sha256: str | None,
    require_checksum: bool,
    is_gzip: bool,
) -> bool:
    """Return True when an on-disk download can be reused without re-fetching."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if is_gzip and not _looks_like_gzip(path):
        return False
    digest = expected_sha256
    origin = "cli (--sha256)"
    if not digest:
        digest, origin = fetch_published_sha256(url)
    if digest:
        actual = _sha256_file(path)
        if actual.lower() != digest.lower():
            log(
                f"Cached file checksum mismatch (source: {origin}); "
                f"will re-download: {path.name}"
            )
            return False
        return True
    if require_checksum:
        return False
    if is_gzip:
        return True
    return False


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    require_checksum: bool = False,
) -> None:
    """Download ``url`` to ``dest`` with resume support (prefers wget/curl)."""
    _require_https(url)
    if _cached_download_is_valid(
        dest,
        url,
        expected_sha256=expected_sha256,
        require_checksum=require_checksum,
        is_gzip=True,
    ):
        log(f"Tarball already present and verified, skipping download: {dest}")
        return
    if dest.exists():
        log(f"Removing invalid or unverified tarball: {dest.name}")
        dest.unlink(missing_ok=True)

    tmp = dest.with_suffix(dest.suffix + ".part")
    wget = shutil.which("wget")
    curl = shutil.which("curl")
    if wget:
        # Live in-place progress bar (nicer than the default "dot" meter). Under a
        # non-TTY log (cron/sbatch) this bar is redrawn via carriage returns, so the
        # captured output is a single messy '\r'-laden line - acceptable trade-off.
        cmd = [
            wget,
            "--continue",
            "--tries=3",
            "-q",
            "--show-progress",
            "--progress=bar:force:noscroll",
            "-O",
            str(tmp),
            url,
        ]
    elif curl:
        # curl's default meter already updates in place (no per-line spam).
        cmd = [curl, "-fL", "--retry", "3", "-C", "-", "-o", str(tmp), url]
    else:
        die(
            "neither 'wget' nor 'curl' is available to download the tarball. "
            "Install one, or pre-stage the tarball with --prepare-only on a "
            "networked node and run with --offline."
        )

    # Run the downloader with inherited stdio so its in-place progress bar renders
    # live on the console, instead of teeing every progress line into the logs.
    log(f"$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode  # noqa: S603
    if rc != 0:
        die(f"download failed (exit {rc}): {url}", rc)
    tmp.rename(dest)


def extract_tarball(tarball: Path, rocm_dir: Path) -> None:
    if rocm_dir.exists():
        shutil.rmtree(rocm_dir)
    rocm_dir.mkdir(parents=True, exist_ok=True)
    log(f"Extracting {tarball.name} -> {rocm_dir} (this can take a while)")
    # Prefer the system tar (much faster for multi-GB archives). GNU tar strips
    # leading '/' and rejects '..' members, so it is safe against tarbombs.
    tar = shutil.which("tar")
    if tar:
        run([tar, "-xf", str(tarball), "-C", str(rocm_dir)])
    else:
        with tarfile.open(tarball) as tf:
            # PEP 706 'data' filter (Python 3.8.17+/3.9.17+/3.10.12+/3.12+) blocks
            # path traversal, absolute paths and unsafe links.
            tf.extractall(rocm_dir, filter="data")  # noqa: S202


def _rocm_marker(workdir: Path) -> Path:
    return workdir / ".rocm-version"


def current_rocm_tarball(workdir: Path) -> str | None:
    marker = _rocm_marker(workdir)
    return marker.read_text().strip() if marker.is_file() else None


def set_rocm_tarball(workdir: Path, filename: str) -> None:
    _rocm_marker(workdir).write_text(filename + "\n")


def cleanup_downloaded_tarballs(workdir: Path) -> None:
    """Delete downloaded dist tarballs (and .part files) to reclaim disk space.

    Called after a successful extraction, since the extracted ``rocm/`` tree is all
    we need going forward.
    """
    for tb in workdir.glob("therock-dist-*.tar.gz*"):
        log(f"Removing extracted tarball to reclaim space: {tb.name}")
        tb.unlink(missing_ok=True)
