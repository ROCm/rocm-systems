#!/usr/bin/env python3
"""Lay out a published RCCL build the way TheRock's CI artifacts come.

test_madengine.py takes each side of its A/B from a directory holding
lib/librccl.so* and, for a library that keeps its kernels in a kpack,
.kpack/rccl_lib_<gfx>.kpack. This fetches those two files from a ROCm build
published outside CI:

  release:<version>  a ROCm release, e.g. release:10.1.0 or release:7.14.1
  nightly:<version>  a TheRock nightly, e.g. nightly:10.1.0a20260822
  url:<url>          a rocm_sdk_libraries-<version>-*.whl, its kpack taken
                     from the rocm_sdk_device_<gfx> wheel of the same version
                     beside it, or a therock-dist-linux-<family>-<version>.tar.gz;
                     https on an amd.com host only

Wheels are read by HTTP range, only the members needed. A release whose
device wheel carries no RCCL kpack (7.13.0) is taken from its dist tarball
instead, which is streamed whole.
"""

import argparse
import hashlib
import html
import io
import json
import logging
import posixpath
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("fetch_rccl_build")

# Searched in order: 10.x releases are on stable.repo.amd.com, 7.x on repo.amd.com.
CHANNELS = {
    "release": {
        "wheels": ("https://stable.repo.amd.com/rocm/core/whl-next/",
                   "https://repo.amd.com/rocm/whl-multi-arch/"),
        "tarballs": "https://repo.amd.com/rocm/tarball-multi-arch/",
    },
    "nightly": {
        "wheels": ("https://rocm.nightlies.amd.com/whl-multi-arch/",),
        "tarballs": "https://rocm.nightlies.amd.com/tarball-multi-arch/",
    },
}
LIBRARIES = "rocm_sdk_libraries"
WHEEL_LIB = "_rocm_sdk_libraries/lib/librccl.so.1"
WHEEL_KPACK = "_rocm_sdk_libraries/.kpack/rccl_lib_{gfx}.kpack"
TAR_LIB = "lib/librccl.so.1.0"
TAR_KPACK = ".kpack/rccl_lib_{gfx}.kpack"
LIB_NAME = "librccl.so.1.0"
VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+]*$")
FAMILY_RE = re.compile(r"^(gfx[0-9a-z]+)(-[a-z]+)?$")
TIMEOUT = 60


class HttpRangeFile(io.RawIOBase):
    """A remote file read by HTTP range, for zipfile to seek in."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                                    timeout=TIMEOUT) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            if response.status != 206:
                raise OSError(f"{self.url} ignored a range request (HTTP {response.status})")
            data = response.read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _get_text(url: str) -> str | None:
    """The page at *url*, or None if there is none."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            return response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None
        raise


def _exists(url: str) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                                    timeout=TIMEOUT):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return False
        raise


def _open_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))


def _open_stream(url: str):
    return urllib.request.urlopen(url, timeout=TIMEOUT)


def gpu_target(family: str) -> str:
    """gfx950 for gfx950-dcgpu."""
    match = FAMILY_RE.match(family)
    if not match:
        raise ValueError(f"not an AMDGPU family: {family!r}")
    return match.group(1)


def parse_source(spec: str) -> tuple[str, str]:
    """``(kind, value)`` of a ``release:``, ``nightly:`` or ``url:`` spec."""
    kind, sep, value = spec.strip().partition(":")
    if not (sep and value):
        raise ValueError(f"{spec!r}: expected release:<version>, nightly:<version> or url:<url>")
    # The spec travels through $GITHUB_OUTPUT, where a line break starts another output.
    if any(c.isspace() or not c.isprintable() for c in value):
        raise ValueError(f"{spec!r}: whitespace or control characters")
    if kind in CHANNELS:
        if not VERSION_RE.fullmatch(value):
            raise ValueError(f"{spec!r}: not a version")
        return kind, value
    if kind != "url":
        raise ValueError(f"{spec!r}: unknown source {kind!r}; expected release, nightly or url")
    parsed = urllib.parse.urlparse(value)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not (host == "amd.com" or host.endswith(".amd.com")):
        raise ValueError(f"{spec!r}: only https URLs on amd.com hosts are fetched")
    name = posixpath.basename(parsed.path)
    if not ((name.startswith(f"{LIBRARIES}-") and name.endswith(".whl"))
            or (name.startswith("therock-dist-linux-") and name.endswith(".tar.gz"))):
        raise ValueError(f"{spec!r}: expected a {LIBRARIES} wheel or a therock-dist tarball")
    return kind, value


def _wheel_name(project: str, version: str) -> str:
    return f"{project}-{version}-py3-none-linux_x86_64.whl"


def _find_wheel(index: str, project: str, version: str) -> str | None:
    """The URL *index* lists for *project*'s Linux wheel of *version*."""
    page = urllib.parse.urljoin(index, project.replace("_", "-") + "/")
    text = _get_text(page)
    if text is None:
        return None
    name = _wheel_name(project, version)
    for href in re.findall(r'href="([^"]+)"', text):
        url = urllib.parse.urldefrag(urllib.parse.urljoin(page, html.unescape(href))).url
        if posixpath.basename(urllib.parse.urlparse(url).path) == name:
            return url
    return None


def _tarball_plan(spec: str, version: str, url: str) -> dict:
    return {"source": spec, "version": version, "archive": "tarball", "tarball_url": url}


def resolve(spec: str, family: str) -> dict:
    """Where *spec*'s RCCL and its *family* kpack are; raises if either is missing."""
    gfx = gpu_target(family)
    device_project = f"rocm_sdk_device_{gfx}"
    kind, value = parse_source(spec)
    tarball = None
    if kind == "url":
        name = posixpath.basename(urllib.parse.urlparse(value).path)
        if name.endswith(".tar.gz"):
            prefix = f"therock-dist-linux-{family}-"
            if not name.startswith(prefix):
                raise ValueError(f"{spec!r}: not a {family} dist tarball")
            return _tarball_plan(spec, name[len(prefix):-len(".tar.gz")], value)
        lib_url, version = value, name.split("-")[1]
        if not VERSION_RE.fullmatch(version):
            raise ValueError(f"{spec!r}: not a version: {version!r}")
        device_name = _wheel_name(device_project, version)
        # repo.amd.com keeps all wheels in one directory, stable.repo.amd.com
        # one directory per project.
        siblings = (urllib.parse.urljoin(value, device_name),
                    urllib.parse.urljoin(value, f"../{device_project.replace('_', '-')}/{device_name}"))
        device = next((url for url in siblings if _exists(url)), None)
    else:
        channel, version = CHANNELS[kind], value
        lib_url = device = None
        for index in channel["wheels"]:
            lib_url = _find_wheel(index, LIBRARIES, version)
            if lib_url:
                device = _find_wheel(index, device_project, version)
                break
        tarball = f"{channel['tarballs']}therock-dist-linux-{family}-{version}.tar.gz"
        if lib_url is None:
            if _exists(tarball):
                return _tarball_plan(spec, version, tarball)
            raise RuntimeError(f"{spec}: no {LIBRARIES} {version} wheel in "
                               f"{', '.join(channel['wheels'])}, nor {tarball}")
    with _open_zip(lib_url) as wheel:
        if WHEEL_LIB not in wheel.namelist():
            raise RuntimeError(f"{spec}: {lib_url} has no {WHEEL_LIB}")
    if device:
        with _open_zip(device) as wheel:
            if WHEEL_KPACK.format(gfx=gfx) in wheel.namelist():
                return {"source": spec, "version": version, "archive": "wheel",
                        "lib_url": lib_url, "kpack_url": device}
    if tarball and _exists(tarball):
        log.info("%s: no RCCL kpack in its %s wheel; using %s", spec, device_project, tarball)
        return _tarball_plan(spec, version, tarball)
    # The base image keeps its own RCCL's kpack under the same kernel name, so
    # a library without one would run the image's kernels.
    raise RuntimeError(f"{spec}: no rccl_lib_{gfx}.kpack in {device or f'a {device_project} wheel'}")


def _copy(src, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)


def _extract_tarball(url: str, members: dict[str, Path]) -> None:
    """Stream *url* until every member is written to its path."""
    wanted = dict(members)
    with _open_stream(url) as stream, tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for entry in tar:
            name = entry.name[2:] if entry.name.startswith("./") else entry.name
            if name in wanted and entry.isfile():
                _copy(tar.extractfile(entry), wanted.pop(name))
                if not wanted:
                    return
    raise RuntimeError(f"{url} has no {', '.join(sorted(wanted))}")


def fetch(spec: str, family: str, output_dir: Path) -> dict:
    """Write *spec*'s RCCL into *output_dir*; returns what was fetched from where."""
    gfx = gpu_target(family)
    plan = resolve(spec, family)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError(f"{output_dir} is not empty")
    lib = output_dir / "lib" / LIB_NAME
    kpack = output_dir / ".kpack" / f"rccl_lib_{gfx}.kpack"
    if plan["archive"] == "wheel":
        for url, member, dest in ((plan["lib_url"], WHEEL_LIB, lib),
                                  (plan["kpack_url"], WHEEL_KPACK.format(gfx=gfx), kpack)):
            log.info("Fetching %s from %s", member, url)
            with _open_zip(url) as wheel, wheel.open(member) as src:
                _copy(src, dest)
    else:
        log.info("Streaming %s", plan["tarball_url"])
        _extract_tarball(plan["tarball_url"], {TAR_LIB: lib, TAR_KPACK.format(gfx=gfx): kpack})
    lib.chmod(0o755)
    (lib.parent / "librccl.so.1").symlink_to(LIB_NAME)
    (lib.parent / "librccl.so").symlink_to("librccl.so.1")
    record = {**plan, "files": {
        path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (lib, kpack)
    }}
    (output_dir / "rccl_source.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="release:, nightly: or url: spec")
    parser.add_argument("--amdgpu-family", required=True, help="e.g. gfx950-dcgpu")
    parser.add_argument("--output-dir", type=Path, help="where to lay the build out")
    parser.add_argument("--check", action="store_true",
                        help="only find the build and print where it is")
    args = parser.parse_args()
    if not (args.check or args.output_dir):
        parser.error("--output-dir is required without --check")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.check:
            result = resolve(args.source, args.amdgpu_family)
        else:
            result = fetch(args.source, args.amdgpu_family, args.output_dir)
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
        log.error("%s", exc)
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
