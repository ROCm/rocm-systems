"""Tests for fetch_rccl_build: finding a published RCCL and laying it out.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import fetch_rccl_build as fb
from rccl_ci_utils import find_rccl_library

STABLE = "https://stable.repo.amd.com/rocm/core/whl-next/"
MULTI = "https://repo.amd.com/rocm/whl-multi-arch/"
TARBALLS = "https://repo.amd.com/rocm/tarball-multi-arch/"
KPACK = "_rocm_sdk_libraries/.kpack/rccl_lib_gfx950.kpack"


def wheel(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def tarball(members: dict[str, bytes], symlinks: dict[str, str] = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tar.addfile(info)
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Remote:
    """Index pages, wheels and tarballs served by URL in place of the network."""

    def __init__(self, pages=None, files=None):
        self.pages, self.files = pages or {}, files or {}
        self.opened = []

    def __enter__(self):
        self.patches = [
            mock.patch.object(fb, "_get_text", self.pages.get),
            mock.patch.object(fb, "_exists", lambda url: url in self.files),
            mock.patch.object(fb, "_open_zip", self.open_zip),
            mock.patch.object(fb, "_open_stream", self.open_stream),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()

    def open_zip(self, url):
        self.opened.append(url)
        return zipfile.ZipFile(io.BytesIO(self.files[url]))

    def open_stream(self, url):
        self.opened.append(url)
        return io.BytesIO(self.files[url])


def libraries(version: str) -> bytes:
    return wheel({fb.WHEEL_LIB: f"lib {version}".encode()})


class ParseSourceTest(unittest.TestCase):
    def test_accepts_each_kind(self):
        self.assertEqual(fb.parse_source(" release:10.1.0 "), ("release", "10.1.0"))
        self.assertEqual(fb.parse_source("nightly:10.1.0a20260822"),
                         ("nightly", "10.1.0a20260822"))
        url = f"{MULTI}rocm_sdk_libraries-7.14.1-py3-none-linux_x86_64.whl"
        self.assertEqual(fb.parse_source(f"url:{url}"), ("url", url))

    def test_rejects(self):
        for spec in ("10.1.0", "release:", "tag:10.1.0", "release:../10.1.0",
                     "release:10.1.0\nartifact_run_id=1",
                     "url:https://repo.amd.com/a\nb/rocm_sdk_libraries-1-x.whl",
                     "url:http://repo.amd.com/rocm_sdk_libraries-1-py3-none-linux_x86_64.whl",
                     "url:https://example.com/rocm_sdk_libraries-1-py3-none-linux_x86_64.whl",
                     "url:https://repo.amd.com.example.com/rocm_sdk_libraries-1-x.whl",
                     "url:https://evilamd.com/rocm_sdk_libraries-1-x.whl",
                     "url:https://repo.amd.com/rocm/whl-multi-arch/rocm_sdk_core-1-x.whl"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                fb.parse_source(spec)

    def test_gpu_target(self):
        self.assertEqual(fb.gpu_target("gfx950-dcgpu"), "gfx950")
        with self.assertRaises(ValueError):
            fb.gpu_target("gfx950/../x")


class ResolveTest(unittest.TestCase):
    def test_a_release_on_stable_repo(self):
        lib = f"{STABLE}rocm-sdk-libraries/rocm_sdk_libraries-10.1.0-py3-none-linux_x86_64.whl"
        dev = f"{STABLE}rocm-sdk-device-gfx950/rocm_sdk_device_gfx950-10.1.0-py3-none-linux_x86_64.whl"
        pages = {
            f"{STABLE}rocm-sdk-libraries/":
                '<a href="rocm_sdk_libraries-10.1.0-py3-none-win_amd64.whl">w</a>'
                '<a href="rocm_sdk_libraries-10.1.0-py3-none-linux_x86_64.whl#sha256=ab">l</a>',
            f"{STABLE}rocm-sdk-device-gfx950/":
                '<a href="rocm_sdk_device_gfx950-10.1.0-py3-none-linux_x86_64.whl">d</a>',
        }
        with Remote(pages, {lib: libraries("10.1.0"), dev: wheel({KPACK: b"k"})}):
            plan = fb.resolve("release:10.1.0", "gfx950-dcgpu")
        self.assertEqual(plan, {"source": "release:10.1.0", "version": "10.1.0",
                                "archive": "wheel", "lib_url": lib, "kpack_url": dev})

    def test_a_release_missing_from_stable_is_looked_up_on_repo(self):
        # repo.amd.com's per-project pages link one directory up.
        lib = f"{MULTI}rocm_sdk_libraries-7.14.1-py3-none-linux_x86_64.whl"
        dev = f"{MULTI}rocm_sdk_device_gfx950-7.14.1-py3-none-linux_x86_64.whl"
        pages = {
            f"{STABLE}rocm-sdk-libraries/": '<a href="rocm_sdk_libraries-10.1.0-py3-none-linux_x86_64.whl">',
            f"{MULTI}rocm-sdk-libraries/": '<a href="../rocm_sdk_libraries-7.14.1-py3-none-linux_x86_64.whl">',
            f"{MULTI}rocm-sdk-device-gfx950/": '<a href="../rocm_sdk_device_gfx950-7.14.1-py3-none-linux_x86_64.whl">',
        }
        with Remote(pages, {lib: libraries("7.14.1"), dev: wheel({KPACK: b"k"})}):
            plan = fb.resolve("release:7.14.1", "gfx950-dcgpu")
        self.assertEqual((plan["lib_url"], plan["kpack_url"]), (lib, dev))

    def test_a_device_wheel_without_the_kpack_falls_back_to_the_tarball(self):
        lib = f"{MULTI}rocm_sdk_libraries-7.13.0-py3-none-linux_x86_64.whl"
        dev = f"{MULTI}rocm_sdk_device_gfx950-7.13.0-py3-none-linux_x86_64.whl"
        tar = f"{TARBALLS}therock-dist-linux-gfx950-dcgpu-7.13.0.tar.gz"
        pages = {
            f"{MULTI}rocm-sdk-libraries/": '<a href="../rocm_sdk_libraries-7.13.0-py3-none-linux_x86_64.whl">',
            f"{MULTI}rocm-sdk-device-gfx950/": '<a href="../rocm_sdk_device_gfx950-7.13.0-py3-none-linux_x86_64.whl">',
        }
        files = {lib: libraries("7.13.0"), dev: wheel({"_rocm_sdk_libraries/.kpack/other.kpack": b""}),
                 tar: b""}
        with Remote(pages, files):
            plan = fb.resolve("release:7.13.0", "gfx950-dcgpu")
        self.assertEqual(plan, {"source": "release:7.13.0", "version": "7.13.0",
                                "archive": "tarball", "tarball_url": tar})

    def test_no_kpack_anywhere_fails(self):
        lib = f"{MULTI}rocm_sdk_libraries-7.13.0-py3-none-linux_x86_64.whl"
        pages = {f"{MULTI}rocm-sdk-libraries/":
                 '<a href="../rocm_sdk_libraries-7.13.0-py3-none-linux_x86_64.whl">'}
        with Remote(pages, {lib: libraries("7.13.0")}), \
             self.assertRaisesRegex(RuntimeError, "no rccl_lib_gfx950.kpack"):
            fb.resolve("release:7.13.0", "gfx950-dcgpu")

    def test_a_wheel_without_rccl_fails(self):
        lib = f"{MULTI}rocm_sdk_libraries-7.14.1-py3-none-linux_x86_64.whl"
        pages = {f"{MULTI}rocm-sdk-libraries/":
                 '<a href="../rocm_sdk_libraries-7.14.1-py3-none-linux_x86_64.whl">'}
        with Remote(pages, {lib: wheel({"_rocm_sdk_libraries/lib/x.so": b""})}), \
             self.assertRaisesRegex(RuntimeError, "has no _rocm_sdk_libraries/lib/librccl.so.1"):
            fb.resolve("release:7.14.1", "gfx950-dcgpu")

    def test_an_unknown_version_fails(self):
        with Remote(), self.assertRaisesRegex(RuntimeError, "no rocm_sdk_libraries 9.9.9 wheel"):
            fb.resolve("release:9.9.9", "gfx950-dcgpu")

    def test_a_nightly_with_no_wheel_but_a_tarball(self):
        tar = ("https://rocm.nightlies.amd.com/tarball-multi-arch/"
               "therock-dist-linux-gfx950-dcgpu-10.1.0a20260801.tar.gz")
        with Remote(files={tar: b""}):
            plan = fb.resolve("nightly:10.1.0a20260801", "gfx950-dcgpu")
        self.assertEqual(plan["tarball_url"], tar)

    def test_a_wheel_url_finds_its_device_wheel_in_the_project_directory(self):
        lib = f"{STABLE}rocm-sdk-libraries/rocm_sdk_libraries-10.0.0-py3-none-linux_x86_64.whl"
        dev = f"{STABLE}rocm-sdk-device-gfx950/rocm_sdk_device_gfx950-10.0.0-py3-none-linux_x86_64.whl"
        with Remote(files={lib: libraries("10.0.0"), dev: wheel({KPACK: b"k"})}):
            plan = fb.resolve(f"url:{lib}", "gfx950-dcgpu")
        self.assertEqual((plan["version"], plan["kpack_url"]), ("10.0.0", dev))

    def test_a_tarball_url_must_be_for_the_family(self):
        url = f"{TARBALLS}therock-dist-linux-gfx942-dcgpu-7.13.0.tar.gz"
        with Remote(), self.assertRaisesRegex(ValueError, "not a gfx950-dcgpu dist tarball"):
            fb.resolve(f"url:{url}", "gfx950-dcgpu")


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "build-baseline"

    def tearDown(self):
        self.tmp.cleanup()

    def assert_laid_out(self, lib: bytes, kpack: bytes):
        self.assertEqual(find_rccl_library(self.out), (self.out / "lib" / "librccl.so.1.0").resolve())
        self.assertEqual((self.out / "lib" / "librccl.so.1.0").read_bytes(), lib)
        self.assertEqual((self.out / ".kpack" / "rccl_lib_gfx950.kpack").read_bytes(), kpack)
        record = json.loads((self.out / "rccl_source.json").read_text())
        self.assertEqual(set(record["files"]), {"lib/librccl.so.1.0", ".kpack/rccl_lib_gfx950.kpack"})
        return record

    def test_from_wheels(self):
        plan = {"source": "release:10.1.0", "version": "10.1.0", "archive": "wheel",
                "lib_url": "L", "kpack_url": "D"}
        files = {"L": wheel({fb.WHEEL_LIB: b"lib"}), "D": wheel({KPACK: b"kernels"})}
        with Remote(files=files), mock.patch.object(fb, "resolve", return_value=plan):
            fb.fetch("release:10.1.0", "gfx950-dcgpu", self.out)
        record = self.assert_laid_out(b"lib", b"kernels")
        self.assertEqual(record["lib_url"], "L")

    def test_from_a_tarball_skips_symlinks_and_stops_once_found(self):
        plan = {"source": "release:7.13.0", "version": "7.13.0", "archive": "tarball",
                "tarball_url": "T"}
        files = {"T": tarball({"./lib/librccl.so.1.0": b"lib",
                               "./.kpack/rccl_lib_gfx950.kpack": b"kernels"},
                              symlinks={"./lib/librccl.so.1": "librccl.so.1.0"})}
        with Remote(files=files), mock.patch.object(fb, "resolve", return_value=plan):
            fb.fetch("release:7.13.0", "gfx950-dcgpu", self.out)
        self.assert_laid_out(b"lib", b"kernels")

    def test_a_tarball_missing_the_kpack_fails(self):
        plan = {"source": "s", "version": "v", "archive": "tarball", "tarball_url": "T"}
        with Remote(files={"T": tarball({"lib/librccl.so.1.0": b"lib"})}), \
             mock.patch.object(fb, "resolve", return_value=plan), \
             self.assertRaisesRegex(RuntimeError, "has no .kpack/rccl_lib_gfx950.kpack"):
            fb.fetch("s", "gfx950-dcgpu", self.out)

    def test_refuses_a_non_empty_directory(self):
        self.out.mkdir()
        (self.out / "stale").write_text("x")
        plan = {"source": "s", "version": "v", "archive": "tarball", "tarball_url": "T"}
        with mock.patch.object(fb, "resolve", return_value=plan), \
             self.assertRaisesRegex(RuntimeError, "is not empty"):
            fb.fetch("s", "gfx950-dcgpu", self.out)


class HttpRangeFileTest(unittest.TestCase):
    def test_a_server_ignoring_ranges_fails(self):
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Length": "4"}

        with mock.patch.object(fb.urllib.request, "urlopen",
                               side_effect=lambda *a, **k: Response(b"data")):
            f = fb.HttpRangeFile("https://repo.amd.com/x.whl")
            with self.assertRaisesRegex(OSError, "ignored a range request"):
                f.readinto(bytearray(4))


if __name__ == "__main__":
    unittest.main()
