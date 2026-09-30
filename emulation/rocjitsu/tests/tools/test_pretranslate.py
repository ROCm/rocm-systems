#!/usr/bin/env python3
"""Unit tests for the installed pretranslation wrapper."""

from __future__ import annotations

import errno
import importlib.util
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).parents[2] / "scripts" / "rocjitsu-pretranslate.py"
SPEC = importlib.util.spec_from_file_location("rocjitsu_pretranslate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pretranslate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pretranslate
SPEC.loader.exec_module(pretranslate)


def amdgpu_object(marker: int) -> bytes:
    data = bytearray(64)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", data, 18, pretranslate.EM_AMDGPU)
    data[24] = marker
    return bytes(data)


class PretranslateTest(unittest.TestCase):
    def test_traversal_error_is_reported_and_fails_main(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            root.mkdir()
            denied = root / "denied"

            def failed_walk(_root, *, followlinks, onerror):
                self.assertFalse(followlinks)
                onerror(PermissionError(errno.EACCES, "permission denied", denied))
                return []

            with mock.patch.object(pretranslate.os, "walk", side_effect=failed_walk):
                found = pretranslate.discover([root], "gfx1250", None, None, None, True)
                code = pretranslate.main(
                    ["--skip-kpack", "--skip-compressed", "--dry-run", str(root)]
                )

        self.assertEqual(code, 1)
        self.assertEqual(len(found.failures), 1)
        self.assertEqual(found.failures[0]["category"], "traversal")
        self.assertEqual(found.failures[0]["path"], str(denied))

    def test_stat_error_is_reported_and_fails_main(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            root.mkdir()
            denied = root / "denied.co"
            denied.write_bytes(amdgpu_object(1))
            original_stat = Path.stat

            def failed_stat(path, *args, **kwargs):
                if path == denied:
                    raise PermissionError(errno.EACCES, "permission denied", path)
                return original_stat(path, *args, **kwargs)

            with mock.patch.object(Path, "stat", failed_stat):
                found = pretranslate.discover([root], "gfx1250", None, None, None, True)
                code = pretranslate.main(
                    ["--skip-kpack", "--skip-compressed", "--dry-run", str(root)]
                )

        self.assertEqual(code, 1)
        self.assertEqual(len(found.failures), 1)
        self.assertEqual(found.failures[0]["category"], "stat")
        self.assertEqual(found.failures[0]["path"], str(denied))

    def test_discovery_streams_and_deduplicates_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "install"
            root.mkdir()
            (root / "first.co").write_bytes(amdgpu_object(1))
            (root / "duplicate.co").write_bytes(amdgpu_object(1))
            (root / "second.co").write_bytes(amdgpu_object(2))

            streamed = {}

            def consume(digest, data, item):
                streamed[digest] = (data, item)

            found = pretranslate.discover([root], "gfx1250", None, None, consume, True)

            self.assertEqual(found.extracted_objects, 3)
            self.assertEqual(found.duplicate_objects, 1)
            self.assertEqual(len(found.objects), 2)
            self.assertEqual(len(streamed), 2)
            for item in found.objects.values():
                self.assertFalse(hasattr(item, "data"))
                self.assertFalse(hasattr(item, "path"))

    def test_host_elf_discovery_never_uses_an_unbounded_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            host = Path(temporary) / "host.so"
            header = bytearray(64)
            header[:6] = b"\x7fELF\x02\x01"
            struct.pack_into("<H", header, 18, 62)  # EM_X86_64
            host.write_bytes(header + bytes(1 << 20))
            original_open = Path.open

            class GuardedReader:
                def __init__(self, wrapped):
                    self.wrapped = wrapped

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return self.wrapped.__exit__(*args)

                def read(self, size=-1):
                    if size < 0:
                        raise AssertionError("host ELF was read without a bound")
                    return self.wrapped.read(size)

                def seek(self, *args):
                    return self.wrapped.seek(*args)

            def guarded_open(path, *args, **kwargs):
                opened = original_open(path, *args, **kwargs)
                return GuardedReader(opened) if path == host else opened

            with mock.patch.object(Path, "open", guarded_open):
                found = pretranslate.discover([host], "gfx1250", None, None, None, True)
            self.assertFalse(found.objects)

    def test_bundler_failure_is_reported(self):
        failed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="bad compressed input"
        )
        with mock.patch.object(pretranslate.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "bad compressed input"):
                pretranslate.unbundle_compressed("bundler", b"CCOB", "gfx1250")

    def test_bundler_keeps_successful_entries_when_another_entry_fails(self):
        targets = "hip-amdgcn-amd-amdhsa--gfx1250 " "hipv4-amdgcn-amd-amdhsa--gfx1250\n"
        extraction = 0

        def run(command, **_kwargs):
            nonlocal extraction
            if "--list" in command:
                return subprocess.CompletedProcess(command, 0, targets, "")
            extraction += 1
            if extraction == 1:
                output = Path(
                    next(
                        arg.removeprefix("--output=")
                        for arg in command
                        if arg.startswith("--output=")
                    )
                )
                output.write_bytes(amdgpu_object(1))
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 1, "", "bad entry")

        with mock.patch.object(pretranslate.subprocess, "run", side_effect=run):
            payloads, failures = pretranslate.unbundle_compressed(
                "bundler", b"CCOB", "gfx1250"
            )
        self.assertEqual(len(payloads), 1)
        self.assertEqual(len(failures), 1)
        self.assertIn("bad entry", failures[0][1])

    def test_wrapper_does_not_override_the_domain_key_policy(self):
        args = pretranslate.parse_arguments(["--portable"])
        self.assertTrue(args.portable)

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="summary written=0 held=1 skipped=0 failed=0\n",
        )
        with mock.patch.object(
            pretranslate.subprocess, "run", return_value=completed
        ) as run:
            pretranslate.run_tool(
                "rj_pretranslate", None, [Path("object.co")], False, False, False
            )
        self.assertNotIn("--portable", run.call_args.args[0])
        self.assertNotIn("--translator-build", run.call_args.args[0])

    def test_dry_run_does_not_stage_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            root.mkdir()
            (root / "object.co").write_bytes(amdgpu_object(1))
            with mock.patch.object(
                pretranslate, "discover", wraps=pretranslate.discover
            ) as discover:
                code = pretranslate.main(
                    ["--skip-kpack", "--skip-compressed", "--dry-run", str(root)]
                )
        self.assertEqual(code, 0)
        self.assertIsNone(discover.call_args.args[4])

    def test_staging_footprint_is_bounded_by_the_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            root.mkdir()
            for marker in range(3):
                (root / f"object{marker}.co").write_bytes(amdgpu_object(marker))
            observed = []

            def run_tool(_tool, _store_root, paths, *_args):
                observed.append(len(list(paths[0].parent.glob("*.co"))))
                return 0, (f"summary written=0 held={len(paths)} skipped=0 failed=0\n")

            with mock.patch.object(pretranslate, "run_tool", side_effect=run_tool):
                code = pretranslate.main(
                    [
                        "--skip-kpack",
                        "--skip-compressed",
                        "--batch",
                        "2",
                        "--tool",
                        sys.executable,
                        str(root),
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(observed, [2, 1])

    def test_explicit_missing_bundler_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            root.mkdir()
            code = pretranslate.main(
                [
                    "--skip-kpack",
                    "--dry-run",
                    "--bundler",
                    str(Path(temporary) / "missing-bundler"),
                    str(root),
                ]
            )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
