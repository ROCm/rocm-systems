#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Unit tests for hrr_torch_regions.py: when the producer treats capture as
active, and what it does with the file system when it writes. POSIX only, like
the checks they cover. Run with pytest or python -m unittest; torch is not
needed."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

PRODUCER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRODUCER_DIR))
os.environ["HRR_REGIONS_AUTOSTART"] = "0"  # no polling thread on import

import hrr_torch_regions as regions  # noqa: E402

RECORD = (regions._ADD, regions._BLOCK, 0, 0x1000, 256, 1)
VICTIM_CONTENTS = b"must survive the producer\n"
# What the tests take this process to be, so they do not depend on /proc.
INSTANCE = "00000000-0000-0000-0000-000000000000 4242"


class _ArchiveCase(unittest.TestCase):
    """A capture root in a temporary directory with this process's pid-<pid>,
    its events.bin and its active marker in it, as the capture writer leaves
    them while it runs, and HIP_HRR_CAPTURE_OUTPUT pointing at the root. The
    marker names INSTANCE, which _process_instance() returns."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hrr_regions_test_"))
        self.root = self.tmp / "capture"
        self.root.mkdir()
        self.archive = self.root / ("pid-%d" % os.getpid())
        self.archive.mkdir(mode=0o700)
        (self.archive / "events.bin").write_bytes(b"")
        self.marker = self.archive / "active"
        self.marker.write_bytes(INSTANCE.encode() + b"\n")
        instance = mock.patch.object(regions, "_process_instance", return_value=INSTANCE)
        instance.start()
        self.addCleanup(instance.stop)
        self.victim = self.tmp / "victim"
        self.victim.write_bytes(VICTIM_CONTENTS)
        env = mock.patch.dict(os.environ, {"HIP_HRR_CAPTURE_OUTPUT": str(self.root)})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write_one(self):
        stream = regions._Stream()
        self.addCleanup(stream.close)
        return stream.write([RECORD])


@unittest.skipIf(os.name == "nt", "POSIX ownership and link checks")
class ArchiveDirTest(_ArchiveCase):
    """_archive_dir: capture counts as active only for a real pid-<pid> of
    ours with a private active marker in it."""

    def test_active_archive(self):
        self.assertEqual(regions._archive_dir(), str(self.archive))

    def test_no_capture_output(self):
        with mock.patch.dict(os.environ, {"HIP_HRR_CAPTURE_OUTPUT": ""}):
            self.assertIsNone(regions._archive_dir())

    def test_events_bin_without_marker(self):
        # What a refused capture leaves behind, and a resume that fails after
        # the writer opened the earlier run's events.bin: pid-<pid> and
        # events.bin, but no marker.
        self.marker.unlink()
        self.assertTrue((self.archive / "events.bin").is_file())
        self.assertIsNone(regions._archive_dir())

    def test_pid_dir_is_a_link(self):
        elsewhere = self.tmp / "elsewhere"
        self.archive.rename(elsewhere)
        self.archive.symlink_to(elsewhere, target_is_directory=True)
        self.assertIsNone(regions._archive_dir())

    def test_marker_is_a_link(self):
        self.marker.unlink()
        self.marker.symlink_to(self.victim)
        self.assertIsNone(regions._archive_dir())

    def test_marker_is_a_hard_link(self):
        self.marker.unlink()
        os.link(self.victim, self.marker)
        self.assertIsNone(regions._archive_dir())

    def test_marker_is_a_directory(self):
        self.marker.unlink()
        self.marker.mkdir()
        self.assertIsNone(regions._archive_dir())

    def test_marker_of_an_earlier_process(self):
        # What a process killed with SIGKILL leaves for a later one with its pid.
        self.marker.write_bytes(b"00000000-0000-0000-0000-000000000000 4241\n")
        self.assertIsNone(regions._archive_dir())

    def test_marker_not_yet_written(self):
        # The writer has created the marker and not yet written the line.
        self.marker.write_bytes(b"")
        self.assertIsNone(regions._archive_dir())

    def test_process_unnamed(self):
        with mock.patch.object(regions, "_process_instance", return_value=None):
            self.assertIsNone(regions._archive_dir())


@unittest.skipUnless(sys.platform.startswith("linux"), "/proc/self/stat")
class ProcessInstanceTest(unittest.TestCase):
    """_process_instance: the boot id and field 22 of /proc/self/stat, also
    when the command name holds spaces and parentheses."""

    def setUp(self):
        with open("/proc/self/comm") as f:
            name = f.read().rstrip("\n")
        self.addCleanup(self.set_comm, name)

    @staticmethod
    def set_comm(name):
        with open("/proc/self/comm", "w") as f:
            f.write(name)

    def test_names_this_process(self):
        with open("/proc/sys/kernel/random/boot_id") as f:
            boot = f.read().strip()
        for name in ("python3", "a) b (c) 1 2"):
            with self.subTest(comm=name):
                self.set_comm(name)
                with open("/proc/self/stat") as f:
                    line = f.read()
                self.assertIn(name, line)
                start = line.rsplit(") ", 1)[1].split(" ")[19]
                self.assertEqual(regions._process_instance(), "%s %s" % (boot, start))


@unittest.skipIf(os.name == "nt", "POSIX permission bits and links")
class StreamTest(_ArchiveCase):
    """_Stream.write: regions/ is 0700 and pytorch.hrrr 0600 whatever the
    umask, and nothing planted at either is written through."""

    def setUp(self):
        super().setUp()
        self.regions = self.archive / "regions"
        self.region_file = self.regions / "pytorch.hrrr"

    def assert_private(self):
        self.assertEqual(stat.S_IMODE(os.lstat(self.regions).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.lstat(self.region_file).st_mode), 0o600)

    def test_new_files_are_private_under_any_umask(self):
        for mask in (0o000, 0o022, 0o277):
            with self.subTest(umask=oct(mask)):
                shutil.rmtree(self.regions, ignore_errors=True)
                old = os.umask(mask)
                try:
                    self.assertTrue(self.write_one())
                finally:
                    os.umask(old)
                self.assert_private()
                data = self.region_file.read_bytes()
                self.assertEqual(data[:len(regions._FILE_HEADER)], regions._FILE_HEADER)
                self.assertEqual(len(data), len(regions._FILE_HEADER)
                                 + regions._BATCH_PREFIX + regions._REC_SIZE)

    def test_existing_files_are_made_private(self):
        self.regions.mkdir(mode=0o755)
        os.chmod(self.regions, 0o755)
        self.region_file.write_bytes(b"")
        os.chmod(self.region_file, 0o644)
        self.assertTrue(self.write_one())
        self.assert_private()

    def test_regions_dir_is_a_link(self):
        elsewhere = self.tmp / "elsewhere"
        elsewhere.mkdir()
        self.regions.symlink_to(elsewhere, target_is_directory=True)
        self.assertFalse(self.write_one())
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_region_file_is_a_link(self):
        self.regions.mkdir(mode=0o700)
        self.region_file.symlink_to(self.victim)
        self.assertFalse(self.write_one())
        self.assertEqual(self.victim.read_bytes(), VICTIM_CONTENTS)

    def test_region_file_is_a_hard_link(self):
        self.regions.mkdir(mode=0o700)
        os.link(self.victim, self.region_file)
        self.assertFalse(self.write_one())
        self.assertEqual(self.victim.read_bytes(), VICTIM_CONTENTS)

    def test_region_file_is_a_fifo(self):
        # Nothing reads the FIFO, so an open that waits for a reader never
        # returns; the write runs on a daemon thread to turn that into a failure.
        self.regions.mkdir(mode=0o700)
        os.mkfifo(self.region_file)
        result = []
        writer = threading.Thread(target=lambda: result.append(self.write_one()),
                                  daemon=True)
        writer.start()
        writer.join(10)
        self.assertFalse(writer.is_alive(), "the write blocked on the FIFO")
        self.assertEqual(result, [False])

    def test_inactive_capture_writes_nothing(self):
        self.marker.unlink()
        self.assertFalse(self.write_one())
        self.assertFalse(self.regions.exists())


if __name__ == "__main__":
    unittest.main()
