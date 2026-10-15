#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tests for ``amdsmi_interface_path``, which locates the interface source a test parses.

Tests that read ``amdsmi_interface.py`` as text cannot hard-code ``py-interface/``: an install
ships the file as ``share/amd_smi/amdsmi/amdsmi_interface.py`` and has no ``py-interface``
directory. These tests build both layouts in a temp dir and check which file is chosen.
"""

import pathlib
import tempfile
import unittest
from unittest import mock

from common import common

_INTERFACE_NAME = "amdsmi_interface.py"


def _touch(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


class TestAmdsmiInterfacePath(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name).resolve()

    def _pin(self, share_dir: pathlib.Path, is_explicit: bool):
        for name, value in (
            ("amdsmi_path", str(share_dir)),
            ("amdsmi_path_is_explicit", is_explicit),
        ):
            patcher = mock.patch.object(common, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _checkout_layout(self) -> pathlib.Path:
        _touch(self.root / "src" / "py-interface" / _INTERFACE_NAME)
        test_file = _touch(self.root / "src" / "tests" / "python" / "unit" / "system" / "t.py")
        return test_file

    def _install_layout(self) -> pathlib.Path:
        share = self.root / "rocm" / "share" / "amd_smi"
        _touch(share / "amdsmi" / _INTERFACE_NAME)
        test_file = _touch(share / "tests" / "python_unittest" / "unit" / "system" / "t.py")
        return test_file

    def test_a_checkout_resolves_to_py_interface(self):
        test_file = self._checkout_layout()
        self._pin(self.root / "nowhere", is_explicit=False)
        expected = self.root / "src" / "py-interface" / _INTERFACE_NAME
        self.assertEqual(common.amdsmi_interface_path(str(test_file)), expected)

    def test_an_install_without_py_interface_resolves_to_the_amdsmi_package(self):
        test_file = self._install_layout()
        share = self.root / "rocm" / "share" / "amd_smi"
        self._pin(share, is_explicit=False)
        expected = share / "amdsmi" / _INTERFACE_NAME
        self.assertEqual(common.amdsmi_interface_path(str(test_file)), expected)

    def test_a_pinned_install_wins_over_a_checkout(self):
        checkout_file = self._checkout_layout()
        share = self.root / "rocm" / "share" / "amd_smi"
        _touch(share / "amdsmi" / _INTERFACE_NAME)
        self._pin(share, is_explicit=True)
        expected = share / "amdsmi" / _INTERFACE_NAME
        self.assertEqual(common.amdsmi_interface_path(str(checkout_file)), expected)

    def test_a_checkout_wins_when_no_install_is_pinned(self):
        checkout_file = self._checkout_layout()
        share = self.root / "rocm" / "share" / "amd_smi"
        _touch(share / "amdsmi" / _INTERFACE_NAME)
        self._pin(share, is_explicit=False)
        expected = self.root / "src" / "py-interface" / _INTERFACE_NAME
        self.assertEqual(common.amdsmi_interface_path(str(checkout_file)), expected)

    def test_a_missing_interface_names_every_place_searched(self):
        test_file = _touch(self.root / "lonely" / "tests" / "t.py")
        share = self.root / "empty_share"
        share.mkdir()
        self._pin(share, is_explicit=False)
        with self.assertRaises(FileNotFoundError) as ctx:
            common.amdsmi_interface_path(str(test_file))
        self.assertIn(str(share / "amdsmi" / _INTERFACE_NAME), str(ctx.exception))
        self.assertIn("py-interface", str(ctx.exception))
