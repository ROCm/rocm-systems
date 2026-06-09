#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Unit tests for the ESMI core-count safety pre-check in amdsmi_init.

``amdsmi_init._check_esmi_safe()`` reproduces the socket-count computation the
esmi library performs for CPU family 0x1A models 0x00-0x1F and 0x50-0x5F: it
divides the number of online cores by the per-socket core capacity reported by
CPUID 0x80000008. When the online core count is below that capacity the division
truncates to zero, which leads esmi to a zero-size allocation and a SIGSEGV.
These tests mock ``/dev/cpu/0/cpuid`` so the prediction logic can be exercised
deterministically without that hardware.
"""

import os
import struct
import sys
import unittest
from unittest import mock

# Resolve the amdsmi_cli and py-interface dirs relative to this test file so the
# pure-Python init logic can be imported without GPU hardware.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "amdsmi_cli"), os.path.join(_REPO_ROOT, "py-interface")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import amdsmi_init


def _pack_leaf(eax=0, ebx=0, ecx=0, edx=0):
    return struct.pack("IIII", eax, ebx, ecx, edx)


# Family 0x1A, model 0x10 - a family/model that esmi reads the CPUID 0x80000008
# capacity field for. Encoded to satisfy esmi's own decode:
#   family = ((eax >> 8) & 0xf) + ((eax >> 20) & 0xff)
#   model  = ((eax >> 16) & 0xf) * 0x10 + ((eax >> 4) & 0xf)
_AFFECTED_LEAF1_EAX = 0x00B10F00
# Family 0x19, model 0x10 (a family that does NOT use the 0x80000008 path).
_UNAFFECTED_LEAF1_EAX = 0x00A10F00


class _FakeCpuid:
    """Emulates /dev/cpu/0/cpuid: the file position selects the CPUID leaf."""

    def __init__(self, leaves):
        self._leaves = leaves
        self._pos = 0

    def open(self, path, flags):
        return 7  # arbitrary sentinel fd

    def lseek(self, fd, pos, whence):
        self._pos = pos
        return pos

    def read(self, fd, n):
        return self._leaves.get(self._pos, _pack_leaf())

    def close(self, fd):
        pass


class TestCheckEsmiSafe(unittest.TestCase):
    def _patch_cpuid(self, leaves, cpu_count):
        fake = _FakeCpuid(leaves)
        return mock.patch.multiple(
            amdsmi_init.os,
            open=fake.open,
            lseek=fake.lseek,
            read=fake.read,
            close=fake.close,
            cpu_count=lambda: cpu_count,
        )

    def test_capacity_exceeds_online_cores_is_unsafe(self):
        # 288-core socket capacity, only 192 online -> esmi would divide to 0.
        leaves = {
            0x1: _pack_leaf(eax=_AFFECTED_LEAF1_EAX),
            0x80000008: _pack_leaf(ecx=287),  # (287 & 0xFFF) + 1 == 288
        }
        with self._patch_cpuid(leaves, cpu_count=192):
            self.assertFalse(amdsmi_init._check_esmi_safe())

    def test_enough_online_cores_is_safe(self):
        leaves = {0x1: _pack_leaf(eax=_AFFECTED_LEAF1_EAX), 0x80000008: _pack_leaf(ecx=287)}
        with self._patch_cpuid(leaves, cpu_count=288):
            self.assertTrue(amdsmi_init._check_esmi_safe())

    def test_unaffected_family_is_always_safe(self):
        # A family esmi does not read the capacity field for must not be gated by
        # the 0x80000008 value even if it looks "too small"; esmi never takes
        # that path here.
        leaves = {0x1: _pack_leaf(eax=_UNAFFECTED_LEAF1_EAX), 0x80000008: _pack_leaf(ecx=287)}
        with self._patch_cpuid(leaves, cpu_count=1):
            self.assertTrue(amdsmi_init._check_esmi_safe())

    def test_cpuid_read_failure_is_safe(self):
        def _boom(path, flags):
            raise OSError("no cpuid device")

        with mock.patch.object(amdsmi_init.os, "open", _boom):
            self.assertTrue(amdsmi_init._check_esmi_safe())


if __name__ == "__main__":
    unittest.main(verbosity=2)
