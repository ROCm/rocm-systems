# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Every handler leaves an unmapped pointer to the one policy.

The unmapped-pointer policy (playback/hip_playback.h) is applied once, in
dispatch_event: translate_ptr notes the miss and the handler does not decide.
This fails if a generated handler again null-checks a translate_ptr result and
returns on its own, the per-API skip lists the policy replaced, and if a
hand-written function outside the known exceptions uses lookup_ptr, the lookup
that leaves no note and so takes a pointer out of the policy.

Run: python3 -m unittest discover -s projects/hrr/tools/tests
"""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
GEN = TOOLS / "gen_hrr_api_args.py"
MANUAL_PLAYBACK = TOOLS.parent / "playback" / "hip_playback.cpp"

# `x = ctx.translate_ptr(...);` followed by a null test of x.
_OWN_NULL_CHECK = re.compile(
    r"(\w+) = ctx\.translate_ptr\([^;]*\);\s*\n\s*if \(\s*(?:!\1\b|\1 == nullptr)")
# The start of a top-level function definition in hip_playback.cpp.
_FUNCTION = re.compile(r"^(?:static\s+)?(?:inline\s+)?[\w:<>*&]+(?:\s+[\w:<>*&]+)*\s+\**(\w+)\(")

# Where a miss is an answer, not a lost pointer: kernel-argument words are
# pointers only by shape, a guard or release has nothing to do for memory
# replay never had, and translate_or_materialize notes the miss itself once the
# region has had its chance.
LOOKUP_ALLOWED = {
    "decode_kernel_args",
    "hrr_block_guard_relocate",
    "playback_hipFreeHost",
    "playback_hipHostUnregister",
    "playback_hipFree",
    "playback_hipFreeAsync",
    "translate_or_materialize",
}


def own_null_checks(text):
    return [m.group(1) for m in _OWN_NULL_CHECK.finditer(text)]


def lookup_users(text):
    """Names of the functions in `text` that call lookup_ptr."""
    users = set()
    current = None
    for line in text.splitlines():
        m = _FUNCTION.match(line)
        if m:
            current = m.group(1)
        if "lookup_ptr(" in line and not line.strip().startswith("//"):
            users.add(current)
    return users


class UnmappedPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        out = Path(cls._tmp.name)
        proc = subprocess.run(
            [sys.executable, str(GEN), "--silent",
             "--output-header", str(out / "hrr_api_args.h"),
             "--output-capture", str(out / "capture.cpp"),
             "--output-playback", str(out / "playback.cpp")],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("generator failed:\n" + proc.stdout + proc.stderr)
        cls.playback = (out / "playback.cpp").read_text()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_no_generated_handler_decides_an_unmapped_pointer(self):
        self.assertEqual(own_null_checks(self.playback), [])

    def test_generated_release_does_not_note_a_miss(self):
        self.assertIn("ctx.lookup_ptr(_rec_ptr)", self.playback)
        self.assertNotIn("ctx.translate_ptr(_rec_ptr)", self.playback)

    def test_manual_lookup_users_are_the_known_exceptions(self):
        users = lookup_users(MANUAL_PLAYBACK.read_text())
        self.assertEqual(users - LOOKUP_ALLOWED, set())
        self.assertIn("translate_or_materialize", users)

    def test_scanner_detects_the_patterns_it_guards_against(self):
        self.assertEqual(own_null_checks(
            "  void* _live_dst = ctx.translate_ptr(a->ptr);\n  if (_live_dst == nullptr) {"),
            ["_live_dst"])
        self.assertEqual(own_null_checks(
            "  void* _live_dst = ctx.translate_ptr(a->dst);\n  if (!_live_dst) {"),
            ["_live_dst"])
        self.assertEqual(lookup_users(
            "static void* helper(int x) {\n    return ctx.lookup_ptr(x);\n}\n"), {"helper"})


if __name__ == "__main__":
    unittest.main()
