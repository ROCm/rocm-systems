#!/usr/bin/env python3
"""Unit tests for src/device/ce_reduce/generate.py.

generate.py splits the 40 (type, redop) instantiations of
ncclCeLocalReduceKernelVec into separate TUs (see generate.py's module
docstring for why) by combining two on-disk templates -- ce_reduce_impl.h.in
(copied verbatim) and ce_reduce_launcher.cpp.in (expanded per-instantiation via
string.Template) -- with the TYPES/REDOPS/VECTORIZE_OK tables in generate.py
itself. These tests guard that pipeline end to end: that every expected file
is produced, that every $-placeholder in the launcher template gets filled in
(a stray literal placeholder, e.g. from an unescaped '$' added to a comment,
would silently leak into every generated .cpp), and that the two review fixes
already applied to generate.py (stale-subdirectory-safe output cleanup;
undefining CE_REDUCE_VECTORIZE_OK so it doesn't leak across TUs) keep working.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATE_PY = os.path.join(HERE, "generate.py")

TYPES = [
    ("f32", "float"),
    ("f64", "double"),
    ("f16", "__half"),
    ("bf16", "hip_bfloat16"),
    ("i32", "int32_t"),
    ("u32", "uint32_t"),
    ("i64", "int64_t"),
    ("u64", "uint64_t"),
    ("i8", "int8_t"),
    ("u8", "uint8_t"),
]

REDOPS = [
    ("Sum", 0),
    ("Prod", 1),
    ("Min", 2),
    ("Max", 3),
]

VECTORIZE_OK = {
    ("i8", "Min"),
    ("i8", "Max"),
    ("i8", "Sum"),
    ("u8", "Sum"),
}

# A leftover, un-substituted string.Template placeholder in generated output
# (e.g. "$ctype" or "${redname}") means the template and the .substitute()
# call fell out of sync -- most likely because a new placeholder was added to
# the .in file but not passed by generate.py, or vice versa.
_STRAY_PLACEHOLDER_RE = re.compile(r"\$\{?[a-zA-Z_]")


def _generate(out_dir):
    subprocess.run(
        [sys.executable, GENERATE_PY, out_dir],
        check=True,
        capture_output=True,
        text=True,
    )


class CeReduceGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")
        cls._dir = tempfile.mkdtemp(prefix="rccl_ce_reduce_")
        _generate(cls._dir)
        with open(os.path.join(cls._dir, "ce_reduce_impl.h")) as f:
            cls.impl_header = f.read()
        cls.launchers = {}
        for tag, _ in TYPES:
            for redname, _ in REDOPS:
                fname = "ce_reduce_%s_%s.cpp" % (tag, redname)
                with open(os.path.join(cls._dir, fname)) as f:
                    cls.launchers[(tag, redname)] = f.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, ignore_errors=True)

    def test_template_files_exist_next_to_generator(self):
        # generate.py reads these relative to its own directory, not argv[0]'s
        # caller cwd, so they must ship alongside it.
        self.assertTrue(os.path.exists(os.path.join(HERE, "ce_reduce_impl.h.in")))
        self.assertTrue(os.path.exists(os.path.join(HERE, "ce_reduce_launcher.cpp.in")))

    def test_generates_one_file_per_instantiation_plus_header(self):
        produced = set(os.listdir(self._dir))
        expected = {"ce_reduce_impl.h"}
        expected.update("ce_reduce_%s_%s.cpp" % (tag, redname) for tag, _ in TYPES for redname, _ in REDOPS)
        self.assertEqual(produced, expected)
        self.assertEqual(len(self.launchers), len(TYPES) * len(REDOPS))

    def test_impl_header_has_shared_definitions(self):
        self.assertIn("#pragma once", self.impl_header)
        self.assertIn("struct VecTrait", self.impl_header)
        self.assertIn("struct ReduceOp", self.impl_header)
        self.assertIn("ncclCeLocalReduceKernelVec", self.impl_header)
        self.assertIn("Copyright", self.impl_header)
        self.assertNotRegex(self.impl_header, _STRAY_PLACEHOLDER_RE)

    def test_every_launcher_fully_substituted(self):
        # No leftover "$identifier"/"${identifier}" anywhere in any generated
        # launcher -- every placeholder in ce_reduce_launcher.cpp.in must be
        # one that generate.py actually passes to .substitute().
        for key, text in self.launchers.items():
            with self.subTest(instantiation=key):
                self.assertNotRegex(text, _STRAY_PLACEHOLDER_RE)

    def test_launcher_uses_correct_type_and_redop(self):
        for tag, ctype in TYPES:
            for redname, redval in REDOPS:
                with self.subTest(tag=tag, redname=redname):
                    text = self.launchers[(tag, redname)]
                    self.assertIn("using T = %s;" % ctype, text)
                    self.assertIn(
                        "ncclResult_t ncclCeLocalReduceLaunch_%s_%s(" % (tag, redname), text
                    )
                    self.assertIn("ncclCeLocalReduceKernelVec<T, %d, UnrollFactor>" % redval, text)

    def test_vectorize_ok_define_matches_table(self):
        for tag, _ in TYPES:
            for redname, _ in REDOPS:
                with self.subTest(tag=tag, redname=redname):
                    text = self.launchers[(tag, redname)]
                    has_define = "#define CE_REDUCE_VECTORIZE_OK" in text
                    self.assertEqual(has_define, (tag, redname) in VECTORIZE_OK)

    def test_vectorize_ok_macro_is_undefined_after_use(self):
        # Regression test for the Copilot review fix: CE_REDUCE_VECTORIZE_OK
        # must not leak past ce_reduce_impl.h's inclusion into the rest of the
        # TU (e.g. into nccl.h or the launcher body below it).
        for key, text in self.launchers.items():
            with self.subTest(instantiation=key):
                include_idx = text.index('#include "ce_reduce_impl.h"')
                undef_idx = text.index("#undef CE_REDUCE_VECTORIZE_OK")
                self.assertGreater(undef_idx, include_idx)

    def test_cleanup_survives_stale_subdirectory_and_symlink(self):
        # Regression test for the Copilot review fix: a leftover subdirectory
        # or symlink (e.g. from an older build's output) must not crash the
        # os.listdir()-and-remove cleanup at the top of generate.py.
        stale_dir = tempfile.mkdtemp(prefix="rccl_ce_reduce_stale_")
        try:
            os.makedirs(os.path.join(stale_dir, "leftover_subdir"))
            with open(os.path.join(stale_dir, "leftover_subdir", "f.txt"), "w") as f:
                f.write("stale")
            os.symlink(
                os.path.join(stale_dir, "leftover_subdir", "f.txt"),
                os.path.join(stale_dir, "leftover_symlink"),
            )
            _generate(stale_dir)  # must not raise
            self.assertTrue(os.path.exists(os.path.join(stale_dir, "ce_reduce_impl.h")))
            self.assertNotIn("leftover_subdir", os.listdir(stale_dir))
            self.assertNotIn("leftover_symlink", os.listdir(stale_dir))
        finally:
            shutil.rmtree(stale_dir, ignore_errors=True)

    def test_generation_is_deterministic(self):
        other_dir = tempfile.mkdtemp(prefix="rccl_ce_reduce_repeat_")
        try:
            _generate(other_dir)
            for name in os.listdir(self._dir):
                with open(os.path.join(self._dir, name)) as f:
                    first = f.read()
                with open(os.path.join(other_dir, name)) as f:
                    second = f.read()
                self.assertEqual(first, second, msg="%s differs between runs" % name)
        finally:
            shutil.rmtree(other_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
