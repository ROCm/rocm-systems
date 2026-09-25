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

from __future__ import annotations

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


def _generate(out_dir: str) -> None:
    subprocess.run(
        [sys.executable, GENERATE_PY, out_dir],
        check=True,
        capture_output=True,
        text=True,
    )


class CeReduceGenerationTest(unittest.TestCase):
    _dir: str
    impl_header: str
    launchers: dict[tuple[str, str], str]

    @classmethod
    def setUpClass(cls) -> None:
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
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._dir, ignore_errors=True)

    def test_template_files_exist_next_to_generator(self) -> None:
        # generate.py reads these relative to its own directory, not argv[0]'s
        # caller cwd, so they must ship alongside it.
        self.assertTrue(os.path.exists(os.path.join(HERE, "ce_reduce_impl.h.in")))
        self.assertTrue(os.path.exists(os.path.join(HERE, "ce_reduce_launcher.cpp.in")))

    def test_generates_one_file_per_instantiation_plus_header(self) -> None:
        produced = set(os.listdir(self._dir))
        expected = {"ce_reduce_impl.h"}
        expected.update("ce_reduce_%s_%s.cpp" % (tag, redname) for tag, _ in TYPES for redname, _ in REDOPS)
        self.assertEqual(produced, expected)
        self.assertEqual(len(self.launchers), len(TYPES) * len(REDOPS))

    def test_impl_header_has_shared_definitions(self) -> None:
        self.assertIn("#pragma once", self.impl_header)
        self.assertIn("struct VecTrait", self.impl_header)
        self.assertIn("struct ReduceOp", self.impl_header)
        self.assertIn("ncclCeLocalReduceKernelVec", self.impl_header)
        self.assertIn("Copyright", self.impl_header)
        self.assertNotRegex(self.impl_header, _STRAY_PLACEHOLDER_RE)

    def test_every_launcher_fully_substituted(self) -> None:
        # No leftover "$identifier"/"${identifier}" anywhere in any generated
        # launcher -- every placeholder in ce_reduce_launcher.cpp.in must be
        # one that generate.py actually passes to .substitute().
        for key, text in self.launchers.items():
            with self.subTest(instantiation=key):
                self.assertNotRegex(text, _STRAY_PLACEHOLDER_RE)

    def test_launcher_uses_correct_type_and_redop(self) -> None:
        for tag, ctype in TYPES:
            for redname, redval in REDOPS:
                with self.subTest(tag=tag, redname=redname):
                    text = self.launchers[(tag, redname)]
                    self.assertIn("using T = %s;" % ctype, text)
                    self.assertIn(
                        "ncclResult_t ncclCeLocalReduceLaunch_%s_%s(" % (tag, redname), text
                    )
                    self.assertIn("ncclCeLocalReduceKernelVec<T, %d, UnrollFactor>" % redval, text)

    def test_vectorize_ok_define_matches_table(self) -> None:
        for tag, _ in TYPES:
            for redname, _ in REDOPS:
                with self.subTest(tag=tag, redname=redname):
                    text = self.launchers[(tag, redname)]
                    has_define = "#define CE_REDUCE_VECTORIZE_OK" in text
                    self.assertEqual(has_define, (tag, redname) in VECTORIZE_OK)

    def test_vectorize_ok_macro_is_undefined_after_use(self) -> None:
        # Regression test for the Copilot review fix: CE_REDUCE_VECTORIZE_OK
        # must not leak past ce_reduce_impl.h's inclusion into the rest of the
        # TU (e.g. into nccl.h or the launcher body below it).
        for key, text in self.launchers.items():
            with self.subTest(instantiation=key):
                include_idx = text.index('#include "ce_reduce_impl.h"')
                undef_idx = text.index("#undef CE_REDUCE_VECTORIZE_OK")
                self.assertGreater(undef_idx, include_idx)

    def test_cleanup_survives_stale_subdirectory_and_symlink(self) -> None:
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

    def test_generation_is_deterministic(self) -> None:
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


# ---------------------------------------------------------------------------
# CE AllReduce persistent-kernel contracts.
#
# The persistent, pipelined kernel adds cross-file contracts the tests above
# cannot cover, because they did not exist when the kernel was a plain
# local reduce with a 5-argument launcher:
#
#   * hipLaunchCooperativeKernel takes its kernel arguments as an untyped
#     void* array, so adding, removing or reordering a kernel parameter
#     without updating that array is not a compile error -- the kernel just
#     receives garbage, on the cooperative path only.
#   * the generated launchers and ce_reduce.cc's extern declarations are
#     written in separate files with nothing tying them together, so drift
#     surfaces as 40 undefined symbols at the end of a full build.
#   * the kernel's slot constants duplicate src/include/ce_coll.h, which the
#     host uses to size the signal buffer and the slot strides. If the two
#     disagree the kernel reads a different slot than the host wrote.
# ---------------------------------------------------------------------------

DEVICE_DIR = os.path.dirname(HERE)
CE_REDUCE_CC = os.path.join(DEVICE_DIR, "ce_reduce.cc")
CE_COLL_H = os.path.join(os.path.dirname(DEVICE_DIR), "include", "ce_coll.h")


def _strip_line_comments(text: str) -> str:
    # Parameter lists carry trailing // comments containing parentheses, which
    # would unbalance the brace matching below.
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def _split_top_level(text: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _trailing_identifier(param: str) -> str | None:
    m = re.search(r"([A-Za-z_]\w*)\s*$", param)
    return m.group(1) if m else None


def _without_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _macro_body(text: str, name: str) -> str | None:
    lines = text.splitlines()
    prefix = "#define " + name
    for idx, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        body, acc = line[len(prefix):], []
        while True:
            stripped = body.rstrip()
            if stripped.endswith("\\"):
                acc.append(stripped[:-1])
                idx += 1
                body = lines[idx]
            else:
                acc.append(stripped)
                return " ".join(acc)
    return None


class CeReducePersistentContractTest(unittest.TestCase):
    _dir: str
    impl: str
    launcher: str

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")
        cls._dir = tempfile.mkdtemp(prefix="rccl_ce_reduce_contract_")
        _generate(cls._dir)
        with open(os.path.join(cls._dir, "ce_reduce_impl.h")) as f:
            cls.impl = _strip_line_comments(f.read())
        # Sum/f32 is the awkward case for the parsing below: its redop value is
        # 0, so "<T, 0, UnrollFactor>" also contains a "0," before the launch
        # config's own "0, stream,".
        with open(os.path.join(cls._dir, "ce_reduce_f32_Sum.cpp")) as f:
            cls.launcher = _strip_line_comments(f.read())

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._dir, ignore_errors=True)

    def _kernel_parameters(self) -> list[str | None]:
        m = re.search(r"void\s+ncclCeLocalReduceKernelVec\s*\((.*?)\)\s*\{", self.impl, re.S)
        self.assertIsNotNone(m, "kernel signature not found in ce_reduce_impl.h")
        assert m is not None
        return [_trailing_identifier(p) for p in _split_top_level(m.group(1))]

    def _ggl_arguments(self) -> list[str]:
        m = re.search(r"hipLaunchKernelGGL\(.*?0,\s*stream,\s*(.*?)\);", self.launcher, re.S)
        self.assertIsNotNone(m, "hipLaunchKernelGGL call not found in the generated launcher")
        assert m is not None
        return [a.strip() for a in m.group(1).split(",") if a.strip()]

    def _cooperative_arguments(self) -> list[str]:
        m = re.search(r"void\*\s*kernelArgs\[\]\s*=\s*\{(.*?)\};", self.launcher, re.S)
        self.assertIsNotNone(m, "kernelArgs array not found in the generated launcher")
        assert m is not None
        return [a.strip().lstrip("&").strip() for a in m.group(1).split(",") if a.strip()]

    def test_cooperative_and_ggl_paths_pass_identical_arguments(self) -> None:
        # Nothing in the compiler makes these two agree: the GGL call is type
        # checked against the kernel, the cooperative array is not.
        self.assertEqual(self._cooperative_arguments(), self._ggl_arguments())

    def test_cooperative_argument_count_matches_kernel_parameters(self) -> None:
        self.assertEqual(len(self._cooperative_arguments()), len(self._kernel_parameters()))

    def test_launcher_signature_matches_dispatcher_declaration(self) -> None:
        with open(CE_REDUCE_CC) as f:
            dispatcher = _strip_line_comments(f.read())
        declared = _macro_body(dispatcher, "NCCL_CE_LAUNCH_PARAMS")
        self.assertIsNotNone(declared, "NCCL_CE_LAUNCH_PARAMS not found in ce_reduce.cc")
        assert declared is not None
        m = re.search(r"ncclResult_t\s+ncclCeLocalReduceLaunch_\w+\s*\((.*?)\)\s*\{", self.launcher, re.S)
        self.assertIsNotNone(m, "launcher signature not found in the generated launcher")
        assert m is not None
        self.assertEqual(
            [_without_whitespace(p) for p in _split_top_level(m.group(1))],
            [_without_whitespace(p) for p in _split_top_level(declared)],
        )

    def test_slot_constants_agree_with_ce_coll_h(self) -> None:
        with open(CE_COLL_H) as f:
            header = f.read()
        for name in ("NCCL_CE_REDUCE_MAX_BLOCKS", "NCCL_CE_NUM_SLOTS"):
            with self.subTest(constant=name):
                pattern = r"#define\s+%s\s+(\d+)" % name
                kernel_side = re.search(pattern, self.impl)
                host_side = re.search(pattern, header)
                self.assertIsNotNone(kernel_side, "%s not defined in ce_reduce_impl.h" % name)
                self.assertIsNotNone(host_side, "%s not defined in ce_coll.h" % name)
                assert kernel_side is not None and host_side is not None
                self.assertEqual(kernel_side.group(1), host_side.group(1))

    def test_launch_bounds_matches_launcher_thread_count(self) -> None:
        bounds = re.search(r"__launch_bounds__\((\d+)\)", self.impl)
        threads = re.search(r"const int threads\s*=\s*(\d+);", self.launcher)
        self.assertIsNotNone(bounds, "__launch_bounds__ not found in ce_reduce_impl.h")
        self.assertIsNotNone(threads, "thread count not found in the generated launcher")
        assert bounds is not None and threads is not None
        self.assertEqual(bounds.group(1), threads.group(1))


if __name__ == "__main__":
    unittest.main()
