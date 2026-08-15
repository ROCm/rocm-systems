#!/usr/bin/env python3
"""Unit tests for src/device/generate.py device_table.h dispatch generation.

These guard the -fgpu-rdc (non device-linker) code path introduced to avoid
taking the address of any ncclDevFunc_* in a function-pointer table (issue
#8129). The generator emits, from a single header:

  * the function-pointer table (ncclDevFuncTable_*), used ONLY when
    RCCL_DEVICE_LINKER (or the legacy USE_INDIRECT_FUNCTION_CALL) is defined,
    and declared `static` so unused copies are dead-stripped; and
  * a compile-time templated binary-search dispatcher (Caller* /
    NCCL_CALL_FUNCTIONS_*) for the pure-RDC build, whose leaves call each
    ncclDevFunc_* directly by name (nothing address-taken).

The header is mode-agnostic (generated once); which arm is active is decided at
compile time by the macros. So these tests assert the *structure/gating* of the
generated text rather than compiling it.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATE_PY = os.path.join(HERE, "generate.py")

# A small, fast slice of collectives. "AllReduce RING SIMPLE Sum f32" expands to
# both an unguarded primary and an arch-guarded variant, which exercises the
# guarded-out (trap) leaf below. The AllReduce LL128 entries are reg-variant
# (see ll128_reg_variant_colls), so they carry a "_1"/"_2" reg suffix while every
# other kernel omits the reg field -- the pure-RDC dispatcher must call each by
# its exact declared name (regression: #ll128-reg-split).
ONLY_FUNCS = "AllReduce RING SIMPLE Sum f32|AllReduce RING LL128 Sum f32|SendRecv"


def _generate(tmpdir, ifc="OFF", local_gpu_only="OFF", only_funcs=ONLY_FUNCS, gfx_name=None):
    """Run generate.py into tmpdir and return the device_table.h contents."""
    env = os.environ.copy()
    if gfx_name is not None:
        rocm = os.path.join(tmpdir, "rocm")
        bin_dir = os.path.join(rocm, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        rocminfo = os.path.join(bin_dir, "rocminfo")
        with open(rocminfo, "w") as f:
            f.write(
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    echo "Name:                    {gfx_name}"
                    echo "Compute Unit:            304"
                    """
                )
            )
        os.chmod(rocminfo, 0o755)
        env["ROCM_PATH"] = rocm
        gensrc = os.path.join(tmpdir, "gensrc")
    else:
        gensrc = tmpdir

    # argv: gensrc, IFC, (unused), local_gpu_only, rocshmem, ONLY_FUNCS
    subprocess.run(
        [sys.executable, GENERATE_PY, gensrc, ifc, "OFF", local_gpu_only, "OFF", only_funcs],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    with open(os.path.join(gensrc, "device_table.h")) as f:
        return f.read(), gensrc


class DeviceTableGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")
        cls._dir = tempfile.mkdtemp(prefix="rccl_devtable_")
        cls.header, cls._gensrc = _generate(cls._dir)

    def test_forward_declarations_are_plain(self):
        # noinline is applied only by DEFINE_ncclDevFunc (common.h), gated on
        # RCCL_DEVICE_LINKER. The generated forward declarations must stay plain:
        # a stray noinline here leaks into the definition (attribute is a union
        # across decl+def) and would force pure-RDC funcs noinline.
        self.assertIn("__device__ void ncclDevFunc_", self.header)
        self.assertNotIn("noinline", self.header)
        self.assertNotIn("RCCL_DEVFUNC_ATTR", self.header)

    def test_table_is_static_and_runtime_dispatch_gated(self):
        # Table only for the runtime-dispatch builds, and internal linkage.
        self.assertIn(
            "#if defined(USE_INDIRECT_FUNCTION_CALL) || defined(RCCL_DEVICE_LINKER)",
            self.header,
        )
        self.assertIn("static __device__ ncclDevFuncPtr_t const ncclDevFuncTable_", self.header)

    def test_pure_rdc_dispatch_block_present(self):
        # Compile-time binary search only when NEITHER runtime-dispatch macro is set.
        self.assertIn(
            "#if !defined(USE_INDIRECT_FUNCTION_CALL) && !defined(RCCL_DEVICE_LINKER)",
            self.header,
        )
        self.assertIn("NCCL_CALL_FUNCTIONS_", self.header)
        # One explicit leaf specialization per index, dispatched by name.
        self.assertIn("struct Caller1<0, 1>", self.header)
        self.assertRegex(self.header, r"Caller1<0, \d+>::call1")

    def test_pure_rdc_dispatch_calls_only_declared_symbols(self):
        # Every ncclDevFunc_* called by name in a pure-RDC Caller leaf must be
        # one of the forward-declared symbols. The reg-variant split makes the
        # symbol name conditional (reg suffix only when reg != 0), so a leaf that
        # reconstructs the name from all fields (appending a stray "_0") would
        # reference an undeclared symbol and fail the -fgpu-rdc / --no-device-linker
        # link. This asserts the two symbol sets agree.
        declared = set(re.findall(r"__device__ void (ncclDevFunc_\w+)\(\);", self.header))
        self.assertTrue(declared, "no forward declarations found")
        called = set(
            re.findall(r"noexcept \{ (ncclDevFunc_\w+)\(\); \}", self.header)
        )
        self.assertTrue(called, "no pure-RDC dispatch leaves found")
        undeclared = called - declared
        self.assertEqual(
            set(),
            undeclared,
            "pure-RDC dispatch calls symbols that were never declared: %s" % sorted(undeclared),
        )
        # Sanity: no kernel ever gets a bogus "_0" reg suffix.
        self.assertFalse(any(re.search(r"_LL128_.*_0$", s) for s in called))
        # Sanity: AllReduce LL128 must appear as a reg-variant PAIR -- a reg=1 and
        # a reg=2 symbol. Match the reg field specifically: a bare endswith("_1"/"_2")
        # is not enough because reg=0 kernels omit the reg field and end in the unroll
        # value (which is also 1/2), so that check passes even if the split were removed.
        self.assertTrue(
            any(re.search(r"_AllReduce_RING_LL128_Sum_f32_\d+_\d+_\d+_1$", s) for s in called),
            "registered (reg=1) AllReduce LL128 symbol missing",
        )
        self.assertTrue(
            any(re.search(r"_AllReduce_RING_LL128_Sum_f32_\d+_\d+_\d+_2$", s) for s in called),
            "non-registered (reg=2) AllReduce LL128 symbol missing",
        )

    def test_guarded_out_leaf_traps_not_noop(self):
        # Arch-guarded-out slots must fail fast (matching the old nullptr table
        # entries), not silently no-op.
        self.assertIn("__builtin_trap();", self.header)

    def test_no_obsolete_table_omit_macro(self):
        # RCCL_DEVICE_TABLE_OMIT was retired by the static-table change.
        self.assertNotIn("RCCL_DEVICE_TABLE_OMIT", self.header)

    def test_specialized_shards_do_not_omit(self):
        # Specialized shards no longer #define RCCL_DEVICE_TABLE_OMIT.
        spec_dir = os.path.join(self._gensrc, "specialized")
        self.assertTrue(os.path.isdir(spec_dir))
        for name in os.listdir(spec_dir):
            with open(os.path.join(spec_dir, name)) as f:
                self.assertNotIn("RCCL_DEVICE_TABLE_OMIT", f.read())

    def test_ifc_build_keeps_table_and_no_rdc_dispatch(self):
        # Don't break the legacy indirect-function-call path: with IFC on, the
        # table is still emitted and the pure-RDC dispatcher is not.
        with tempfile.TemporaryDirectory(prefix="rccl_devtable_ifc_") as d:
            header, _ = _generate(d, ifc="ON")
        self.assertIn("static __device__ ncclDevFuncPtr_t const ncclDevFuncTable_", header)


class Gfx1250Fp8UnrollGenerationTest(unittest.TestCase):
    """FP8 kernels on gfx1250 local builds must be generated only at unroll 8."""

    FP8_ONLY_FUNCS = "AllReduce RING SIMPLE Sum f8e4m3|AllReduce RING SIMPLE Sum f8e5m2"
    F32_ONLY_FUNCS = "AllReduce RING SIMPLE Sum f32"

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")
        cls._gensrc_root = tempfile.mkdtemp(prefix="rccl_fp8_unroll_root_")

    def _specialized_unrolls(self, only_funcs):
        with tempfile.TemporaryDirectory(prefix="rccl_fp8_unroll_") as tmpdir:
            _, gensrc = _generate(
                tmpdir,
                local_gpu_only="ON",
                only_funcs=only_funcs,
                gfx_name="gfx1250",
            )
            spec_dir = os.path.join(gensrc, "specialized")
            return sorted(os.listdir(spec_dir))

    def test_fp8_kernels_generated_only_at_unroll_8(self):
        names = self._specialized_unrolls(self.FP8_ONLY_FUNCS)
        fp8_names = [n for n in names if "f8e4m3" in n or "f8e5m2" in n]
        self.assertTrue(fp8_names, "expected FP8 specialized kernels to be generated")
        for name in fp8_names:
            self.assertRegex(name, r"_8\.cpp$", msg=name)
            self.assertNotRegex(name, r"_(16|32)\.cpp$", msg=name)

    def test_non_fp8_kernels_keep_gfx1250_unroll_set(self):
        names = self._specialized_unrolls(self.F32_ONLY_FUNCS)
        f32_names = [n for n in names if "f32" in n]
        unrolls = sorted({re.search(r"_(\d+)\.cpp$", n).group(1) for n in f32_names})
        self.assertEqual(["16", "32", "8"], unrolls)

    def test_host_table_complete_when_unroll_buckets_are_uneven(self):
        # FP8 kernels only exist at unroll 8, so the unroll-8/16/32 slices of
        # func_rows are different sizes. host_table.cpp generation must select
        # rows by identity (fn.unroll == func_id_unroll), not by a positional
        # slice of func_rows: with a positional slice, identities past the
        # shortest bucket's length are silently missing from
        # ncclDevFuncNameToId entirely (not merely mapped to -1), which causes
        # "ncclDevFuncId: ... not found" at runtime for those identities.
        rocm = os.path.join(self._gensrc_root, "rocm")
        bin_dir = os.path.join(rocm, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        rocminfo = os.path.join(bin_dir, "rocminfo")
        with open(rocminfo, "w") as f:
            f.write(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    echo "Name:                    gfx1250"
                    echo "Compute Unit:            304"
                    """
                )
            )
        os.chmod(rocminfo, 0o755)
        env = os.environ.copy()
        env["ROCM_PATH"] = rocm
        gensrc = os.path.join(self._gensrc_root, "gensrc")
        script = textwrap.dedent(
            f"""
            import importlib.util, os, sys
            spec = importlib.util.spec_from_file_location("gen", {GENERATE_PY!r})
            mod = importlib.util.module_from_spec(spec)
            sys.argv = ["generate.py", {gensrc!r}, "OFF", "OFF", "ON", "OFF", "AllGather|AllReduce|AlltoAllPivot|Broadcast|Reduce|ReduceScatter|SendRecv"]
            spec.loader.exec_module(mod)
            expected = {{fn for fn in mod.func_rows if fn.unroll == mod.func_id_unroll}}
            print(len(expected))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env
        )
        expected_count = int(result.stdout.strip().splitlines()[-1])
        with open(os.path.join(gensrc, "host_table.cpp")) as f:
            host_table = f.read()
        actual_count = len(re.findall(r"^\s*\{\d+,\s*-?\d+\},", host_table, re.M))
        self.assertEqual(
            expected_count, actual_count,
            "ncclDevFuncNameToId is missing entries: expected one row per "
            "identity at func_id_unroll, got a different count",
        )


def _extract_table_rows(header, unroll):
    """Parse funcId -> symbol rows out of a generated ncclDevFuncTable_<unroll>[]
    (handling the "#if defined(__gfx1250__) ... #else ... #endif" guard form)."""
    rows = {}
    m = re.search(
        rf"ncclDevFuncTable_{unroll}\[\] = \{{(.*?)\nnullptr\}};",
        header,
        re.S,
    )
    if m is None:
        return None
    body = m.group(1)
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#if "):
            idx_line = lines[i + 1].strip()
            mm = re.match(r"/\*(\s*\d+)\*/\s*(.+?),", idx_line)
            if mm:
                rows[int(mm.group(1))] = mm.group(2).strip()
            i += 4
            continue
        mm = re.match(r"/\*(\s*\d+)\*/\s*(.+?),", line)
        if mm:
            rows[int(mm.group(1))] = mm.group(2).strip()
        i += 1
    return rows


class UnrollClampTest(unittest.TestCase):
    MINMAX_U8 = "AllReduce TREE SIMPLE MinMax u8"
    MINMAX_U8_ONLY = MINMAX_U8 + "|AllReduce TREE SIMPLE MinMax i8"

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")

    def _specialized(self, only_funcs):
        with tempfile.TemporaryDirectory(prefix="rccl_unroll_clamp_") as tmpdir:
            _, gensrc = _generate(
                tmpdir,
                local_gpu_only="ON",
                only_funcs=only_funcs,
                gfx_name="gfx1250",
            )
            return sorted(os.listdir(os.path.join(gensrc, "specialized")))

    def test_builtin_clamp_skips_unroll_16_and_32(self):
        names = self._specialized(self.MINMAX_U8_ONLY)
        u8 = [n for n in names if "minmax_u8_1_0_" in n]
        self.assertTrue(any(n.endswith("_8.cpp") for n in u8))
        self.assertFalse(any(n.endswith("_16.cpp") for n in u8))
        self.assertFalse(any(n.endswith("_32.cpp") for n in u8))

    def test_builtin_clamp_does_not_apply_on_multiarch(self):
        # Multi-arch func_id_unroll is "1"; clamping 16/32 there would alias
        # missing rows to the wrong axis. The blocklist is gfx1250-local only.
        with tempfile.TemporaryDirectory(prefix="rccl_unroll_clamp_ma_") as tmpdir:
            _, gensrc = _generate(
                tmpdir,
                local_gpu_only="OFF",
                only_funcs=self.MINMAX_U8_ONLY,
            )
            names = sorted(os.listdir(os.path.join(gensrc, "specialized")))
        u8 = [n for n in names if "minmax_u8_1_0_" in n]
        self.assertTrue(any(n.endswith("_16.cpp") for n in u8), names)
        self.assertTrue(any(n.endswith("_32.cpp") for n in u8), names)

    def _extract_gfx1250_table(self, header, unroll):
        rows = _extract_table_rows(header, unroll)
        self.assertIsNotNone(rows, f"table_{unroll} missing")
        return rows

    def test_clamped_kernels_alias_in_higher_unroll_tables(self):
        """Unroll-overridden kernels must occupy the same funcId in every table,
        aliased to their func-id-axis (unroll-8) symbol when no native variant."""
        with tempfile.TemporaryDirectory(prefix="rccl_clamp_table_") as tmpdir:
            header, _ = _generate(
                tmpdir,
                local_gpu_only="ON",
                only_funcs=(
                    "AllReduce TREE SIMPLE MinMax u8|AllReduce TREE SIMPLE MinMax i8|"
                    "AllReduce RING SIMPLE Sum f32"
                ),
                gfx_name="gfx1250",
            )
        t8 = self._extract_gfx1250_table(header, "8")
        t16 = self._extract_gfx1250_table(header, "16")
        t32 = self._extract_gfx1250_table(header, "32")
        self.assertEqual(len(t8), len(t16))
        self.assertEqual(len(t8), len(t32))

        # Built-in override: MinMax u8 acc=1 is clamped to unroll 8 only.
        clamp_idx = next(
            i for i, sym in t8.items() if sym.endswith("_MinMax_u8_1_0_8")
        )
        self.assertEqual(t8[clamp_idx], t16[clamp_idx])
        self.assertEqual(t8[clamp_idx], t32[clamp_idx])

        # Non-clamped: Sum f32 acc=1 keeps distinct native unroll variants.
        native_idx = next(
            i for i, sym in t8.items() if sym.endswith("_Sum_f32_1_0_8")
        )
        self.assertTrue(t16[native_idx].endswith("_Sum_f32_1_0_16"))
        self.assertTrue(t32[native_idx].endswith("_Sum_f32_1_0_32"))


class ArbitraryUnrollOverrideTest(unittest.TestCase):
    """The unroll override table is a general (identity -> target unroll)
    mechanism, not just a fixed "clamp to 8" blocklist: any identity can be
    redirected to any unroll, including one this build wouldn't otherwise
    produce -- that variant is compiled on demand and logged, then folded
    into every generic kernel's dispatch table for that identity.

    Exercises this via a throwaway extra entry appended to the "gfx1250" set
    inside _UNROLL_OVERRIDES in a patched copy of generate.py, so the
    built-in 17-entry table (all of which target unroll 8, already covered by
    UnrollClampTest) doesn't need to be changed just to test the general
    mechanism.
    """

    # acc=0 so the override target's arch guard is unambiguous (see
    # get_arch_guard(): unroll 8/16/32 implies gfx1250, acc=1 implies
    # gfx942/950/1250 -- neither applies to a bare unroll-4 acc=0 kernel).
    EXTRA_IDENTITY = ("AllReduce", "RING", "SIMPLE", "Sum", "f32", "0", "0")
    EXTRA_UNROLL = "4"  # not in gfx1250's local_unroll (8/16/32)
    ONLY_FUNCS = "AllReduce RING SIMPLE Sum f32"

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")

    def _patch_generate_py(self, tmpdir):
        with open(GENERATE_PY) as f:
            src = f.read()
        marker = '"gfx1250": {\n'
        self.assertIn(marker, src, "generate.py layout changed; update this test")
        extra = "    UnrollOverride(%s),\n" % ", ".join(
            repr(x) for x in self.EXTRA_IDENTITY + (self.EXTRA_UNROLL,)
        )
        patched_path = os.path.join(tmpdir, "generate_patched.py")
        with open(patched_path, "w") as f:
            f.write(src.replace(marker, marker + extra, 1))
        return patched_path

    def _generate(self, tmpdir):
        patched = self._patch_generate_py(tmpdir)
        rocm = os.path.join(tmpdir, "rocm")
        bin_dir = os.path.join(rocm, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        rocminfo = os.path.join(bin_dir, "rocminfo")
        with open(rocminfo, "w") as f:
            f.write(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    echo "Name:                    gfx1250"
                    echo "Compute Unit:            304"
                    """
                )
            )
        os.chmod(rocminfo, 0o755)
        env = os.environ.copy()
        env["ROCM_PATH"] = rocm
        gensrc = os.path.join(tmpdir, "gensrc")
        result = subprocess.run(
            [sys.executable, patched, gensrc, "OFF", "OFF", "ON", "OFF", self.ONLY_FUNCS],
            check=True, capture_output=True, text=True, env=env,
        )
        return result, gensrc

    def test_missing_target_unroll_is_forced_and_logged(self):
        with tempfile.TemporaryDirectory(prefix="rccl_override_arb_") as tmpdir:
            result, gensrc = self._generate(tmpdir)
            names = sorted(os.listdir(os.path.join(gensrc, "specialized")))
        overridden = [n for n in names if "_sum_f32_0_0_" in n]
        # unroll 4 (the override target) is force-compiled even though it's
        # not in gfx1250's local_unroll (8/16/32).
        self.assertTrue(any(n.endswith("_4.cpp") for n in overridden), overridden)
        # unroll 8 (func_id_unroll) is still compiled too -- it anchors this
        # identity's funcId/host_table bookkeeping -- but every generic
        # kernel's dispatch table is folded onto unroll 4 instead (see
        # test_generic_tables_fold_overridden_identity_everywhere), so 16/32
        # are redundant and skipped.
        self.assertTrue(any(n.endswith("_8.cpp") for n in overridden), overridden)
        self.assertFalse(any(n.endswith("_16.cpp") for n in overridden), overridden)
        self.assertFalse(any(n.endswith("_32.cpp") for n in overridden), overridden)
        self.assertIn("Unroll override", result.stdout)
        self.assertIn("unroll 4", result.stdout)

    def test_generic_tables_fold_overridden_identity_everywhere(self):
        # func_id_unroll (8) is skipped too: an override always wins, even on
        # its own identity's func-id-axis table (see
        # dispatch_fn_for_unroll_table()), so nothing needs to natively exist
        # at unroll 8 for this identity either.
        with tempfile.TemporaryDirectory(prefix="rccl_override_arb2_") as tmpdir:
            _, gensrc = self._generate(tmpdir)
            with open(os.path.join(gensrc, "device_table.h")) as f:
                header = f.read()
        for unroll in ("8", "16", "32"):
            rows = _extract_table_rows(header, unroll)
            self.assertIsNotNone(rows, f"table_{unroll} missing")
            idx = next(
                i for i, sym in rows.items() if sym.endswith("_Sum_f32_0_0_4")
            )
            self.assertTrue(
                rows[idx].endswith("_Sum_f32_0_0_4"),
                f"table_{unroll} funcId {idx} = {rows[idx]!r}, expected the "
                "override's unroll-4 symbol",
            )

    def test_unrelated_identity_is_unaffected(self):
        # Only the overridden acc=0 identity is redirected; the acc=1 variant
        # of the same coll/algo/proto/redop/ty keeps its native unroll set.
        with tempfile.TemporaryDirectory(prefix="rccl_override_arb3_") as tmpdir:
            _, gensrc = self._generate(tmpdir)
            names = sorted(os.listdir(os.path.join(gensrc, "specialized")))
        native = [n for n in names if "_sum_f32_1_0_" in n]
        self.assertTrue(any(n.endswith("_8.cpp") for n in native), names)
        self.assertTrue(any(n.endswith("_16.cpp") for n in native), names)
        self.assertTrue(any(n.endswith("_32.cpp") for n in native), names)


class HostDeviceDispatchAlignmentTest(unittest.TestCase):
    """host_table.cpp must map every device dispatch-table funcId.

    Guards against non-uniform per-unroll func_rows slices truncating
    host_table.cpp and dropping ncclDevFuncNameToId keys for real device
    funcIds (SendRecv, AllReduce SIMPLE, ...), which makes ncclDevFuncId()
    return -1 and the collective fail with ncclInvalidUsage at launch.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")
        cls._multiarch_dir = tempfile.mkdtemp(prefix="rccl_align_multiarch_")
        cls.multiarch_header, cls.multiarch_gensrc = _generate(
            cls._multiarch_dir, local_gpu_only="OFF", only_funcs="")
        cls._gfx1250_dir = tempfile.mkdtemp(prefix="rccl_align_gfx1250_")
        cls.gfx1250_header, cls.gfx1250_gensrc = _generate(
            cls._gfx1250_dir, local_gpu_only="ON", only_funcs="", gfx_name="gfx1250")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._multiarch_dir, ignore_errors=True)
        shutil.rmtree(cls._gfx1250_dir, ignore_errors=True)

    def _device_funcids(self, header):
        m = re.search(r"ncclDevFuncTable_8\[\] = \{(.*?)\nnullptr\};", header, re.S)
        self.assertIsNotNone(m, "ncclDevFuncTable_8 not found in device_table.h")
        # Arch-guarded rows print the index twice (symbol + nullptr); set() dedups.
        return {int(i) for i in re.findall(r"/\*\s*(\d+)\s*\*/", m.group(1))}

    def _host_funcids(self, gensrc):
        with open(os.path.join(gensrc, "host_table.cpp")) as f:
            body = f.read()
        return {int(fid) for _, fid in re.findall(r"\{(\d+),\s*(-?\d+)\}", body)
                if int(fid) >= 0}

    def _assert_complete(self, header, gensrc, label):
        missing = sorted(self._device_funcids(header) - self._host_funcids(gensrc))
        self.assertEqual(missing, [], "%s: %d device funcId(s) have no "
                         "ncclDevFuncNameToId key: %s" % (label, len(missing), missing[:16]))

    def test_multiarch_host_table_is_complete(self):
        self._assert_complete(self.multiarch_header, self.multiarch_gensrc, "multi-arch")

    def test_gfx1250_local_host_table_is_complete(self):
        self._assert_complete(self.gfx1250_header, self.gfx1250_gensrc, "gfx1250 -l")

    def test_p2p_sendrecv_key_present_multiarch(self):
        with open(os.path.join(self.multiarch_gensrc, "host_table.cpp")) as f:
            body = f.read()
        self.assertRegex(body, r"\{5,\s*\d+\}",
                         "SendRecv key (5) missing -> ncclDevFuncId_P2p() returns -1")


if __name__ == "__main__":
    unittest.main()
