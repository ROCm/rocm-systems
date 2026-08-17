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

import contextlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from importlib.machinery import SourceFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATE_PY = os.path.join(HERE, "generate.py")

# A small, fast slice of collectives. "AllReduce RING SIMPLE Sum f32" expands to
# both an unguarded primary and an arch-guarded variant, which exercises the
# guarded-out (trap) leaf below. The AllReduce LL128 entries are reg-variant
# (see ll128_reg_variant_colls), so they carry a "_1"/"_2" reg suffix while every
# other kernel omits the reg field -- the pure-RDC dispatcher must call each by
# its exact declared name (regression: #ll128-reg-split).
ONLY_FUNCS = "AllReduce RING SIMPLE Sum f32|AllReduce RING LL128 Sum f32|SendRecv"


def _write_fake_rocminfo(tmpdir, agents):
    """Create a stub $ROCM_PATH/bin/rocminfo announcing `agents`, and return ROCM_PATH.

    `agents` is a list of (name, compute_units). A leading CPU agent (a "Name:"
    line with no "gfx" in it) is always emitted, mirroring real rocminfo output:
    calc_unroll_and_pipeline_for_local_arch() must attribute each "Compute Unit:"
    line to the preceding gfx "Name:" line and ignore the CPU's.
    """
    rocm = os.path.join(tmpdir, "rocm")
    bin_dir = os.path.join(rocm, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    lines = ["#!/bin/sh", 'echo "Name:                    AMD EPYC 9654"',
             'echo "Compute Unit:            192"']
    for name, cu in agents:
        lines.append('echo "Name:                    %s"' % name)
        lines.append('echo "Compute Unit:            %s"' % cu)
    rocminfo = os.path.join(bin_dir, "rocminfo")
    with open(rocminfo, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(rocminfo, 0o755)
    return rocm


def _generate(tmpdir, ifc="OFF", local_gpu_only="OFF", only_funcs=ONLY_FUNCS, gfx_name=None):
    """Run generate.py into tmpdir and return (device_table.h contents, gensrc).

    gfx_name is either a single gfx name (304 CUs assumed) or a list of
    (name, compute_units) tuples for multi-agent / heterogeneous systems.
    """
    env = os.environ.copy()
    if gfx_name is not None:
        agents = [(gfx_name, 304)] if isinstance(gfx_name, str) else list(gfx_name)
        env["ROCM_PATH"] = _write_fake_rocminfo(tmpdir, agents)
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


def _import_generate(tmpdir, only_funcs="", local_gpu_only="OFF", gfx_name=None):
    """Import generate.py in-process so a test can inspect its own tables.

    generate.py does all of its work at import time, driven by sys.argv, so this
    performs a full generation into tmpdir as a side effect; the returned module
    exposes that run's state (mod.gensrc, mod.primary_funcs, mod.get_arch_guard,
    ...). Used where asserting on generated *text* would only re-derive what the
    generator already computed.
    """
    gensrc = os.path.join(tmpdir, "gensrc")
    saved_argv = sys.argv
    saved_rocm = os.environ.get("ROCM_PATH")
    if gfx_name is not None:
        agents = [(gfx_name, 304)] if isinstance(gfx_name, str) else list(gfx_name)
        os.environ["ROCM_PATH"] = _write_fake_rocminfo(tmpdir, agents)
    try:
        sys.argv = ["generate.py", gensrc, "OFF", "OFF", local_gpu_only, "OFF",
                    only_funcs]
        loader = SourceFileLoader("rccl_generate_under_test", GENERATE_PY)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            loader.exec_module(mod)
        return mod
    finally:
        sys.argv = saved_argv
        if saved_rocm is None:
            os.environ.pop("ROCM_PATH", None)
        else:
            os.environ["ROCM_PATH"] = saved_rocm


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
    """FP8 kernels on a gfx1250 local build must be generated only at unroll 8.

    Fp8FixedUnrollArchRuleTest covers the same rule in a multi-arch/fat build.
    """

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


class Fp8FixedUnrollArchRuleTest(unittest.TestCase):
    """"gfx1250 launches FP8 only at unroll 8" is a property of the ARCH, so it
    must hold however the build was invoked -- not just under --local_gpu_only.

    A multi-arch/fat build never learns its GPU_TARGETS list, so it compiles the
    gfx1250-exclusive tiers (8/16/32) unconditionally. Keying this rule on
    local_gfx_name therefore left every FP8 kernel at unroll 16/32 in the fat
    binary, where enqueue.cc's runtime clamp guarantees nothing can ever dispatch
    it -- pure compile time at the two slowest tiers. The tiers gfx1250 shares
    with other archs (1/2/4) must keep their FP8 variants, since those archs do
    dispatch them.
    """

    # SIMPLE only: LL128 kernels are reg-variants whose filenames carry a
    # trailing reg field, which would confuse a naive unroll-from-filename read.
    FUNCS = "AllReduce RING SIMPLE Sum f8e4m3/f8e5m2/f32"

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")
        cls._dir = tempfile.mkdtemp(prefix="rccl_fp8_arch_rule_")
        # One generation per build mode, shared by every test below.
        cls.multiarch_header, cls.multiarch = cls._gen("multiarch")
        _, cls.gfx1250_local = cls._gen("gfx1250_local", local="ON", gfx="gfx1250")
        _, cls.gfx950_local = cls._gen("gfx950_local", local="ON", gfx="gfx950")

    @classmethod
    def _gen(cls, name, local="OFF", gfx=None):
        tmpdir = os.path.join(cls._dir, name)
        os.makedirs(tmpdir, exist_ok=True)
        return _generate(tmpdir, local_gpu_only=local, only_funcs=cls.FUNCS,
                         gfx_name=gfx)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, ignore_errors=True)

    @staticmethod
    def _unrolls_for_type(gensrc, ty):
        """The unroll factors a specialized kernel was generated at for `ty`.

        Filenames are specialized_<coll>_<algo>_<proto>_<redop>_<ty>_<acc>_
        <pipeline>_<unroll>[_<reg>].cpp; anchoring on the type keeps the unroll
        field unambiguous.
        """
        pattern = re.compile(rf"_{ty}_\d+_\d+_(\d+)(?:_\d+)?\.cpp$")
        found = set()
        for name in os.listdir(os.path.join(gensrc, "specialized")):
            m = pattern.search(name)
            if m:
                found.add(m.group(1))
        return found

    def _fp8_unrolls(self, gensrc):
        return (self._unrolls_for_type(gensrc, "f8e4m3")
                | self._unrolls_for_type(gensrc, "f8e5m2"))

    def test_multiarch_omits_fp8_at_gfx1250_exclusive_tiers(self):
        # The regression this fix exists for: unroll 16/32 are compiled only for
        # gfx1250, which can never launch FP8 there.
        unrolls = self._fp8_unrolls(self.multiarch)
        self.assertTrue(unrolls, "expected FP8 kernels in a multi-arch build")
        self.assertEqual(set(), unrolls & {"16", "32"},
                         "FP8 kernels generated at a gfx1250-exclusive tier that "
                         "the runtime clamp makes unreachable")

    def test_multiarch_keeps_fp8_at_shared_tiers_and_clamp_target(self):
        # The other side of the rule: narrowing it must not strand gfx950/gfx942,
        # which dispatch FP8 from the shared tiers, nor drop unroll 8 itself --
        # the variant gfx1250 clamps TO.
        self.assertEqual({"1", "2", "4", "8"}, self._fp8_unrolls(self.multiarch))

    def test_multiarch_keeps_non_fp8_types_at_every_unroll(self):
        # Guards against the rule leaking past FP8 and thinning out real work.
        self.assertEqual({"1", "2", "4", "8", "16", "32"},
                         self._unrolls_for_type(self.multiarch, "f32"))

    def test_exclusive_tier_fp8_set_is_independent_of_build_mode(self):
        # The point of the fix, stated directly: restricted to the tiers gfx1250
        # owns, a fat build and a gfx1250 -l build agree on which FP8 variants
        # exist. Before the fix the fat build had all three tiers here.
        exclusive = {"8", "16", "32"}
        self.assertEqual(self._fp8_unrolls(self.gfx1250_local) & exclusive,
                         self._fp8_unrolls(self.multiarch) & exclusive)
        self.assertEqual({"8"}, self._fp8_unrolls(self.gfx1250_local) & exclusive)

    def test_non_gfx1250_local_build_keeps_every_fp8_variant(self):
        # gfx950 owns none of the exclusive tiers, so one arch's launch rule must
        # not follow it into its own -l build.
        self.assertEqual(self._unrolls_for_type(self.gfx950_local, "f32"),
                         self._fp8_unrolls(self.gfx950_local))

    def test_exclusive_tier_tables_dispatch_fp8_to_the_clamp_target(self):
        # Dropping the native variant leaves those tables' FP8 slots to be
        # filled: they must land on the unroll-8 kernel the runtime clamp picks,
        # not drift to the func-id axis (unroll 1 in a fat build).
        for unroll in ("16", "32"):
            rows = _extract_table_rows(self.multiarch_header, unroll)
            self.assertIsNotNone(rows, "no ncclDevFuncTable_%s" % unroll)
            fp8_rows = [sym for sym in rows.values()
                        if "f8e4m3" in sym or "f8e5m2" in sym]
            self.assertTrue(fp8_rows, "table %s has no FP8 rows" % unroll)
            for sym in fp8_rows:
                self.assertTrue(sym.endswith("_8"),
                                "table %s dispatches FP8 to %s" % (unroll, sym))

    def test_alias_log_reports_the_fp8_redirects(self):
        # install.sh surfaces this log, so a silently-empty one would hide the
        # redirect from anyone auditing a build.
        with open(os.path.join(self.multiarch, "unroll_table_aliases.log")) as f:
            log = f.read()
        for unroll in ("16", "32"):
            self.assertIn("ncclDevKernel_Generic_%s" % unroll, log)
        self.assertRegex(log, r"f8e4m3.*-> \S+_8 \(compiled at unroll 8\)")


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

    def test_builtin_clamp_also_applies_on_multiarch(self):
        # gfx1250's unroll 8/16/32 tier is compiled exclusively for gfx1250
        # (see _EXCLUSIVE_UNROLL_TIERS / get_arch_guard()) regardless of build
        # mode, so no OTHER arch sharing a multi-arch/fat binary could ever
        # have dispatched to the redundant 16/32 native variants either --
        # skipping them is just as safe here as in a gfx1250-local build (see
        # dispatch_branches_for_unroll_table()/maybe_remap_unroll()). Multi-arch
        # func_id_unroll is "1", not "8", but that's a different (shared,
        # always-generated) tier and doesn't change this.
        with tempfile.TemporaryDirectory(prefix="rccl_unroll_clamp_ma_") as tmpdir:
            _, gensrc = _generate(
                tmpdir,
                local_gpu_only="OFF",
                only_funcs=self.MINMAX_U8_ONLY,
            )
            names = sorted(os.listdir(os.path.join(gensrc, "specialized")))
        u8 = [n for n in names if "minmax_u8_1_0_" in n]
        self.assertTrue(any(n.endswith("_8.cpp") for n in u8), names)
        self.assertFalse(any(n.endswith("_16.cpp") for n in u8), names)
        self.assertFalse(any(n.endswith("_32.cpp") for n in u8), names)
        # The *shared* tier (1/2/4) is used by every OTHER arch in this same
        # fat binary and must NOT be skipped just because gfx1250 has an
        # override -- only gfx1250's own private 8/16/32 tier is exclusive.
        self.assertTrue(any(n.endswith("_1.cpp") for n in u8), names)
        self.assertTrue(any(n.endswith("_2.cpp") for n in u8), names)
        self.assertTrue(any(n.endswith("_4.cpp") for n in u8), names)

    def test_multiarch_clamp_redirects_unconditionally_in_exclusive_tier(self):
        # gfx1250's 8/16/32 tier is exclusively its own even in a multi-arch
        # build (no other arch's compile pass ever reaches a real entry
        # there), so the redirect needs no #if defined(__gfxN__) branch of
        # its own -- it's guarded the exact same way an unoverridden kernel
        # at that unroll would be (see get_arch_guard()).
        with tempfile.TemporaryDirectory(prefix="rccl_unroll_clamp_ma2_") as tmpdir:
            header, _ = _generate(
                tmpdir,
                local_gpu_only="OFF",
                only_funcs=self.MINMAX_U8_ONLY,
            )
        t8 = _extract_table_rows(header, "8")
        t16 = _extract_table_rows(header, "16")
        t32 = _extract_table_rows(header, "32")
        clamp_idx = next(i for i, sym in t8.items() if sym.endswith("_MinMax_u8_1_0_8"))
        self.assertEqual(t8[clamp_idx], t16[clamp_idx])
        self.assertEqual(t8[clamp_idx], t32[clamp_idx])
        # No stray per-arch branch: the redirect appears as a normal
        # single-candidate guarded row, not wrapped in an extra
        # `#if defined(__gfx1250__)` selector on top of get_arch_guard()'s own.
        self.assertNotIn(
            "#if defined(__gfx1250__)\n#if defined(__gfx1250__)", header)

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


class Gfx950LocalArchOverrideTest(unittest.TestCase):
    """The override mechanism must keep working for a true single-local-arch
    (-l) build targeting gfx950, not just gfx1250: both take the same
    local_gfx_name-driven "exactly one arch is ever compiled" branch of
    dispatch_branches_for_unroll_table()/maybe_remap_unroll()/
    forced_override_funcs() (see MultiArchOverrideTest for the *multi*-arch,
    gfx950-alongside-gfx1250 scenario). But gfx950's native unroll set (1/2)
    is a *shared* tier (unlike gfx1250's private 8/16/32 -- see
    _EXCLUSIVE_UNROLL_TIERS), so this exercises the single-arch code path
    against a different tier shape rather than assuming gfx1250's tests
    (all of which run on the 8/16/32 tier) generalize.

    Exercises this via a throwaway "gfx950" entry patched into
    _UNROLL_OVERRIDES in a copy of generate.py (the real table has no gfx950
    entries yet), mirroring ArbitraryUnrollOverrideTest's approach for
    gfx1250.
    """

    EXTRA_IDENTITY = ("AllReduce", "RING", "SIMPLE", "Sum", "f32", "0", "0")
    EXTRA_UNROLL = "4"  # not in gfx950-local's local_unroll (1/2)
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
        extra = (
            '  "gfx950": {\n'
            "    UnrollOverride(%s),\n"
            "  },\n"
        ) % ", ".join(repr(x) for x in self.EXTRA_IDENTITY + (self.EXTRA_UNROLL,))
        patched_path = os.path.join(tmpdir, "generate_patched.py")
        with open(patched_path, "w") as f:
            f.write(src.replace(marker, extra + marker, 1))
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
                    echo "Name:                    gfx950"
                    echo "Compute Unit:            256"
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
        with tempfile.TemporaryDirectory(prefix="rccl_gfx950_override_") as tmpdir:
            result, gensrc = self._generate(tmpdir)
            names = sorted(os.listdir(os.path.join(gensrc, "specialized")))
        overridden = [n for n in names if "_sum_f32_0_0_" in n]
        # unroll 4 (the override target) is force-compiled even though it's
        # not in gfx950-local's local_unroll (1/2).
        self.assertTrue(any(n.endswith("_4.cpp") for n in overridden), overridden)
        # unroll 1 (func_id_unroll for gfx950-local) still anchors this
        # identity's funcId/host_table bookkeeping, but every generic
        # kernel's dispatch table is folded onto unroll 4 instead (see
        # test_generic_table_folds_overridden_identity), so 2 is redundant
        # and skipped -- same skip privilege as a gfx1250-local build gets,
        # just exercised against gfx950's shared (1/2) tier instead.
        self.assertTrue(any(n.endswith("_1.cpp") for n in overridden), overridden)
        self.assertFalse(any(n.endswith("_2.cpp") for n in overridden), overridden)
        self.assertIn("Unroll override (gfx950)", result.stdout)
        self.assertIn("unroll 4", result.stdout)

    def test_generic_table_folds_overridden_identity(self):
        with tempfile.TemporaryDirectory(prefix="rccl_gfx950_override_table_") as tmpdir:
            _, gensrc = self._generate(tmpdir)
            with open(os.path.join(gensrc, "device_table.h")) as f:
                header = f.read()
        rows = _extract_table_rows(header, "1")
        self.assertIsNotNone(rows, "table_1 missing")
        idx = next(i for i, sym in rows.items() if sym.endswith("_Sum_f32_0_0_4"))
        self.assertTrue(rows[idx].endswith("_Sum_f32_0_0_4"))
        # A true single-local-arch build applies its override unconditionally
        # (exactly one arch is ever compiled) -- no per-arch #if/#elif
        # selector should be emitted anywhere, unlike the multi-arch/shared-
        # tier case in MultiArchOverrideTest. #elif is only ever emitted for
        # a multi-branch (multi-arch, shared-tier) result, so its absence
        # here is a precise signal (a bare "defined(__gfx950__)" substring
        # search would also match acc=1 kernels' unrelated own arch guard).
        self.assertNotIn("#elif defined(__gfx", header)

    def test_unrelated_identity_is_unaffected(self):
        # Only the overridden acc=0 identity is redirected; the acc=1 variant
        # of the same coll/algo/proto/redop/ty keeps its native (1/2) unroll set.
        with tempfile.TemporaryDirectory(prefix="rccl_gfx950_override_unrel_") as tmpdir:
            _, gensrc = self._generate(tmpdir)
            names = sorted(os.listdir(os.path.join(gensrc, "specialized")))
        native = [n for n in names if "_sum_f32_1_0_" in n]
        self.assertTrue(any(n.endswith("_1.cpp") for n in native), names)
        self.assertTrue(any(n.endswith("_2.cpp") for n in native), names)


class MultiArchOverrideTest(unittest.TestCase):
    """Unroll overrides used to only take effect in a true single-local-arch
    (-l) build: local_gfx_name gated every lookup, so a multi-arch/fat build
    (e.g. GPU_TARGETS listing several archs from a dev box that's itself a
    different arch) silently ignored the whole _UNROLL_OVERRIDES table.

    generate.py is never told the actual GPU_TARGETS list (that's a
    CMake-level concept), so overrides for every gfx key in _UNROLL_OVERRIDES
    must now apply unconditionally in that mode -- each guarded by its own
    `#if defined(__gfxN__)` branch, which is simply never taken during some
    OTHER arch's compile pass. This exercises that with a throwaway "gfx950"
    entry patched in alongside the real (built-in) "gfx1250" set, so a single
    multi-arch generate.py run has to keep BOTH archs' overrides intact at
    once -- the user-facing scenario of building gfx950 + gfx1250 together
    from a gfx942 dev system.
    """

    # Shared tier (1/2/4): gfx950 and every other non-gfx1250 arch all read
    # the same ncclDevFuncTable_1/2/4 rows, so this needs a real per-arch
    # branch (unlike gfx1250's private 8/16/32 tier).
    GFX950_IDENTITY = ("AllReduce", "RING", "SIMPLE", "Sum", "f32", "0", "0")
    GFX950_OVERRIDE_UNROLL = "1"
    # A real (already-shipped) gfx1250 override, so this test also proves the
    # built-in table keeps working unmodified alongside a second arch's.
    GFX1250_OVERRIDDEN_SYMBOL_SUFFIX = "_MinMax_u8_1_0_8"
    ONLY_FUNCS = (
        "AllReduce RING SIMPLE Sum f32|"
        "AllReduce TREE SIMPLE MinMax u8|AllReduce TREE SIMPLE MinMax i8"
    )

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")

    def _patch_generate_py(self, tmpdir, override_unroll):
        with open(GENERATE_PY) as f:
            src = f.read()
        marker = '"gfx1250": {\n'
        self.assertIn(marker, src, "generate.py layout changed; update this test")
        extra = (
            '  "gfx950": {\n'
            "    UnrollOverride(%s),\n"
            "  },\n"
        ) % ", ".join(repr(x) for x in self.GFX950_IDENTITY + (override_unroll,))
        patched_path = os.path.join(tmpdir, "generate_patched.py")
        with open(patched_path, "w") as f:
            f.write(src.replace(marker, extra + marker, 1))
        return patched_path

    def _generate(self, tmpdir, override_unroll=None):
        if override_unroll is None:
            override_unroll = self.GFX950_OVERRIDE_UNROLL
        patched = self._patch_generate_py(tmpdir, override_unroll)
        gensrc = os.path.join(tmpdir, "gensrc")
        # Multi-arch: no -l, no gfx_name/rocminfo -- generate.py isn't told
        # (and doesn't need to know) the real GPU_TARGETS list.
        result = subprocess.run(
            [sys.executable, patched, gensrc, "OFF", "OFF", "OFF", "OFF", self.ONLY_FUNCS],
            capture_output=True, text=True,
        )
        return result, gensrc

    def test_both_archs_overrides_present_in_same_multiarch_build(self):
        with tempfile.TemporaryDirectory(prefix="rccl_multiarch_override_") as tmpdir:
            result, gensrc = self._generate(tmpdir)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(gensrc, "device_table.h")) as f:
                header = f.read()

        # gfx950's shared-tier override: a real per-arch branch, since gfx942
        # (and every other non-overridden arch) still needs its own native
        # row in this same table.
        t2 = _extract_table_rows(header, "2")
        self.assertTrue(any(sym.endswith("_Sum_f32_0_0_1") for sym in t2.values()))
        self.assertIn("defined(__gfx950__)", header)

        # gfx1250's built-in override (unaffected by the gfx950 patch): its
        # private 8/16/32 tier needs no branch of its own.
        t8 = _extract_table_rows(header, "8")
        t16 = _extract_table_rows(header, "16")
        minmax_idx = next(
            i for i, sym in t8.items() if sym.endswith(self.GFX1250_OVERRIDDEN_SYMBOL_SUFFIX)
        )
        self.assertEqual(t8[minmax_idx], t16[minmax_idx])

    def test_shared_tier_override_branch_structure(self):
        with tempfile.TemporaryDirectory(prefix="rccl_multiarch_override_struct_") as tmpdir:
            result, gensrc = self._generate(tmpdir)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(gensrc, "device_table.h")) as f:
                header = f.read()
        m = re.search(
            r"#if defined\(__gfx950__\)\n"
            r"/\*\s*\d+\*/ ncclDevFunc_AllReduce_RING_SIMPLE_Sum_f32_0_0_1,\n"
            r"#else\n"
            r"/\*\s*\d+\*/ ncclDevFunc_AllReduce_RING_SIMPLE_Sum_f32_0_0_2,\n"
            r"#endif\n",
            header,
        )
        self.assertIsNotNone(
            m, "expected an unconditional #if defined(__gfx950__)/#else branch "
            "redirecting only gfx950 to the unroll-1 override, falling back to "
            "the native unroll-2 row for every other arch"
        )
        # No redundant nested guard: get_arch_guard() of a plain acc=0 kernel
        # is None, so the branch content must not re-wrap itself in another
        # #if for the same (or any) condition.
        self.assertNotIn(
            "#if defined(__gfx950__)\n#if", header,
            "override branch should not re-guard a kernel with no arch_guard of its own",
        )

    def test_unrelated_identity_still_unbranched(self):
        # Only the specifically-overridden identities get a per-arch branch;
        # everything else keeps its plain single-candidate row.
        with tempfile.TemporaryDirectory(prefix="rccl_multiarch_override_unrel_") as tmpdir:
            result, gensrc = self._generate(tmpdir)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(gensrc, "device_table.h")) as f:
                header = f.read()
        t2 = _extract_table_rows(header, "2")
        acc1_idx = next(i for i, sym in t2.items() if sym.endswith("_Sum_f32_1_0_2"))
        # acc=1 Sum f32 has no override for either arch: plain get_arch_guard()
        # gating only, no "defined(__gfx950__)"/"defined(__gfx1250__)" selector.
        self.assertNotIn("gfx950", t2[acc1_idx])

    def test_shared_tier_keeps_native_variant_for_other_archs(self):
        # Unlike gfx1250's exclusive 8/16/32 tier, a shared-tier (1/2/4)
        # override must NOT skip compiling the redundant native variant --
        # gfx942 (and everyone else sharing this table) still needs it.
        with tempfile.TemporaryDirectory(prefix="rccl_multiarch_override_native_") as tmpdir:
            result, gensrc = self._generate(tmpdir)
            self.assertEqual(result.returncode, 0, result.stderr)
            names = sorted(os.listdir(os.path.join(gensrc, "specialized")))
        f32_acc0 = [n for n in names if "_sum_f32_0_0_" in n]
        self.assertTrue(any(n.endswith("_1.cpp") for n in f32_acc0), f32_acc0)
        self.assertTrue(any(n.endswith("_2.cpp") for n in f32_acc0), f32_acc0)

    def test_host_table_complete_with_two_active_override_archs(self):
        with tempfile.TemporaryDirectory(prefix="rccl_multiarch_override_host_") as tmpdir:
            result, gensrc = self._generate(tmpdir)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(gensrc, "device_table.h")) as f:
                header = f.read()
            with open(os.path.join(gensrc, "host_table.cpp")) as f:
                host_table = f.read()
        m = re.search(r"ncclDevFuncTable_1\[\] = \{(.*?)\nnullptr\};", header, re.S)
        device_funcids = {int(i) for i in re.findall(r"/\*\s*(\d+)\s*\*/", m.group(1))}
        host_funcids = {
            int(fid) for _, fid in re.findall(r"\{(\d+),\s*(-?\d+)\}", host_table)
            if int(fid) >= 0
        }
        missing = sorted(device_funcids - host_funcids)
        self.assertEqual(missing, [])

    def test_incompatible_override_arch_rejected_at_generate_time(self):
        # An override for a SHARED-tier arch (gfx950) that targets an
        # EXCLUSIVE-tier unroll (8, compiled only under defined(__gfx1250__))
        # would reference an undeclared symbol during the gfx950 compile
        # pass. build_unroll_override_index() must catch this at generate.py
        # load time with a clear message, not let it become a silent
        # runtime trap/null-dispatch on real gfx950 hardware.
        with tempfile.TemporaryDirectory(prefix="rccl_multiarch_override_bad_") as tmpdir:
            result, _ = self._generate(tmpdir, override_unroll="8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AssertionError", result.stderr)
        self.assertIn("compiled exclusively for gfx1250", result.stderr)

    @unittest.skipUnless(shutil.which("cpp"), "no C preprocessor available")
    def test_cpp_preprocessing_selects_correct_branch_per_arch(self):
        # Belt-and-suspenders: actually run the generated header through a C
        # preprocessor under each arch's macro (mirroring one hipcc
        # --offload-arch compile pass) and check the winning row, instead of
        # just trusting the generator's own bookkeeping.
        with tempfile.TemporaryDirectory(prefix="rccl_multiarch_override_cpp_") as tmpdir:
            result, gensrc = self._generate(tmpdir)
            self.assertEqual(result.returncode, 0, result.stderr)
            header_path = os.path.join(gensrc, "device_table.h")

            def winning_row(unroll_table, funcid_symbol_suffix, arch_macro):
                cmd = [
                    "cpp", "-P", "-DUSE_INDIRECT_FUNCTION_CALL", header_path,
                ]
                if arch_macro:
                    cmd.insert(1, "-D%s" % arch_macro)
                out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
                m = re.search(
                    r"ncclDevFuncTable_%s\[\] = \{(.*?)nullptr\};" % unroll_table, out, re.S)
                rows = [r.strip().rstrip(",") for r in m.group(1).splitlines() if r.strip()]
                match = [r for r in rows if funcid_symbol_suffix in r]
                return match[0] if match else None

            # ncclDevFuncTable_2 (the identity's own native/shared row): gfx950
            # is redirected to the unroll-1 override; every other arch (gfx942,
            # gfx1250, ...) keeps the plain native unroll-2 row.
            self.assertEqual(
                winning_row("2", "_Sum_f32_0_0_", "__gfx950__"),
                "ncclDevFunc_AllReduce_RING_SIMPLE_Sum_f32_0_0_1",
            )
            self.assertEqual(
                winning_row("2", "_Sum_f32_0_0_", "__gfx942__"),
                "ncclDevFunc_AllReduce_RING_SIMPLE_Sum_f32_0_0_2",
            )
            self.assertEqual(
                winning_row("2", "_Sum_f32_0_0_", "__gfx1250__"),
                "ncclDevFunc_AllReduce_RING_SIMPLE_Sum_f32_0_0_2",
            )
            # ncclDevFuncTable_1 IS the override's own target unroll: every
            # arch (overridden or not) resolves to the same unroll-1 row here.
            self.assertEqual(
                winning_row("1", "_Sum_f32_0_0_", "__gfx950__"),
                "ncclDevFunc_AllReduce_RING_SIMPLE_Sum_f32_0_0_1",
            )
            self.assertEqual(
                winning_row("1", "_Sum_f32_0_0_", "__gfx942__"),
                "ncclDevFunc_AllReduce_RING_SIMPLE_Sum_f32_0_0_1",
            )


class LocalArchUnrollSelectionTest(unittest.TestCase):
    """calc_unroll_and_pipeline_for_local_arch() maps the rocminfo-detected
    agent set to this build's unroll/pipeline set.

    Every arch branch other than gfx1250's was previously unexercised, as was
    the "not exactly one gfx target" fallback -- which is the branch that
    leaves local_gfx_name None and therefore switches the whole unroll-override
    mechanism into its multi-arch mode (see override_archs). A regression here
    silently changes which kernels get built for every non-gfx1250 target.
    """

    ONLY_FUNCS = "AllReduce RING SIMPLE Sum f32"
    ALL_UNROLLS = ["1", "16", "2", "32", "4", "8"]  # sorted as strings

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")

    # Pipelining is only ever generated for the types in `pipelined_types`, so
    # observing local_pipeline at all requires a bf16 kernel: with f32 the
    # pipeline field is 0 no matter what local_pipeline says.
    PIPELINED_ONLY_FUNCS = "AllReduce RING SIMPLE Sum bf16"

    def _unrolls_and_pipelines(self, agents, only_funcs=None, ty="f32"):
        """Return (sorted unroll suffixes, sorted pipeline fields) of the
        generated specialized kernels for a --local_gpu_only build on `agents`."""
        with tempfile.TemporaryDirectory(prefix="rccl_local_arch_") as tmpdir:
            _, gensrc = _generate(
                tmpdir,
                local_gpu_only="ON",
                only_funcs=self.ONLY_FUNCS if only_funcs is None else only_funcs,
                gfx_name=agents,
            )
            names = sorted(os.listdir(os.path.join(gensrc, "specialized")))
        unrolls, pipelines = set(), set()
        for name in names:
            m = re.search(r"_sum_%s_(\d+)_(\d+)_(\d+)\.cpp$" % ty, name)
            self.assertIsNotNone(m, name)
            pipelines.add(m.group(2))
            unrolls.add(m.group(3))
        return sorted(unrolls), sorted(pipelines)

    def test_gfx950_uses_unroll_1_2(self):
        unrolls, _ = self._unrolls_and_pipelines([("gfx950", 256)])
        self.assertEqual(["1", "2"], unrolls)

    def test_gfx950_disables_pipelining(self):
        _, pipelines = self._unrolls_and_pipelines(
            [("gfx950", 256)], only_funcs=self.PIPELINED_ONLY_FUNCS, ty="bf16")
        self.assertEqual(["0"], pipelines)

    def test_other_archs_keep_pipelining(self):
        # Counterpart to the gfx950 case: proves the ["0"] above is gfx950's
        # own disable, not just an artifact of the kernel selection.
        for agent in (("gfx942", 304), ("gfx1250", 304)):
            _, pipelines = self._unrolls_and_pipelines(
                [agent], only_funcs=self.PIPELINED_ONLY_FUNCS, ty="bf16")
            self.assertEqual(["0", "1"], pipelines, agent[0])

    def test_gfx908_uses_unroll_2(self):
        unrolls, _ = self._unrolls_and_pipelines([("gfx908", 120)])
        self.assertEqual(["2"], unrolls)

    def test_gfx942_above_80_cu_uses_unroll_2(self):
        unrolls, _ = self._unrolls_and_pipelines([("gfx942", 304)])
        self.assertEqual(["2"], unrolls)

    def test_gfx942_at_80_cu_falls_through_to_unroll_4(self):
        # The condition is `cu_count > 80`, so 80 itself is NOT the >80 tier.
        # Pins the boundary: an off-by-one here changes the shipped kernel set
        # for small-partition (SPX/CPX) gfx942 configurations.
        unrolls, _ = self._unrolls_and_pipelines([("gfx942", 80)])
        self.assertEqual(["4"], unrolls)
        unrolls, _ = self._unrolls_and_pipelines([("gfx942", 81)])
        self.assertEqual(["2"], unrolls)

    def test_gfx1250_uses_unroll_8_16_32(self):
        unrolls, _ = self._unrolls_and_pipelines([("gfx1250", 304)])
        self.assertEqual(["16", "32", "8"], unrolls)

    def test_unlisted_arch_defaults_to_unroll_4(self):
        unrolls, _ = self._unrolls_and_pipelines([("gfx90a", 104)])
        self.assertEqual(["4"], unrolls)

    def test_homogeneous_multi_gpu_is_still_a_single_arch_build(self):
        # Several identical agents dedupe to one (gfx_name, cu_count) key.
        unrolls, _ = self._unrolls_and_pipelines([("gfx942", 304)] * 8)
        self.assertEqual(["2"], unrolls)

    def test_heterogeneous_archs_fall_back_to_all_unrolls(self):
        unrolls, _ = self._unrolls_and_pipelines(
            [("gfx942", 304), ("gfx1250", 304)])
        self.assertEqual(self.ALL_UNROLLS, unrolls)

    def test_same_arch_with_differing_cu_counts_is_heterogeneous(self):
        # gfx_targets is keyed by (name, cu_count), so one gfx942 partitioned
        # differently from another counts as two targets and must not be
        # collapsed into a single-arch build.
        unrolls, _ = self._unrolls_and_pipelines(
            [("gfx942", 304), ("gfx942", 38)])
        self.assertEqual(self.ALL_UNROLLS, unrolls)

    def test_no_gpu_detected_falls_back_to_all_unrolls(self):
        # rocminfo listing no gfx agent at all (CPU-only / no driver) must not
        # crash or emit an empty kernel set -- it degrades to the full build.
        unrolls, _ = self._unrolls_and_pipelines([])
        self.assertEqual(self.ALL_UNROLLS, unrolls)

    def _unrolls_without_rocminfo(self, rocm_path):
        """Run a --local_gpu_only build whose ROCM_PATH has no usable rocminfo."""
        env = os.environ.copy()
        if rocm_path is None:
            env.pop("ROCM_PATH", None)
        else:
            env["ROCM_PATH"] = rocm_path
        with tempfile.TemporaryDirectory(prefix="rccl_no_rocminfo_") as tmpdir:
            gensrc = os.path.join(tmpdir, "gensrc")
            result = subprocess.run(
                [sys.executable, GENERATE_PY, gensrc, "OFF", "OFF", "ON", "OFF",
                 self.ONLY_FUNCS],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(
                0, result.returncode,
                "a --local_gpu_only build must not fail when rocminfo is "
                "unavailable:\n" + result.stderr)
            names = sorted(os.listdir(os.path.join(gensrc, "specialized")))
        unrolls = {re.search(r"_(\d+)\.cpp$", n).group(1) for n in names}
        return sorted(unrolls), result.stderr

    def test_unset_rocm_path_warns_and_builds_all_unrolls(self):
        # CMake invokes generate.py with the ambient environment and does not
        # set ROCM_PATH, so `-l` from a shell without it used to abort configure
        # with a bare TypeError from string-concatenating None.
        unrolls, stderr = self._unrolls_without_rocminfo(None)
        self.assertEqual(self.ALL_UNROLLS, unrolls)
        self.assertIn("WARNING", stderr)

    def test_missing_rocminfo_binary_warns_and_builds_all_unrolls(self):
        with tempfile.TemporaryDirectory(prefix="rccl_empty_rocm_") as rocm:
            unrolls, stderr = self._unrolls_without_rocminfo(rocm)
        self.assertEqual(self.ALL_UNROLLS, unrolls)
        self.assertIn("rocminfo", stderr)


class UnrollAliasLogTest(unittest.TestCase):
    """gensrc/unroll_table_aliases.log is a build-facing artifact: install.sh
    reads it (`-s` test) and prints either its path or "none". Nothing
    previously asserted its content or its emptiness contract.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")

    def _alias_log(self, **kwargs):
        with tempfile.TemporaryDirectory(prefix="rccl_alias_log_") as tmpdir:
            _, gensrc = _generate(tmpdir, **kwargs)
            path = os.path.join(gensrc, "unroll_table_aliases.log")
            self.assertTrue(os.path.isfile(path), "alias log was not written")
            with open(path) as f:
                return f.read()

    def test_override_is_reported_for_each_higher_unroll_table(self):
        body = self._alias_log(
            local_gpu_only="ON",
            only_funcs="AllReduce TREE SIMPLE MinMax u8|AllReduce TREE SIMPLE MinMax i8",
            gfx_name="gfx1250",
        )
        # Tables 16/32 have no native variant for the overridden identity, so
        # both redirect to the unroll-8 kernel and both must say so.
        self.assertIn("ncclDevKernel_Generic_16:", body)
        self.assertIn("ncclDevKernel_Generic_32:", body)
        # Table 8 IS the override target, so nothing is redirected there.
        self.assertNotIn("ncclDevKernel_Generic_8:", body)
        self.assertIn("AllReduce TREE SIMPLE MinMax u8", body)
        self.assertIn(
            "-> ncclDevFunc_AllReduce_TREE_SIMPLE_MinMax_u8_1_0_8 "
            "(compiled at unroll 8)", body)

    def test_missing_native_variant_is_reported_even_without_an_override(self):
        # FP8 on gfx1250 is generated only at unroll 8 (func_validate), with no
        # _UNROLL_OVERRIDES entry involved. Tables 16/32 still fall back to the
        # func-id-axis kernel, and that redirect must be logged too -- it is the
        # same "this table does not use its native variant" fact.
        body = self._alias_log(
            local_gpu_only="ON",
            only_funcs="AllReduce RING SIMPLE Sum f8e4m3",
            gfx_name="gfx1250",
        )
        self.assertIn("ncclDevKernel_Generic_16:", body)
        self.assertIn("f8e4m3", body)
        self.assertIn("(compiled at unroll 8)", body)

    def test_log_is_empty_when_nothing_is_redirected(self):
        # install.sh distinguishes "no clamps" from "clamps happened" purely by
        # `[[ -s $log ]]`, so a build with no redirects must leave the file
        # genuinely zero-length, not containing a blank line or a header.
        body = self._alias_log(
            local_gpu_only="ON",
            only_funcs="AllReduce RING SIMPLE Sum f32",
            gfx_name="gfx950",
        )
        self.assertEqual("", body)

    def test_reported_symbols_are_declared_in_device_table(self):
        # A redirect naming a symbol that was never generated would be a
        # silently wrong log; cross-check every reported target against the
        # header's forward declarations.
        with tempfile.TemporaryDirectory(prefix="rccl_alias_log_sym_") as tmpdir:
            header, gensrc = _generate(
                tmpdir, local_gpu_only="ON", only_funcs="", gfx_name="gfx1250")
            with open(os.path.join(gensrc, "unroll_table_aliases.log")) as f:
                body = f.read()
        declared = set(re.findall(r"__device__ void (ncclDevFunc_\w+)\(\);", header))
        reported = set(re.findall(r"-> (ncclDevFunc_\w+) ", body))
        self.assertTrue(reported, "expected a gfx1250 build to log redirects")
        self.assertEqual(set(), reported - declared)


class TwoArchOverrideSameIdentityTest(unittest.TestCase):
    """Two archs overriding the SAME identity to different shared-tier unrolls.

    This is the only shape that produces a three-way dispatch chain
    (#if/#elif/#else). The shipped table has a single arch, and
    MultiArchOverrideTest's patched second arch targets a different identity,
    so the #elif arm of both emitters (function-pointer table and pure-RDC
    Caller specializations) was never generated. A malformed #elif breaks the
    build for every arch at once.
    """

    IDENTITY = ("AllReduce", "RING", "SIMPLE", "Sum", "f32", "0", "0")
    ONLY_FUNCS = "AllReduce RING SIMPLE Sum f32"

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")

    def _generate_two_arch(self, tmpdir):
        with open(GENERATE_PY) as f:
            src = f.read()
        marker = '"gfx1250": {\n'
        self.assertIn(marker, src, "generate.py layout changed; update this test")
        # gfx90a -> unroll 2, gfx950 -> unroll 1; native stays unroll 4.
        extra = ""
        for gfx, unroll in (("gfx90a", "2"), ("gfx950", "1")):
            extra += (
                '  %r: {\n    UnrollOverride(%s),\n  },\n'
                % (gfx, ", ".join(repr(x) for x in self.IDENTITY + (unroll,)))
            )
        patched = os.path.join(tmpdir, "generate_patched.py")
        with open(patched, "w") as f:
            f.write(src.replace(marker, extra + marker, 1))
        gensrc = os.path.join(tmpdir, "gensrc")
        result = subprocess.run(
            [sys.executable, patched, gensrc, "OFF", "OFF", "OFF", "OFF", self.ONLY_FUNCS],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(os.path.join(gensrc, "device_table.h")) as f:
            return f.read()

    def test_function_pointer_table_emits_elif_chain(self):
        with tempfile.TemporaryDirectory(prefix="rccl_two_arch_") as tmpdir:
            header = self._generate_two_arch(tmpdir)
        sym = "ncclDevFunc_AllReduce_RING_SIMPLE_Sum_f32_0_0"
        m = re.search(
            r"#if defined\(__gfx90a__\)\n"
            r"/\*\s*\d+\*/ %s_2,\n"
            r"#elif defined\(__gfx950__\)\n"
            r"/\*\s*\d+\*/ %s_1,\n"
            r"#else\n"
            r"/\*\s*\d+\*/ %s_4,\n"
            r"#endif\n" % (sym, sym, sym),
            header,
        )
        self.assertIsNotNone(
            m,
            "expected a three-way #if/#elif/#else chain in ncclDevFuncTable_4: "
            "gfx90a -> unroll 2, gfx950 -> unroll 1, everyone else -> native unroll 4",
        )

    def test_pure_rdc_caller_emits_elif_chain(self):
        with tempfile.TemporaryDirectory(prefix="rccl_two_arch_rdc_") as tmpdir:
            header = self._generate_two_arch(tmpdir)
        # The Caller specializations are the pure-RDC (--no-device-linker) arm;
        # they must carry the same three-way selection as the pointer table.
        block = re.search(
            r"#if defined\(__gfx90a__\)\n"
            r"template<> struct Caller4<\d+, \d+> \{[^\n]*Sum_f32_0_0_2\(\);[^\n]*\n"
            r"#elif defined\(__gfx950__\)\n"
            r"template<> struct Caller4<\d+, \d+> \{[^\n]*Sum_f32_0_0_1\(\);[^\n]*\n"
            r"#else\n"
            r"template<> struct Caller4<\d+, \d+> \{[^\n]*Sum_f32_0_0_4\(\);[^\n]*\n"
            r"#endif\n",
            header,
        )
        self.assertIsNotNone(
            block, "expected a three-way Caller4 specialization chain")

    @unittest.skipUnless(shutil.which("cpp"), "no C preprocessor available")
    def test_each_arch_preprocesses_to_its_own_override(self):
        with tempfile.TemporaryDirectory(prefix="rccl_two_arch_cpp_") as tmpdir:
            header = self._generate_two_arch(tmpdir)
            header_path = os.path.join(tmpdir, "table.h")
            with open(header_path, "w") as f:
                f.write(header)

            def winning_row(arch_macro):
                out = subprocess.run(
                    ["cpp", "-P", "-D%s" % arch_macro,
                     "-DUSE_INDIRECT_FUNCTION_CALL", header_path],
                    capture_output=True, text=True, check=True).stdout
                m = re.search(
                    r"ncclDevFuncTable_4\[\] = \{(.*?)nullptr\};", out, re.S)
                rows = [r.strip().rstrip(",") for r in m.group(1).splitlines()
                        if "Sum_f32_0_0_" in r]
                self.assertEqual(1, len(rows), rows)
                return rows[0]

            sym = "ncclDevFunc_AllReduce_RING_SIMPLE_Sum_f32_0_0"
            self.assertEqual(sym + "_2", winning_row("__gfx90a__"))
            self.assertEqual(sym + "_1", winning_row("__gfx950__"))
            # Not overridden for this identity: keeps table 4's native row.
            self.assertEqual(sym + "_4", winning_row("__gfx942__"))
            self.assertEqual(sym + "_4", winning_row("__gfx1250__"))


class OnlyFuncsValidationTest(unittest.TestCase):
    """ONLY_FUNCS is a developer-facing build knob; a typo must fail loudly at
    generate time rather than silently building a smaller kernel set."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")

    def _run(self, only_funcs):
        with tempfile.TemporaryDirectory(prefix="rccl_only_funcs_") as tmpdir:
            return subprocess.run(
                [sys.executable, GENERATE_PY, os.path.join(tmpdir, "gensrc"),
                 "OFF", "OFF", "OFF", "OFF", only_funcs],
                capture_output=True, text=True,
            )

    def test_unknown_token_is_rejected(self):
        result = self._run("AllReduce RING SIMPLE Sum notatype")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("notatype", result.stderr)
        self.assertIn("unrecognized", result.stderr)

    def test_token_from_the_wrong_category_is_rejected(self):
        # "TREE" is a valid algo but not a valid proto; the positional check
        # must catch a field swap, not just an unknown word.
        result = self._run("AllReduce RING TREE Sum f32")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("TREE", result.stderr)

    def test_valid_pattern_succeeds(self):
        self.assertEqual(0, self._run("AllReduce RING SIMPLE Sum f32").returncode)


class ExclusiveUnrollTierConsistencyTest(unittest.TestCase):
    """_EXCLUSIVE_UNROLL_TIERS, get_arch_guard() and the specialized_files.txt
    guard column each encode "unroll 8/16/32 belongs to gfx1250". They are
    separate literals in three places, and generate.py's own comment asks for
    them to be kept in sync by hand. This asserts they actually agree, so a
    future arch added to one place cannot silently disagree with the others.

    Also pins _FIXED_UNROLL_TYPES against the same tier table, since the unroll
    it names becomes a dispatch target for tiers the same arch owns.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")
        cls._dir = tempfile.mkdtemp(prefix="rccl_tier_sync_")
        # LL128 must be in the mix: it is the one proto whose specialized_files
        # guard differs from get_arch_guard() (the ENABLE_LL128 qualifier), so a
        # slice without it cannot detect that qualifier going missing. acc=1
        # kernels contribute the non-exclusive arch-guard case.
        cls.mod = _import_generate(cls._dir, only_funcs=(
            "AllReduce RING SIMPLE Sum f32|AllReduce RING LL128 Sum f32"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, ignore_errors=True)

    def test_get_arch_guard_matches_the_tier_table(self):
        mod = self.mod
        for unroll, owner in mod._EXCLUSIVE_UNROLL_TIERS.items():
            fn = mod.Fn("AllReduce", "RING", "SIMPLE", "Sum", "f32", "0", "0",
                        unroll, "0")
            self.assertEqual("defined(__%s__)" % owner, mod.get_arch_guard(fn),
                             "unroll %s" % unroll)

    def test_non_exclusive_unrolls_have_no_arch_guard(self):
        mod = self.mod
        for unroll in mod.all_unrolls:
            if unroll in mod._EXCLUSIVE_UNROLL_TIERS:
                continue
            fn = mod.Fn("AllReduce", "RING", "SIMPLE", "Sum", "f32", "0", "0",
                        unroll, "0")
            self.assertIsNone(mod.get_arch_guard(fn), "unroll %s" % unroll)

    def test_fixed_unroll_targets_a_tier_its_own_arch_owns(self):
        # Same hazard build_unroll_override_index() asserts on for explicit
        # overrides: dispatch_branches_for_unroll_table() redirects an exclusive
        # tier's rows to this unroll, so if the target were a tier some OTHER arch
        # owns, that row would name a symbol guarded out of its own compile pass
        # -- a null dispatch or trap on hardware rather than a build failure.
        mod = self.mod
        for gfx, (_, unroll) in mod._FIXED_UNROLL_TYPES.items():
            self.assertEqual(
                gfx, mod._EXCLUSIVE_UNROLL_TIERS.get(unroll),
                "%s pins its FP8 kernels to unroll %s, which %s owns" % (
                    gfx, unroll,
                    mod._EXCLUSIVE_UNROLL_TIERS.get(unroll) or "no single arch"))

    def test_fixed_unroll_types_are_real_datatypes(self):
        # A typo here would silently constrain nothing at all.
        mod = self.mod
        for gfx, (tys, _) in mod._FIXED_UNROLL_TYPES.items():
            self.assertTrue(tys, "%s constrains no datatypes" % gfx)
            for ty in tys:
                self.assertIn(ty, mod.all_tys, "%s: %s" % (gfx, ty))

    def test_fixed_unroll_unconstrained_without_a_sole_owning_arch(self):
        # A tier several archs share cannot be narrowed to one arch's rule; that
        # is what keeps FP8 alive at unrolls 1/2/4 in a fat build.
        mod = self.mod
        self.assertIsNone(mod.fixed_unroll_for_type(None, "f8e4m3"))
        self.assertIsNone(mod.fixed_unroll_for_type("gfx1250", "f32"))
        self.assertEqual("8", mod.fixed_unroll_for_type("gfx1250", "f8e4m3"))

    def test_specialized_files_guard_column_matches_arch_guard(self):
        # specialized_files.txt is consumed by CMake to decide which kernels a
        # given --offload-arch pass compiles; it must not disagree with the
        # guard the generated source itself carries.
        by_func = {}
        with open(os.path.join(self.mod.gensrc, "specialized_files.txt")) as f:
            for line in f:
                if not line.strip():
                    continue
                # "<filename> <func_name> [<guard>]" -- the guard column is
                # empty for kernels that need no #if at all.
                parts = line.rstrip("\n").split(" ", 2)
                by_func[parts[1]] = parts[2].strip() if len(parts) > 2 else ""
        self.assertTrue(by_func)
        for fn in self.mod.primary_funcs:
            guard = self.mod.get_arch_guard(fn)
            actual = by_func["ncclDevFunc_" + self.mod.fn_sym(fn)]
            if fn.unroll in self.mod._EXCLUSIVE_UNROLL_TIERS:
                owner = self.mod._EXCLUSIVE_UNROLL_TIERS[fn.unroll]
                expected = "defined(__%s__)" % owner
                if fn.proto == "LL128":
                    expected += " && defined(ENABLE_LL128)"
                self.assertEqual(expected, actual, self.mod.fn_sym(fn))
            else:
                self.assertEqual(guard or "", actual, self.mod.fn_sym(fn))


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
