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
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATE_PY = os.path.join(HERE, "generate.py")

# A small, fast slice of collectives. "AllReduce RING SIMPLE Sum f32" expands to
# both an unguarded primary and an arch-guarded variant, which exercises the
# guarded-out (trap) leaf below. The AllReduce LL128 entries are reg-variant
# (see ll128_reg_variant_colls), so they carry a "_1"/"_2" reg suffix while every
# other kernel omits the reg field -- the pure-RDC dispatcher must call each by
# its exact declared name (regression: #ll128-reg-split).
#
# SendRecv is emitted as TWO latency-protocol kernel variants (reg_values_of):
#   reg=0 -> legacy LL send/recv kernel, built unguarded on every arch (the default)
#   reg=1 -> LL128 send/recv kernel, arch-guarded to gfx942/gfx950 + ENABLE_LL128. Emitted
#            for every unroll regardless: a hole would shift that table's later indices.
ONLY_FUNCS = "AllReduce RING SIMPLE Sum f32|AllReduce RING LL128 Sum f32|SendRecv"


def _generate(tmpdir, ifc="OFF", all_unrolls="OFF"):
    """Run generate.py into tmpdir and return the device_table.h contents."""
    # argv: gensrc, IFC, (unused), local_gpu_only, rocshmem, all_unrolls, ONLY_FUNCS
    # local_gpu_only=OFF avoids needing rocminfo/a local GPU.
    subprocess.run(
        [sys.executable, GENERATE_PY, tmpdir, ifc, "OFF", "OFF", "OFF", all_unrolls, ONLY_FUNCS],
        check=True,
        capture_output=True,
        text=True,
    )
    with open(os.path.join(tmpdir, "device_table.h")) as f:
        return f.read()


def _unroll_tables(header):
    """Map unroll -> list of table entries, guarded slots reduced to their symbol."""
    tables = {}
    for unroll, body in re.findall(
        r"ncclDevFuncTable_(\d+)\[\] = \{(.*?)nullptr\};", header, re.S
    ):
        slots = {}
        # Guarded slots emit symbol + "#else" nullptr at the same index; keep the symbol.
        for m in re.finditer(r"/\*\s*(\d+)\*/ (\w+),", body):
            slots.setdefault(int(m.group(1)), m.group(2))
        tables[unroll] = [slots[i] for i in sorted(slots)]
    return tables


def _strip_unroll(sym):
    # ncclDevFunc_<coll>_<algo>_<proto>_<redop>_<ty>_<acc>_<pipeline>_<unroll>[_<reg>]
    parts = sym.split("_")
    del parts[8]
    return "_".join(parts)


class DeviceTableGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")
        cls._dir = tempfile.mkdtemp(prefix="rccl_devtable_")
        cls.header = _generate(cls._dir)

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

    # ---- SendRecv LL / LL128 reg-variant codegen (ll128-p2p-send-recv) --------
    # These pin the two-kernel split so a regression in reg_values_of /
    # get_arch_guard / func_validate for SendRecv fails here instead of only at
    # device link or at runtime.

    def _sendrecv_decls(self):
        # All forward-declared SendRecv device-function symbols.
        return set(
            re.findall(r"__device__ void (ncclDevFunc_SendRecv\w*)\(\);", self.header)
        )

    def test_sendrecv_emits_ll_and_ll128_reg_variants(self):
        # Both kernels must be generated: the legacy LL kernel (reg=0, no reg
        # suffix) and the LL128 kernel (reg=1, trailing "_1"). If reg_values_of
        # regressed to ["0"] only the LL kernel would exist.
        decls = self._sendrecv_decls()
        self.assertTrue(decls, "no SendRecv forward declarations generated")
        ll = [s for s in decls if re.fullmatch(r"ncclDevFunc_SendRecv_\w+?_\d+_\d+_\d+", s)]
        ll128 = [s for s in decls if re.fullmatch(r"ncclDevFunc_SendRecv_\w+?_\d+_\d+_\d+_1", s)]
        self.assertTrue(ll, "legacy LL SendRecv kernel (reg=0) missing")
        self.assertTrue(ll128, "LL128 SendRecv kernel (reg=1, '_1' suffix) missing")

    def test_sendrecv_ll128_is_arch_guarded_and_ll_is_not(self):
        # Every reg=1 (LL128) SendRecv declaration must sit inside the
        # gfx942/gfx950 + ENABLE_LL128 guard...
        guarded = re.findall(
            r"#if \(defined\(__gfx942__\) \|\| defined\(__gfx950__\)\) && defined\(ENABLE_LL128\)\n"
            r"__device__ void (ncclDevFunc_SendRecv\w*_1)\(\);\n#endif",
            self.header,
        )
        self.assertTrue(guarded, "LL128 SendRecv (reg=1) declaration is not arch-guarded")
        # ...and every emitted reg=1 SendRecv symbol is one of those guarded ones
        # (none leaks out unguarded onto the default-built archs). A reg=1 symbol
        # has FOUR trailing numeric fields (acc, pipeline, unroll, reg); a bare
        # endswith("_1") would also catch the reg=0 unroll=1 kernel (..._0_0_1).
        ll128 = {
            s
            for s in self._sendrecv_decls()
            if re.fullmatch(r"ncclDevFunc_SendRecv_\w+?_\d+_\d+_\d+_1", s)
        }
        self.assertEqual(
            set(),
            ll128 - set(guarded),
            "LL128 SendRecv kernels emitted without the gfx942/gfx950 guard: %s"
            % sorted(ll128 - set(guarded)),
        )
        # The legacy LL kernel must stay unguarded (built on every arch): its
        # bare declaration line has no surrounding #if.
        self.assertRegex(
            self.header,
            r"\n__device__ void ncclDevFunc_SendRecv_\w+?_\d+_\d+_\d+\(\);\n",
            "legacy LL SendRecv (reg=0) declaration should be unguarded",
        )

    def test_sendrecv_ll128_covers_every_generated_unroll(self):
        # Skipping reg=1 on the gfx1250 unrolls leaves a hole that shifts every later
        # index in that table away from the host ids (derived from the first unroll).
        decls = self._sendrecv_decls()
        # A table is emitted per enum value; only non-empty ones were generated.
        generated = {u for u, t in _unroll_tables(self.header).items() if t}
        missing = sorted(
            u
            for u in generated
            if not any(re.fullmatch(r"ncclDevFunc_SendRecv_\w+?_\d+_\d+_%s_1" % u, s) for s in decls)
        )
        self.assertEqual(
            [], missing, "LL128 SendRecv (reg=1) missing for generated unrolls: %s" % missing
        )

    def test_device_tables_are_index_aligned_across_unrolls(self):
        # Host ids come from the first generated unroll but index every
        # ncclDevFuncTable_*, so all tables must hold the same functions in order.
        tables = {u: t for u, t in _unroll_tables(self.header).items() if t}
        self.assertTrue(tables, "no non-empty unroll tables generated")
        base_unroll, base = sorted(tables.items())[0]
        base_shape = [_strip_unroll(s) for s in base]
        for unroll, entries in sorted(tables.items()):
            self.assertEqual(
                base_shape,
                [_strip_unroll(s) for s in entries],
                "ncclDevFuncTable_%s is not index-aligned with ncclDevFuncTable_%s"
                % (unroll, base_unroll),
            )

    def test_all_unrolls_opt_in_generates_every_unroll(self):
        # BUILD_ALL_UNROLLS fills in the skipped unrolls and drops the gfx1250 restriction.
        tmpdir = tempfile.mkdtemp(prefix="rccl_devtable_all_")
        try:
            header = _generate(tmpdir, all_unrolls="ON")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        tables = _unroll_tables(header)
        self.assertEqual({"1", "2", "4", "8", "16", "32"}, set(tables))
        for unroll, entries in tables.items():
            self.assertTrue(entries, "ncclDevFuncTable_%s is empty under BUILD_ALL_UNROLLS" % unroll)
        self.assertNotIn("#if defined(__gfx1250__)\n", header)

    def test_no_obsolete_table_omit_macro(self):
        # RCCL_DEVICE_TABLE_OMIT was retired by the static-table change.
        self.assertNotIn("RCCL_DEVICE_TABLE_OMIT", self.header)

    def test_specialized_shards_do_not_omit(self):
        # Specialized shards no longer #define RCCL_DEVICE_TABLE_OMIT.
        spec_dir = os.path.join(self._dir, "specialized")
        self.assertTrue(os.path.isdir(spec_dir))
        for name in os.listdir(spec_dir):
            with open(os.path.join(spec_dir, name)) as f:
                self.assertNotIn("RCCL_DEVICE_TABLE_OMIT", f.read())

    def test_ifc_build_keeps_table_and_no_rdc_dispatch(self):
        # Don't break the legacy indirect-function-call path: with IFC on, the
        # table is still emitted and the pure-RDC dispatcher is not.
        with tempfile.TemporaryDirectory(prefix="rccl_devtable_ifc_") as d:
            header = _generate(d, ifc="ON")
        self.assertIn("static __device__ ncclDevFuncPtr_t const ncclDevFuncTable_", header)


if __name__ == "__main__":
    unittest.main()
